"""Training entry point.

Usage
-----
::

    python -m src.train --config configs/unet.yaml
    python -m src.train --config configs/unet_resnet34.yaml --epochs 30 --batch-size 4
    python -m src.train --dry-run          # two batches, no checkpoints written
    python -m src.train --resume checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim

from .config import Config, load_config
from .dataset import DatasetPairingError, build_dataloaders
from .engine import EarlyStopping, log_epoch, make_grad_scaler, train_one_epoch, validate
from .losses import build_loss
from .model import build_model
from .utils import (
    count_parameters,
    get_device,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    setup_logging,
)

LOGGER = logging.getLogger("src.train")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(
        prog="python -m src.train",
        description="Train a crack-segmentation model on the DeepCrack dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data.data_dir")
    parser.add_argument("--model", type=str, default=None, help="Override model.name")
    parser.add_argument("--loss", type=str, default=None, help="Override loss.name")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override train.lr")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay")
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["cosine", "plateau", "none"],
        help="Override train.scheduler",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--device", type=str, default=None, help="auto | cuda | cpu | mps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument(
        "--experiment-name", type=str, default=None, help="Name used for checkpoints and TB runs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run two batches of train/val for one epoch and exit without saving",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG | INFO | WARNING")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Fold argparse flags into the YAML config."""
    overrides: dict[str, Any] = {
        "data": {
            "data_dir": args.data_dir,
            "num_workers": args.num_workers,
        },
        "model": {"name": args.model},
        "loss": {"name": args.loss},
        "train": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
        },
        "device": args.device,
        "seed": args.seed,
        "experiment_name": args.experiment_name,
    }
    if args.no_amp:
        overrides["train"]["amp"] = False
    if args.no_augment:
        overrides["augment"] = {"enabled": False}
    # Strip empty sections so load_config's merge leaves YAML values intact.
    cleaned = {
        key: ({k: v for k, v in value.items() if v is not None} if isinstance(value, dict) else value)
        for key, value in overrides.items()
    }
    cleaned = {k: v for k, v in cleaned.items() if v not in (None, {})}
    return load_config(args.config, overrides=cleaned)


def build_optimizer(model: nn.Module, cfg: Config) -> optim.Optimizer:
    """AdamW with weight decay disabled for norm layers and biases."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": cfg.train.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return optim.AdamW(groups, lr=cfg.train.lr)


def build_scheduler(optimizer: optim.Optimizer, cfg: Config):
    """Create the LR schedule named in the config (or ``None``)."""
    if cfg.train.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.train.epochs), eta_min=cfg.train.min_lr
        )
    if cfg.train.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=cfg.train.min_lr
        )
    return None


def _make_writer(log_dir: Path):
    """Return a TensorBoard ``SummaryWriter``, or ``None`` when unavailable."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:  # pragma: no cover - tensorboard is an optional extra
        LOGGER.warning("tensorboard not installed; skipping TensorBoard logging")
        return None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(log_dir))
    except OSError as exc:  # pragma: no cover - filesystem dependent
        LOGGER.warning("Could not open TensorBoard log dir %s: %s", log_dir, exc)
        return None


def _apply_warmup(optimizer: optim.Optimizer, cfg: Config, epoch: int) -> None:
    """Linearly ramp the LR over ``train.warmup_epochs`` starting epochs."""
    if cfg.train.warmup_epochs <= 0 or epoch > cfg.train.warmup_epochs:
        return
    factor = epoch / max(1, cfg.train.warmup_epochs)
    for group in optimizer.param_groups:
        group["lr"] = cfg.train.lr * factor


def train(cfg: Config, *, resume: str | None = None, dry_run: bool = False) -> dict[str, float]:
    """Train a model end to end and return the best validation metrics."""
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    LOGGER.info("Experiment %s on %s", cfg.experiment_name, device)

    train_loader, val_loader = build_dataloaders(cfg)
    LOGGER.info(
        "train batches: %d | val batches: %d | image size: %s",
        len(train_loader),
        len(val_loader),
        cfg.data.image_size,
    )

    model = build_model(cfg.model.name, **cfg.model.kwargs()).to(device)
    LOGGER.info(
        "Model %s with %s trainable parameters",
        cfg.model.name,
        f"{count_parameters(model):,}",
    )

    criterion = build_loss(
        cfg.loss.name,
        bce_weight=cfg.loss.bce_weight,
        dice_weight=cfg.loss.dice_weight,
        focal_alpha=cfg.loss.focal_alpha,
        focal_gamma=cfg.loss.focal_gamma,
        tversky_alpha=cfg.loss.tversky_alpha,
        tversky_beta=cfg.loss.tversky_beta,
        pos_weight=cfg.loss.pos_weight,
    ).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    # bf16 autocast on CPU is slower than fp32 on most CPUs, so AMP is CUDA-only.
    amp_enabled = bool(cfg.train.amp and device.type == "cuda")
    scaler = make_grad_scaler(device, amp_enabled)

    start_epoch = 1
    best_dice = 0.0
    if resume:
        payload = load_checkpoint(resume, map_location=device)
        model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and "scheduler_state" in payload:
            try:
                scheduler.load_state_dict(payload["scheduler_state"])
            except (ValueError, KeyError) as exc:
                LOGGER.warning("Could not restore scheduler state: %s", exc)
        if "scaler_state" in payload and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(payload["scaler_state"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_dice = float(payload.get("metrics", {}).get("dice", 0.0))
        LOGGER.info("Resumed from %s at epoch %d (best dice %.4f)", resume, start_epoch, best_dice)

    if dry_run:
        LOGGER.info("Dry run: two batches of train and validation, nothing is saved")
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, amp=amp_enabled, grad_clip=cfg.train.grad_clip,
            epoch=0, max_batches=2,
        )
        val_metrics = validate(model, val_loader, criterion, device, amp=amp_enabled, max_batches=2)
        log_epoch(0, 0, train_metrics, val_metrics, cfg.train.lr)
        return val_metrics

    writer = _make_writer(cfg.train.log_dir / cfg.experiment_name)
    stopper = EarlyStopping(patience=cfg.train.early_stopping_patience, mode="max")
    stopper.best = best_dice
    checkpoint_dir = cfg.train.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_metrics: dict[str, float] = {}

    try:
        for epoch in range(start_epoch, cfg.train.epochs + 1):
            _apply_warmup(optimizer, cfg, epoch)
            current_lr = optimizer.param_groups[0]["lr"]

            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                scaler=scaler, amp=amp_enabled, grad_clip=cfg.train.grad_clip,
                epoch=epoch, threshold=cfg.eval.threshold,
            )
            val_metrics = validate(
                model, val_loader, criterion, device,
                amp=amp_enabled, threshold=cfg.eval.threshold,
            )
            log_epoch(epoch, cfg.train.epochs, train_metrics, val_metrics, current_lr)

            if scheduler is not None and epoch > cfg.train.warmup_epochs:
                if cfg.train.scheduler == "plateau":
                    scheduler.step(val_metrics["dice"])
                else:
                    scheduler.step()

            if writer is not None:
                for split, metrics in (("train", train_metrics), ("val", val_metrics)):
                    for key in ("loss", "dice", "iou", "precision", "recall"):
                        if key in metrics:
                            writer.add_scalar(f"{split}/{key}", metrics[key], epoch)
                writer.add_scalar("lr", current_lr, epoch)

            improved = stopper.step(val_metrics["dice"])
            common: dict[str, Any] = {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "scaler": scaler,
                "epoch": epoch,
                "config": cfg.to_dict(),
            }
            save_checkpoint(checkpoint_dir / "last.pt", metrics=val_metrics, **common)
            if improved:
                best_metrics = dict(val_metrics)
                save_checkpoint(checkpoint_dir / "best.pt", metrics=val_metrics, **common)
                LOGGER.info("New best validation dice: %.4f", val_metrics["dice"])
            if cfg.train.save_every > 0 and epoch % cfg.train.save_every == 0:
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:03d}.pt", metrics=val_metrics, **common
                )
            if stopper.should_stop:
                break
    except KeyboardInterrupt:  # pragma: no cover - interactive
        LOGGER.warning("Interrupted by user; last.pt holds the most recent epoch")
    finally:
        if writer is not None:
            writer.close()

    if best_metrics:
        summary_path = checkpoint_dir / f"{cfg.experiment_name}_best_metrics.json"
        try:
            summary_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - filesystem dependent
            LOGGER.warning("Could not write %s: %s", summary_path, exc)
        LOGGER.info("Best validation dice %.4f (epoch metrics in %s)", stopper.best, summary_path)
    return best_metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        cfg = config_from_args(args)
        train(cfg, resume=args.resume, dry_run=args.dry_run)
    except (FileNotFoundError, DatasetPairingError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except torch.cuda.OutOfMemoryError:  # pragma: no cover - hardware dependent
        LOGGER.error("CUDA out of memory. Lower --batch-size or data.image_size and retry.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
