"""Held-out test-set evaluation with JSON metrics and figures.

Usage
-----
::

    python -m src.evaluate --checkpoint checkpoints/best.pt
    python -m src.evaluate --threshold 0.4 --num-qualitative 8

Outputs
-------
``reports/metrics.json``
    Fixed-threshold metrics, the best-F1 threshold and the full sweep.
``reports/figures/threshold_f1.png``
    Threshold-vs-F1 (and IoU) curve.
``reports/figures/qualitative_grid.png``
    ``image | ground truth | prediction | overlay`` for a handful of samples.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config, load_config, resolve_checkpoint_path
from .dataset import DatasetPairingError, build_test_dataloader
from .engine import validate
from .losses import build_loss
from .metrics import ThresholdSweep, format_metrics
from .model import build_model
from .utils import (
    get_device,
    load_checkpoint,
    mask_to_overlay,
    seed_everything,
    setup_logging,
    tensor_to_uint8_image,
)

LOGGER = logging.getLogger("src.evaluate")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for evaluation."""
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Evaluate a trained crack-segmentation model on the held-out test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to evaluate")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data.data_dir")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default=None, help="auto | cuda | cpu | mps")
    parser.add_argument(
        "--num-qualitative", type=int, default=None, help="Samples in the qualitative grid"
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="Write metrics.json but skip matplotlib figures"
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG | INFO | WARNING")
    return parser


def load_model_from_checkpoint(
    checkpoint_path: Path, cfg: Config, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild the architecture recorded in a checkpoint and load its weights.

    The checkpoint's own ``config`` block wins over *cfg* for the architecture,
    so evaluating a ResNet checkpoint with the default U-Net YAML still works.
    """
    payload = load_checkpoint(checkpoint_path, map_location=device)
    ckpt_model_cfg = dict(payload.get("config", {}).get("model", {}))
    name = ckpt_model_cfg.pop("name", cfg.model.name)
    kwargs = {**cfg.model.kwargs(), **ckpt_model_cfg}
    kwargs["pretrained"] = False  # weights come from the checkpoint

    model = build_model(name, **kwargs)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    if missing or unexpected:
        LOGGER.warning(
            "State dict mismatch: %d missing, %d unexpected keys", len(missing), len(unexpected)
        )
    model.to(device).eval()
    _adopt_input_geometry(cfg, payload.get("config", {}))
    LOGGER.info(
        "Loaded %s from %s (epoch %s)", name, checkpoint_path, payload.get("epoch", "unknown")
    )
    return model, payload


def _adopt_input_geometry(cfg: Config, ckpt_cfg: dict[str, Any]) -> None:
    """Use the input size and tile size the checkpoint was trained with.

    A model trained at 256x256 should be run at 256x256 whichever YAML is
    active at inference time, so these values follow the checkpoint (in place).
    """
    size = ckpt_cfg.get("data", {}).get("image_size")
    if size and tuple(size) != tuple(cfg.data.image_size):
        LOGGER.info("Using the checkpoint's training size %s (config had %s)", list(size), list(cfg.data.image_size))
        cfg.data.image_size = tuple(int(v) for v in size)
    tile = ckpt_cfg.get("inference", {}).get("tile_size")
    if tile:
        cfg.inference.tile_size = int(tile)


def plot_threshold_curve(sweep: ThresholdSweep, output_path: Path) -> Path | None:
    """Save a threshold-vs-F1/IoU curve. Returns the path, or ``None`` on failure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is a declared dependency
        LOGGER.warning("matplotlib not installed; skipping the threshold curve")
        return None

    thresholds, f1_values = sweep.curve("f1")
    _, iou_values = sweep.curve("iou")
    _, precision = sweep.curve("precision")
    _, recall = sweep.curve("recall")
    best_t, best_row = sweep.best_threshold("f1")

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    ax.plot(thresholds, f1_values, marker="o", markersize=3, label="F1 / Dice")
    ax.plot(thresholds, iou_values, marker="s", markersize=3, label="IoU")
    ax.plot(thresholds, precision, linestyle="--", linewidth=1, label="Precision")
    ax.plot(thresholds, recall, linestyle=":", linewidth=1.5, label="Recall")
    ax.axvline(best_t, color="0.4", linewidth=1, linestyle="-.")
    ax.annotate(
        f"best F1 {best_row['f1']:.3f} @ {best_t:.2f}",
        xy=(best_t, best_row["f1"]),
        xytext=(6, -14),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Test-set metrics vs decision threshold")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    except OSError as exc:
        LOGGER.warning("Could not write %s: %s", output_path, exc)
        return None
    finally:
        plt.close(fig)
    LOGGER.info("Wrote %s", output_path)
    return output_path


@torch.no_grad()
def plot_qualitative_grid(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    output_path: Path,
    *,
    threshold: float = 0.5,
    num_samples: int = 6,
) -> Path | None:
    """Save an ``image | ground truth | prediction | overlay`` grid."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        LOGGER.warning("matplotlib not installed; skipping the qualitative grid")
        return None

    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    model.eval()
    for batch in loader:
        images, masks = batch[0], batch[1]
        metas = batch[2] if len(batch) > 2 else [{} for _ in range(images.size(0))]
        probs = torch.sigmoid(model(images.to(device)).float()).cpu()
        for i in range(images.size(0)):
            if len(rows) >= num_samples:
                break
            rgb = tensor_to_uint8_image(images[i])
            truth = masks[i, 0].numpy()
            pred = (probs[i, 0].numpy() >= threshold).astype(np.float32)
            overlay = mask_to_overlay(rgb, pred)
            rows.append((metas[i].get("stem", f"sample_{i}"), rgb, truth, pred, overlay))
        if len(rows) >= num_samples:
            break

    if not rows:
        LOGGER.warning("No samples available for the qualitative grid")
        return None

    titles = ("Image", "Ground truth", "Prediction", "Overlay")
    fig, axes = plt.subplots(len(rows), 4, figsize=(11, 2.8 * len(rows)), dpi=130)
    axes = np.atleast_2d(axes)
    for row_index, (stem, rgb, truth, pred, overlay) in enumerate(rows):
        panels = (rgb, truth, pred, overlay)
        for col_index, panel in enumerate(panels):
            ax = axes[row_index, col_index]
            ax.imshow(panel, cmap=None if panel.ndim == 3 else "gray", vmin=None if panel.ndim == 3 else 0, vmax=None if panel.ndim == 3 else 1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(titles[col_index], fontsize=10)
        axes[row_index, 0].set_ylabel(stem, fontsize=8)
    fig.suptitle(f"Qualitative results (threshold = {threshold:.2f})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    except OSError as exc:
        LOGGER.warning("Could not write %s: %s", output_path, exc)
        return None
    finally:
        plt.close(fig)
    LOGGER.info("Wrote %s", output_path)
    return output_path


def evaluate(
    cfg: Config,
    checkpoint_path: Path,
    *,
    make_figures: bool = True,
) -> dict[str, Any]:
    """Run the full test-set evaluation and write the report artefacts."""
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    model, payload = load_model_from_checkpoint(checkpoint_path, cfg, device)

    loader = build_test_dataloader(cfg)
    criterion = build_loss(
        cfg.loss.name,
        bce_weight=cfg.loss.bce_weight,
        dice_weight=cfg.loss.dice_weight,
        pos_weight=cfg.loss.pos_weight,
    ).to(device)

    start, stop, num = cfg.eval.threshold_sweep
    sweep = ThresholdSweep.from_range(start, stop, num)
    metrics = validate(
        model, loader, criterion, device,
        amp=False, threshold=cfg.eval.threshold, sweep=sweep,
    )
    best_threshold, best_row = sweep.best_threshold("f1")

    LOGGER.info("Test metrics @ %.2f: %s", cfg.eval.threshold, format_metrics(metrics))
    LOGGER.info("Best F1 %.4f at threshold %.2f", best_row["f1"], best_threshold)

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": payload.get("epoch"),
        "model": cfg.model.name,
        "device": str(device),
        "num_test_images": len(loader.dataset),
        "image_size": list(cfg.data.image_size),
        "threshold": cfg.eval.threshold,
        "metrics": metrics,
        "best_threshold": best_threshold,
        "best_threshold_metrics": best_row,
        "threshold_sweep": sweep.compute(),
    }

    reports_dir = cfg.eval.reports_dir
    metrics_path = reports_dir / "metrics.json"
    try:
        reports_dir.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        LOGGER.info("Wrote %s", metrics_path)
    except OSError as exc:
        LOGGER.error("Could not write %s: %s", metrics_path, exc)

    if make_figures:
        plot_threshold_curve(sweep, cfg.eval.figures_dir / "threshold_f1.png")
        plot_qualitative_grid(
            model,
            loader,
            device,
            cfg.eval.figures_dir / "qualitative_grid.png",
            threshold=cfg.eval.threshold,
            num_samples=cfg.eval.num_qualitative,
        )
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    overrides: dict[str, Any] = {
        "data": {"data_dir": args.data_dir},
        "eval": {
            "threshold": args.threshold,
            "batch_size": args.batch_size,
            "num_qualitative": args.num_qualitative,
        },
        "device": args.device,
    }
    overrides = {
        k: ({kk: vv for kk, vv in v.items() if vv is not None} if isinstance(v, dict) else v)
        for k, v in overrides.items()
    }
    overrides = {k: v for k, v in overrides.items() if v not in (None, {})}

    try:
        cfg = load_config(args.config, overrides=overrides)
        checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint)
        evaluate(cfg, checkpoint_path, make_figures=not args.no_figures)
    except (FileNotFoundError, DatasetPairingError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
