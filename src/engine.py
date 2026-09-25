"""Training and validation loops.

Both loops take an explicit device, an optional AMP scaler and a metrics
accumulator, so :mod:`src.train` and :mod:`src.evaluate` share the same
code path and cannot drift apart.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import SegmentationMetrics, ThresholdSweep
from .utils import AverageMeter

LOGGER = logging.getLogger(__name__)


def _unpack(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept ``(image, mask)`` or ``(image, mask, meta)`` batches."""
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(f"Unsupported batch type {type(batch)!r}; expected a (image, mask) tuple")


def _autocast(device: torch.device, enabled: bool):
    """AMP context manager for the device type, or a no-op when disabled."""
    device_type = device.type
    if not enabled or device_type not in {"cuda", "cpu"}:
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)


def make_grad_scaler(device: torch.device, enabled: bool):
    """Create a :class:`torch.amp.GradScaler` appropriate for *device*.

    Gradient scaling is only meaningful for fp16 on CUDA; elsewhere a disabled
    scaler is returned so the calling code stays branch-free.
    """
    use_scaler = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=use_scaler)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader | Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler: Any | None = None,
    amp: bool = False,
    grad_clip: float = 0.0,
    epoch: int = 0,
    threshold: float = 0.5,
    max_batches: int | None = None,
    log_interval: int = 20,
    on_batch_end: Callable[[int, float], None] | None = None,
) -> dict[str, float]:
    """Run one training epoch and return loss plus dataset-level metrics.

    Parameters
    ----------
    max_batches:
        Stop after this many batches. Used by ``--dry-run``.
    on_batch_end:
        Optional callback receiving ``(global_step_within_epoch, loss)``, used
        for per-step TensorBoard logging.

    Raises
    ------
    RuntimeError
        If the loss becomes non-finite, which otherwise silently poisons every
        subsequent weight update.
    """
    model.train()
    loss_meter = AverageMeter("train_loss")
    metrics = SegmentationMetrics(threshold=threshold)
    started = time.perf_counter()
    num_batches = 0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images, masks = _unpack(batch)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            logits = model(images)
            loss = criterion(logits, masks)

        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            raise RuntimeError(
                f"Non-finite loss ({loss_value}) at epoch {epoch}, step {step}. "
                "Lower the learning rate or disable AMP."
            )

        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            metrics.update(torch.sigmoid(logits.detach().float()), masks)
        loss_meter.update(loss_value, images.size(0))
        num_batches += 1

        if on_batch_end is not None:
            on_batch_end(step, loss_value)
        if log_interval and step % log_interval == 0:
            LOGGER.debug("epoch %d | batch %d | loss %.4f", epoch, step, loss_value)

    results = metrics.compute()
    results["loss"] = loss_meter.avg
    results["batches"] = float(num_batches)
    results["seconds"] = time.perf_counter() - started
    return results


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader | Iterable,
    criterion: nn.Module | None,
    device: torch.device,
    *,
    amp: bool = False,
    threshold: float = 0.5,
    max_batches: int | None = None,
    sweep: ThresholdSweep | None = None,
) -> dict[str, float]:
    """Evaluate *model* over *loader* without touching gradients.

    Parameters
    ----------
    criterion:
        Optional; when ``None`` the returned dict has ``loss == 0.0``.
    sweep:
        Optional :class:`~src.metrics.ThresholdSweep` updated alongside the
        fixed-threshold metrics, so a single pass produces the whole curve.
    """
    model.eval()
    loss_meter = AverageMeter("val_loss")
    metrics = SegmentationMetrics(threshold=threshold)
    started = time.perf_counter()
    num_batches = 0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images, masks = _unpack(batch)
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with _autocast(device, amp):
            logits = model(images)
            if criterion is not None:
                loss = criterion(logits, masks)
                loss_meter.update(float(loss.detach().item()), images.size(0))

        probs = torch.sigmoid(logits.float())
        metrics.update(probs, masks)
        if sweep is not None:
            sweep.update(probs, masks)
        num_batches += 1

    results = metrics.compute()
    results["loss"] = loss_meter.avg
    results["batches"] = float(num_batches)
    results["seconds"] = time.perf_counter() - started
    return results


def log_epoch(
    epoch: int,
    epochs: int,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    lr: float,
) -> None:
    """Emit a single human-readable summary line for an epoch."""
    LOGGER.info(
        "epoch %3d/%d | lr %.2e | train loss %.4f dice %.4f | "
        "val loss %.4f dice %.4f iou %.4f prec %.4f rec %.4f | %.1fs",
        epoch,
        epochs,
        lr,
        train_metrics.get("loss", float("nan")),
        train_metrics.get("dice", float("nan")),
        val_metrics.get("loss", float("nan")),
        val_metrics.get("dice", float("nan")),
        val_metrics.get("iou", float("nan")),
        val_metrics.get("precision", float("nan")),
        val_metrics.get("recall", float("nan")),
        train_metrics.get("seconds", 0.0) + val_metrics.get("seconds", 0.0),
    )


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience:
        Epochs without improvement to tolerate. ``<= 0`` disables the check.
    mode:
        ``"max"`` for metrics like Dice, ``"min"`` for losses.
    min_delta:
        Minimum change that counts as an improvement.
    """

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best: float = -math.inf if mode == "max" else math.inf
        self.num_bad_epochs = 0
        self.should_stop = False

    def is_improvement(self, value: float) -> bool:
        """Whether *value* beats the best seen so far by ``min_delta``."""
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        """Register *value*; returns ``True`` when it is a new best."""
        if self.is_improvement(value):
            self.best = float(value)
            self.num_bad_epochs = 0
            return True
        self.num_bad_epochs += 1
        if self.patience > 0 and self.num_bad_epochs >= self.patience:
            self.should_stop = True
            LOGGER.info(
                "Early stopping: no improvement in %d epochs (best %.4f)",
                self.num_bad_epochs,
                self.best,
            )
        return False


__all__ = ["EarlyStopping", "log_epoch", "make_grad_scaler", "train_one_epoch", "validate"]
