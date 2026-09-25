"""Segmentation metrics accumulated over confusion counts.

Averaging per-batch IoU or Dice is a common but misleading shortcut: batches
that happen to contain almost no crack pixels dominate the mean and the number
you report is not the metric over the dataset. Everything here accumulates
``tp``, ``fp``, ``fn`` and ``tn`` across batches and computes the metric once,
at the end, from those totals (the "dataset-level" or micro-averaged metric).

:class:`SegmentationMetrics` also exposes per-image Dice so you can report the
macro average alongside, since that is what several crack-segmentation papers
quote.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

EPS: float = 1e-7


def _as_binary(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    return (tensor >= threshold).to(torch.bool)


def confusion_counts(
    probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, fn, tn)`` for a batch of probabilities.

    Parameters
    ----------
    probs:
        Sigmoid probabilities in ``[0, 1]``, any shape.
    targets:
        Ground truth of the same shape; anything ``>= 0.5`` counts as crack.
    threshold:
        Decision threshold applied to *probs*.
    """
    if probs.shape != targets.shape:
        raise ValueError(
            f"probs shape {tuple(probs.shape)} must match targets shape {tuple(targets.shape)}"
        )
    pred = _as_binary(probs.detach(), threshold)
    truth = _as_binary(targets.detach(), 0.5)
    tp = int(torch.logical_and(pred, truth).sum().item())
    fp = int(torch.logical_and(pred, ~truth).sum().item())
    fn = int(torch.logical_and(~pred, truth).sum().item())
    tn = int(torch.logical_and(~pred, ~truth).sum().item())
    return tp, fp, fn, tn


def iou_from_counts(tp: int, fp: int, fn: int) -> float:
    """Intersection over union (Jaccard index) of the positive class."""
    denominator = tp + fp + fn
    return float(tp / denominator) if denominator > 0 else 1.0


def dice_from_counts(tp: int, fp: int, fn: int) -> float:
    """Dice coefficient, numerically identical to the F1 score."""
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator > 0 else 1.0


def precision_from_counts(tp: int, fp: int) -> float:
    """Fraction of predicted crack pixels that are really crack."""
    denominator = tp + fp
    return float(tp / denominator) if denominator > 0 else 1.0


def recall_from_counts(tp: int, fn: int) -> float:
    """Fraction of true crack pixels that were found."""
    denominator = tp + fn
    return float(tp / denominator) if denominator > 0 else 1.0


def accuracy_from_counts(tp: int, fp: int, fn: int, tn: int) -> float:
    """Pixel accuracy. Near-useless on its own here: >95% by predicting nothing."""
    total = tp + fp + fn + tn
    return float((tp + tn) / total) if total > 0 else 0.0


@dataclass
class SegmentationMetrics:
    """Accumulator for dataset-level binary segmentation metrics.

    Example
    -------
    >>> import torch
    >>> m = SegmentationMetrics(threshold=0.5)
    >>> probs = torch.tensor([[[[0.9, 0.1], [0.8, 0.2]]]])
    >>> target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    >>> m.update(probs, target)
    >>> round(m.compute()["precision"], 4)
    0.5
    """

    threshold: float = 0.5
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    per_image_dice: list[float] = field(default_factory=list)

    def reset(self) -> None:
        """Zero every accumulator."""
        self.tp = self.fp = self.fn = self.tn = 0
        self.per_image_dice.clear()

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch of probabilities against its targets."""
        tp, fp, fn, tn = confusion_counts(probs, targets, self.threshold)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

        # Per-image Dice for the macro average.
        if probs.dim() >= 3:
            pred = _as_binary(probs.detach(), self.threshold)
            truth = _as_binary(targets.detach(), 0.5)
            dims = tuple(range(1, pred.dim()))
            inter = torch.logical_and(pred, truth).sum(dim=dims).double()
            total = pred.sum(dim=dims).double() + truth.sum(dim=dims).double()
            dice = torch.where(
                total > 0, 2.0 * inter / total.clamp_min(EPS), torch.ones_like(total)
            )
            self.per_image_dice.extend(dice.cpu().tolist())

    def compute(self) -> dict[str, float]:
        """Return every metric computed from the accumulated counts."""
        results = {
            "iou": iou_from_counts(self.tp, self.fp, self.fn),
            "dice": dice_from_counts(self.tp, self.fp, self.fn),
            "f1": dice_from_counts(self.tp, self.fp, self.fn),
            "precision": precision_from_counts(self.tp, self.fp),
            "recall": recall_from_counts(self.tp, self.fn),
            "pixel_accuracy": accuracy_from_counts(self.tp, self.fp, self.fn, self.tn),
            "threshold": float(self.threshold),
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(self.fn),
            "tn": float(self.tn),
        }
        if self.per_image_dice:
            results["dice_per_image"] = float(np.mean(self.per_image_dice))
        return results

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        values = self.compute()
        return (
            f"SegmentationMetrics(iou={values['iou']:.4f}, dice={values['dice']:.4f}, "
            f"precision={values['precision']:.4f}, recall={values['recall']:.4f})"
        )


def compute_metrics(
    probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    """One-shot metrics for a single prediction/target pair."""
    meter = SegmentationMetrics(threshold=threshold)
    meter.update(probs, targets)
    return meter.compute()


class ThresholdSweep:
    """Accumulate confusion counts for several thresholds at once.

    A single pass over the test set then yields the whole threshold-vs-F1
    curve, which is what :mod:`src.evaluate` plots and what
    :meth:`best_threshold` reads off.
    """

    def __init__(self, thresholds: Sequence[float]) -> None:
        thresholds = [float(t) for t in thresholds]
        if not thresholds:
            raise ValueError("ThresholdSweep needs at least one threshold")
        self.thresholds = thresholds
        self._meters = [SegmentationMetrics(threshold=t) for t in thresholds]

    @classmethod
    def from_range(cls, start: float, stop: float, num: int) -> ThresholdSweep:
        """Build a sweep of *num* thresholds linearly spaced in ``[start, stop]``."""
        if num < 2:
            raise ValueError(f"num must be >= 2, got {num}")
        return cls(np.linspace(float(start), float(stop), int(num)).tolist())

    def reset(self) -> None:
        for meter in self._meters:
            meter.reset()

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate one batch against every threshold."""
        probs = probs.detach()
        targets = targets.detach()
        for meter in self._meters:
            meter.update(probs, targets)

    def compute(self) -> list[dict[str, float]]:
        """Per-threshold metric dictionaries, in threshold order."""
        return [meter.compute() for meter in self._meters]

    def curve(self, key: str = "f1") -> tuple[list[float], list[float]]:
        """Return ``(thresholds, values)`` for *key*, ready to plot."""
        rows = self.compute()
        if key not in rows[0]:
            raise KeyError(f"unknown metric {key!r}; available: {sorted(rows[0])}")
        return list(self.thresholds), [row[key] for row in rows]

    def best_threshold(self, key: str = "f1") -> tuple[float, dict[str, float]]:
        """Return the threshold maximising *key* and its full metric row."""
        rows = self.compute()
        best_index = max(range(len(rows)), key=lambda i: rows[i][key])
        return self.thresholds[best_index], rows[best_index]


def find_best_threshold(
    probs: Iterable[torch.Tensor] | torch.Tensor,
    targets: Iterable[torch.Tensor] | torch.Tensor,
    *,
    thresholds: Sequence[float] | None = None,
    key: str = "f1",
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    """Sweep thresholds over (possibly batched) predictions.

    Parameters
    ----------
    probs, targets:
        Either single tensors or matching iterables of batches.
    thresholds:
        Thresholds to test; defaults to 19 values from 0.05 to 0.95.
    key:
        Metric to maximise (``"f1"``, ``"iou"``, ...).

    Returns
    -------
    tuple
        ``(best_threshold, best_metrics, all_rows)``.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19).tolist()
    sweep = ThresholdSweep(thresholds)

    if isinstance(probs, torch.Tensor):
        if not isinstance(targets, torch.Tensor):
            raise TypeError("probs and targets must both be tensors or both be iterables")
        sweep.update(probs, targets)
    else:
        for prob_batch, target_batch in zip(probs, targets, strict=False):
            sweep.update(prob_batch, target_batch)

    best_t, best_row = sweep.best_threshold(key)
    return best_t, best_row, sweep.compute()


def format_metrics(metrics: Mapping[str, float], *, keys: Sequence[str] | None = None) -> str:
    """Render a metrics mapping as a compact one-line log string."""
    keys = keys or ("iou", "dice", "precision", "recall", "pixel_accuracy")
    parts = [f"{key}={metrics[key]:.4f}" for key in keys if key in metrics]
    return " ".join(parts)


__all__ = [
    "SegmentationMetrics",
    "ThresholdSweep",
    "compute_metrics",
    "confusion_counts",
    "find_best_threshold",
    "format_metrics",
    "iou_from_counts",
    "dice_from_counts",
    "precision_from_counts",
    "recall_from_counts",
    "accuracy_from_counts",
]
