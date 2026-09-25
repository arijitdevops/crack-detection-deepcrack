"""Metric tests against a hand-computed confusion matrix.

The fixture below is small enough to verify by eye. Probabilities
``[[0.9, 0.7, 0.1, 0.2], [0.8, 0.3, 0.05, 0.4]]`` thresholded at 0.5 give::

    prediction               ground truth
        1 1 0 0                  1 0 0 0
        1 0 0 0                  1 1 0 0

    tp = 2   (0, 0) and (1, 0)
    fp = 1   (0, 1) predicted crack, labelled background
    fn = 1   (1, 1) labelled crack, predicted background
    tn = 4   the remaining pixels
"""

from __future__ import annotations

import pytest
import torch

from src.metrics import (
    SegmentationMetrics,
    ThresholdSweep,
    accuracy_from_counts,
    compute_metrics,
    confusion_counts,
    dice_from_counts,
    find_best_threshold,
    iou_from_counts,
    precision_from_counts,
    recall_from_counts,
)


@pytest.fixture
def hand_computed() -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    probs = torch.tensor([[[[0.9, 0.7, 0.1, 0.2], [0.8, 0.3, 0.05, 0.4]]]])
    truth = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]]])
    # At threshold 0.5 the prediction is [[1, 1, 0, 0], [1, 0, 0, 0]].
    expected = {"tp": 2, "fp": 1, "fn": 1, "tn": 4}
    return probs, truth, expected


def test_confusion_counts_match_by_hand(hand_computed) -> None:
    probs, truth, expected = hand_computed
    tp, fp, fn, tn = confusion_counts(probs, truth, threshold=0.5)
    assert (tp, fp, fn, tn) == (expected["tp"], expected["fp"], expected["fn"], expected["tn"])
    assert tp + fp + fn + tn == probs.numel()


def test_derived_metrics_match_by_hand(hand_computed) -> None:
    probs, truth, expected = hand_computed
    tp, fp, fn, tn = expected["tp"], expected["fp"], expected["fn"], expected["tn"]

    metrics = compute_metrics(probs, truth, threshold=0.5)
    assert metrics["precision"] == pytest.approx(tp / (tp + fp))          # 2/3
    assert metrics["recall"] == pytest.approx(tp / (tp + fn))             # 2/3
    assert metrics["iou"] == pytest.approx(tp / (tp + fp + fn))           # 2/4 = 0.5
    assert metrics["dice"] == pytest.approx(2 * tp / (2 * tp + fp + fn))  # 4/6
    assert metrics["f1"] == pytest.approx(metrics["dice"])
    assert metrics["pixel_accuracy"] == pytest.approx((tp + tn) / 8)      # 6/8


def test_count_helpers_handle_empty_denominators() -> None:
    assert iou_from_counts(0, 0, 0) == 1.0
    assert dice_from_counts(0, 0, 0) == 1.0
    assert precision_from_counts(0, 0) == 1.0
    assert recall_from_counts(0, 0) == 1.0
    assert accuracy_from_counts(0, 0, 0, 0) == 0.0


def test_metrics_accumulate_over_batches_rather_than_averaging() -> None:
    """Dataset-level metrics must not be the mean of per-batch metrics.

    Batch A is a perfect prediction on a mask with a single crack pixel;
    batch B misses 99 of 100 crack pixels. Naively averaging the two batch
    Dice scores gives ~0.51, while the correct dataset-level Dice is far
    lower because batch B dominates the pixel counts.
    """
    meter = SegmentationMetrics(threshold=0.5)

    batch_a_probs = torch.zeros(1, 1, 10, 10)
    batch_a_truth = torch.zeros(1, 1, 10, 10)
    batch_a_probs[0, 0, 0, 0] = 0.99
    batch_a_truth[0, 0, 0, 0] = 1.0

    batch_b_probs = torch.zeros(1, 1, 10, 10)
    batch_b_truth = torch.ones(1, 1, 10, 10)
    batch_b_probs[0, 0, 0, 0] = 0.99

    meter.update(batch_a_probs, batch_a_truth)
    meter.update(batch_b_probs, batch_b_truth)
    results = meter.compute()

    assert results["tp"] == 2
    assert results["fn"] == 99
    assert results["dice"] == pytest.approx(2 * 2 / (2 * 2 + 0 + 99))
    naive_average = (1.0 + 2 * 1 / (1 + 100)) / 2
    assert results["dice"] < naive_average / 10


def test_reset_clears_the_accumulator(hand_computed) -> None:
    probs, truth, _ = hand_computed
    meter = SegmentationMetrics()
    meter.update(probs, truth)
    meter.reset()
    assert meter.compute()["tp"] == 0.0
    assert meter.per_image_dice == []


def test_per_image_dice_is_reported_alongside_the_dataset_metric() -> None:
    meter = SegmentationMetrics(threshold=0.5)
    probs = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 0.0]]]])
    truth = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 1.0]]]])
    meter.update(probs, truth)
    results = meter.compute()
    assert len(meter.per_image_dice) == 2
    assert results["dice_per_image"] == pytest.approx((1.0 + 0.0) / 2)


def test_confusion_counts_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        confusion_counts(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 2, 2))


def test_threshold_sweep_finds_the_best_f1_threshold() -> None:
    """A prediction whose crack pixels sit at p=0.3 is only found below 0.3."""
    truth = torch.zeros(1, 1, 8, 8)
    truth[:, :, 2:5, :] = 1.0
    probs = torch.full_like(truth, 0.05)
    probs[truth > 0.5] = 0.3

    sweep = ThresholdSweep([0.1, 0.2, 0.5, 0.8])
    sweep.update(probs, truth)
    best_t, best_row = sweep.best_threshold("f1")

    assert best_t in (0.1, 0.2)
    assert best_row["f1"] == pytest.approx(1.0)
    # Above 0.3 nothing is predicted, so recall collapses.
    rows = sweep.compute()
    assert rows[-1]["recall"] == pytest.approx(0.0)


def test_threshold_sweep_curve_lengths_match() -> None:
    sweep = ThresholdSweep.from_range(0.1, 0.9, 5)
    sweep.update(torch.rand(1, 1, 8, 8), (torch.rand(1, 1, 8, 8) > 0.5).float())
    thresholds, values = sweep.curve("f1")
    assert len(thresholds) == len(values) == 5
    assert thresholds[0] == pytest.approx(0.1) and thresholds[-1] == pytest.approx(0.9)


def test_threshold_sweep_rejects_unknown_metric() -> None:
    sweep = ThresholdSweep([0.5])
    sweep.update(torch.rand(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    with pytest.raises(KeyError):
        sweep.curve("auroc")


def test_find_best_threshold_accepts_batched_iterables() -> None:
    truth = [torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)]
    probs = [torch.full((1, 1, 4, 4), 0.6), torch.full((1, 1, 4, 4), 0.1)]
    best_t, best_row, rows = find_best_threshold(probs, truth, thresholds=[0.2, 0.5, 0.9])
    assert best_t in (0.2, 0.5)
    assert best_row["f1"] == pytest.approx(1.0)
    assert len(rows) == 3


@pytest.mark.parametrize("scale", [1, 255])
def test_mask_statistics_accept_0_1_and_0_255_uint8_masks(scale: int) -> None:
    """predict_array returns 0/1 uint8 masks; saved PNG masks are 0/255."""
    import numpy as np

    from src.utils import crack_pixel_ratio, estimate_crack_length_px, mask_to_overlay

    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[9:11, 5:35] = scale
    assert crack_pixel_ratio(mask) == pytest.approx(60 / 800)
    assert estimate_crack_length_px(mask) > 20
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    overlay = mask_to_overlay(image, mask)
    assert overlay[10, 20].any() and not overlay[0, 0].any()
