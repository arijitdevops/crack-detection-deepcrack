"""Loss-function behaviour: perfect predictions, ordering and stability."""

from __future__ import annotations

import math

import pytest
import torch

from src.losses import BCEDiceLoss, DiceLoss, FocalLoss, TverskyLoss, build_loss

#: Logit that sigmoids to ~1.0 without overflowing.
BIG = 20.0


def _perfect_logits(target: torch.Tensor) -> torch.Tensor:
    """Logits that are confidently correct for every pixel of *target*."""
    return torch.where(target > 0.5, torch.tensor(BIG), torch.tensor(-BIG))


@pytest.fixture
def target() -> torch.Tensor:
    mask = torch.zeros(2, 1, 16, 16)
    mask[:, :, 4:8, 2:14] = 1.0
    return mask


def test_dice_loss_of_a_perfect_prediction_is_almost_zero(target: torch.Tensor) -> None:
    loss = DiceLoss()(_perfect_logits(target), target)
    assert float(loss) == pytest.approx(0.0, abs=1e-4)


def test_dice_loss_of_an_inverted_prediction_is_near_one(target: torch.Tensor) -> None:
    loss = DiceLoss()(_perfect_logits(1.0 - target), target)
    assert float(loss) > 0.99


def test_dice_loss_of_empty_prediction_on_empty_target_is_zero() -> None:
    empty = torch.zeros(1, 1, 8, 8)
    loss = DiceLoss()(_perfect_logits(empty), empty)
    assert float(loss) == pytest.approx(0.0, abs=1e-3)


def test_dice_loss_decreases_as_the_prediction_improves(target: torch.Tensor) -> None:
    good = _perfect_logits(target)
    mediocre = good * 0.1
    dice = DiceLoss()
    assert float(dice(good, target)) < float(dice(mediocre, target))


def test_bce_dice_is_the_weighted_sum_of_its_parts(target: torch.Tensor) -> None:
    logits = torch.randn_like(target)
    bce_only = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)(logits, target)
    dice_only = BCEDiceLoss(bce_weight=0.0, dice_weight=1.0)(logits, target)
    mixed = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7)(logits, target)
    assert float(mixed) == pytest.approx(
        0.3 * float(bce_only) + 0.7 * float(dice_only), rel=1e-5
    )


def test_bce_dice_rejects_two_zero_weights() -> None:
    with pytest.raises(ValueError):
        BCEDiceLoss(bce_weight=0.0, dice_weight=0.0)


def test_pos_weight_increases_the_penalty_for_missed_cracks(target: torch.Tensor) -> None:
    # Predict "background everywhere": every crack pixel is a false negative.
    logits = torch.full_like(target, -BIG)
    plain = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)(logits, target)
    weighted = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0, pos_weight=10.0)(logits, target)
    assert float(weighted) > float(plain)


def test_focal_loss_downweights_easy_pixels(target: torch.Tensor) -> None:
    easy = _perfect_logits(target)
    hard = torch.zeros_like(target)  # p = 0.5 everywhere
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    assert float(focal(easy, target)) < float(focal(hard, target))


def test_focal_gamma_zero_matches_alpha_weighted_bce(target: torch.Tensor) -> None:
    logits = torch.randn_like(target)
    focal = FocalLoss(alpha=None, gamma=0.0)
    bce = torch.nn.BCEWithLogitsLoss()
    assert float(focal(logits, target)) == pytest.approx(float(bce(logits, target)), rel=1e-5)


def test_focal_loss_is_finite_for_extreme_logits(target: torch.Tensor) -> None:
    extreme = torch.full_like(target, 60.0)
    value = float(FocalLoss()(extreme, target))
    assert math.isfinite(value)


def test_tversky_with_equal_weights_matches_dice(target: torch.Tensor) -> None:
    logits = torch.randn_like(target)
    # (TP + s/2) / (TP + FP/2 + FN/2 + s/2) == (2TP + s) / (2TP + FP + FN + s)
    tversky = TverskyLoss(alpha=0.5, beta=0.5, smooth=0.5)
    dice = DiceLoss(smooth=1.0)
    assert float(tversky(logits, target)) == pytest.approx(float(dice(logits, target)), rel=1e-4)


def test_tversky_beta_penalises_false_negatives_more(target: torch.Tensor) -> None:
    missed = torch.full_like(target, -BIG)  # all false negatives
    recall_biased = TverskyLoss(alpha=0.1, beta=0.9)
    precision_biased = TverskyLoss(alpha=0.9, beta=0.1)
    assert float(recall_biased(missed, target)) > float(precision_biased(missed, target))


def test_losses_reject_shape_mismatches() -> None:
    logits = torch.randn(1, 1, 8, 8)
    target = torch.zeros(1, 1, 4, 4)
    for loss in (DiceLoss(), BCEDiceLoss(), FocalLoss(), TverskyLoss()):
        with pytest.raises(ValueError, match="shape"):
            loss(logits, target)


def test_losses_are_differentiable(target: torch.Tensor) -> None:
    for loss_fn in (DiceLoss(), BCEDiceLoss(), FocalLoss(), TverskyLoss()):
        logits = torch.randn_like(target, requires_grad=True)
        loss_fn(logits, target).backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("name", ["dice", "bce_dice", "focal", "tversky", "bce"])
def test_build_loss_returns_a_working_module(name: str, target: torch.Tensor) -> None:
    loss_fn = build_loss(name, pos_weight=2.0)
    value = float(loss_fn(torch.randn_like(target), target))
    assert math.isfinite(value)


def test_build_loss_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss("lovasz")
