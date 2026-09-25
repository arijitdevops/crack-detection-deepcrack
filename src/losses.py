"""Segmentation objectives for sparse binary masks.

Every loss in this module consumes **raw logits**, never probabilities. The
binary cross-entropy terms go through :class:`~torch.nn.BCEWithLogitsLoss`,
which fuses the sigmoid into the loss for numerical stability; calling
``sigmoid`` and then ``BCELoss`` would lose that and can produce ``inf``
gradients once the network becomes confident.

Crack pixels make up well under 5% of a DeepCrack mask, so a plain BCE model
is heavily biased toward the background. The Dice, Focal and Tversky variants
here all exist to counteract that imbalance.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.nn import functional as F

LOGGER = logging.getLogger(__name__)

#: Smoothing constant added to both numerator and denominator of Dice-like terms.
DEFAULT_SMOOTH: float = 1.0


def _check_shapes(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} must match targets shape {tuple(targets.shape)}"
        )
    if logits.dim() < 3:
        raise ValueError(f"expected at least a 3-D tensor, got {tuple(logits.shape)}")


def _flatten_per_sample(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape ``(N, C, H, W)`` to ``(N, C*H*W)`` for per-sample reductions."""
    return tensor.reshape(tensor.shape[0], -1)


class DiceLoss(nn.Module):
    """Soft Dice loss, ``1 - dice``, averaged over the batch.

    Parameters
    ----------
    smooth:
        Added to numerator and denominator. Also defines the loss of an
        all-empty prediction on an all-empty target (which becomes 0).
    from_logits:
        ``True`` (default) applies a sigmoid internally.
    per_sample:
        Compute Dice per image and average (default), rather than over the
        whole flattened batch. Per-sample is the stricter, more common choice.
    """

    def __init__(
        self,
        smooth: float = DEFAULT_SMOOTH,
        from_logits: bool = True,
        per_sample: bool = True,
    ) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError(f"smooth must be positive, got {smooth}")
        self.smooth = float(smooth)
        self.from_logits = from_logits
        self.per_sample = per_sample

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _check_shapes(logits, targets)
        probs = torch.sigmoid(logits) if self.from_logits else logits
        targets = targets.to(probs.dtype)

        if self.per_sample:
            probs_f = _flatten_per_sample(probs)
            targets_f = _flatten_per_sample(targets)
            dim = 1
        else:
            probs_f = probs.reshape(1, -1)
            targets_f = targets.reshape(1, -1)
            dim = 1

        intersection = (probs_f * targets_f).sum(dim=dim)
        cardinality = probs_f.sum(dim=dim) + targets_f.sum(dim=dim)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCE-with-logits and :class:`DiceLoss`.

    ``loss = bce_weight * BCE + dice_weight * Dice``

    Parameters
    ----------
    bce_weight, dice_weight:
        Non-negative mixing weights. At least one must be positive.
    pos_weight:
        Optional positive-class weight handed to
        :class:`~torch.nn.BCEWithLogitsLoss`. Values above 1 push the model to
        recall more crack pixels at the cost of precision.
    smooth:
        Dice smoothing constant.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = None,
        smooth: float = DEFAULT_SMOOTH,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("bce_weight and dice_weight must be non-negative")
        if bce_weight == 0 and dice_weight == 0:
            raise ValueError("at least one of bce_weight / dice_weight must be positive")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.dice = DiceLoss(smooth=smooth)
        if pos_weight is not None:
            # Registered as a buffer so it follows the module across devices.
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _check_shapes(logits, targets)
        targets = targets.to(logits.dtype)
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017) computed from logits.

    ``FL = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    The implementation reuses ``binary_cross_entropy_with_logits`` for the
    ``log(p_t)`` term so it stays stable for large-magnitude logits.

    Parameters
    ----------
    alpha:
        Weight of the positive class in ``[0, 1]``; ``None`` disables the
        alpha balancing.
    gamma:
        Focusing exponent. ``0`` reduces the loss to weighted BCE.
    reduction:
        ``"mean"``, ``"sum"`` or ``"none"``.
    """

    def __init__(
        self,
        alpha: float | None = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1] or None, got {alpha}")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be mean/sum/none, got {reduction!r}")
        self.alpha = alpha
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _check_shapes(logits, targets)
        targets = targets.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t = p for positives, 1 - p for negatives; exp(-bce) computes it
        # without a second sigmoid and without underflow for confident pixels.
        p_t = torch.exp(-bce)
        loss = ((1.0 - p_t) ** self.gamma) * bce
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class TverskyLoss(nn.Module):
    """Tversky loss (Salehi et al., 2017), a Dice generalisation.

    ``TI = TP / (TP + alpha * FP + beta * FN)``

    ``alpha == beta == 0.5`` recovers Dice (with half the smoothing constant). Raising *beta* penalises false
    negatives harder, which is usually what you want for thin cracks that the
    network is tempted to erase.

    Parameters
    ----------
    alpha:
        False-positive weight.
    beta:
        False-negative weight.
    smooth:
        Smoothing constant.
    gamma:
        Focal-Tversky exponent; ``1.0`` is the plain Tversky loss.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = DEFAULT_SMOOTH,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _check_shapes(logits, targets)
        probs = _flatten_per_sample(torch.sigmoid(logits))
        targets_f = _flatten_per_sample(targets.to(probs.dtype))

        true_pos = (probs * targets_f).sum(dim=1)
        false_pos = (probs * (1.0 - targets_f)).sum(dim=1)
        false_neg = ((1.0 - probs) * targets_f).sum(dim=1)

        index = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        loss = (1.0 - index) ** self.gamma
        return loss.mean()


#: Name -> constructor for :func:`build_loss`.
LOSS_REGISTRY = {
    "dice": DiceLoss,
    "bce_dice": BCEDiceLoss,
    "focal": FocalLoss,
    "tversky": TverskyLoss,
    "bce": nn.BCEWithLogitsLoss,
}


def available_losses() -> list[str]:
    """Names accepted by :func:`build_loss`."""
    return sorted(LOSS_REGISTRY)


def build_loss(name: str = "bce_dice", **kwargs) -> nn.Module:
    """Construct a loss from a config-style name.

    Recognised keys per loss (extras are ignored):

    * ``dice``      -- ``smooth``
    * ``bce_dice``  -- ``bce_weight``, ``dice_weight``, ``pos_weight``, ``smooth``
    * ``focal``     -- ``alpha``/``focal_alpha``, ``gamma``/``focal_gamma``
    * ``tversky``   -- ``alpha``/``tversky_alpha``, ``beta``/``tversky_beta``, ``smooth``
    * ``bce``       -- ``pos_weight``

    Raises
    ------
    ValueError
        If *name* is unknown.
    """
    key = (name or "").strip().lower()
    if key not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss {name!r}. Available: {', '.join(available_losses())}")

    if key == "dice":
        return DiceLoss(smooth=kwargs.get("smooth", DEFAULT_SMOOTH))
    if key == "bce_dice":
        return BCEDiceLoss(
            bce_weight=kwargs.get("bce_weight", 0.5),
            dice_weight=kwargs.get("dice_weight", 0.5),
            pos_weight=kwargs.get("pos_weight"),
            smooth=kwargs.get("smooth", DEFAULT_SMOOTH),
        )
    if key == "focal":
        return FocalLoss(
            alpha=kwargs.get("alpha", kwargs.get("focal_alpha", 0.25)),
            gamma=kwargs.get("gamma", kwargs.get("focal_gamma", 2.0)),
        )
    if key == "tversky":
        return TverskyLoss(
            alpha=kwargs.get("alpha", kwargs.get("tversky_alpha", 0.3)),
            beta=kwargs.get("beta", kwargs.get("tversky_beta", 0.7)),
            smooth=kwargs.get("smooth", DEFAULT_SMOOTH),
        )
    pos_weight = kwargs.get("pos_weight")
    return nn.BCEWithLogitsLoss(
        pos_weight=None if pos_weight is None else torch.tensor(float(pos_weight))
    )


__all__ = [
    "DiceLoss",
    "BCEDiceLoss",
    "FocalLoss",
    "TverskyLoss",
    "LOSS_REGISTRY",
    "available_losses",
    "build_loss",
]
