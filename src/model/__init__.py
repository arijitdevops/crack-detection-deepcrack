"""Model factory for the crack-segmentation architectures.

>>> from src.model import build_model, available_models
>>> sorted(available_models())
['unet', 'unet_resnet34']
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from torch import nn

from .blocks import DoubleConv, Down, OutConv, Up
from .unet import UNet
from .unet_resnet import UNetResNet34

LOGGER = logging.getLogger(__name__)

#: Registry of architecture name -> constructor.
MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "unet": UNet,
    "unet_resnet34": UNetResNet34,
}

#: Keyword arguments each constructor accepts, used to drop irrelevant config.
_ACCEPTED_KWARGS: dict[str, set[str]] = {
    "unet": {"in_channels", "num_classes", "base_channels", "depth", "bilinear", "dropout"},
    "unet_resnet34": {
        "in_channels",
        "num_classes",
        "pretrained",
        "decoder_channels",
        "bilinear",
        "freeze_encoder",
        "dropout",
    },
}


def available_models() -> list[str]:
    """Names accepted by :func:`build_model`."""
    return sorted(MODEL_REGISTRY)


def build_model(name: str = "unet", **kwargs: Any) -> nn.Module:
    """Instantiate an architecture by name.

    Unknown keyword arguments are dropped with a debug message rather than
    raising, so a single YAML ``model:`` block can describe either
    architecture (``base_channels`` is meaningless for the ResNet variant, and
    ``pretrained`` is meaningless for the plain U-Net).

    Parameters
    ----------
    name:
        ``"unet"`` or ``"unet_resnet34"`` (case-insensitive).
    **kwargs:
        Forwarded to the constructor after filtering.

    Raises
    ------
    ValueError
        If *name* is not registered.
    """
    key = (name or "").strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Available models: {', '.join(available_models())}"
        )
    accepted = _ACCEPTED_KWARGS[key]
    used = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = set(kwargs) - set(used)
    if dropped:
        LOGGER.debug("build_model(%s): ignoring %s", key, ", ".join(sorted(dropped)))
    return MODEL_REGISTRY[key](**used)


__all__ = [
    "MODEL_REGISTRY",
    "UNet",
    "UNetResNet34",
    "DoubleConv",
    "Down",
    "Up",
    "OutConv",
    "available_models",
    "build_model",
]
