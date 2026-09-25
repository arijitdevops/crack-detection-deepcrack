"""Joint image/mask augmentation for binary segmentation.

Every geometric operation is applied to the image *and* the mask with the same
parameters; photometric operations (brightness, contrast) touch the image only.
Masks are always resampled with nearest-neighbour interpolation so they stay
strictly binary.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .utils import IMAGENET_MEAN, IMAGENET_STD

LOGGER = logging.getLogger(__name__)


def _as_chw_float(image: np.ndarray) -> torch.Tensor:
    """Convert an ``(H, W, 3)`` uint8 array to a ``(3, H, W)`` float tensor in [0, 1]."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {image.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    return tensor.float().div_(255.0) if tensor.dtype == torch.uint8 else tensor.float()


def _as_chw_mask(mask: np.ndarray) -> torch.Tensor:
    """Convert an ``(H, W)`` mask to a ``(1, H, W)`` float tensor of 0.0/1.0.

    Accepts 0/255 uint8 (the DeepCrack convention), 0/1 integers or floats.
    """
    if mask.ndim == 3:
        mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()
    threshold = 127.0 if tensor.max() > 1.0 else 0.5
    return (tensor > threshold).float().unsqueeze(0)


def resize_pair(
    image: torch.Tensor, mask: torch.Tensor, size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize a ``(3, H, W)`` image bilinearly and a ``(1, H, W)`` mask by nearest."""
    height, width = size
    if image.shape[-2:] == (height, width):
        return image, mask
    image = F.interpolate(
        image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
    ).squeeze(0)
    mask = F.interpolate(mask.unsqueeze(0), size=(height, width), mode="nearest").squeeze(0)
    return image, mask


def normalize(image: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalisation to a ``(3, H, W)`` tensor in ``[0, 1]``."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(-1, 1, 1)
    return (image - mean) / std


@dataclass
class JointTransform:
    """Configurable augmentation pipeline shared by images and masks.

    Parameters
    ----------
    image_size:
        ``(height, width)`` of the tensors handed to the network.
    train:
        When ``False`` only the deterministic resize + normalise path runs.
    random_crop / crop_scale:
        Random square-ish crop covering ``crop_scale`` of each side before the
        resize, which gives the network a mix of scales.
    hflip_prob / vflip_prob / rot90_prob:
        Probabilities of the corresponding geometric flip or rotation.
    brightness / contrast:
        Maximum relative jitter applied to the **image only**.
    seed:
        Optional seed for a transform-local :class:`random.Random`, so unit
        tests can reproduce an augmentation exactly.
    """

    image_size: tuple[int, int] = (384, 384)
    train: bool = False
    random_crop: bool = True
    crop_scale: tuple[float, float] = (0.6, 1.0)
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rot90_prob: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    seed: int | None = None

    def __post_init__(self) -> None:
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        self.crop_scale = (float(self.crop_scale[0]), float(self.crop_scale[1]))
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------ #
    # individual operations
    # ------------------------------------------------------------------ #
    def _random_crop(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, height, width = image.shape
        lo, hi = self.crop_scale
        scale = self._rng.uniform(lo, hi)
        crop_h = max(8, min(height, int(round(height * scale))))
        crop_w = max(8, min(width, int(round(width * scale))))
        top = self._rng.randint(0, height - crop_h)
        left = self._rng.randint(0, width - crop_w)
        return (
            image[:, top : top + crop_h, left : left + crop_w],
            mask[:, top : top + crop_h, left : left + crop_w],
        )

    def _photometric(self, image: torch.Tensor) -> torch.Tensor:
        """Brightness and contrast jitter. Never applied to a mask."""
        if self.brightness > 0:
            factor = 1.0 + self._rng.uniform(-self.brightness, self.brightness)
            image = image * factor
        if self.contrast > 0:
            factor = 1.0 + self._rng.uniform(-self.contrast, self.contrast)
            mean = image.mean()
            image = (image - mean) * factor + mean
        return image.clamp_(0.0, 1.0)

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #
    def __call__(
        self, image: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(image_tensor, mask_tensor)`` ready for the network.

        The image comes back ImageNet-normalised with shape ``(3, H, W)``; the
        mask is float 0.0/1.0 with shape ``(1, H, W)``.
        """
        image_t = image if isinstance(image, torch.Tensor) else _as_chw_float(np.asarray(image))
        mask_t = mask if isinstance(mask, torch.Tensor) else _as_chw_mask(np.asarray(mask))
        if image_t.shape[-2:] != mask_t.shape[-2:]:
            raise ValueError(
                f"image size {tuple(image_t.shape[-2:])} does not match "
                f"mask size {tuple(mask_t.shape[-2:])}"
            )

        if self.train:
            if self.random_crop:
                image_t, mask_t = self._random_crop(image_t, mask_t)
            if self._rng.random() < self.hflip_prob:
                image_t = torch.flip(image_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
            if self._rng.random() < self.vflip_prob:
                image_t = torch.flip(image_t, dims=[1])
                mask_t = torch.flip(mask_t, dims=[1])
            if self._rng.random() < self.rot90_prob:
                k = self._rng.choice((1, 2, 3))
                image_t = torch.rot90(image_t, k, dims=(1, 2))
                mask_t = torch.rot90(mask_t, k, dims=(1, 2))

        image_t, mask_t = resize_pair(image_t.contiguous(), mask_t.contiguous(), self.image_size)

        if self.train:
            image_t = self._photometric(image_t)

        image_t = normalize(image_t)
        mask_t = (mask_t > 0.5).float()
        return image_t, mask_t


def build_transforms(
    image_size: Sequence[int],
    *,
    train: bool,
    augment_enabled: bool = True,
    random_crop: bool = True,
    crop_scale: Sequence[float] = (0.6, 1.0),
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
    rot90_prob: float = 0.5,
    brightness: float = 0.2,
    contrast: float = 0.2,
    seed: int | None = None,
) -> JointTransform:
    """Convenience factory mirroring the ``augment`` block of the YAML config."""
    if train and not augment_enabled:
        LOGGER.info("Augmentation disabled; training pipeline is resize + normalise only")
    return JointTransform(
        image_size=(int(image_size[0]), int(image_size[1])),
        train=train and augment_enabled,
        random_crop=random_crop,
        crop_scale=(float(crop_scale[0]), float(crop_scale[1])),
        hflip_prob=hflip_prob,
        vflip_prob=vflip_prob,
        rot90_prob=rot90_prob,
        brightness=brightness,
        contrast=contrast,
        seed=seed,
    )


def preprocess_image(image: np.ndarray, size: tuple[int, int] | None = None) -> torch.Tensor:
    """Prepare a standalone RGB array for inference (no mask involved)."""
    tensor = _as_chw_float(np.asarray(image))
    if size is not None:
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=tuple(size), mode="bilinear", align_corners=False
        ).squeeze(0)
    return normalize(tensor)
