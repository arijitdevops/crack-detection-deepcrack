"""DeepCrack dataset loading.

Layout expected under ``DATA_DIR`` (see :mod:`src.config`)::

    deep_crack_dataset/
        train_img/  11111.jpg  ...   (300 RGB photographs)
        train_lab/  11111.png  ...   (300 binary masks, 0 = background, 255 = crack)
        test_img/   11125-1.jpg ...  (237 RGB photographs)
        test_lab/   11125-1.png ...  (237 binary masks)

An image and its mask correspond when their **file stems match exactly**: the
image ``train_img/11111.jpg`` pairs with ``train_lab/11111.png``. Frames are
544x384 (landscape) or 384x544 (portrait); the transform pipeline resizes them
to a single configurable training size.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .transforms import JointTransform, build_transforms

LOGGER = logging.getLogger(__name__)


class DatasetPairingError(RuntimeError):
    """Raised when images and masks cannot be paired one-to-one."""


@dataclass(frozen=True)
class SamplePair:
    """A single image/mask pair on disk."""

    stem: str
    image_path: Path
    mask_path: Path


def _list_by_stem(directory: Path, suffix: str) -> dict[str, Path]:
    """Map ``stem -> path`` for every ``*suffix`` file in *directory*.

    The suffix match is case-insensitive so that ``.JPG`` files are picked up.
    """
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {directory}. "
            "Set DATA_DIR or run scripts/download_data.py."
        )
    suffix = suffix.lower()
    found: dict[str, Path] = {}
    try:
        entries = sorted(directory.iterdir())
    except OSError as exc:
        raise FileNotFoundError(f"Could not list {directory}: {exc}") from exc
    for path in entries:
        if path.is_file() and path.suffix.lower() == suffix:
            if path.stem in found:
                raise DatasetPairingError(
                    f"Duplicate stem {path.stem!r} in {directory}: "
                    f"{found[path.stem].name} and {path.name}"
                )
            found[path.stem] = path
    return found


def discover_pairs(
    image_dir: str | os.PathLike[str],
    mask_dir: str | os.PathLike[str],
    *,
    image_suffix: str = ".jpg",
    mask_suffix: str = ".png",
    max_reported: int = 10,
) -> list[SamplePair]:
    """Pair images with masks by file stem.

    Raises
    ------
    DatasetPairingError
        If any image has no mask or any mask has no image. The message lists
        up to *max_reported* offending stems on each side rather than silently
        dropping them, because a half-loaded dataset is a subtle way to get
        meaningless metrics.
    FileNotFoundError
        If either directory is missing.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    images = _list_by_stem(image_dir, image_suffix)
    masks = _list_by_stem(mask_dir, mask_suffix)

    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        parts: list[str] = [
            f"Unpaired files between {image_dir} ({len(images)} images) "
            f"and {mask_dir} ({len(masks)} masks)."
        ]
        if missing_masks:
            shown = ", ".join(missing_masks[:max_reported])
            suffix = f" (+{len(missing_masks) - max_reported} more)" if len(missing_masks) > max_reported else ""
            parts.append(f"Images without a mask: {shown}{suffix}")
        if missing_images:
            shown = ", ".join(missing_images[:max_reported])
            suffix = f" (+{len(missing_images) - max_reported} more)" if len(missing_images) > max_reported else ""
            parts.append(f"Masks without an image: {shown}{suffix}")
        raise DatasetPairingError(" ".join(parts))

    if not images:
        raise DatasetPairingError(
            f"No {image_suffix} files found in {image_dir}; is DATA_DIR pointing at the right place?"
        )

    return [SamplePair(stem, images[stem], masks[stem]) for stem in sorted(images)]


class CrackSegmentationDataset(Dataset):
    """Binary crack-segmentation dataset returning ``(image, mask)`` tensors.

    Each item is a tuple of

    * ``image``: ``(3, H, W)`` float32, ImageNet-normalised;
    * ``mask``:  ``(1, H, W)`` float32 containing only 0.0 and 1.0.

    Parameters
    ----------
    pairs:
        The image/mask pairs this dataset instance serves. Use
        :func:`discover_pairs` or the :meth:`from_config` helpers to build it.
    transform:
        A :class:`~src.transforms.JointTransform`. When ``None`` a
        deterministic resize-and-normalise transform is created.
    image_size:
        Used only when *transform* is ``None``.
    return_meta:
        When ``True`` items become ``(image, mask, meta)`` where ``meta`` holds
        the stem and the original size. Useful for qualitative reports.
    """

    def __init__(
        self,
        pairs: Sequence[SamplePair],
        *,
        transform: JointTransform | None = None,
        image_size: tuple[int, int] = (384, 384),
        return_meta: bool = False,
    ) -> None:
        if not pairs:
            raise ValueError("CrackSegmentationDataset requires at least one sample pair")
        self.pairs: list[SamplePair] = list(pairs)
        self.transform = transform or build_transforms(image_size, train=False)
        self.return_meta = return_meta

    # ------------------------------------------------------------------ #
    # constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_directories(
        cls,
        image_dir: str | os.PathLike[str],
        mask_dir: str | os.PathLike[str],
        *,
        image_suffix: str = ".jpg",
        mask_suffix: str = ".png",
        **kwargs,
    ) -> CrackSegmentationDataset:
        """Build a dataset from a pair of directories."""
        pairs = discover_pairs(
            image_dir, mask_dir, image_suffix=image_suffix, mask_suffix=mask_suffix
        )
        return cls(pairs, **kwargs)

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        image = self._read_image(pair.image_path)
        mask = self._read_mask(pair.mask_path)
        if image.shape[:2] != mask.shape[:2]:
            raise DatasetPairingError(
                f"Size mismatch for {pair.stem}: image {image.shape[:2]} vs mask {mask.shape[:2]}"
            )
        image_t, mask_t = self.transform(image, mask)
        if self.return_meta:
            meta = {
                "stem": pair.stem,
                "image_path": str(pair.image_path),
                "mask_path": str(pair.mask_path),
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
            }
            return image_t, mask_t, meta
        return image_t, mask_t

    # ------------------------------------------------------------------ #
    # I/O helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        """Read an RGB image as ``(H, W, 3)`` uint8."""
        try:
            with Image.open(path) as handle:
                return np.array(handle.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            raise OSError(f"Could not read image {path}: {exc}") from exc

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        """Read a binary mask as ``(H, W)`` uint8 (DeepCrack uses 0 and 255)."""
        try:
            with Image.open(path) as handle:
                return np.array(handle.convert("L"))
        except (OSError, UnidentifiedImageError) as exc:
            raise OSError(f"Could not read mask {path}: {exc}") from exc

    def positive_pixel_fraction(self, limit: int | None = 50) -> float:
        """Average fraction of crack pixels over the first *limit* masks.

        Handy for choosing ``loss.pos_weight``: DeepCrack masks are sparse.
        """
        subset = self.pairs if limit is None else self.pairs[:limit]
        total = 0.0
        for pair in subset:
            mask = self._read_mask(pair.mask_path)
            total += float((mask > 127).mean())
        return total / max(len(subset), 1)


def split_pairs(
    pairs: Sequence[SamplePair], val_split: float, seed: int = 42
) -> tuple[list[SamplePair], list[SamplePair]]:
    """Deterministically split *pairs* into ``(train, val)``.

    The shuffle uses a dedicated :class:`random.Random` so the split does not
    depend on, or disturb, the global RNG state.
    """
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")
    ordered = sorted(pairs, key=lambda p: p.stem)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_val = int(round(len(ordered) * val_split))
    if val_split > 0.0:
        n_val = max(1, min(n_val, len(ordered) - 1))
    return ordered[n_val:], ordered[:n_val]


def build_datasets(
    cfg: Config,
) -> tuple[CrackSegmentationDataset, CrackSegmentationDataset]:
    """Create the training and validation datasets described by *cfg*.

    ``train_img``/``train_lab`` are split with ``data.val_split``; the
    ``test_*`` directories are never touched here so they stay a clean
    held-out set for :mod:`src.evaluate`.
    """
    pairs = discover_pairs(
        cfg.data.train_image_dir,
        cfg.data.train_mask_dir,
        image_suffix=cfg.data.image_suffix,
        mask_suffix=cfg.data.mask_suffix,
    )
    train_pairs, val_pairs = split_pairs(pairs, cfg.data.val_split, seed=cfg.seed)
    LOGGER.info(
        "Discovered %d training pairs -> %d train / %d val",
        len(pairs),
        len(train_pairs),
        len(val_pairs),
    )

    train_tf = build_transforms(
        cfg.data.image_size,
        train=True,
        augment_enabled=cfg.augment.enabled,
        random_crop=cfg.augment.random_crop,
        crop_scale=cfg.augment.crop_scale,
        hflip_prob=cfg.augment.hflip_prob,
        vflip_prob=cfg.augment.vflip_prob,
        rot90_prob=cfg.augment.rot90_prob,
        brightness=cfg.augment.brightness,
        contrast=cfg.augment.contrast,
    )
    val_tf = build_transforms(cfg.data.image_size, train=False)

    train_ds = CrackSegmentationDataset(train_pairs, transform=train_tf)
    val_ds = CrackSegmentationDataset(val_pairs or train_pairs[:1], transform=val_tf)
    return train_ds, val_ds


def build_test_dataset(cfg: Config, *, return_meta: bool = True) -> CrackSegmentationDataset:
    """Create the held-out test dataset from ``test_img``/``test_lab``."""
    pairs = discover_pairs(
        cfg.data.test_image_dir,
        cfg.data.test_mask_dir,
        image_suffix=cfg.data.image_suffix,
        mask_suffix=cfg.data.mask_suffix,
    )
    LOGGER.info("Discovered %d test pairs", len(pairs))
    transform = build_transforms(cfg.data.image_size, train=False)
    return CrackSegmentationDataset(pairs, transform=transform, return_meta=return_meta)


def collate_with_meta(batch: list):
    """Collate ``(image, mask, meta)`` items, keeping ``meta`` as a list of dicts."""
    images = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    metas = [item[2] for item in batch]
    return images, masks, metas


def build_dataloaders(
    cfg: Config,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation :class:`~torch.utils.data.DataLoader` objects."""
    train_ds, val_ds = build_datasets(cfg)
    common = {
        "num_workers": cfg.data.num_workers,
        "pin_memory": cfg.data.pin_memory and torch.cuda.is_available(),
    }
    if cfg.data.num_workers > 0:
        common["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=len(train_ds) > cfg.train.batch_size,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader


def build_test_dataloader(cfg: Config) -> DataLoader:
    """Create the held-out test loader, yielding ``(images, masks, metas)``."""
    test_ds = build_test_dataset(cfg, return_meta=True)
    return DataLoader(
        test_ds,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and torch.cuda.is_available(),
        collate_fn=collate_with_meta,
    )
