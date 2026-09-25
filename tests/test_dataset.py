"""Dataset pairing, splitting and augmentation-alignment tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset import (
    CrackSegmentationDataset,
    DatasetPairingError,
    discover_pairs,
    split_pairs,
)
from src.transforms import JointTransform


def test_discover_pairs_matches_by_stem(train_dirs: tuple[Path, Path]) -> None:
    image_dir, mask_dir = train_dirs
    pairs = discover_pairs(image_dir, mask_dir)
    assert len(pairs) == 6
    for pair in pairs:
        assert pair.image_path.suffix == ".jpg"
        assert pair.mask_path.suffix == ".png"
        assert pair.image_path.stem == pair.mask_path.stem == pair.stem


def test_discover_pairs_is_sorted(train_dirs: tuple[Path, Path]) -> None:
    stems = [p.stem for p in discover_pairs(*train_dirs)]
    assert stems == sorted(stems)


def test_missing_mask_raises_and_names_the_file(
    train_dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    image_dir, mask_dir = train_dirs
    orphan = image_dir / "orphan_image.jpg"
    orphan.write_bytes((image_dir / "train_000.jpg").read_bytes())

    with pytest.raises(DatasetPairingError) as excinfo:
        discover_pairs(image_dir, mask_dir)
    message = str(excinfo.value)
    assert "orphan_image" in message
    assert "without a mask" in message


def test_missing_image_raises_and_names_the_file(train_dirs: tuple[Path, Path]) -> None:
    image_dir, mask_dir = train_dirs
    orphan = mask_dir / "orphan_mask.png"
    orphan.write_bytes((mask_dir / "train_000.png").read_bytes())

    with pytest.raises(DatasetPairingError) as excinfo:
        discover_pairs(image_dir, mask_dir)
    assert "orphan_mask" in str(excinfo.value)


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_pairs(tmp_path / "nope_img", tmp_path / "nope_lab")


def test_item_shapes_and_mask_is_binary(train_dirs: tuple[Path, Path]) -> None:
    dataset = CrackSegmentationDataset.from_directories(*train_dirs, image_size=(48, 48))
    image, mask = dataset[0]
    assert image.shape == (3, 48, 48)
    assert mask.shape == (1, 48, 48)
    assert image.dtype == torch.float32 and mask.dtype == torch.float32
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
    assert mask.sum() > 0, "the synthetic crack should survive the resize"


def test_return_meta_carries_stem_and_original_size(train_dirs: tuple[Path, Path]) -> None:
    dataset = CrackSegmentationDataset.from_directories(
        *train_dirs, image_size=(32, 32), return_meta=True
    )
    _, _, meta = dataset[0]
    assert meta["stem"].startswith("train_")
    assert (meta["height"], meta["width"]) == (64, 96)


def test_split_is_deterministic_and_disjoint(train_dirs: tuple[Path, Path]) -> None:
    pairs = discover_pairs(*train_dirs)
    train_a, val_a = split_pairs(pairs, 0.34, seed=7)
    train_b, val_b = split_pairs(pairs, 0.34, seed=7)

    assert [p.stem for p in train_a] == [p.stem for p in train_b]
    assert [p.stem for p in val_a] == [p.stem for p in val_b]
    assert len(train_a) + len(val_a) == len(pairs)
    assert not set(p.stem for p in train_a) & set(p.stem for p in val_a)


def test_split_depends_on_the_seed(train_dirs: tuple[Path, Path]) -> None:
    """Different seeds must be able to produce different validation sets.

    Any single pair of seeds can coincide by chance on six samples, so this
    checks that the seed has an effect across a range of seeds instead.
    """
    pairs = discover_pairs(*train_dirs)
    splits = {
        tuple(sorted(p.stem for p in split_pairs(pairs, 0.5, seed=seed)[1]))
        for seed in range(10)
    }
    assert len(splits) > 1


def test_augmentation_keeps_image_and_mask_aligned() -> None:
    """A geometric transform must move image and mask identically.

    The image is built so its crack pixels are exactly the mask's crack
    pixels; after augmentation the dark pixels must still coincide with the
    positive mask, whatever flip or rotation was drawn.
    """
    height, width = 32, 40
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[4:8, 10:30] = 255              # a horizontal bar, not flip-symmetric
    mask[20:24, 2:6] = 255              # plus an off-centre blob
    image = np.full((height, width, 3), 200, dtype=np.uint8)
    image[mask > 0] = 0                  # crack pixels are black

    for seed in range(12):
        transform = JointTransform(
            image_size=(height, width),
            train=True,
            random_crop=False,
            brightness=0.0,
            contrast=0.0,
            seed=seed,
        )
        image_t, mask_t = transform(image, mask)
        # Un-normalise back to a dark/light decision.
        dark = image_t.mean(dim=0) < image_t.mean()
        positive = mask_t[0] > 0.5
        overlap = torch.logical_and(dark, positive).sum().item()
        assert positive.sum().item() > 0
        assert overlap >= 0.9 * positive.sum().item(), f"misaligned for seed {seed}"


def test_validation_transform_is_deterministic() -> None:
    image = np.random.default_rng(0).integers(0, 255, (24, 24, 3), dtype=np.uint8)
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[10:14, :] = 255
    transform = JointTransform(image_size=(24, 24), train=False)
    first = transform(image, mask)
    second = transform(image, mask)
    assert torch.allclose(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_transform_rejects_mismatched_sizes() -> None:
    transform = JointTransform(image_size=(16, 16), train=False)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match"):
        transform(image, mask)


def test_build_datasets_holds_out_the_test_split(sample_config) -> None:
    from src.dataset import build_datasets, build_test_dataset

    train_ds, val_ds = build_datasets(sample_config)
    test_ds = build_test_dataset(sample_config)

    train_stems = {p.stem for p in train_ds.pairs} | {p.stem for p in val_ds.pairs}
    test_stems = {p.stem for p in test_ds.pairs}
    assert len(train_stems) == 6
    assert len(test_stems) == 3
    assert not train_stems & test_stems
