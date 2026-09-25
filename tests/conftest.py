"""Shared pytest fixtures.

The suite never touches the real DeepCrack dataset: every test builds a tiny
synthetic dataset on disk that mirrors the real layout and conventions
(``*.jpg`` images, ``*.png`` masks with values 0 and 255, matching stems).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Make the repository importable when pytest is run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_pair(image_dir: Path, mask_dir: Path, stem: str, size: tuple[int, int]) -> None:
    """Write one synthetic image/mask pair with a diagonal 'crack'."""
    height, width = size
    rng = np.random.default_rng(abs(hash(stem)) % (2**32))
    image = rng.integers(60, 200, size=(height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    for offset in range(-1, 2):
        rows = np.arange(height)
        cols = np.clip((rows * width // height) + offset, 0, width - 1)
        mask[rows, cols] = 255
    image[mask > 0] = 30  # make the crack visibly darker

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(image_dir / f"{stem}.jpg", quality=95)
    Image.fromarray(mask).save(mask_dir / f"{stem}.png")


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a DeepCrack-shaped dataset root and return its path."""
    root = tmp_path / "deep_crack_dataset"
    for split, count in (("train", 6), ("test", 3)):
        image_dir = root / f"{split}_img"
        mask_dir = root / f"{split}_lab"
        for index in range(count):
            _write_pair(image_dir, mask_dir, f"{split}_{index:03d}", (64, 96))
    return root


@pytest.fixture
def train_dirs(synthetic_dataset: Path) -> tuple[Path, Path]:
    """``(train_img, train_lab)`` of the synthetic dataset."""
    return synthetic_dataset / "train_img", synthetic_dataset / "train_lab"


@pytest.fixture
def sample_config(synthetic_dataset: Path):
    """A :class:`~src.config.Config` pointed at the synthetic dataset."""
    from src.config import load_config

    cfg = load_config(
        overrides={
            "data": {"num_workers": 0, "image_size": [64, 64], "val_split": 0.34},
            "train": {"batch_size": 2, "epochs": 1, "amp": False},
            "model": {"base_channels": 4, "depth": 2},
        }
    )
    cfg.data.data_dir = synthetic_dataset
    return cfg


@pytest.fixture
def tiny_model():
    """A very small U-Net that trains and runs quickly on CPU."""
    from src.model import build_model

    return build_model("unet", in_channels=3, num_classes=1, base_channels=4, depth=2)


@pytest.fixture
def flask_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A Flask test app whose checkpoint path deliberately does not exist."""
    from app import create_app
    from app.routes import reset_model_cache

    monkeypatch.setenv("SECRET_KEY", "test-key-not-a-real-secret")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    reset_model_cache()
    app = create_app(
        {
            "TESTING": True,
            "CHECKPOINT_PATH": tmp_path / "missing" / "best.pt",
            "UPLOAD_DIR": tmp_path / "uploads",
        }
    )
    yield app
    reset_model_cache()


@pytest.fixture
def client(flask_app):
    """Flask test client for the app above."""
    return flask_app.test_client()
