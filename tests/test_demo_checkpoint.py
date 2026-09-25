"""The bundled CPU demo checkpoint loads and runs on the bundled samples."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import DEMO_CHECKPOINT_PATH, PROJECT_ROOT, load_config, resolve_checkpoint_path
from src.evaluate import load_model_from_checkpoint
from src.predict import predict_array

pytestmark = pytest.mark.skipif(
    not DEMO_CHECKPOINT_PATH.is_file(), reason="demo checkpoint not present"
)


def test_demo_checkpoint_is_small_and_weights_only() -> None:
    assert DEMO_CHECKPOINT_PATH.stat().st_size < 25 * 1024 * 1024
    payload = torch.load(DEMO_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    assert "optimizer_state" not in payload
    assert payload["config"]["data"]["image_size"] == [256, 256]


def test_demo_checkpoint_is_the_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CHECKPOINT_PATH", raising=False)
    cfg = load_config(use_env=False, overrides={"train": {"checkpoint_dir": str(tmp_path)}})
    assert resolve_checkpoint_path(cfg) == DEMO_CHECKPOINT_PATH
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "custom.pt"))
    assert resolve_checkpoint_path(cfg) == tmp_path / "custom.pt"


def test_demo_model_segments_a_sample_crack() -> None:
    cfg = load_config(use_env=False)
    model, _ = load_model_from_checkpoint(DEMO_CHECKPOINT_PATH, cfg, torch.device("cpu"))
    assert tuple(cfg.data.image_size) == (256, 256)  # adopted from the checkpoint

    image = np.array(Image.open(PROJECT_ROOT / "samples" / "11304.jpg").convert("RGB"))
    truth = np.array(Image.open(PROJECT_ROOT / "samples" / "masks" / "11304.png")) > 127
    _, mask = predict_array(image, model, cfg, device=torch.device("cpu"), threshold=0.5)
    assert mask.shape == truth.shape
    pred = mask > 0
    dice = 2 * (pred & truth).sum() / (pred.sum() + truth.sum())
    assert dice > 0.6
