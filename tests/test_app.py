"""Flask application tests.

These run without any checkpoint on disk: the point is that the app starts,
answers its health probe and explains how to train a model instead of
returning a traceback.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image


def _png_upload(size: tuple[int, int] = (32, 32), name: str = "sample.png") -> tuple:
    """Build an in-memory PNG suitable for ``data={"image": ...}``."""
    array = np.random.default_rng(0).integers(0, 255, (*size, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer, name


def test_health_reports_degraded_without_a_checkpoint(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "degraded"
    assert payload["checkpoint_available"] is False
    assert payload["model_loaded"] is False
    assert "python -m src.train" in payload["message"]
    assert payload["version"]


def test_health_lists_the_configured_model_and_device(client) -> None:
    payload = client.get("/api/health").get_json()
    assert payload["model"] in {"unet", "unet_resnet34"}
    assert payload["device"]
    assert 0.0 < payload["default_threshold"] < 1.0


def test_index_renders_the_missing_checkpoint_banner(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "No trained model yet" in body
    assert "python -m src.train" in body


def test_predict_without_a_checkpoint_returns_503_not_a_traceback(client) -> None:
    buffer, name = _png_upload()
    response = client.post(
        "/predict",
        data={"image": (buffer, name), "threshold": "0.5"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 503
    assert "python -m src.train" in response.get_data(as_text=True)


def test_api_predict_without_a_checkpoint_returns_a_json_error(client) -> None:
    buffer, name = _png_upload()
    response = client.post(
        "/api/predict",
        data={"image": (buffer, name)},
        content_type="multipart/form-data",
    )
    assert response.status_code == 503
    payload = response.get_json()
    assert payload["error"] == "checkpoint_missing"
    assert "python -m src.train" in payload["message"]


def test_predict_rejects_a_missing_file(client) -> None:
    response = client.post("/predict", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Choose an image" in response.get_data(as_text=True)


def test_predict_rejects_a_disallowed_extension(client) -> None:
    response = client.post(
        "/predict",
        data={"image": (io.BytesIO(b"MZ not an image"), "payload.exe")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.get_data(as_text=True)


def test_api_predict_rejects_a_disallowed_extension(client) -> None:
    response = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(b"not an image"), "script.sh")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "unsupported_type"


def test_api_predict_requires_an_image(client) -> None:
    response = client.post("/api/predict", json={"threshold": 0.5})
    assert response.status_code == 400
    assert response.get_json()["error"] == "missing_image"


def test_api_predict_rejects_undecodable_base64(client) -> None:
    encoded = base64.b64encode(b"definitely not a png").decode("ascii")
    response = client.post("/api/predict", json={"image_base64": encoded})
    assert response.status_code == 400
    assert response.get_json()["error"] == "bad_image"


def test_upload_size_cap_is_configured(flask_app) -> None:
    assert flask_app.config["MAX_CONTENT_LENGTH"] == int(
        flask_app.config["MAX_UPLOAD_MB"] * 1024 * 1024
    )


def test_unknown_route_returns_the_404_page(client) -> None:
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert "Page not found" in response.get_data(as_text=True)


def test_unknown_api_route_returns_json_404(client) -> None:
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["error"] == "not_found"


def test_expired_download_token_returns_404(client) -> None:
    assert client.get("/download/deadbeef").status_code == 404


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("photo.jpg", True),
        ("photo.JPEG", True),
        ("scan.tif", True),
        ("archive.zip", False),
        ("noextension", False),
    ],
)
def test_allowed_file_extension_rules(flask_app, filename: str, expected: bool) -> None:
    from app.routes import allowed_file

    with flask_app.app_context():
        assert allowed_file(filename) is expected


def test_threshold_is_clamped_into_range(flask_app) -> None:
    from app.routes import _parse_threshold

    assert _parse_threshold("0.0", 0.5) == pytest.approx(0.05)
    assert _parse_threshold("2.0", 0.5) == pytest.approx(0.95)
    assert _parse_threshold("not a number", 0.42) == pytest.approx(0.42)
    assert _parse_threshold(None, 0.3) == pytest.approx(0.3)


def test_end_to_end_prediction_with_a_trained_checkpoint(tmp_path, monkeypatch) -> None:
    """Save a (randomly initialised) checkpoint and exercise the full path."""
    import torch

    from app import create_app
    from app.routes import reset_model_cache
    from src.config import load_config
    from src.model import build_model
    from src.utils import save_checkpoint

    cfg = load_config(overrides={"model": {"base_channels": 4, "depth": 2}})
    model = build_model("unet", base_channels=4, depth=2)
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(checkpoint, model=model, epoch=1, metrics={"dice": 0.0}, config=cfg.to_dict())

    monkeypatch.setenv("SECRET_KEY", "test-key-not-a-real-secret")
    reset_model_cache()
    app = create_app({"TESTING": True, "CHECKPOINT_PATH": checkpoint})
    client = app.test_client()
    try:
        health = client.get("/api/health").get_json()
        assert health["status"] == "ok"

        buffer, name = _png_upload((48, 64))
        response = client.post(
            "/api/predict",
            data={"image": (buffer, name), "threshold": "0.5"},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["width"] == 64 and payload["height"] == 48
        assert 0.0 <= payload["crack_pixel_ratio"] <= 1.0
        assert payload["mask_png_base64"] and payload["overlay_png_base64"]

        decoded = base64.b64decode(payload["overlay_png_base64"])
        with Image.open(io.BytesIO(decoded)) as handle:
            assert handle.size == (64, 48)

        buffer, name = _png_upload((48, 64))
        html = client.post(
            "/predict",
            data={"image": (buffer, name), "threshold": "0.4"},
            content_type="multipart/form-data",
        )
        assert html.status_code == 200
        body = html.get_data(as_text=True)
        assert "Detection result" in body
        assert "Download overlay" in body
        torch.manual_seed(0)
    finally:
        reset_model_cache()
