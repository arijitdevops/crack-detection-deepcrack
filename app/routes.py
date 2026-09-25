"""HTTP routes for the crack-detection UI.

=========================  ======  ====================================================
Route                      Method  Purpose
=========================  ======  ====================================================
``/``                      GET     Upload form and model status.
``/predict``               POST    Run inference on an upload, render the result page.
``/download/<token>``      GET     Download the generated overlay PNG.
``/api/predict``           POST    JSON inference endpoint (multipart or base64).
``/api/health``            GET     Liveness probe and checkpoint status.
=========================  ======  ====================================================
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
)
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from src import __version__
from src.predict import predict_array
from src.utils import crack_pixel_ratio, estimate_crack_length_px, get_device, mask_to_overlay

LOGGER = logging.getLogger("app.routes")

bp = Blueprint("main", __name__)

#: Lazily-created model singleton, guarded because Flask serves concurrently.
_MODEL: torch.nn.Module | None = None
_MODEL_LOCK = threading.Lock()

#: In-memory store of generated overlays, keyed by a random token.
_DOWNLOADS: dict[str, tuple[str, bytes]] = {}
_MAX_DOWNLOADS = 32


class CheckpointMissingError(RuntimeError):
    """Raised when inference is requested but no checkpoint exists."""


def checkpoint_path() -> Path:
    """Path of the checkpoint this app instance would load."""
    return Path(current_app.config["CHECKPOINT_PATH"])


def checkpoint_available() -> bool:
    """Whether a usable checkpoint is present on disk."""
    return checkpoint_path().is_file()


def get_model() -> torch.nn.Module:
    """Return the loaded model, loading it on first use.

    Raises
    ------
    CheckpointMissingError
        If no checkpoint file exists. Callers turn this into a friendly page
        rather than a 500.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:  # another thread won the race
            return _MODEL
        path = checkpoint_path()
        if not path.is_file():
            raise CheckpointMissingError(str(path))
        cfg = current_app.config["CRACK_CONFIG"]
        device = get_device(cfg.device)
        from src.evaluate import load_model_from_checkpoint

        model, _ = load_model_from_checkpoint(path, cfg, device)
        _MODEL = model
        LOGGER.info("Model loaded from %s on %s", path, device)
        return _MODEL


def reset_model_cache() -> None:
    """Drop the cached model. Used by tests and after retraining."""
    global _MODEL
    with _MODEL_LOCK:
        _MODEL = None


def allowed_file(filename: str) -> bool:
    """Extension allowlist check on a (possibly hostile) filename."""
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in current_app.config["ALLOWED_EXTENSIONS"]


def _safe_name(filename: str | None) -> str:
    """Sanitise an upload filename, falling back to a generated one."""
    cleaned = secure_filename(filename or "")
    return cleaned or f"upload_{uuid.uuid4().hex[:8]}.png"


def _read_upload(storage) -> np.ndarray:
    """Decode an uploaded file into an ``(H, W, 3)`` uint8 RGB array."""
    try:
        data = storage.read()
        with Image.open(io.BytesIO(data)) as handle:
            handle.load()
            return np.array(handle.convert("RGB"))
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"Could not decode the uploaded image: {exc}") from exc


def _png_bytes(array: np.ndarray) -> bytes:
    """Encode a uint8 array as PNG bytes."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _data_uri(array: np.ndarray) -> str:
    """Encode an array as an inline ``data:`` URI for templates."""
    return "data:image/png;base64," + base64.b64encode(_png_bytes(array)).decode("ascii")


def _register_download(name: str, payload: bytes) -> str:
    """Store *payload* for later download and return its token."""
    token = uuid.uuid4().hex
    _DOWNLOADS[token] = (name, payload)
    while len(_DOWNLOADS) > _MAX_DOWNLOADS:
        _DOWNLOADS.pop(next(iter(_DOWNLOADS)))
    return token


def _parse_threshold(raw: Any, default: float) -> float:
    """Clamp a user-supplied threshold into ``[0.05, 0.95]``."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return float(min(max(value, 0.05), 0.95))


def _run_inference(
    image: np.ndarray, threshold: float
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Run the model and return (stats, mask_rgb, overlay_rgb)."""
    cfg = current_app.config["CRACK_CONFIG"]
    model = get_model()
    device = get_device(cfg.device)
    probs, mask = predict_array(image, model, cfg, device=device, threshold=threshold)

    overlay = mask_to_overlay(image, mask)
    mask_rgb = (mask.astype(np.uint8) * 255)
    ratio = crack_pixel_ratio(mask)
    length_px = estimate_crack_length_px(mask)

    result: dict[str, Any] = {
        "threshold": round(float(threshold), 3),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "crack_pixels": int(mask.sum()),
        "crack_pixel_ratio": float(ratio),
        "crack_pixel_percent": round(float(ratio) * 100.0, 4),
        "estimated_crack_length_px": round(float(length_px), 1),
        "mean_probability": round(float(probs.mean()), 6),
        "max_probability": round(float(probs.max()), 6),
        "detected": bool(mask.sum() > 0),
    }
    if cfg.inference.pixel_size_mm:
        result["estimated_crack_length_mm"] = round(length_px * cfg.inference.pixel_size_mm, 1)
    return result, mask_rgb, overlay


# ---------------------------------------------------------------------- #
# HTML routes
# ---------------------------------------------------------------------- #
@bp.route("/", methods=["GET"])
def index():
    """Upload form, plus a banner when the model has not been trained yet."""
    return render_template(
        "index.html",
        checkpoint_ready=checkpoint_available(),
        checkpoint_path=str(checkpoint_path()),
        default_threshold=current_app.config["DEFAULT_THRESHOLD"],
        max_upload_mb=current_app.config["MAX_UPLOAD_MB"],
        allowed=sorted(current_app.config["ALLOWED_EXTENSIONS"]),
    )


@bp.route("/predict", methods=["POST"])
def predict():
    """Run inference on an uploaded image and render the comparison page."""
    threshold = _parse_threshold(
        request.form.get("threshold"), current_app.config["DEFAULT_THRESHOLD"]
    )

    storage = request.files.get("image")
    if storage is None or not storage.filename:
        abort(400, description="Choose an image to analyse.")
    filename = _safe_name(storage.filename)
    if not allowed_file(filename):
        allowed = ", ".join(sorted(current_app.config["ALLOWED_EXTENSIONS"]))
        abort(400, description=f"Unsupported file type. Allowed extensions: {allowed}.")

    try:
        image = _read_upload(storage)
    except ValueError as exc:
        abort(400, description=str(exc))

    try:
        result, mask_rgb, overlay = _run_inference(image, threshold)
    except CheckpointMissingError:
        return (
            render_template(
                "index.html",
                checkpoint_ready=False,
                checkpoint_path=str(checkpoint_path()),
                default_threshold=threshold,
                max_upload_mb=current_app.config["MAX_UPLOAD_MB"],
                allowed=sorted(current_app.config["ALLOWED_EXTENSIONS"]),
            ),
            503,
        )
    except (RuntimeError, ValueError) as exc:
        LOGGER.exception("Inference failed for %s", filename)
        abort(400, description=f"Inference failed: {exc}")

    token = _register_download(f"{Path(filename).stem}_overlay.png", _png_bytes(overlay))
    return render_template(
        "result.html",
        filename=filename,
        result=result,
        original_uri=_data_uri(image),
        mask_uri=_data_uri(mask_rgb),
        overlay_uri=_data_uri(overlay),
        download_token=token,
        default_threshold=threshold,
    )


@bp.route("/download/<token>", methods=["GET"])
def download(token: str):
    """Serve a previously generated overlay PNG."""
    entry = _DOWNLOADS.get(token)
    if entry is None:
        abort(404)
    name, payload = entry
    return send_file(
        io.BytesIO(payload),
        mimetype="image/png",
        as_attachment=True,
        download_name=name,
    )


# ---------------------------------------------------------------------- #
# JSON API
# ---------------------------------------------------------------------- #
@bp.route("/api/health", methods=["GET"])
def api_health():
    """Liveness probe reporting whether a checkpoint is loadable."""
    ready = checkpoint_available()
    cfg = current_app.config["CRACK_CONFIG"]
    return (
        jsonify(
            status="ok" if ready else "degraded",
            version=__version__,
            model=cfg.model.name,
            checkpoint=str(checkpoint_path()),
            checkpoint_available=ready,
            model_loaded=_MODEL is not None,
            device=str(get_device(cfg.device)),
            default_threshold=current_app.config["DEFAULT_THRESHOLD"],
            message=None if ready else "No checkpoint found. Run: python -m src.train",
        ),
        200,
    )


@bp.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON inference endpoint.

    Accepts either ``multipart/form-data`` with an ``image`` file, or a JSON
    body ``{"image_base64": "...", "threshold": 0.5}``. Returns the crack
    statistics plus base64 PNGs of the mask and overlay.
    """
    threshold_raw: Any = None
    image: np.ndarray | None = None

    if request.files.get("image") is not None:
        storage = request.files["image"]
        filename = _safe_name(storage.filename)
        if not allowed_file(filename):
            return jsonify(error="unsupported_type", message="Unsupported file extension."), 400
        try:
            image = _read_upload(storage)
        except ValueError as exc:
            return jsonify(error="bad_image", message=str(exc)), 400
        threshold_raw = request.form.get("threshold")
    else:
        payload = request.get_json(silent=True) or {}
        encoded = payload.get("image_base64")
        if not encoded:
            return (
                jsonify(
                    error="missing_image",
                    message="Provide an 'image' file upload or an 'image_base64' JSON field.",
                ),
                400,
            )
        try:
            if "," in encoded[:64]:  # strip a data: URI prefix if present
                encoded = encoded.split(",", 1)[1]
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as handle:
                handle.load()
                image = np.array(handle.convert("RGB"))
        except (ValueError, OSError, UnidentifiedImageError) as exc:
            return jsonify(error="bad_image", message=f"Could not decode image: {exc}"), 400
        threshold_raw = payload.get("threshold")

    threshold = _parse_threshold(threshold_raw, current_app.config["DEFAULT_THRESHOLD"])

    try:
        result, mask_rgb, overlay = _run_inference(image, threshold)
    except CheckpointMissingError as exc:
        return (
            jsonify(
                error="checkpoint_missing",
                message="No trained checkpoint is available. Run: python -m src.train",
                checkpoint=str(exc),
            ),
            503,
        )
    except (RuntimeError, ValueError) as exc:
        LOGGER.exception("API inference failed")
        return jsonify(error="inference_failed", message=str(exc)), 500

    result["mask_png_base64"] = base64.b64encode(_png_bytes(mask_rgb)).decode("ascii")
    result["overlay_png_base64"] = base64.b64encode(_png_bytes(overlay)).decode("ascii")
    return jsonify(result), 200
