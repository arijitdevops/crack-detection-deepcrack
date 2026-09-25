"""Flask application factory for the crack-detection web UI.

The app deliberately keeps the model in a lazily-initialised singleton: the
process starts (and ``/api/health`` answers) even when no checkpoint exists
yet, so a fresh clone shows an explanatory page instead of a traceback.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template

from src.config import PROJECT_ROOT, load_config, resolve_checkpoint_path
from src.utils import setup_logging

LOGGER = logging.getLogger("app")

#: Upload extensions the UI accepts.
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"jpg", "jpeg", "png", "bmp", "tif", "tiff"})


def _default_secret_key() -> str:
    """Return ``$SECRET_KEY``, or a random one with a warning."""
    key = os.environ.get("SECRET_KEY")
    if key and key != "change-me-in-production":
        return key
    LOGGER.warning(
        "SECRET_KEY is unset or still the placeholder; generating an ephemeral key. "
        "Sessions will not survive a restart."
    )
    return secrets.token_hex(32)


def create_app(config_overrides: Mapping[str, Any] | None = None) -> Flask:
    """Build and configure the Flask application.

    Parameters
    ----------
    config_overrides:
        Values merged into ``app.config`` after the defaults. Tests use this to
        point ``CHECKPOINT_PATH`` at a temporary file or to force
        ``TESTING=True``.
    """
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    cfg = load_config(os.environ.get("CONFIG_PATH"))
    max_upload_mb = float(os.environ.get("MAX_UPLOAD_MB", "16"))
    upload_dir = Path(os.environ.get("UPLOAD_DIR", PROJECT_ROOT / "uploads"))

    app.config.update(
        SECRET_KEY=_default_secret_key(),
        MAX_CONTENT_LENGTH=int(max_upload_mb * 1024 * 1024),
        MAX_UPLOAD_MB=max_upload_mb,
        ALLOWED_EXTENSIONS=ALLOWED_EXTENSIONS,
        UPLOAD_DIR=upload_dir,
        CRACK_CONFIG=cfg,
        CHECKPOINT_PATH=resolve_checkpoint_path(cfg),
        DEFAULT_THRESHOLD=cfg.eval.threshold,
        JSON_SORT_KEYS=False,
    )
    if config_overrides:
        app.config.update(dict(config_overrides))

    try:
        Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        LOGGER.warning("Could not create upload directory: %s", exc)

    from .routes import bp as main_bp

    app.register_blueprint(main_bp)
    _register_error_handlers(app)

    LOGGER.info(
        "Flask app ready | checkpoint=%s (%s) | max upload %.0f MB",
        app.config["CHECKPOINT_PATH"],
        "found" if Path(app.config["CHECKPOINT_PATH"]).is_file() else "missing",
        max_upload_mb,
    )
    return app


def _wants_json(request) -> bool:
    """True when the caller asked for JSON (API route or Accept header)."""
    return request.path.startswith("/api/") or request.accept_mimetypes.best == "application/json"


def _register_error_handlers(app: Flask) -> None:
    """Attach HTML/JSON error handlers for the statuses the UI can produce."""
    from flask import request

    @app.errorhandler(400)
    def bad_request(error):  # type: ignore[no-untyped-def]
        message = getattr(error, "description", "The request could not be understood.")
        if _wants_json(request):
            return jsonify(error="bad_request", message=message), 400
        return render_template("errors/400.html", message=message), 400

    @app.errorhandler(404)
    def not_found(error):  # type: ignore[no-untyped-def]
        if _wants_json(request):
            return jsonify(error="not_found", message="No such endpoint."), 404
        return render_template("errors/404.html"), 404

    @app.errorhandler(413)
    def too_large(error):  # type: ignore[no-untyped-def]
        limit = app.config.get("MAX_UPLOAD_MB", 16)
        message = f"That file is larger than the {limit:.0f} MB upload limit."
        if _wants_json(request):
            return jsonify(error="payload_too_large", message=message), 413
        return render_template("errors/413.html", message=message), 413

    @app.errorhandler(500)
    def server_error(error):  # type: ignore[no-untyped-def]
        LOGGER.exception("Unhandled server error: %s", error)
        if _wants_json(request):
            return jsonify(error="internal_error", message="Inference failed."), 500
        return render_template("errors/500.html"), 500


__all__ = ["ALLOWED_EXTENSIONS", "create_app"]
