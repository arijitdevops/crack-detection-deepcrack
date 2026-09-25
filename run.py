"""Development entry point for the Flask UI.

::

    python run.py                 # http://127.0.0.1:5000
    python run.py --port 8000 --host 0.0.0.0

For anything beyond local development use a real WSGI server::

    waitress-serve --port=8000 "app:create_app()"       # Windows-friendly
    gunicorn -w 2 -b 0.0.0.0:8000 "app:create_app()"    # POSIX
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from app import create_app
from src.utils import setup_logging

LOGGER = logging.getLogger("run")


def build_parser() -> argparse.ArgumentParser:
    """CLI options for the development server."""
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description="Run the crack-detection web UI (development server).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=os.environ.get("FLASK_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FLASK_PORT", "5000")))
    parser.add_argument("--debug", action="store_true", help="Enable the Flask reloader/debugger")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Start the development server. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    debug = args.debug or os.environ.get("FLASK_ENV", "").lower() == "development"
    try:
        application = create_app()
    except (OSError, ValueError) as exc:
        LOGGER.error("Could not start the application: %s", exc)
        return 1

    LOGGER.info("Serving on http://%s:%d (debug=%s)", args.host, args.port, debug)
    application.run(host=args.host, port=args.port, debug=debug, use_reloader=debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
