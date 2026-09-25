"""Inference on a single image or a directory of images.

Usage
-----
::

    python -m src.predict --input samples/11289-11.jpg --output-dir reports/predictions
    python -m src.predict --input samples/ --threshold 0.4 --csv reports/predictions.csv
    python -m src.predict --input big_photo.jpg --tile   # force sliding-window inference

For every input the CLI writes ``<stem>_mask.png`` (binary mask) and
``<stem>_overlay.png`` (crack pixels tinted over the original) and reports the
crack-pixel ratio.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from .config import Config, load_config, resolve_checkpoint_path
from .transforms import preprocess_image
from .utils import (
    crack_pixel_ratio,
    estimate_crack_length_px,
    get_device,
    mask_to_overlay,
    setup_logging,
    sliding_window_inference,
)

LOGGER = logging.getLogger("src.predict")

#: Extensions the CLI will pick up when pointed at a directory.
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for prediction."""
    parser = argparse.ArgumentParser(
        prog="python -m src.predict",
        description="Predict crack masks for an image or a directory of images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, required=True, help="Image file or directory")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load")
    parser.add_argument(
        "--output-dir", "-o", type=str, default="reports/predictions", help="Where to write results"
    )
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold")
    parser.add_argument("--device", type=str, default=None, help="auto | cuda | cpu | mps")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV summary path")
    parser.add_argument(
        "--tile", action="store_true", help="Force sliding-window inference regardless of size"
    )
    parser.add_argument(
        "--no-tile", action="store_true", help="Never tile; always resize to the training size"
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay blend factor")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG | INFO | WARNING")
    return parser


def collect_inputs(path: str | Path) -> list[Path]:
    """Return the image files addressed by *path*.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist or a directory holds no supported images.
    """
    path = Path(path).expanduser()
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise FileNotFoundError(
                f"No images with extensions {', '.join(IMAGE_EXTENSIONS)} found in {path}"
            )
        return files
    raise FileNotFoundError(f"Input path does not exist: {path}")


def read_rgb(path: Path) -> np.ndarray:
    """Read an image file as an ``(H, W, 3)`` uint8 RGB array."""
    try:
        with Image.open(path) as handle:
            return np.array(handle.convert("RGB"))
    except (OSError, UnidentifiedImageError) as exc:
        raise OSError(f"Could not read image {path}: {exc}") from exc


@torch.no_grad()
def predict_array(
    image: np.ndarray,
    model: torch.nn.Module,
    cfg: Config,
    *,
    device: torch.device,
    threshold: float = 0.5,
    force_tile: bool = False,
    disable_tile: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a crack mask for one RGB array.

    Large images go through :func:`~src.utils.sliding_window_inference` so the
    network still sees crack-sized detail instead of a heavily downscaled view.

    Returns
    -------
    tuple
        ``(probability_map, binary_mask)``, both ``(H, W)`` float arrays at the
        original image resolution.
    """
    model.eval()
    height, width = image.shape[:2]
    tensor = preprocess_image(image)  # (3, H, W) at native resolution

    use_tiling = force_tile or (
        not disable_tile and max(height, width) >= cfg.inference.sliding_window_min_size
    )
    if use_tiling:
        LOGGER.debug("Sliding-window inference on a %dx%d image", width, height)
        logits = sliding_window_inference(
            tensor,
            model,
            tile_size=cfg.inference.tile_size,
            overlap=cfg.inference.tile_overlap,
            device=device,
        )
    else:
        resized = F.interpolate(
            tensor.unsqueeze(0),
            size=tuple(cfg.data.image_size),
            mode="bilinear",
            align_corners=False,
        ).to(device)
        logits = model(resized).float().cpu()
        logits = F.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )

    probs = torch.sigmoid(logits)[0, 0].numpy()
    mask = (probs >= threshold).astype(np.uint8)
    return probs, mask


def save_outputs(
    stem: str,
    image: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    *,
    alpha: float = 0.5,
) -> tuple[Path, Path]:
    """Write ``<stem>_mask.png`` and ``<stem>_overlay.png``; return both paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{stem}_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    try:
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        Image.fromarray(mask_to_overlay(image, mask, alpha=alpha)).save(overlay_path)
    except OSError as exc:
        raise OSError(f"Could not write predictions for {stem}: {exc}") from exc
    return mask_path, overlay_path


def load_model_for_inference(
    cfg: Config, checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    """Load a checkpoint into the architecture it was trained with."""
    from .evaluate import load_model_from_checkpoint

    model, _ = load_model_from_checkpoint(checkpoint_path, cfg, device)
    return model


def write_csv(rows: Sequence[dict[str, Any]], csv_path: Path) -> Path | None:
    """Write the per-image summary as CSV; returns the path or ``None``."""
    if not rows:
        return None
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        LOGGER.error("Could not write %s: %s", csv_path, exc)
        return None
    LOGGER.info("Wrote %s", csv_path)
    return csv_path


def run(
    inputs: Iterable[Path],
    cfg: Config,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    threshold: float = 0.5,
    alpha: float = 0.5,
    force_tile: bool = False,
    disable_tile: bool = False,
    csv_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Predict over every input and return one summary row per image."""
    device = get_device(cfg.device)
    model = load_model_for_inference(cfg, checkpoint_path, device)

    rows: list[dict[str, Any]] = []
    for path in inputs:
        try:
            image = read_rgb(path)
        except OSError as exc:
            LOGGER.error("Skipping %s: %s", path.name, exc)
            continue

        probs, mask = predict_array(
            image, model, cfg, device=device, threshold=threshold,
            force_tile=force_tile, disable_tile=disable_tile,
        )
        try:
            mask_path, overlay_path = save_outputs(
                path.stem, image, mask, output_dir, alpha=alpha
            )
        except OSError as exc:
            LOGGER.error("%s", exc)
            continue

        ratio = crack_pixel_ratio(mask)
        length_px = estimate_crack_length_px(mask)
        row: dict[str, Any] = {
            "image": path.name,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "threshold": round(float(threshold), 4),
            "crack_pixel_ratio": round(ratio, 6),
            "crack_pixels": int(mask.sum()),
            "estimated_crack_length_px": round(length_px, 2),
            "mean_probability": round(float(probs.mean()), 6),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
        }
        if cfg.inference.pixel_size_mm:
            row["estimated_crack_length_mm"] = round(length_px * cfg.inference.pixel_size_mm, 2)
        rows.append(row)
        LOGGER.info(
            "%s -> crack ratio %.4f%% (%d px), length ~%.0f px",
            path.name,
            ratio * 100.0,
            int(mask.sum()),
            length_px,
        )

    if csv_path is not None:
        write_csv(rows, csv_path)
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if args.tile and args.no_tile:
        LOGGER.error("--tile and --no-tile are mutually exclusive")
        return 2

    overrides: dict[str, Any] = {}
    if args.device:
        overrides["device"] = args.device
    if args.threshold is not None:
        overrides["eval"] = {"threshold": args.threshold}

    try:
        cfg = load_config(args.config, overrides=overrides)
        checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint)
        inputs = collect_inputs(args.input)
        threshold = args.threshold if args.threshold is not None else cfg.eval.threshold
        rows = run(
            inputs,
            cfg,
            checkpoint_path,
            Path(args.output_dir),
            threshold=threshold,
            alpha=args.alpha,
            force_tile=args.tile,
            disable_tile=args.no_tile,
            csv_path=Path(args.csv) if args.csv else None,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 1

    if not rows:
        LOGGER.warning("No predictions were produced")
        return 1
    LOGGER.info("Processed %d image(s); results in %s", len(rows), args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
