"""Shared helpers: seeding, devices, checkpoints, overlays and tiled inference."""

from __future__ import annotations

import logging
import os
import random
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LOGGER = logging.getLogger(__name__)

#: ImageNet statistics used by every encoder in this project.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def setup_logging(level: str | int = "INFO", *, log_file: str | os.PathLike[str] | None = None) -> None:
    """Configure the root logger once, for CLIs and the web app alike.

    Parameters
    ----------
    level:
        Level name or numeric level. ``$LOG_LEVEL`` wins when set.
    log_file:
        Optional file that receives the same records as stderr.
    """
    env_level = os.environ.get("LOG_LEVEL")
    if env_level:
        level = env_level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stderr)]
    if log_file is not None:
        path = Path(log_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(path, encoding="utf-8"))
        except OSError as exc:  # pragma: no cover - depends on the filesystem
            LOGGER.warning("Could not open log file %s: %s", path, exc)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def seed_everything(seed: int = 42, *, deterministic: bool = False) -> int:
    """Seed ``random``, ``numpy`` and ``torch`` and return the seed used."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    LOGGER.debug("Seeded RNGs with %d (deterministic=%s)", seed, deterministic)
    return seed


def get_device(preference: str = "auto") -> torch.device:
    """Resolve a torch device from ``auto`` / ``cuda`` / ``mps`` / ``cpu``.

    Requests that cannot be honoured fall back to CPU with a warning rather
    than raising, so a CUDA config still runs on a laptop.
    """
    preference = (preference or "auto").strip().lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if preference.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(preference)
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    if preference == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        LOGGER.warning("MPS requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device("cpu")


class AverageMeter:
    """Running mean of a scalar, weighted by batch size."""

    def __init__(self, name: str = "meter") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.last = 0.0

    def update(self, value: float, n: int = 1) -> None:
        """Add *value* observed over *n* items."""
        if n <= 0:
            return
        self.last = float(value)
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def __float__(self) -> float:
        return self.avg

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"AverageMeter(name={self.name!r}, avg={self.avg:.4f}, count={self.count})"


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters in *model*."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    metrics: Mapping[str, float] | None = None,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a checkpoint atomically and return its path.

    The payload always carries ``model_state``, ``epoch``, ``metrics`` and the
    serialised ``config`` so that inference can rebuild the architecture
    without the original YAML.
    """
    path = Path(path)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics or {}),
        "config": dict(config or {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        payload["scaler_state"] = scaler.state_dict()
    if extra:
        payload.update(dict(extra))

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except OSError as exc:
        raise OSError(f"Could not write checkpoint to {path}: {exc}") from exc
    LOGGER.info("Saved checkpoint -> %s (epoch %d)", path, epoch)
    return path


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file exists but does not look like one of our checkpoints.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train a model first: python -m src.train"
        )
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:  # torch raises a variety of types here
        raise ValueError(f"Could not read checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"{path} is not a crack-detection checkpoint (no 'model_state' key)")
    return payload


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalisation; returns a ``[0, 1]`` tensor of the same shape."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return (tensor * std + mean).clamp(0.0, 1.0)


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised ``(3, H, W)`` tensor to an ``(H, W, 3)`` uint8 array."""
    if tensor.dim() != 3:
        raise ValueError(f"expected a (C, H, W) tensor, got shape {tuple(tensor.shape)}")
    array = denormalize(tensor.detach().cpu()).permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def mask_to_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (220, 30, 60),
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a binary *mask* over an RGB *image*.

    Parameters
    ----------
    image:
        ``(H, W, 3)`` uint8 RGB, or ``(H, W)`` greyscale which is expanded.
    mask:
        ``(H, W)`` array, either 0/1 (any dtype) or 0/255 uint8.
    color:
        RGB colour painted over crack pixels.
    alpha:
        Blend factor in ``[0, 1]``; 1.0 paints the mask opaquely.

    Returns
    -------
    numpy.ndarray
        ``(H, W, 3)`` uint8 overlay.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3) RGB, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.squeeze()
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {image.shape[:2]}")
    binary = _binarize(mask)

    overlay = image.astype(np.float32).copy()
    tint = np.array(color, dtype=np.float32)
    overlay[binary] = (1.0 - alpha) * overlay[binary] + alpha * tint
    return overlay.round().clip(0, 255).astype(np.uint8)


def _binarize(mask: np.ndarray) -> np.ndarray:
    """Boolean crack mask from a 0/1 or 0/255 array (any dtype)."""
    mask = np.asarray(mask)
    if mask.size == 0:
        return mask.astype(bool)
    if mask.dtype == np.uint8 and mask.max() > 1:
        return mask > 127
    return mask > 0.5


def crack_pixel_ratio(mask: np.ndarray | torch.Tensor) -> float:
    """Fraction of pixels classified as crack, in ``[0, 1]``."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.size == 0:
        return 0.0
    binary = _binarize(mask)
    return float(binary.sum()) / float(binary.size)


def estimate_crack_length_px(mask: np.ndarray) -> float:
    """Rough crack length in pixels from the skeleton of a binary mask.

    Uses an OpenCV thinning implementation when ``opencv-contrib`` is present
    and otherwise falls back to ``area / mean_width``, which is adequate for
    the thin, roughly constant-width cracks in DeepCrack. The value is an
    indicative figure, not a calibrated measurement.
    """
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.squeeze()
    binary = _binarize(mask).astype(np.uint8)
    area = float(binary.sum())
    if area == 0.0:
        return 0.0

    try:
        import cv2

        thinning = getattr(getattr(cv2, "ximgproc", None), "thinning", None)
        if thinning is not None:
            skeleton = thinning(binary * 255)
            return float((skeleton > 0).sum())
        # Perimeter-based estimate: a thin ribbon of length L and width w has a
        # perimeter of roughly 2L, so L ~ perimeter / 2.
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        if perimeter > 0:
            return float(perimeter / 2.0)
    except ImportError:  # pragma: no cover - OpenCV is a declared dependency
        LOGGER.debug("OpenCV unavailable; using the area/width length estimate")

    rows = np.flatnonzero(binary.any(axis=1))
    cols = np.flatnonzero(binary.any(axis=0))
    extent = max(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)
    mean_width = max(area / max(float(extent), 1.0), 1.0)
    return float(area / mean_width)


def _gaussian_window(height: int, width: int, sigma_scale: float = 0.125) -> np.ndarray:
    """2-D Gaussian weight map used to feather sliding-window tile seams."""
    sigma_y = max(height * sigma_scale, 1e-6)
    sigma_x = max(width * sigma_scale, 1e-6)
    ys = np.arange(height, dtype=np.float32) - (height - 1) / 2.0
    xs = np.arange(width, dtype=np.float32) - (width - 1) / 2.0
    gy = np.exp(-(ys**2) / (2.0 * sigma_y**2))
    gx = np.exp(-(xs**2) / (2.0 * sigma_x**2))
    window = np.outer(gy, gx)
    # Keep a strictly positive floor so that every pixel has non-zero weight.
    return np.maximum(window, 1e-4).astype(np.float32)


def _tile_starts(total: int, tile: int, stride: int) -> list[int]:
    """Start offsets covering ``[0, total)`` with the last tile flush to the edge."""
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, stride))
    if starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


@torch.no_grad()
def sliding_window_inference(
    image: torch.Tensor,
    model: nn.Module,
    *,
    tile_size: int = 384,
    overlap: float = 0.25,
    batch_size: int = 4,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Run *model* over a large image in overlapping tiles with Gaussian blending.

    Parameters
    ----------
    image:
        ``(C, H, W)`` or ``(1, C, H, W)`` normalised input tensor.
    model:
        A module mapping ``(N, C, h, w)`` to ``(N, 1, h, w)`` logits.
    tile_size:
        Square tile side in pixels. Images smaller than this in either
        dimension are reflection-padded up to ``tile_size``.
    overlap:
        Fractional tile overlap in ``[0, 1)``.
    batch_size:
        Number of tiles forwarded at once.
    device:
        Device to run on; defaults to the device of the model's parameters.

    Returns
    -------
    torch.Tensor
        ``(1, 1, H, W)`` logits on the CPU, blended across tiles.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.shape[0] != 1:
        raise ValueError(f"expected a (C, H, W) or (1, C, H, W) tensor, got {tuple(image.shape)}")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:  # pragma: no cover - parameterless model
            device = torch.device("cpu")
    device = torch.device(device)
    model.eval()

    _, _, height, width = image.shape
    pad_bottom = max(0, tile_size - height)
    pad_right = max(0, tile_size - width)
    if pad_bottom or pad_right:
        # Reflection needs the pad to be smaller than the input; fall back to
        # edge replication for images much smaller than the tile.
        mode = "reflect" if pad_bottom < height and pad_right < width else "replicate"
        image = F.pad(image, (0, pad_right, 0, pad_bottom), mode=mode)
    padded_h, padded_w = image.shape[-2:]

    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    ys = _tile_starts(padded_h, tile_size, stride)
    xs = _tile_starts(padded_w, tile_size, stride)

    window = torch.from_numpy(_gaussian_window(tile_size, tile_size))
    accumulator = torch.zeros((1, 1, padded_h, padded_w), dtype=torch.float32)
    weights = torch.zeros((1, 1, padded_h, padded_w), dtype=torch.float32)

    coords = [(y, x) for y in ys for x in xs]
    for start in range(0, len(coords), batch_size):
        chunk = coords[start : start + batch_size]
        tiles = torch.cat(
            [image[:, :, y : y + tile_size, x : x + tile_size] for y, x in chunk], dim=0
        ).to(device)
        logits = model(tiles)
        if logits.shape[-2:] != (tile_size, tile_size):
            logits = F.interpolate(
                logits, size=(tile_size, tile_size), mode="bilinear", align_corners=False
            )
        logits = logits.detach().float().cpu()
        for idx, (y, x) in enumerate(chunk):
            accumulator[:, :, y : y + tile_size, x : x + tile_size] += logits[idx : idx + 1] * window
            weights[:, :, y : y + tile_size, x : x + tile_size] += window

    blended = accumulator / weights.clamp_min(1e-8)
    return blended[:, :, :height, :width]


def chunked(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    """Yield lists of at most *size* elements from *items*."""
    if size <= 0:
        raise ValueError("size must be positive")
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
