"""Typed configuration loaded from YAML with environment-variable overrides.

Precedence (lowest to highest):

1. the dataclass defaults in this module,
2. the YAML file passed to :func:`load_config` (default ``configs/unet.yaml``),
3. environment variables (optionally read from a ``.env`` file),
4. explicit keyword overrides supplied by a CLI.

Example
-------
>>> from src.config import load_config
>>> cfg = load_config("configs/unet.yaml")           # doctest: +SKIP
>>> cfg.data.data_dir.name                            # doctest: +SKIP
'deep_crack_dataset'
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)

#: Repository root (the directory that contains ``src/``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Fallback config shipped with the repository.
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "unet.yaml"


def _load_dotenv() -> None:
    """Load ``.env`` from the project root if ``python-dotenv`` is installed.

    Missing dependency or missing file are both non-fatal: the process simply
    relies on the ambient environment.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.debug("python-dotenv not installed; skipping %s", env_path)
        return
    load_dotenv(env_path, override=False)
    LOGGER.debug("Loaded environment overrides from %s", env_path)


def resolve_path(value: str | os.PathLike[str]) -> Path:
    """Resolve *value* against :data:`PROJECT_ROOT` when it is relative."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


@dataclass
class DataConfig:
    """Where the dataset lives and how it is split."""

    data_dir: Path = PROJECT_ROOT.parent / "_datasets" / "deep_crack_dataset"
    train_images: str = "train_img"
    train_masks: str = "train_lab"
    test_images: str = "test_img"
    test_masks: str = "test_lab"
    image_suffix: str = ".jpg"
    mask_suffix: str = ".png"
    image_size: tuple[int, int] = (384, 384)
    val_split: float = 0.15
    num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self) -> None:
        self.data_dir = resolve_path(self.data_dir)
        self.image_size = tuple(int(v) for v in self.image_size)  # type: ignore[assignment]
        if len(self.image_size) != 2:
            raise ValueError(f"image_size must be [height, width], got {self.image_size!r}")
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError(f"val_split must be in [0, 1), got {self.val_split}")

    @property
    def train_image_dir(self) -> Path:
        return self.data_dir / self.train_images

    @property
    def train_mask_dir(self) -> Path:
        return self.data_dir / self.train_masks

    @property
    def test_image_dir(self) -> Path:
        return self.data_dir / self.test_images

    @property
    def test_mask_dir(self) -> Path:
        return self.data_dir / self.test_masks


@dataclass
class AugmentConfig:
    """Joint image/mask augmentation strength."""

    enabled: bool = True
    random_crop: bool = True
    crop_scale: tuple[float, float] = (0.6, 1.0)
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rot90_prob: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2

    def __post_init__(self) -> None:
        self.crop_scale = tuple(float(v) for v in self.crop_scale)  # type: ignore[assignment]
        lo, hi = self.crop_scale
        if not 0.0 < lo <= hi <= 1.0:
            raise ValueError(f"crop_scale must satisfy 0 < lo <= hi <= 1, got {self.crop_scale!r}")


@dataclass
class ModelConfig:
    """Architecture selection."""

    name: str = "unet"
    in_channels: int = 3
    num_classes: int = 1
    base_channels: int = 32
    depth: int = 4
    bilinear: bool = True
    pretrained: bool = False

    def kwargs(self) -> dict[str, Any]:
        """Keyword arguments accepted by :func:`src.model.build_model`."""
        return {
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "bilinear": self.bilinear,
            "pretrained": self.pretrained,
        }


@dataclass
class LossConfig:
    """Objective function selection and its hyper-parameters."""

    name: str = "bce_dice"
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    pos_weight: float | None = None


@dataclass
class TrainConfig:
    """Optimisation schedule and checkpointing."""

    epochs: int = 40
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    amp: bool = True
    early_stopping_patience: int = 10
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    log_dir: Path = PROJECT_ROOT / "runs"
    save_every: int = 0

    def __post_init__(self) -> None:
        self.checkpoint_dir = resolve_path(self.checkpoint_dir)
        self.log_dir = resolve_path(self.log_dir)
        if self.scheduler not in {"cosine", "plateau", "none"}:
            raise ValueError(
                f"scheduler must be one of 'cosine', 'plateau', 'none'; got {self.scheduler!r}"
            )


@dataclass
class EvalConfig:
    """Test-set evaluation and report generation."""

    threshold: float = 0.5
    threshold_sweep: tuple[float, float, int] = (0.05, 0.95, 19)
    batch_size: int = 8
    reports_dir: Path = PROJECT_ROOT / "reports"
    num_qualitative: int = 6

    def __post_init__(self) -> None:
        self.reports_dir = resolve_path(self.reports_dir)
        start, stop, num = self.threshold_sweep
        self.threshold_sweep = (float(start), float(stop), int(num))
        if int(num) < 2:
            raise ValueError("threshold_sweep must request at least 2 thresholds")

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"


@dataclass
class InferenceConfig:
    """Prediction-time behaviour, shared by the CLI and the web app."""

    tile_size: int = 384
    tile_overlap: float = 0.25
    sliding_window_min_size: int = 768
    pixel_size_mm: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.tile_overlap < 1.0:
            raise ValueError(f"tile_overlap must be in [0, 1), got {self.tile_overlap}")


@dataclass
class Config:
    """Top-level configuration object handed to every entry point."""

    experiment_name: str = "unet_baseline"
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    config_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serialisable view of the configuration."""
        return _asdict(self)


def _asdict(obj: Any) -> Any:
    """``dataclasses.asdict`` variant that stringifies :class:`~pathlib.Path`."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    return obj


def _coerce(value: Any, target: Any) -> Any:
    """Best-effort conversion of an environment string to the default's type."""
    if not isinstance(value, str):
        return value
    if isinstance(target, bool):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"cannot interpret {value!r} as a boolean")
    if isinstance(target, Path):
        return resolve_path(value)
    if isinstance(target, int):
        return int(value)
    if isinstance(target, float):
        return float(value)
    return value


def _build_section(cls: type, raw: Mapping[str, Any] | None) -> Any:
    """Instantiate dataclass *cls*, keeping only keys it declares."""
    raw = dict(raw or {})
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        LOGGER.warning("Ignoring unknown %s keys: %s", cls.__name__, ", ".join(sorted(unknown)))
    return cls(**{k: v for k, v in raw.items() if k in known})


#: Environment variable -> ``(section, attribute)``. ``None`` means top level.
ENV_OVERRIDES: dict[str, tuple[str | None, str]] = {
    "DATA_DIR": ("data", "data_dir"),
    "NUM_WORKERS": ("data", "num_workers"),
    "CHECKPOINT_PATH": (None, "_checkpoint_path"),
    "DEVICE": (None, "device"),
    "RANDOM_SEED": (None, "seed"),
}


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    use_env: bool = True,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Build a :class:`Config`.

    Parameters
    ----------
    path:
        YAML file to read. Falls back to ``$CONFIG_PATH`` and then to
        :data:`DEFAULT_CONFIG_PATH`.
    use_env:
        Apply the :data:`ENV_OVERRIDES` environment variables on top of YAML.
    overrides:
        Nested mapping merged last, e.g. ``{"train": {"epochs": 1}}``. Used by
        CLIs to fold argparse flags into the config.

    Raises
    ------
    FileNotFoundError
        If an explicit config path does not exist.
    ValueError
        If the YAML file does not parse to a mapping, or a value is invalid.
    """
    if use_env:
        _load_dotenv()

    if path is None:
        path = os.environ.get("CONFIG_PATH") or DEFAULT_CONFIG_PATH
    config_path = resolve_path(path)

    raw: dict[str, Any] = {}
    if config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Could not parse YAML config {config_path}: {exc}") from exc
        except OSError as exc:
            raise ValueError(f"Could not read config {config_path}: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Config {config_path} must contain a mapping at the top level")
        raw = dict(loaded)
    elif path is not None and Path(path) != DEFAULT_CONFIG_PATH:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    else:  # pragma: no cover - only when the shipped default was deleted
        LOGGER.warning("No config file at %s; using built-in defaults", config_path)

    if overrides:
        raw = _deep_merge(raw, overrides)

    cfg = Config(
        experiment_name=str(raw.get("experiment_name", "unet_baseline")),
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "auto")),
        data=_build_section(DataConfig, raw.get("data")),
        augment=_build_section(AugmentConfig, raw.get("augment")),
        model=_build_section(ModelConfig, raw.get("model")),
        loss=_build_section(LossConfig, raw.get("loss")),
        train=_build_section(TrainConfig, raw.get("train")),
        eval=_build_section(EvalConfig, raw.get("eval")),
        inference=_build_section(InferenceConfig, raw.get("inference")),
        config_path=config_path if config_path.is_file() else None,
    )

    if use_env:
        _apply_env(cfg)
    return cfg


def _apply_env(cfg: Config) -> None:
    """Mutate *cfg* in place from the environment, ignoring unset variables."""
    for env_name, (section, attr) in ENV_OVERRIDES.items():
        raw_value = os.environ.get(env_name)
        if raw_value is None or raw_value == "":
            continue
        if attr.startswith("_"):
            continue  # handled by resolve_checkpoint_path
        target_obj = cfg if section is None else getattr(cfg, section)
        current = getattr(target_obj, attr)
        try:
            setattr(target_obj, attr, _coerce(raw_value, current))
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_name}={raw_value!r}: {exc}") from exc
        LOGGER.debug("Applied %s override from environment", env_name)
    # data_dir may have been replaced with a relative string; normalise again.
    cfg.data.data_dir = resolve_path(cfg.data.data_dir)


def _deep_merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge *extra* into *base* without mutating either."""
    merged = dict(base)
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


#: Small CPU-trained model committed to the repository so the CLIs and the web
#: UI work straight after cloning. See the README for how it was produced.
DEMO_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "demo" / "unet_cpu_demo.pt"


def resolve_checkpoint_path(cfg: Config, explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the checkpoint path from CLI flag, ``$CHECKPOINT_PATH`` or config.

    When neither a flag nor ``$CHECKPOINT_PATH`` is given and the config's
    ``<checkpoint_dir>/best.pt`` does not exist yet, the bundled CPU demo
    checkpoint is used instead (with a log message), so a fresh clone can run
    inference before any training.
    """
    if explicit:
        return resolve_path(explicit)
    env_value = os.environ.get("CHECKPOINT_PATH")
    if env_value:
        return resolve_path(env_value)
    trained = cfg.train.checkpoint_dir / "best.pt"
    if not trained.is_file() and DEMO_CHECKPOINT_PATH.is_file():
        LOGGER.info(
            "%s not found; using the bundled CPU demo checkpoint %s", trained, DEMO_CHECKPOINT_PATH
        )
        return DEMO_CHECKPOINT_PATH
    return trained
