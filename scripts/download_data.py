"""Fetch the DeepCrack dataset (Kaggle) or register a local copy of it.

Dataset: `rukiyeaydn/deepcrack-dataset
<https://www.kaggle.com/datasets/rukiyeaydn/deepcrack-dataset>`_ (not
redistributed with this repository).

Three ways to get the data into place:

1. **kagglehub** (default)::

       python scripts/download_data.py

2. **Kaggle CLI**::

       pip install kaggle
       python scripts/download_data.py --method cli

   Both need a Kaggle API token: ``kaggle.json`` in ``~/.kaggle/``
   (``%USERPROFILE%\\.kaggle\\kaggle.json`` on Windows) or the
   ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables.

3. **A copy you already have** (folder or the downloaded ``.zip``)::

       python scripts/download_data.py --local D:\\sample_projects\\_datasets\\deep_crack_dataset
       python scripts/download_data.py --local D:\\Downloads\\archive.zip

   A folder that already has the right layout is only verified and used in
   place (point ``DATA_DIR`` at it); add ``--copy`` to copy it into
   ``--data-dir``. A ``.zip`` is always extracted into ``--data-dir``.

Expected layout::

    DATA_DIR/train_img/*.jpg   DATA_DIR/train_lab/*.png   (300 pairs)
    DATA_DIR/test_img/*.jpg    DATA_DIR/test_lab/*.png    (237 pairs)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging  # noqa: E402  (path setup must come first)

LOGGER = logging.getLogger("scripts.download_data")

#: Kaggle dataset slug.
KAGGLE_DATASET = "rukiyeaydn/deepcrack-dataset"

#: Subdirectories a complete dataset must contain.
REQUIRED_DIRS: tuple[str, ...] = ("train_img", "train_lab", "test_img", "test_lab")

DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "_datasets" / "deep_crack_dataset"


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the downloader."""
    parser = argparse.ArgumentParser(
        prog="python scripts/download_data.py",
        description="Download the DeepCrack dataset from Kaggle, or verify/import a local copy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Destination directory. Defaults to $DATA_DIR, else ../_datasets/deep_crack_dataset "
        "(relative paths resolve against the repository root).",
    )
    parser.add_argument(
        "--method", choices=["kagglehub", "cli"], default="kagglehub", help="How to download"
    )
    parser.add_argument(
        "--local",
        "--archive",
        dest="local",
        default=None,
        help="Use an existing local copy (dataset folder or downloaded .zip) instead of Kaggle",
    )
    parser.add_argument(
        "--copy", action="store_true", help="With --local <folder>: copy it into --data-dir"
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if data exists")
    parser.add_argument("--log-level", default="INFO", help="DEBUG | INFO | WARNING")
    return parser


def resolve_data_dir(raw: str | None) -> Path:
    """Resolve the destination; relative paths are anchored at the repository root."""
    path = Path(raw or os.environ.get("DATA_DIR") or DEFAULT_DATA_DIR).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def dataset_is_present(data_dir: Path) -> bool:
    """True when every required subdirectory exists and holds files."""
    return all(
        (data_dir / name).is_dir() and any((data_dir / name).iterdir()) for name in REQUIRED_DIRS
    )


def find_dataset_root(start: Path, max_depth: int = 3) -> Path | None:
    """Locate the folder holding the four split directories at or below *start*."""
    if dataset_is_present(start):
        return start
    if max_depth == 0 or not start.is_dir():
        return None
    for child in sorted(p for p in start.iterdir() if p.is_dir()):
        found = find_dataset_root(child, max_depth - 1)
        if found is not None:
            return found
    return None


def extract_archive(archive: Path, data_dir: Path) -> None:
    """Extract *archive* into *data_dir*, rejecting unsafe member paths."""
    if not archive.is_file():
        raise FileNotFoundError(f"Archive not found: {archive}")
    data_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Extracting %s -> %s", archive, data_dir)
    root = data_dir.resolve()
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                target = (data_dir / member).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"Refusing to extract outside the target: {member}")
            bundle.extractall(data_dir)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"{archive} is not a valid zip archive: {exc}") from exc


def copy_dataset(source: Path, data_dir: Path) -> None:
    """Copy the four split directories from *source* into *data_dir*."""
    for name in REQUIRED_DIRS:
        LOGGER.info("Copying %s -> %s", source / name, data_dir / name)
        shutil.copytree(source / name, data_dir / name, dirs_exist_ok=True)
    readme = source / "README.md"
    if readme.is_file():
        shutil.copy2(readme, data_dir / "README.md")


def download_with_kagglehub() -> Path:
    """Download with kagglehub (cached after the first run) and return the local path."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Run 'pip install kagglehub' or use --method cli."
        ) from exc
    LOGGER.info("Downloading %s with kagglehub", KAGGLE_DATASET)
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def download_with_cli(data_dir: Path) -> None:
    """Download and unzip with the Kaggle CLI into *data_dir*."""
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The 'kaggle' command was not found. Install it with 'pip install kaggle' "
            "and place your API token in ~/.kaggle/kaggle.json "
            "(%USERPROFILE%\\.kaggle\\kaggle.json on Windows)."
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(data_dir), "--unzip"]
    LOGGER.info("Running: %s", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"'kaggle datasets download' failed with exit code {exc.returncode}. "
            "Check your credentials and that you have accepted the dataset's terms."
        ) from exc
    for archive in sorted(data_dir.glob("*.zip")):  # older CLI versions ignore --unzip
        extract_archive(archive, data_dir)
        archive.unlink(missing_ok=True)


def summarise(data_dir: Path) -> None:
    """Log how many files ended up in each split."""
    for name in REQUIRED_DIRS:
        directory = data_dir / name
        count = sum(1 for p in directory.iterdir() if p.is_file()) if directory.is_dir() else 0
        LOGGER.info("%-10s %4d files", name, count)


def finish(root: Path | None, searched: Path) -> int:
    """Report the final location, or an error if the layout was not found."""
    if root is None:
        LOGGER.error("Could not find %s under %s.", ", ".join(REQUIRED_DIRS), searched)
        return 1
    summarise(root)
    LOGGER.info("Dataset ready at %s", root)
    LOGGER.info("Set DATA_DIR=%s in your .env (or pass --data-dir) to use it.", root)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    data_dir = resolve_data_dir(args.data_dir)

    try:
        if args.local:
            source = Path(args.local).expanduser().resolve()
            if not source.exists():
                LOGGER.error("Local path does not exist: %s", source)
                return 1
            if source.is_file():
                extract_archive(source, data_dir)
                return finish(find_dataset_root(data_dir), data_dir)
            root = find_dataset_root(source)
            if root is not None and args.copy:
                copy_dataset(root, data_dir)
                root = data_dir
            return finish(root, source)

        if dataset_is_present(data_dir) and not args.force:
            LOGGER.info("Dataset already present at %s (use --force to download again)", data_dir)
            return finish(data_dir, data_dir)

        if args.method == "kagglehub":
            cached = download_with_kagglehub()
            root = find_dataset_root(cached)
            if root is None:
                return finish(None, cached)
            copy_dataset(root, data_dir)
            return finish(data_dir, data_dir)

        download_with_cli(data_dir)
        return finish(find_dataset_root(data_dir), data_dir)
    except (RuntimeError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
