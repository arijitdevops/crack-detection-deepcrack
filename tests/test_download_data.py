"""Tests for scripts/download_data.py (no network: Kaggle is mocked)."""

from __future__ import annotations

import importlib.util
import sys
import types
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
SPLITS = ("train_img", "train_lab", "test_img", "test_lab")


@pytest.fixture(scope="module")
def dl():
    spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_layout(root: Path) -> Path:
    for split in SPLITS:
        folder = root / split
        folder.mkdir(parents=True)
        (folder / ("1.png" if split.endswith("lab") else "1.jpg")).write_bytes(b"fake")
    return root


def test_local_folder_is_used_in_place(dl, tmp_path):
    source = _make_layout(tmp_path / "wrapper" / "deep_crack_dataset")
    dest = tmp_path / "dest"
    assert dl.main(["--local", str(tmp_path / "wrapper"), "--data-dir", str(dest)]) == 0
    assert not dest.exists()
    assert dl.find_dataset_root(tmp_path / "wrapper") == source


def test_local_folder_copy(dl, tmp_path):
    _make_layout(tmp_path / "src")
    dest = tmp_path / "dest"
    assert dl.main(["--local", str(tmp_path / "src"), "--data-dir", str(dest), "--copy"]) == 0
    assert dl.dataset_is_present(dest)


def test_local_zip_is_extracted(dl, tmp_path):
    _make_layout(tmp_path / "raw")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for file in (tmp_path / "raw").rglob("*.*"):
            bundle.write(file, file.relative_to(tmp_path / "raw"))
    dest = tmp_path / "dest"
    assert dl.main(["--archive", str(archive), "--data-dir", str(dest)]) == 0
    assert dl.dataset_is_present(dest)


def test_kagglehub_download_is_copied_to_data_dir(dl, tmp_path, monkeypatch):
    cache = _make_layout(tmp_path / "cache")
    monkeypatch.setitem(
        sys.modules, "kagglehub", types.SimpleNamespace(dataset_download=lambda slug: str(cache))
    )
    dest = tmp_path / "dest"
    assert dl.main(["--data-dir", str(dest)]) == 0
    assert dl.dataset_is_present(dest)


def test_existing_dataset_is_not_downloaded_again(dl, tmp_path, monkeypatch):
    dest = _make_layout(tmp_path / "dest")

    def boom(slug):  # pragma: no cover - must not be called
        raise AssertionError("should not download")

    monkeypatch.setitem(sys.modules, "kagglehub", types.SimpleNamespace(dataset_download=boom))
    assert dl.main(["--data-dir", str(dest)]) == 0


def test_incomplete_layout_fails(dl, tmp_path):
    (tmp_path / "train_img").mkdir()
    assert dl.main(["--local", str(tmp_path)]) == 1
