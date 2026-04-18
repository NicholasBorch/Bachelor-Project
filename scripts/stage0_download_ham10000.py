"""Download HAM10000 from Kaggle into ``data/raw/HAM10000/``.

Headless alternative to the notebook ``notebooks/HAM10000_data_loader.ipynb``.
Use this on HPC or anywhere a notebook would be awkward.

Layout produced (what Stage 0 expects)::

    data/raw/HAM10000/
    ├── HAM10000_metadata.csv
    ├── HAM10000_images_part_1/   *.jpg
    └── HAM10000_images_part_2/   *.jpg

Prerequisites:
    1. A Kaggle account.
    2. An API token file at ``~/.kaggle/kaggle.json`` with permissions 600.
       Create one via Kaggle -> Account -> "Create New API Token".

Run:
    python -m scripts.stage0_download_ham10000

Idempotent: if the expected files already exist under ``data/raw/HAM10000``,
the script does nothing and exits 0. Pass ``--force`` to re-download.

This script only populates ``data/raw``. After it finishes, run
``python -m scripts.stage0_prepare_dataset`` to deduplicate and build the
balanced subset.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.utils.io import load_config, project_root


KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"

# Files Stage 0 reads. If all of these exist under data/raw/HAM10000/, we're done.
EXPECTED_METADATA = "HAM10000_metadata.csv"
EXPECTED_IMAGE_DIRS = ("HAM10000_images_part_1", "HAM10000_images_part_2")


def _already_present(raw_dir: Path) -> bool:
    """True if everything Stage 0 needs is already on disk."""
    if not (raw_dir / EXPECTED_METADATA).exists():
        return False
    for sub in EXPECTED_IMAGE_DIRS:
        d = raw_dir / sub
        if not d.is_dir() or not any(d.glob("*.jpg")):
            return False
    return True


def _find_in_download(src_root: Path, name: str) -> Path | None:
    """Locate ``name`` (file or directory) anywhere in the Kaggle cache tree.

    kagglehub sometimes places things a level deep depending on the dataset.
    We walk the tree to find the canonical artifacts by name.
    """
    for candidate in src_root.rglob(name):
        return candidate
    return None


def _link_or_copy(src: Path, dst: Path) -> None:
    """Prefer a symlink (fast, no disk cost); fall back to copy on failure.

    On HPC the kagglehub cache typically lives under ``~/.cache/kagglehub/``
    — that is on home, not on work3. Symlinking keeps ``data/raw`` small
    while still letting Stage 0 read the files transparently. If the
    symlink fails (e.g. cross-filesystem restriction or Windows), we copy.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink() if dst.is_symlink() else shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    try:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copyfile(src, dst)


def main(args: argparse.Namespace) -> int:
    try:
        import kagglehub
    except ImportError:
        print(
            "ERROR: kagglehub is not installed. Run:\n"
            "    pip install kagglehub\n"
            "(or: pip install -r requirements.txt)",
            file=sys.stderr,
        )
        return 1

    cfg = load_config("base.yaml")
    root = project_root()
    raw_dir = root / cfg["paths"]["data_raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    if _already_present(raw_dir) and not args.force:
        print(f"[stage0-download] {raw_dir} already contains HAM10000. Nothing to do.")
        print("[stage0-download] Pass --force to re-download.")
        return 0

    # kagglehub reads credentials from ~/.kaggle/kaggle.json automatically.
    # If they're missing it will raise a clear error — we just let it bubble up.
    print(f"[stage0-download] downloading {KAGGLE_DATASET} via kagglehub...")
    print("[stage0-download] (first-time downloads are ~3 GB and can take a while)")
    download_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"[stage0-download] kagglehub cache: {download_path}")

    # Locate the artifacts inside the cache. kagglehub layouts vary; search by name.
    metadata_src = _find_in_download(download_path, EXPECTED_METADATA)
    if metadata_src is None:
        print(
            f"ERROR: could not find {EXPECTED_METADATA} under {download_path}. "
            "The Kaggle dataset layout may have changed.",
            file=sys.stderr,
        )
        return 2

    image_dir_srcs = []
    for sub in EXPECTED_IMAGE_DIRS:
        d = _find_in_download(download_path, sub)
        if d is None or not d.is_dir():
            print(
                f"ERROR: could not find directory {sub} under {download_path}. "
                "The Kaggle dataset layout may have changed.",
                file=sys.stderr,
            )
            return 2
        image_dir_srcs.append(d)

    # Lay them out where Stage 0 expects.
    print(f"[stage0-download] linking artifacts into {raw_dir}...")
    _link_or_copy(metadata_src, raw_dir / EXPECTED_METADATA)
    for sub, src in zip(EXPECTED_IMAGE_DIRS, image_dir_srcs):
        _link_or_copy(src, raw_dir / sub)

    # Sanity check — count images per part.
    for sub in EXPECTED_IMAGE_DIRS:
        n = len(list((raw_dir / sub).glob("*.jpg")))
        print(f"[stage0-download]   {sub}: {n} jpgs")

    print("[stage0-download] DONE")
    print(f"[stage0-download] next: python -m scripts.stage0_prepare_dataset")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download HAM10000 from Kaggle into data/raw/HAM10000/."
    )
    p.add_argument(
        "--force", action="store_true",
        help="re-run even if the expected files already exist",
    )
    sys.exit(main(p.parse_args()))
