"""
Download HAM10000 from Kaggle into data/raw/HAM10000/ (metadata CSV + two image dirs).

Alternative to notebooks/HAM10000_data_loader.ipynb. Reads Kaggle
credentials from ~/.kaggle/kaggle.json. Idempotent: skips if the expected files
already exist unless --force. Run stage0_prepare_dataset next.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.utils.io import load_config, project_root


KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"

# Files Stage 0 reads.
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
    """Locate name (file or directory) anywhere in the Kaggle cache tree."""
    for candidate in src_root.rglob(name):
        return candidate
    return None


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink src into dst, falling back to a copy if symlinking fails."""
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

    print(f"[stage0-download] downloading {KAGGLE_DATASET} via kagglehub...")
    print("[stage0-download] (first-time downloads are ~3 GB and can take a while)")
    download_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"[stage0-download] kagglehub cache: {download_path}")

    # Locate the artifacts inside the cache.
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