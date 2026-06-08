"""
Stage 0: dataset preparation.

Reads the raw Kaggle HAM10000 download, deduplicates to one image per lesion,
copies the selected images to data/processed/.../images/, and writes the
deduplicated and balanced-subset metadata CSVs.
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

from src.data.ham10000 import CLASS_NAMES
from src.utils.io import load_config, project_root
from src.utils.manifest import write_manifest


def _find_raw_image(raw_dir: Path, image_id: str) -> Path | None:
    """HAM10000 is split across HAM10000_images_part_1 and _part_2. Search both."""
    for sub in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        candidate = raw_dir / sub / f"{image_id}.jpg"
        if candidate.exists():
            return candidate
    return None


def deduplicate_one_image_per_lesion(
    raw_metadata: pd.DataFrame,
    seed: int = 10,
) -> pd.DataFrame:
    """For each unique lesion_id, select one image at random."""
    rng = random.Random(seed)
    rows = []
    for lesion_id, group in raw_metadata.groupby("lesion_id"):
        picked = rng.choice(group.index.tolist())
        rows.append(raw_metadata.loc[picked])
    return pd.DataFrame(rows).reset_index(drop=True)


def create_balanced_subset(
    metadata: pd.DataFrame,
    seed: int = 10,
) -> pd.DataFrame:
    """Downsample each class to the minimum class count."""
    min_count = metadata["dx"].value_counts().min()
    rng = random.Random(seed)
    pieces = []
    for cls in CLASS_NAMES:
        cls_rows = metadata[metadata["dx"] == cls].index.tolist()
        if len(cls_rows) < min_count:
            raise RuntimeError(f"Class {cls} has only {len(cls_rows)} samples, need {min_count}")
        picked = rng.sample(cls_rows, min_count)
        pieces.append(metadata.loc[picked])
    return pd.concat(pieces).sort_values("image_id").reset_index(drop=True)


def main(args: argparse.Namespace) -> int:
    cfg = load_config("base.yaml")
    root = project_root()
    raw_dir = root / cfg["paths"]["data_raw"]
    processed_dir = root / cfg["paths"]["data_processed"] / "one_image_per_lesion"
    images_out = processed_dir / "images"

    if not raw_dir.exists():
        print(f"ERROR: raw data directory {raw_dir} does not exist.", file=sys.stderr)
        print("Please download HAM10000 via the Kaggle notebook first:", file=sys.stderr)
        print("  notebooks/HAM10000_data_loader.ipynb", file=sys.stderr)
        return 1

    raw_meta_path = raw_dir / "HAM10000_metadata.csv"
    if not raw_meta_path.exists():
        print(f"ERROR: expected {raw_meta_path} not found.", file=sys.stderr)
        return 1

    print(f"[stage0] reading raw metadata from {raw_meta_path}")
    raw_meta = pd.read_csv(raw_meta_path)
    print(f"[stage0] raw metadata: {len(raw_meta)} rows, {raw_meta['lesion_id'].nunique()} unique lesions")

    # Deduplicate
    print("[stage0] deduplicating to one image per lesion (seed=42)...")
    dedup = deduplicate_one_image_per_lesion(raw_meta, seed=42)
    print(f"[stage0] after dedup: {len(dedup)} rows")

    # Class distribution check
    class_counts = dedup["dx"].value_counts()
    print("[stage0] class distribution after dedup:")
    for cls in CLASS_NAMES:
        c = int(class_counts.get(cls, 0))
        print(f"           {cls}: {c}")
    unknown = set(dedup["dx"]) - set(CLASS_NAMES)
    if unknown:
        print(f"ERROR: unknown classes in dedup metadata: {unknown}", file=sys.stderr)
        return 1

    # Copy images
    images_out.mkdir(parents=True, exist_ok=True)
    print(f"[stage0] copying images to {images_out}...")
    copied = 0
    missing = []
    for iid in dedup["image_id"]:
        dst = images_out / f"{iid}.jpg"
        if dst.exists() and not args.force:
            copied += 1
            continue
        src = _find_raw_image(raw_dir, iid)
        if src is None:
            missing.append(iid)
            continue
        shutil.copyfile(src, dst)
        copied += 1
    print(f"[stage0] copied/verified {copied} images, {len(missing)} missing")
    if missing:
        print("ERROR: some images not found in raw data. First 5:", missing[:5], file=sys.stderr)
        return 1

    # Write deduplicated metadata
    dedup_out = processed_dir / "HAM10000_metadata_one_per_lesion.csv"
    dedup.to_csv(dedup_out, index=False)
    print(f"[stage0] wrote {dedup_out}")

    # Create balanced subset
    print("[stage0] creating balanced subset (seed=10)...")
    balanced = create_balanced_subset(dedup, seed=10)
    balanced_out = processed_dir / "metadata_balanced.csv"
    balanced.to_csv(balanced_out, index=False)
    print(f"[stage0] balanced subset: {len(balanced)} rows")
    print("[stage0] balanced class distribution:")
    for cls, c in balanced["dx"].value_counts().items():
        print(f"           {cls}: {c}")
    print(f"[stage0] wrote {balanced_out}")

    # Manifest
    manifest_path = root / cfg["paths"]["manifests"] / "stage0.json"
    write_manifest(
        manifest_path,
        stage="stage0",
        params={"force": args.force},
        outputs=[
            str(dedup_out.relative_to(root)),
            str(balanced_out.relative_to(root)),
            str(images_out.relative_to(root)),
        ],
        extra={
            "n_images": len(dedup),
            "n_balanced": len(balanced),
            "n_per_class_balanced": int(balanced["dx"].value_counts().iloc[0]),
        },
    )
    print(f"[stage0] wrote manifest {manifest_path}")
    print("[stage0] DONE")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 0: HAM10000 dataset preparation")
    p.add_argument("--force", action="store_true", help="re-copy images even if they exist")
    sys.exit(main(p.parse_args()))