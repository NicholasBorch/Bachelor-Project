"""
create_balanced_dataset.py

Downsamples the one-image-per-lesion HAM10000 metadata to the minimum class count,
producing a class-balanced dataset used for the balanced experiment arm.

Usage:
    python -m src.utils.create_balanced_dataset

Output:
    data/processed/HAM10000/one_image_per_lesion/metadata_balanced.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.common.io import project_root
from src.common.seed import seed_everything

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 10
CLASS_COL = "dx"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = project_root()
METADATA_IN  = ROOT / "data/processed/HAM10000/one_image_per_lesion/HAM10000_metadata_one_per_lesion.csv"
METADATA_OUT = ROOT / "data/processed/HAM10000/one_image_per_lesion/metadata_balanced.csv"


def create_balanced_metadata(
    metadata_path: Path = METADATA_IN,
    output_path:   Path = METADATA_OUT,
    seed:          int  = SEED,
) -> pd.DataFrame:
    seed_everything(seed)

    df = pd.read_csv(metadata_path)

    # Validate expected columns
    if CLASS_COL not in df.columns:
        raise ValueError(f"Expected column '{CLASS_COL}' not found. Got: {list(df.columns)}")

    # Count per class
    counts = df[CLASS_COL].value_counts()
    min_class = counts.idxmin()
    min_count = counts.min()

    print("\nClass distribution (one image per lesion):")
    for cls in sorted(counts.index, key=lambda c: -counts[c]):
        print(f"  {cls:>6}: {counts[cls]:>5} → {min_count}")
    print(f"  {'Total':>6}: {len(df):>5} → {min_count * len(counts)}")
    print(f"\nMinimum class: '{min_class}' with {min_count} samples.")
    print(f"Downsampling all classes to {min_count}.")

    # Sample each class down to min_count
    sampled_dfs = []
    for cls in CLASS_NAMES:
        cls_df = df[df[CLASS_COL] == cls]
        if len(cls_df) < min_count:
            raise RuntimeError(
                f"Class '{cls}' has {len(cls_df)} samples, less than min_count={min_count}."
            )
        sampled_dfs.append(cls_df.sample(n=min_count, replace=False, random_state=seed))

    balanced_df = pd.concat(sampled_dfs, ignore_index=True)

    # Validation
    assert len(balanced_df) == min_count * len(CLASS_NAMES), "Row count mismatch after balancing."
    for cls in CLASS_NAMES:
        assert (balanced_df[CLASS_COL] == cls).sum() == min_count, \
            f"Class '{cls}' count mismatch after balancing."

    output_path.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(output_path, index=False)
    print(f"\nSaved balanced metadata → {output_path}")
    print(f"Total rows: {len(balanced_df)}  ({len(CLASS_NAMES)} classes × {min_count} samples)")

    return balanced_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create balanced HAM10000 metadata CSV.")
    parser.add_argument("--metadata_in",  type=str, default=str(METADATA_IN))
    parser.add_argument("--metadata_out", type=str, default=str(METADATA_OUT))
    parser.add_argument("--seed",         type=int, default=SEED)
    args = parser.parse_args()

    create_balanced_metadata(
        metadata_path=Path(args.metadata_in),
        output_path=Path(args.metadata_out),
        seed=args.seed,
    )
