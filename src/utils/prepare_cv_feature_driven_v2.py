"""
prepare_cv_feature_driven_v2.py

Generates feature-driven IDN v2 (argmax variant) cross-validation fold CSVs
for the FULL imbalanced HAM10000 dataset (7,470 samples).

Uses the existing imbalanced OOF softmax probabilities
(fold_probs/fold_probs_full.npy) and the same lesion-stratified fold
assignments as the v1 feature-driven pipeline.

Must be run AFTER merge_fold_probs.py has produced fold_probs_full.npy.

Usage (one job per fold on HPC, or all sequentially):
    python -m src.utils.prepare_cv_feature_driven_v2 --fold 0
    python -m src.utils.prepare_cv_feature_driven_v2 --fold all

Output:
    data/processed/HAM10000/cv_feature_driven_v2/
    ├── clean/fold_00/{train_noisy.csv, test_clean.csv}
    ├── idn_feature_tau05/fold_00/{train_noisy.csv, test_clean.csv}
    └── idn_feature_tau30/fold_09/{train_noisy.csv, test_clean.csv}
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import project_root
from src.common.seed import seed_everything
from src.classification.noise_idn_feature_driven_v2 import (
    generate_feature_driven_noisy_labels_v2,
)
from src.classification.folds import make_folds_lesion_stratified
from configs.classification_default import (
    SEED,
    FOLDS,
    NOISE_RATES,
    NORM_STD,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = project_root()
METADATA   = ROOT / "data/processed/HAM10000/one_image_per_lesion/HAM10000_metadata_one_per_lesion.csv"
PROBS_FILE = ROOT / "data/processed/HAM10000/fold_probs/fold_probs_full.npy"
OUTPUT_DIR = ROOT / "data/processed/HAM10000/cv_feature_driven_v2"


def process_fold(
    fold_id: int,
    df: pd.DataFrame,
    fold_probs_full: np.ndarray,
    out_root: Path,
) -> None:
    """Generate v2 noisy fold CSVs for one fold of the full imbalanced dataset."""
    # Use the same lesion-stratified folds as all other imbalanced experiments
    df_folds = make_folds_lesion_stratified(df, n_splits=FOLDS, seed=SEED)

    test_df       = df_folds[df_folds["fold"] == fold_id].copy().reset_index(drop=True)
    train_mask    = df_folds["fold"] != fold_id
    train_df      = df_folds[train_mask].copy().reset_index(drop=True)
    train_indices = df_folds[train_mask].index.tolist()

    # Slice OOF probs for training samples only — row i matches train_df row i
    fold_train_probs = fold_probs_full[train_indices]  # (N_train, C)

    print(f"\nFold {fold_id} | feature-driven v2 (argmax) | "
          f"train={len(train_df)} | test={len(test_df)}")

    if fold_id == 0:
        assign_path = out_root / "fold_assignments.csv"
        df_folds[["image_id", "lesion_id", "dx", "fold"]].to_csv(assign_path, index=False)
        print(f"  Saved fold assignments: {assign_path}")

    for tau in NOISE_RATES:
        # Use same folder naming as v1 feature-driven: idn_feature_tauXX
        folder_name = "clean" if tau == 0.0 else f"idn_feature_tau{int(tau * 100):02d}"
        fold_dir = out_root / folder_name / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        df_corrupted, report = generate_feature_driven_noisy_labels_v2(
            df=train_df[["image_id", "lesion_id", "dx"]].copy(),
            tau=tau,
            seed=(SEED * 10_000 + fold_id),
            oof_probs=fold_train_probs.copy(),
            norm_std=NORM_STD,
        )

        keep_cols = ["image_id", "lesion_id", "dx", "dx_clean", "dx_noisy"]

        train_clean       = df_corrupted.copy()
        train_clean["dx"] = train_clean["dx_clean"]
        train_clean       = train_clean[[c for c in keep_cols if c in train_clean.columns]]

        train_noisy       = df_corrupted.copy()
        train_noisy["dx"] = train_noisy["dx_noisy"]
        train_noisy       = train_noisy[[c for c in keep_cols if c in train_noisy.columns]]

        train_clean.to_csv(fold_dir / "train_clean.csv", index=False)
        train_noisy.to_csv(fold_dir / "train_noisy.csv", index=False)
        test_df[["image_id", "lesion_id", "dx"]].to_csv(fold_dir / "test_clean.csv", index=False)
        with open(fold_dir / "noise_report.json", "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)

        n_flip  = report.n_flipped
        n_train = report.n_train
        print(f"  tau={tau:.2f} | flipped {n_flip}/{n_train} "
              f"({100 * n_flip / max(n_train, 1):.1f}%) → {fold_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create feature-driven IDN v2 (argmax) CV folds on the full dataset."
    )
    parser.add_argument("--fold", type=str, default="all",
                        help="Fold index (0-9) or 'all'")
    args = parser.parse_args()

    seed_everything(SEED)

    if not PROBS_FILE.exists():
        raise FileNotFoundError(
            f"Fold probs not found at {PROBS_FILE}.\n"
            "Run collect_fold_probs.py (all folds) then merge_fold_probs.py first."
        )

    print(f"=== Prepare Feature-Driven IDN v2 (argmax) ===")
    print(f"SEED={SEED} | FOLDS={FOLDS}")
    print(f"Metadata: {METADATA}")
    print(f"OOF probs: {PROBS_FILE}")
    print(f"Output: {OUTPUT_DIR}")

    df = pd.read_csv(METADATA)
    df["image_id"]  = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"]        = df["dx"].astype(str)

    fold_probs_full = np.load(PROBS_FILE)
    print(f"Loaded fold probs: {fold_probs_full.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.fold == "all":
        for fold_id in range(FOLDS):
            process_fold(fold_id, df, fold_probs_full, OUTPUT_DIR)
    else:
        fold_id = int(args.fold)
        if not (0 <= fold_id < FOLDS):
            raise ValueError(f"--fold must be in [0, {FOLDS - 1}], got {fold_id}")
        process_fold(fold_id, df, fold_probs_full, OUTPUT_DIR)

    print(f"\nDone — written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
