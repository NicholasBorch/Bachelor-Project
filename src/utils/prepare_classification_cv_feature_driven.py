# src/utils/prepare_classification_cv_feature_driven.py
#
# Generates CV fold artifacts for ONE fold using feature-driven IDN.
# Designed to run as a parallel HPC job — submit 5 jobs each with --fold 0..4.
#
# PREREQUISITES: must run before this script:
#   1. src/utils/collect_oof_probs.py  --fold 0..4  (5 parallel jobs)
#   2. src/utils/merge_oof_probs.py                  (1 sequential job)
#
# The merged oof_probs_full.npy contains a softmax probability vector for every
# sample in the dataset, collected from a ResNet-18 that never saw that sample.
# These probs replace the random W projection from standard IDN, grounding flip
# targets in visual similarity rather than random geometry.
#
# Usage (from repo root):
#   python -m src.utils.prepare_classification_cv_feature_driven --fold 0
#
# Output structure:
#   data/processed/HAM10000/cv_feature_driven/
#     fold_assignments.csv          (written by fold 0 only)
#     clean/fold_00/{train_clean,train_noisy,test_clean}.csv + noise_report.json
#     idn_feature_tau05/fold_00/...
#     ...

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import project_root
from src.common.seed import seed_everything
from src.classification.noise_idn_feature_driven import generate_feature_driven_noisy_labels
from src.classification.folds import make_outer_folds_lesion_stratified
from configs.classification_default import (
    SEED,
    OUTER_FOLDS,
    NOISE_RATES,
    NORM_STD,
)


def process_fold(
    fold_id: int,
    df: pd.DataFrame,
    oof_probs_full: np.ndarray,
    out_root: Path,
) -> None:
    """
    Generates feature-driven noisy training splits for fold_id at all tau values.
    One call per HPC job.
    """
    # Rebuild the same fold split used in collect_oof_probs.py
    # Identical seed + function guarantees OOF probs were collected leakage-free
    # for every training sample in this fold.
    df_folds = make_outer_folds_lesion_stratified(df, n_splits=OUTER_FOLDS, seed=SEED)

    test_df  = df_folds[df_folds["outer_fold"] == fold_id].copy().reset_index(drop=True)

    # Extract training rows AND their positions in the full dataset.
    # We need the original indices to slice the correct OOF probs.
    train_mask    = df_folds["outer_fold"] != fold_id
    train_df      = df_folds[train_mask].copy().reset_index(drop=True)
    train_indices = df_folds[train_mask].index.tolist()  # positions in full df

    # Slice OOF probs for training samples only
    # Row i of oof_train_probs corresponds to train_df row i
    oof_train_probs = oof_probs_full[train_indices]  # (N_train, C)

    print(f"\nFold {fold_id} | feature-driven | "
          f"train={len(train_df)} | test={len(test_df)}")

    # ── Save fold assignments on fold 0 only ─────────────────────────────
    if fold_id == 0:
        assign_path = out_root / "fold_assignments.csv"
        df_folds[["image_id", "lesion_id", "dx", "outer_fold"]].to_csv(
            assign_path, index=False
        )
        print(f"  Saved fold assignments: {assign_path}")

    # ── Process each tau value for this fold ─────────────────────────────
    for tau in NOISE_RATES:
        folder_name = "clean" if tau == 0.0 else f"idn_feature_tau{int(tau * 100):02d}"
        fold_dir = out_root / folder_name / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        df_corrupted, report = generate_feature_driven_noisy_labels(
            df=train_df[["image_id", "lesion_id", "dx"]].copy(),
            tau=tau,
            seed=(SEED * 10_000 + fold_id),  # unique seed per fold
            oof_probs=oof_train_probs.copy(),
            norm_std=NORM_STD,
        )

        # Build clean and noisy training DataFrames
        keep_cols = ["image_id", "lesion_id", "dx", "dx_clean", "dx_noisy"]

        train_clean      = df_corrupted.copy()
        train_clean["dx"] = train_clean["dx_clean"]
        train_clean      = train_clean[[c for c in keep_cols if c in train_clean.columns]]

        train_noisy      = df_corrupted.copy()
        train_noisy["dx"] = train_noisy["dx_noisy"]
        train_noisy      = train_noisy[[c for c in keep_cols if c in train_noisy.columns]]

        # Write artifacts
        train_clean.to_csv(fold_dir / "train_clean.csv", index=False)
        train_noisy.to_csv(fold_dir / "train_noisy.csv", index=False)
        test_df[["image_id", "lesion_id", "dx"]].to_csv(
            fold_dir / "test_clean.csv", index=False
        )
        with open(fold_dir / "noise_report.json", "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)

        n_flip  = report.n_flipped
        n_train = report.n_train
        print(f"  tau={tau:.2f} | flipped {n_flip}/{n_train} "
              f"({100 * n_flip / max(n_train, 1):.1f}%) → {fold_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare one CV fold with feature-driven IDN noise."
    )
    parser.add_argument("--fold", type=int, required=True,
                        help="Fold index (0-indexed)")
    args = parser.parse_args()

    seed_everything(SEED)

    root    = project_root()
    ham_one = root / "data" / "processed" / "HAM10000" / "one_image_per_lesion"

    meta_path = ham_one / "HAM10000_metadata_one_per_lesion.csv"
    oof_path  = root / "data" / "processed" / "HAM10000" / "oof_probs" / "oof_probs_full.npy"
    out_root  = root / "data" / "processed" / "HAM10000" / "cv_feature_driven"
    out_root.mkdir(parents=True, exist_ok=True)

    if not oof_path.exists():
        raise FileNotFoundError(
            f"OOF probs not found at {oof_path}.\n"
            "Run collect_oof_probs.py (all folds) then merge_oof_probs.py first."
        )

    print(f"=== Prepare Feature-Driven CV ===")
    print(f"fold={args.fold} | SEED={SEED} | NOISE_RATES={NOISE_RATES}")
    print(f"OOF probs: {oof_path}")
    print(f"Output:    {out_root}")

    if not (0 <= args.fold < OUTER_FOLDS):
        raise ValueError(f"--fold must be in [0, {OUTER_FOLDS - 1}], got {args.fold}")

    df = pd.read_csv(meta_path)
    df["image_id"]  = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"]        = df["dx"].astype(str)

    print(f"Loading OOF probs from {oof_path} ...")
    oof_probs_full = np.load(oof_path)  # (N_total, C)
    print(f"OOF probs shape: {oof_probs_full.shape}")

    process_fold(
        fold_id=args.fold,
        df=df,
        oof_probs_full=oof_probs_full,
        out_root=out_root,
    )

    print(f"\nDone — fold {args.fold} written to {out_root}")


if __name__ == "__main__":
    main()