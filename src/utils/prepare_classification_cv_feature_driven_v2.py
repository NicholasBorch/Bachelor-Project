# src/utils/prepare_classification_cv_feature_driven_v2.py
#
# Generates CV fold artifacts for ONE fold using feature-driven IDN v2
# (argmax variant). Designed to run as a parallel HPC job — submit 10 jobs
# each with --fold 0..9.
#
# Prerequisites:
#   - data/processed/HAM10000/fold_probs/fold_probs_full.npy
#     (already produced by the original master_noise_submit.sh pipeline —
#      collect_fold_probs.py + merge_fold_probs.py)
#
# Usage (from repo root):
#   python -m src.utils.prepare_classification_cv_feature_driven_v2 --fold 0
#
# Output structure:
#   data/processed/HAM10000/cv_feature_driven_v2/
#     fold_assignments.csv
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


def process_fold(
    fold_id: int,
    df: pd.DataFrame,
    fold_probs_full: np.ndarray,
    out_root: Path,
) -> None:
    df_folds = make_folds_lesion_stratified(df, n_splits=FOLDS, seed=SEED)

    test_df       = df_folds[df_folds["fold"] == fold_id].copy().reset_index(drop=True)
    train_mask    = df_folds["fold"] != fold_id
    train_df      = df_folds[train_mask].copy().reset_index(drop=True)
    train_indices = df_folds[train_mask].index.tolist()

    # Slice probs for training samples only — row i matches train_df row i
    fold_train_probs = fold_probs_full[train_indices]  # (N_train, C)

    print(f"\nFold {fold_id} | feature-driven v2 (argmax) | "
          f"train={len(train_df)} | test={len(test_df)}")

    if fold_id == 0:
        assign_path = out_root / "fold_assignments.csv"
        df_folds[["image_id", "lesion_id", "dx", "fold"]].to_csv(assign_path, index=False)
        print(f"  Saved fold assignments: {assign_path}")

    for tau in NOISE_RATES:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()

    if not (0 <= args.fold < FOLDS):
        raise ValueError(f"--fold must be in [0, {FOLDS - 1}], got {args.fold}")

    seed_everything(SEED)

    root       = project_root()
    ham_one    = root / "data" / "processed" / "HAM10000" / "one_image_per_lesion"
    meta_path  = ham_one / "HAM10000_metadata_one_per_lesion.csv"
    probs_path = root / "data" / "processed" / "HAM10000" / "fold_probs" / "fold_probs_full.npy"
    out_root   = root / "data" / "processed" / "HAM10000" / "cv_feature_driven_v2"
    out_root.mkdir(parents=True, exist_ok=True)

    if not probs_path.exists():
        raise FileNotFoundError(
            f"Fold probs not found at {probs_path}.\n"
            "Run collect_fold_probs.py (all folds) then merge_fold_probs.py first "
            "(already done as part of the original master_noise_submit.sh pipeline)."
        )

    print(f"=== Prepare Feature-Driven CV v2 (argmax) ===")
    print(f"fold={args.fold} | SEED={SEED} | FOLDS={FOLDS}")

    df = pd.read_csv(meta_path)
    df["image_id"]  = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"]        = df["dx"].astype(str)

    fold_probs_full = np.load(probs_path)
    print(f"Loaded fold probs: {fold_probs_full.shape}")

    process_fold(
        fold_id=args.fold,
        df=df,
        fold_probs_full=fold_probs_full,
        out_root=out_root,
    )

    print(f"\nDone — fold {args.fold} written to {out_root}")


if __name__ == "__main__":
    main()
