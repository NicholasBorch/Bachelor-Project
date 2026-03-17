# src/utils/prepare_classification_cv.py
#
# Generates CV fold artifacts for ONE fold using either standard or normalized IDN.
# Designed to run as a parallel HPC job — submit 5 jobs each with --fold 0..4.
#
# Two noise methods are supported:
#   --method standard    → Xia et al. (2020), raw pixel values in [0, 1]
#   --method normalized  → Xia et al. + ImageNet channel normalisation
#                          Reduces concentration bias from class imbalance.
#
# For each fold, all tau values in NOISE_RATES are processed sequentially within
# the job (tau values are fast — no model training, just pixel projection ops).
#
# Usage (from repo root):
#   python -m src.utils.prepare_classification_cv --fold 0 --method standard
#   python -m src.utils.prepare_classification_cv --fold 2 --method normalized
#
# Output structure:
#   data/processed/HAM10000/cv_standard/
#     fold_assignments.csv          (written by fold 0 only)
#     clean/fold_00/{train_clean,train_noisy,test_clean}.csv + noise_report.json
#     idn_tau05/fold_00/...
#     idn_tau10/fold_00/...
#     ...
#   data/processed/HAM10000/cv_normalized/
#     (same structure)

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.common.io import project_root
from src.common.seed import seed_everything
from src.classification.noise_idn import generate_instance_dependent_noisy_labels
from src.classification.folds import make_outer_folds_lesion_stratified
from configs.classification_default import (
    SEED,
    OUTER_FOLDS,
    NOISE_RATES,
    NORM_STD,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)

# Output root per method — keeps standard and normalized results clearly separated
METHOD_OUT_DIRS = {
    "standard":   "cv_standard",
    "normalized": "cv_normalized",
}


def process_fold(
    fold_id: int,
    method: str,
    df: pd.DataFrame,
    images_dir: Path,
    out_root: Path,
) -> None:
    """
    Generates noisy training splits for fold_id at all tau values.
    One call per HPC job.
    """
    normalize = (method == "normalized")

    # Rebuild the same fold split used across the whole pipeline
    df_folds = make_outer_folds_lesion_stratified(df, n_splits=OUTER_FOLDS, seed=SEED)

    test_df  = df_folds[df_folds["outer_fold"] == fold_id].copy().reset_index(drop=True)
    train_df = df_folds[df_folds["outer_fold"] != fold_id].copy().reset_index(drop=True)

    print(f"\nFold {fold_id} | method={method} | "
          f"train={len(train_df)} | test={len(test_df)}")

    # ── Save fold assignments on fold 0 only (same across all folds/methods) ─
    # Other folds skip this to avoid write conflicts in parallel execution
    if fold_id == 0:
        assign_path = out_root / "fold_assignments.csv"
        df_folds[["image_id", "lesion_id", "dx", "outer_fold"]].to_csv(
            assign_path, index=False
        )
        print(f"  Saved fold assignments: {assign_path}")

    # ── Process each tau value for this fold ──────────────────────────────
    for tau in NOISE_RATES:
        folder_name = "clean" if tau == 0.0 else f"idn_tau{int(tau * 100):02d}"
        fold_dir = out_root / folder_name / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        df_corrupted, report = generate_instance_dependent_noisy_labels(
            df=train_df[["image_id", "lesion_id", "dx"]].copy(),
            images_dir=images_dir,
            tau=tau,
            seed=(SEED * 10_000 + fold_id),  # unique seed per fold
            normalize=normalize,
            image_size=IMAGE_SIZE,
            norm_std=NORM_STD,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        report.outer_fold = int(fold_id)

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

        n_flip = report.n_flipped
        n_train = report.n_train
        print(f"  tau={tau:.2f} | flipped {n_flip}/{n_train} "
              f"({100 * n_flip / max(n_train, 1):.1f}%) → {fold_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare one CV fold with standard or normalized IDN noise."
    )
    parser.add_argument("--fold", type=int, required=True,
                        help="Fold index (0-indexed)")
    parser.add_argument("--method", choices=["standard", "normalized"], required=True,
                        help="Noise method: standard (Xia et al.) or normalized")
    args = parser.parse_args()

    seed_everything(SEED)

    root       = project_root()
    ham_one    = root / "data" / "processed" / "HAM10000" / "one_image_per_lesion"
    meta_path  = ham_one / "HAM10000_metadata_one_per_lesion.csv"
    images_dir = ham_one / "images"
    out_root   = root / "data" / "processed" / "HAM10000" / METHOD_OUT_DIRS[args.method]
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"=== Prepare Classification CV ===")
    print(f"method={args.method} | fold={args.fold} | SEED={SEED}")
    print(f"NOISE_RATES={NOISE_RATES} | NORM_STD={NORM_STD}")
    print(f"Output: {out_root}")

    if not (0 <= args.fold < OUTER_FOLDS):
        raise ValueError(f"--fold must be in [0, {OUTER_FOLDS - 1}], got {args.fold}")

    df = pd.read_csv(meta_path)
    df["image_id"]  = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"]        = df["dx"].astype(str)

    process_fold(
        fold_id=args.fold,
        method=args.method,
        df=df,
        images_dir=images_dir,
        out_root=out_root,
    )

    print(f"\nDone — fold {args.fold} written to {out_root}")


if __name__ == "__main__":
    main()