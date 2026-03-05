# src/utils/prepare_classification_cv_feature_driven.py
# One-time script to generate outer CV folds with feature-driven IDN-corrupted training splits.
# Run this after prepare_classification_cv.py — it reuses the same fold structure
# but replaces random projections with OOF softmax probabilities from inner-fold ResNets.

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.common.io import project_root
from src.common.seed import seed_everything
from src.classification.noise_idn_feature_driven import generate_feature_driven_idn_outercv
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

# Inner fold count for OOF probability estimation
INNER_FOLDS = 5

# Training settings for inner fold baseline models
EPOCHS = 30
LR     = 1e-4


def main() -> None:
    seed_everything(SEED)

    root = project_root()

    # Input: same preprocessed HAM10000 as the standard IDN script
    ham_one    = root / "data" / "processed" / "HAM10000" / "one_image_per_lesion"
    meta_path  = ham_one / "HAM10000_metadata_one_per_lesion.csv"
    images_dir = ham_one / "images"

    # Output: written to a separate folder to keep standard and feature-driven results distinct
    out_root = root / "data" / "processed" / "HAM10000" / "cv_feature_driven"
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print("Prepare HAM10000 Outer CV + Feature-Driven IDN")
    print("========================================")
    print(f"Input metadata  : {meta_path}")
    print(f"Input images    : {images_dir}")
    print(f"Output folder   : {out_root}")
    print(f"SEED={SEED} | OUTER_FOLDS={OUTER_FOLDS} | INNER_FOLDS={INNER_FOLDS}")
    print(f"NOISE_RATES(tau)={NOISE_RATES} | NORM_STD={NORM_STD}")
    print(f"IMAGE_SIZE={IMAGE_SIZE} | BATCH_SIZE={BATCH_SIZE} | EPOCHS={EPOCHS}")
    print("========================================\n")

    df = pd.read_csv(meta_path)

    # Run the full nested CV pipeline — OOF probs collected once per outer fold,
    # then reused across all tau values to avoid redundant training runs
    all_outputs = generate_feature_driven_idn_outercv(
        df=df,
        images_dir=images_dir,
        outer_folds=OUTER_FOLDS,
        inner_folds=INNER_FOLDS,
        seed=SEED,
        tau_values=NOISE_RATES,
        image_size=IMAGE_SIZE,
        norm_std=NORM_STD,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        num_workers=NUM_WORKERS,
    )

    # Save fold assignments once — identical structure to the standard IDN script
    fold_assign_path = out_root / "fold_assignments.csv"
    first_outputs = next(iter(all_outputs.values()))
    first_outputs.fold_assignments.to_csv(fold_assign_path, index=False)
    print(f"\nSaved fold assignments: {fold_assign_path}")

    # Write per-tau, per-fold artifacts
    for tau, outputs in all_outputs.items():
        folder_name = "clean" if tau == 0.0 else f"idn_feature_tau{int(tau * 100):02d}"
        rate_folder = out_root / folder_name
        rate_folder.mkdir(parents=True, exist_ok=True)

        for fold_id, fold_data in tqdm(outputs.folds.items(), desc=f"Writing folds (tau={tau:.2f})", leave=False):
            fold_dir = rate_folder / f"fold_{fold_id:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            fold_data.train_clean.to_csv(fold_dir / "train_clean.csv", index=False)
            fold_data.train_noisy.to_csv(fold_dir / "train_noisy.csv", index=False)
            fold_data.test_clean.to_csv(fold_dir  / "test_clean.csv",  index=False)

            with open(fold_dir / "noise_report.json", "w", encoding="utf-8") as f:
                json.dump(asdict(fold_data.report), f, indent=2)

        # Print summary to verify actual flip rate matches target tau
        flipped_total = sum(v.report.n_flipped for v in outputs.folds.values())
        train_total   = sum(v.report.n_train   for v in outputs.folds.values())
        actual_rate   = (flipped_total / max(train_total, 1)) * 100
        print(f"tau={tau:.2f} | flipped {flipped_total}/{train_total} ({actual_rate:.1f}%) across all folds")

    print("\nDone. Feature-driven IDN artifacts written to:", out_root.resolve())


if __name__ == "__main__":
    main()