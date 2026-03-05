# src/utils/prepare_classification_cv.py
# One-time script to generate outer CV folds with IDN-corrupted training splits.
# Run this once before any training to produce the fold artifacts used by all methods.

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.common.io import project_root
from src.common.seed import seed_everything
from src.classification.noise_idn import generate_idn_outercv
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


def main() -> None:
    seed_everything(SEED)

    root = project_root()

    # Input: preprocessed HAM10000 with one image per lesion
    ham_one = root / "data" / "processed" / "HAM10000" / "one_image_per_lesion"
    meta_path = ham_one / "HAM10000_metadata_one_per_lesion.csv"
    images_dir = ham_one / "images"

    # Output: CV fold artifacts written here
    out_root = root / "data" / "processed" / "HAM10000" / "cv"
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print("Prepare HAM10000 Outer CV + IDN (Algorithm 2, Xia et al. 2020)")
    print("========================================")
    print(f"Input metadata : {meta_path}")
    print(f"Input images   : {images_dir}")
    print(f"Output folder  : {out_root}")
    print(f"SEED={SEED} | OUTER_FOLDS={OUTER_FOLDS}")
    print(f"NOISE_RATES(tau)={NOISE_RATES} | NORM_STD={NORM_STD}")
    print(f"IMAGE_SIZE={IMAGE_SIZE} | BATCH_SIZE={BATCH_SIZE}")
    print("========================================\n")

    df = pd.read_csv(meta_path)

    # Fold assignments are seed/fold-count dependent, not tau dependent — write once
    fold_assign_written = False

    for tau in NOISE_RATES:
        print(f"\nApplying IDN with tau = {tau:.2f}")

        # tau=0.0 produces clean folds, all other tau values produce noisy folds
        folder_name = "clean" if tau == 0.0 else f"idn_tau{int(tau * 100):02d}"
        rate_folder = out_root / folder_name
        rate_folder.mkdir(parents=True, exist_ok=True)

        outputs = generate_idn_outercv(
            df=df,
            images_dir=images_dir,
            outer_folds=OUTER_FOLDS,
            seed=SEED,
            tau=tau,
            image_size=IMAGE_SIZE,
            norm_std=NORM_STD,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        # Save shared fold assignments on first tau iteration
        if not fold_assign_written:
            fold_assign_path = out_root / "fold_assignments.csv"
            outputs.fold_assignments.to_csv(fold_assign_path, index=False)
            print(f"Saved fold assignments: {fold_assign_path}")
            fold_assign_written = True

        # Write per-fold CSVs and noise report for each outer fold
        for fold_id, fold_data in tqdm(outputs.folds.items(), desc=f"Writing folds (tau={tau:.2f})", leave=False):
            fold_dir = rate_folder / f"fold_{fold_id:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            fold_data.train_clean.to_csv(fold_dir / "train_clean.csv", index=False)
            fold_data.train_noisy.to_csv(fold_dir / "train_noisy.csv", index=False)
            fold_data.test_clean.to_csv(fold_dir / "test_clean.csv", index=False)

            with open(fold_dir / "noise_report.json", "w", encoding="utf-8") as f:
                json.dump(asdict(fold_data.report), f, indent=2)

        # Print summary to verify actual flip rate is close to target tau
        flipped_total = sum(v.report.n_flipped for v in outputs.folds.values())
        train_total = sum(v.report.n_train for v in outputs.folds.values())
        actual_rate = (flipped_total / max(train_total, 1)) * 100
        print(f"tau={tau:.2f} | flipped {flipped_total}/{train_total} ({actual_rate:.1f}%) across all folds")

    print("\nDone. CV fold artifacts written to:", out_root.resolve())


if __name__ == "__main__":
    main()