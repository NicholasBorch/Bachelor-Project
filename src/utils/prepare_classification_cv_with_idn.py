from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.classification.noise_idn import (
    generate_oof_nestedcv,
    apply_idn_from_oof,
)

# CONFIGS
SEED = 42

# Outer evaluation CV
OUTER_FOLDS = 2  # set to 5 normally, but 2 for quick testing.

# Inner CV 
INNER_FOLDS = 2  # set to 5 normally, but 2 for quick testing.

# Noise settings (IDN)
NOISE_RATES = [0.10, 0.20, 0.30, 0.40]  # Multiple global noise levels.
ETA_MAX = 0.30 # No more than 30% of a given class is flipped.
SCORE_TYPE = "p_true"

# Teacher model for uncertainty ranking
ARCH = "resnet18"
PRETRAINED = True
TEACHER_EPOCHS = 1  # Set to 3 normally, but 1 for quick testing.

# Training hyperparams for teacher
BATCH_SIZE = 16  # Set to 64 normally, but 16 for quick testing.
LR = 3e-4
WEIGHT_DECAY = 1e-4

# Runtime
NUM_WORKERS = 2
USE_AMP = True
PIN_MEMORY = True


# Project Paths
def project_root() -> Path:
    """
    src/utils/prepare_classification_cv_with_idn.py -> parents[2] is repo root.
    """
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = project_root()

    # Input: one-image-per-lesion processed dataset
    ham_one = root / "data" / "processed" / "HAM10000" / "one_image_per_lesion"
    meta_path = ham_one / "HAM10000_metadata_one_per_lesion.csv"
    images_dir = ham_one / "images"

    # Output: CV folds
    out_root = root / "data" / "processed" / "HAM10000" / "cv"
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print("Prepare HAM10000 CV folds + IDN noise")
    print("========================================")
    print("Input metadata:", meta_path)
    print("Input images  :", images_dir)
    print("Output folder :", out_root)
    print("----------------------------------------")
    print(f"SEED={SEED}")
    print(f"OUTER_FOLDS={OUTER_FOLDS} | INNER_FOLDS={INNER_FOLDS}")
    print(f"NOISE_RATES={NOISE_RATES} | ETA_MAX={ETA_MAX}")
    print(f"SCORE_TYPE={SCORE_TYPE}")
    print(f"TEACHER={ARCH} | PRETRAINED={PRETRAINED} | EPOCHS={TEACHER_EPOCHS}")
    print(f"BATCH_SIZE={BATCH_SIZE} | LR={LR} | WEIGHT_DECAY={WEIGHT_DECAY}")
    print(f"NUM_WORKERS={NUM_WORKERS} | AMP={USE_AMP} | PIN_MEMORY={PIN_MEMORY}")
    print("========================================\n")

    df = pd.read_csv(meta_path)

    # ============================================================
    # STAGE 1 — Generate OOF probabilities (expensive, done once)
    # ============================================================
    print("\nGenerating OOF predictions (nested CV)...\n")

    oof_outputs = generate_oof_nestedcv(
        df=df,
        images_dir=images_dir,
        outer_folds=OUTER_FOLDS,
        inner_folds=INNER_FOLDS,
        seed=SEED,
        score_type=SCORE_TYPE,
        arch=ARCH,
        pretrained=PRETRAINED,
        teacher_epochs=TEACHER_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        num_workers=NUM_WORKERS,
        use_amp=USE_AMP,
        pin_memory=PIN_MEMORY,
    )

    # Save fold assignments (shared across all noise levels)
    fold_assign_path = out_root / "fold_assignments.csv"
    oof_outputs.fold_assignments.to_csv(fold_assign_path, index=False)
    print("Saved:", fold_assign_path)

    # ============================================================
    # STAGE 2 — Apply IDN for multiple global noise rates
    # ============================================================

    for noise_rate in NOISE_RATES:
        print(f"\nApplying IDN noise for global rate = {noise_rate:.2f}")

        rate_folder = out_root / f"idn_r{int(noise_rate * 100):02d}"
        rate_folder.mkdir(parents=True, exist_ok=True)

        for fold_id, fold_data in oof_outputs.folds.items():
            fold_dir = rate_folder / f"fold_{fold_id:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_clean, train_noisy, report = apply_idn_from_oof(
                train_df=fold_data.train_df,
                oof_scores=fold_data.oof_scores,
                oof_probs=fold_data.oof_probs,
                outer_fold=fold_id,
                seed=SEED,
                noise_rate=noise_rate,
                eta_max=ETA_MAX,
                score_type=SCORE_TYPE,
                arch=ARCH,
            )

            test_clean = fold_data.test_df.copy()

            train_clean_path = fold_dir / "train_clean.csv"
            train_noisy_path = fold_dir / "train_noisy.csv"
            test_clean_path = fold_dir / "test_clean.csv"
            report_path = fold_dir / "noise_report.json"

            train_clean.to_csv(train_clean_path, index=False)
            train_noisy.to_csv(train_noisy_path, index=False)
            test_clean.to_csv(test_clean_path, index=False)

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2)

            print(
                f"[r={noise_rate:.2f} | fold {fold_id:02d}] "
                f"flipped {report.n_flipped}/{report.n_train}"
            )

    print("\nDone.")
    print("CV + multi-rate IDN artifacts written to:", out_root.resolve())


if __name__ == "__main__":
    main()