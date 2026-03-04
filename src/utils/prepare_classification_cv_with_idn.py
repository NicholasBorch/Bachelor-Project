# src/utils/prepare_classification_cv_with_idn.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.common.seed import seed_everything
from src.classification.noise_idn import generate_idn_outercv


# =========================
# CONFIG (edit and run)
# =========================
SEED = 42

# Outer evaluation CV
OUTER_FOLDS = 5  # 5 normally, smaller for quick tests

# Noise settings (standard synthetic IDN, Algorithm 2)
NOISE_RATES = [0.05, 0.10, 0.15, 0.20]  # tau values
NORM_STD = 0.10                         # std in TruncNorm(tau, norm_std^2)

# IDN feature extraction (pixel-vector)
IMAGE_SIZE = 224

# Runtime
BATCH_SIZE = 64
NUM_WORKERS = 2
PIN_MEMORY = True


def project_root() -> Path:
    """
    src/utils/prepare_classification_cv_with_idn.py -> parents[2] is repo root.
    """
    return Path(__file__).resolve().parents[2]


def main() -> None:
    seed_everything(SEED)

    root = project_root()

    # Input: one-image-per-lesion processed dataset
    ham_one = root / "data" / "processed" / "HAM10000" / "one_image_per_lesion"
    meta_path = ham_one / "HAM10000_metadata_one_per_lesion.csv"
    images_dir = ham_one / "images"

    # Output: CV folds
    out_root = root / "data" / "processed" / "HAM10000" / "cv"
    out_root.mkdir(parents=True, exist_ok=True)

    print("\n========================================")
    print("Prepare HAM10000 Outer CV + Standard Synthetic IDN (Algorithm 2)")
    print("========================================")
    print("Input metadata:", meta_path)
    print("Input images  :", images_dir)
    print("Output folder :", out_root)
    print("----------------------------------------")
    print(f"SEED={SEED}")
    print(f"OUTER_FOLDS={OUTER_FOLDS}")
    print(f"NOISE_RATES(tau)={NOISE_RATES} | NORM_STD={NORM_STD}")
    print(f"IMAGE_SIZE={IMAGE_SIZE}")
    print(f"BATCH_SIZE={BATCH_SIZE} | NUM_WORKERS={NUM_WORKERS} | PIN_MEMORY={PIN_MEMORY}")
    print("========================================\n")

    df = pd.read_csv(meta_path)

    # Save fold assignments once per run (same across tau)
    # Note: fold assignments depend only on SEED/OUTER_FOLDS, not tau.
    # We write it from the first tau run and reuse it for consistency.
    fold_assign_written = False

    for tau in NOISE_RATES:
        print(f"\nApplying standard synthetic IDN with tau = {tau:.2f}")

        rate_folder = out_root / f"idn_algo2_tau{int(tau * 100):02d}"
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

        if not fold_assign_written:
            fold_assign_path = out_root / "fold_assignments.csv"
            outputs.fold_assignments.to_csv(fold_assign_path, index=False)
            print("Saved:", fold_assign_path)
            fold_assign_written = True

        # Write per-fold artifacts
        for fold_id, fold_data in tqdm(outputs.folds.items(), desc=f"Write folds (tau={tau:.2f})", leave=False):
            fold_dir = rate_folder / f"fold_{fold_id:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_clean_path = fold_dir / "train_clean.csv"
            train_noisy_path = fold_dir / "train_noisy.csv"
            test_clean_path = fold_dir / "test_clean.csv"
            report_path = fold_dir / "noise_report.json"

            fold_data.train_clean.to_csv(train_clean_path, index=False)
            fold_data.train_noisy.to_csv(train_noisy_path, index=False)
            fold_data.test_clean.to_csv(test_clean_path, index=False)

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(asdict(fold_data.report), f, indent=2)

        # Quick summary
        flipped_total = sum(v.report.n_flipped for v in outputs.folds.values())
        train_total = sum(v.report.n_train for v in outputs.folds.values())
        print(f"tau={tau:.2f} | flipped total {flipped_total}/{train_total} "
              f"({(flipped_total / max(train_total, 1)) * 100:.1f}%) across all outer folds")

    print("\nDone.")
    print("CV + standard synthetic IDN artifacts written to:", out_root.resolve())


if __name__ == "__main__":
    main()