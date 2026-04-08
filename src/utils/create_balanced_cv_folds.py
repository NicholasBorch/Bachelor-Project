"""
create_balanced_cv_folds.py

Generates stratified 10-fold cross-validation splits with BOTH standard IDN and
normalized IDN noise injection for the balanced HAM10000 dataset.

Standard IDN  (normalize=False) → cv_balanced_standard/    (reference/visual analysis only)
Normalized IDN (normalize=True)  → cv_balanced_normalized/  (primary experiment noise type)

Both variants are created in a single pass so fold splits are identical and noise
seeds are consistent.

Usage:
    # Single fold (for HPC parallelism):
    python -m src.utils.create_balanced_cv_folds --fold 0

    # All folds sequentially:
    python -m src.utils.create_balanced_cv_folds --fold all

Output:
    data/processed/HAM10000/cv_balanced_standard/
    ├── clean/fold_00/{train_noisy.csv, test_clean.csv}
    ├── idn_tau05/fold_00/{train_noisy.csv, test_clean.csv}
    └── idn_tau30/fold_09/{train_noisy.csv, test_clean.csv}

    data/processed/HAM10000/cv_balanced_normalized/
    ├── clean/fold_00/{train_noisy.csv, test_clean.csv}
    ├── idn_tau05/fold_00/{train_noisy.csv, test_clean.csv}
    └── idn_tau30/fold_09/{train_noisy.csv, test_clean.csv}
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.common.io import project_root
from src.common.seed import seed_everything
from src.classification.noise_idn import generate_instance_dependent_noisy_labels

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 10
FOLDS       = 10
NOISE_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
NORM_STD    = 0.10   # truncated normal std for per-sample flip rates
IMAGE_SIZE  = 224
BATCH_SIZE  = 64
NUM_WORKERS = 2
PIN_MEMORY  = True

CLASS_COL   = "dx"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT                  = project_root()
METADATA_IN           = ROOT / "data/processed/HAM10000/one_image_per_lesion/metadata_balanced.csv"
IMAGES_DIR            = ROOT / "data/processed/HAM10000/one_image_per_lesion/images"
OUTPUT_DIR_STANDARD   = ROOT / "data/processed/HAM10000/cv_balanced_standard"
OUTPUT_DIR_NORMALIZED = ROOT / "data/processed/HAM10000/cv_balanced_normalized"

# Both IDN variants to generate in a single pass.
# Each entry: (output_dir, normalize_flag, label)
IDN_VARIANTS = [
    (OUTPUT_DIR_STANDARD,   False, "standard"),
    (OUTPUT_DIR_NORMALIZED, True,  "normalized"),
]


def _tau_tag(tau: float) -> str:
    """Convert noise rate to directory name (e.g. 0.05 → 'idn_tau05', 0.0 → 'clean')."""
    if tau == 0.0:
        return "clean"
    return f"idn_tau{int(round(tau * 100)):02d}"


def prepare_fold(
    fold_id:     int,
    df:          pd.DataFrame,
    train_idx:   np.ndarray,
    test_idx:    np.ndarray,
    images_dir:  Path,
    noise_rates: list  = NOISE_RATES,
    norm_std:    float = NORM_STD,
    seed:        int   = SEED,
) -> None:
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df  = df.iloc[test_idx].copy().reset_index(drop=True)

    fold_tag   = f"fold_{fold_id:02d}"
    noise_seed = seed * 10_000 + fold_id

    print(f"  Processing {fold_tag}: {len(train_df)} train / {len(test_df)} test")

    for output_dir, normalize, variant_label in IDN_VARIANTS:
        print(f"    Variant: {variant_label} IDN  →  {output_dir.name}")

        for tau in noise_rates:
            tag     = _tau_tag(tau)
            out_dir = output_dir / tag / fold_tag
            out_dir.mkdir(parents=True, exist_ok=True)

            if tau == 0.0:
                # Clean: train_noisy has original labels (name kept for compatibility)
                train_noisy = train_df.copy()
            else:
                df_corrupted, _ = generate_instance_dependent_noisy_labels(
                    df=train_df[["image_id", "lesion_id", "dx"]].copy(),
                    images_dir=images_dir,
                    tau=tau,
                    seed=noise_seed,
                    normalize=normalize,
                    image_size=IMAGE_SIZE,
                    norm_std=norm_std,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY,
                )

                # CRITICAL: Overwrite dx with the noisy labels before saving.
                # generate_instance_dependent_noisy_labels returns dx unchanged,
                # with corrupted labels in dx_noisy. The training code reads dx,
                # so we must set dx = dx_noisy for the noise to take effect.
                train_noisy = df_corrupted.copy()
                train_noisy["dx"] = train_noisy["dx_noisy"]

                noisy_count = int((df_corrupted["dx_clean"] != df_corrupted["dx_noisy"]).sum())
                actual_rate = noisy_count / len(train_df)
                print(f"      τ={tau:.2f}: {noisy_count}/{len(train_df)} flipped "
                      f"(actual={actual_rate:.3f})")

            train_noisy.to_csv(out_dir / "train_noisy.csv", index=False)
            test_df[["image_id", "lesion_id", "dx"]].to_csv(
                out_dir / "test_clean.csv", index=False
            )

    print(f"  {fold_tag} done.")


def run_fold(fold_id: int) -> None:
    seed_everything(SEED)

    df = pd.read_csv(METADATA_IN)
    if CLASS_COL not in df.columns:
        raise ValueError(f"Column '{CLASS_COL}' not found in {METADATA_IN}")

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(df, df[CLASS_COL]))

    train_idx, test_idx = splits[fold_id]
    prepare_fold(
        fold_id=fold_id,
        df=df,
        train_idx=train_idx,
        test_idx=test_idx,
        images_dir=IMAGES_DIR,
    )


def run_all_folds() -> None:
    seed_everything(SEED)

    df = pd.read_csv(METADATA_IN)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    print(f"Creating balanced IDN CV folds (standard + normalized).")
    print(f"  Metadata: {METADATA_IN}  ({len(df)} samples)")
    print(f"  Outputs:  {OUTPUT_DIR_STANDARD.name}  |  {OUTPUT_DIR_NORMALIZED.name}")
    print(f"  Folds: {FOLDS}  |  Noise rates: {NOISE_RATES}")

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(df, df[CLASS_COL])):
        prepare_fold(
            fold_id=fold_id,
            df=df,
            train_idx=train_idx,
            test_idx=test_idx,
            images_dir=IMAGES_DIR,
        )

    print("\nAll folds complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create balanced CV folds (standard IDN + normalized IDN)."
    )
    parser.add_argument(
        "--fold", type=str, default="all",
        help="Fold index (0–9) or 'all' to run all folds sequentially.",
    )
    args = parser.parse_args()

    if args.fold == "all":
        run_all_folds()
    else:
        fold_id = int(args.fold)
        assert 0 <= fold_id < FOLDS, f"fold must be 0–{FOLDS - 1}, got {fold_id}"
        print(f"Creating balanced IDN folds (standard + normalized) for fold {fold_id}.")
        run_fold(fold_id)