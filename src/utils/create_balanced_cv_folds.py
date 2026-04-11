"""
create_balanced_cv_folds.py

Generates stratified 10-fold CV splits with BOTH standard IDN and normalized IDN
noise injection for the balanced HAM10000 dataset.

Each (variant, tau, fold) directory contains:
    train_noisy.csv, test_clean.csv, noise_report.json
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.common.io import project_root
from src.common.seed import seed_everything
from src.classification.noise_idn import (
    NoiseReport,
    generate_instance_dependent_noisy_labels,
)

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 10
FOLDS       = 10
NOISE_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
NORM_STD    = 0.10
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

IDN_VARIANTS = [
    (OUTPUT_DIR_STANDARD,   False, "standard"),
    (OUTPUT_DIR_NORMALIZED, True,  "normalized"),
]


def _tau_tag(tau: float) -> str:
    if tau == 0.0:
        return "clean"
    return f"idn_tau{int(round(tau * 100)):02d}"


def _make_clean_report(train_df: pd.DataFrame, normalize: bool) -> NoiseReport:
    """Report stub for τ=0.0 so every dir has noise_report.json with the same schema."""
    counts = train_df["dx"].value_counts().to_dict()
    return NoiseReport(
        seed=0,
        tau=0.0,
        norm_std=float(NORM_STD),
        normalize=bool(normalize),
        num_classes=int(len(CLASS_NAMES)),
        feature_size=int(3 * IMAGE_SIZE * IMAGE_SIZE),
        n_train=int(len(train_df)),
        n_flipped=0,
        class_counts_clean=counts,
        class_counts_noisy=counts,
        flip_confusion={},
        flip_rate_min=0.0,
        flip_rate_median=0.0,
        flip_rate_max=0.0,
    )


def prepare_fold(
    fold_id:    int,
    df:         pd.DataFrame,
    train_idx:  np.ndarray,
    test_idx:   np.ndarray,
    images_dir: Path,
) -> None:
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df  = df.iloc[test_idx].copy().reset_index(drop=True)

    fold_tag   = f"fold_{fold_id:02d}"
    noise_seed = SEED * 10_000 + fold_id

    print(f"  Processing {fold_tag}: {len(train_df)} train / {len(test_df)} test")

    for output_dir, normalize, variant_label in IDN_VARIANTS:
        print(f"    Variant: {variant_label} IDN  →  {output_dir.name}")

        for tau in NOISE_RATES:
            out_dir = output_dir / _tau_tag(tau) / fold_tag
            out_dir.mkdir(parents=True, exist_ok=True)

            if tau == 0.0:
                train_noisy  = train_df.copy()
                noise_report = _make_clean_report(train_df, normalize=normalize)
            else:
                df_corrupted, noise_report = generate_instance_dependent_noisy_labels(
                    df=train_df[["image_id", "lesion_id", "dx"]].copy(),
                    images_dir=images_dir,
                    tau=tau,
                    seed=noise_seed,
                    normalize=normalize,
                    image_size=IMAGE_SIZE,
                    norm_std=NORM_STD,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY,
                )

                train_noisy = df_corrupted.copy()
                train_noisy["dx"] = train_noisy["dx_noisy"]

                flipped = int((df_corrupted["dx_clean"] != df_corrupted["dx_noisy"]).sum())
                print(f"      τ={tau:.2f}: {flipped}/{len(train_df)} flipped "
                      f"(actual={flipped / len(train_df):.3f})")

            train_noisy.to_csv(out_dir / "train_noisy.csv", index=False)
            test_df[["image_id", "lesion_id", "dx"]].to_csv(
                out_dir / "test_clean.csv", index=False
            )
            with open(out_dir / "noise_report.json", "w", encoding="utf-8") as f:
                json.dump(asdict(noise_report), f, indent=2)

    print(f"  {fold_tag} done.")


def run_fold(fold_id: int) -> None:
    seed_everything(SEED)
    df = pd.read_csv(METADATA_IN)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(df, df[CLASS_COL]))
    train_idx, test_idx = splits[fold_id]
    prepare_fold(fold_id, df, train_idx, test_idx, IMAGES_DIR)


def run_all_folds() -> None:
    seed_everything(SEED)
    df = pd.read_csv(METADATA_IN)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    print(f"Creating balanced IDN CV folds (standard + normalized).")
    print(f"  Metadata: {METADATA_IN}  ({len(df)} samples)")
    print(f"  Outputs:  {OUTPUT_DIR_STANDARD.name}  |  {OUTPUT_DIR_NORMALIZED.name}")
    print(f"  Folds: {FOLDS}  |  Noise rates: {NOISE_RATES}")

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(df, df[CLASS_COL])):
        prepare_fold(fold_id, df, train_idx, test_idx, IMAGES_DIR)

    print("\nAll folds complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create balanced CV folds (standard IDN + normalized IDN)."
    )
    parser.add_argument("--fold", type=str, default="all",
                        help="Fold index (0-9) or 'all'.")
    args = parser.parse_args()

    if args.fold == "all":
        run_all_folds()
    else:
        fold_id = int(args.fold)
        assert 0 <= fold_id < FOLDS, f"fold must be 0-{FOLDS - 1}, got {fold_id}"
        run_fold(fold_id)
