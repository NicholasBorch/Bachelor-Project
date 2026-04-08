"""
prepare_balanced_cv_feature_driven.py

Generates feature-driven IDN cross-validation fold CSVs for the balanced HAM10000 dataset.
Uses the balanced OOF softmax probabilities (fold_probs_balanced/fold_probs_full.npy)
as the flip-target signal instead of random projection matrices.

Must be run AFTER merge_balanced_fold_probs.py has produced fold_probs_full.npy.

Usage (one job per fold on HPC, or all sequentially):
    python -m src.utils.prepare_balanced_cv_feature_driven --fold 0
    python -m src.utils.prepare_balanced_cv_feature_driven --fold all

Output:
    data/processed/HAM10000/cv_balanced_feature_driven/
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
from src.classification.noise_idn_feature_driven import generate_feature_driven_noisy_labels

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 10
FOLDS       = 10
NOISE_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
NORM_STD    = 0.10
CLASS_COL   = "dx"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = project_root()
METADATA_IN = ROOT / "data/processed/HAM10000/one_image_per_lesion/metadata_balanced.csv"
PROBS_FILE  = ROOT / "data/processed/HAM10000/fold_probs_balanced/fold_probs_full.npy"
OUTPUT_DIR  = ROOT / "data/processed/HAM10000/cv_balanced_feature_driven"


def _tau_tag(tau: float) -> str:
    if tau == 0.0:
        return "clean"
    return f"idn_tau{int(round(tau * 100)):02d}"


def prepare_fold(
    fold_id:    int,
    df:         pd.DataFrame,
    oof_probs:  np.ndarray,
    train_idx:  np.ndarray,
    test_idx:   np.ndarray,
    output_dir: Path,
) -> None:
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df  = df.iloc[test_idx].copy().reset_index(drop=True)

    # OOF probs for training samples (indexed by global balanced dataset position)
    train_probs = oof_probs[train_idx]   # (n_train, 7)

    fold_tag = f"fold_{fold_id:02d}"
    print(f"  Processing {fold_tag}: {len(train_df)} train / {len(test_df)} test")

    for tau in NOISE_RATES:
        tag     = _tau_tag(tau)
        out_dir = output_dir / tag / fold_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        noise_seed = SEED * 10_000 + fold_id

        if tau == 0.0:
            train_noisy = train_df.copy()
        else:
            df_corrupted, noise_report = generate_feature_driven_noisy_labels(
                df=train_df[["image_id", "lesion_id", "dx"]].copy(),
                tau=tau,
                seed=noise_seed,
                oof_probs=train_probs,
                norm_std=NORM_STD,
            )

            # CRITICAL: Overwrite dx with the noisy labels before saving.
            # generate_feature_driven_noisy_labels returns dx unchanged,
            # with corrupted labels in dx_noisy. The training code reads dx,
            # so we must set dx = dx_noisy for the noise to take effect.
            train_noisy = df_corrupted.copy()
            train_noisy["dx"] = train_noisy["dx_noisy"]

            noisy_count = int((df_corrupted["dx_clean"] != df_corrupted["dx_noisy"]).sum())
            actual_rate = noisy_count / len(train_df)
            print(f"    τ={tau:.2f}: {noisy_count}/{len(train_df)} labels flipped "
                  f"(actual rate={actual_rate:.3f})")

        train_noisy.to_csv(out_dir / "train_noisy.csv", index=False)
        test_df[["image_id", "lesion_id", "dx"]].to_csv(
            out_dir / "test_clean.csv", index=False
        )

    print(f"  {fold_tag} done.")


def run_fold(fold_id: int) -> None:
    seed_everything(SEED)

    if not PROBS_FILE.exists():
        raise FileNotFoundError(
            f"OOF probs not found: {PROBS_FILE}\n"
            "Run merge_balanced_fold_probs.py first."
        )

    df         = pd.read_csv(METADATA_IN)
    oof_probs  = np.load(PROBS_FILE)

    assert oof_probs.shape[0] == len(df), (
        f"OOF probs shape {oof_probs.shape} does not match "
        f"balanced metadata length {len(df)}."
    )

    skf    = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(df, df[CLASS_COL]))

    train_idx, test_idx = splits[fold_id]
    prepare_fold(
        fold_id=fold_id,
        df=df,
        oof_probs=oof_probs,
        train_idx=train_idx,
        test_idx=test_idx,
        output_dir=OUTPUT_DIR,
    )


def run_all_folds() -> None:
    seed_everything(SEED)

    if not PROBS_FILE.exists():
        raise FileNotFoundError(
            f"OOF probs not found: {PROBS_FILE}\n"
            "Run merge_balanced_fold_probs.py first."
        )

    df        = pd.read_csv(METADATA_IN)
    oof_probs = np.load(PROBS_FILE)

    assert oof_probs.shape[0] == len(df)

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    print(f"Creating balanced feature-driven IDN CV folds.")
    print(f"  Metadata:  {METADATA_IN}  ({len(df)} samples)")
    print(f"  OOF probs: {PROBS_FILE}   shape={oof_probs.shape}")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"  Folds: {FOLDS}  |  Noise rates: {NOISE_RATES}")

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(df, df[CLASS_COL])):
        prepare_fold(
            fold_id=fold_id,
            df=df,
            oof_probs=oof_probs,
            train_idx=train_idx,
            test_idx=test_idx,
            output_dir=OUTPUT_DIR,
        )

    print("\nAll folds complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create balanced feature-driven IDN CV folds."
    )
    parser.add_argument(
        "--fold", type=str, default="all",
        help="Fold index (0-9) or 'all' to run sequentially.",
    )
    args = parser.parse_args()

    if args.fold == "all":
        run_all_folds()
    else:
        fold_id = int(args.fold)
        assert 0 <= fold_id < FOLDS, f"fold must be 0–{FOLDS - 1}, got {fold_id}"
        print(f"Creating balanced feature-driven IDN folds for fold {fold_id}.")
        run_fold(fold_id)