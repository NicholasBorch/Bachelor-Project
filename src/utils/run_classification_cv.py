# src/utils/run_classification_cv.py
#
# Runs classification experiments for ONE fold across all tau levels.
# Designed to run as a parallel HPC job — submit one job per fold.
#
# Usage (from repo root):
#   python -m src.utils.run_classification_cv \
#       --fold 0 --noise_type standard_idn --method baseline

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.io import project_root
from src.common.logging import make_output_dir
from src.common.seed import seed_everything
from src.methods.baseline import run_baseline_fold
from src.methods.asyco import run_asyco_fold
from src.methods.sce import run_sce_fold
from configs.classification_default import (
    SEED,
    FOLDS,
    NOISE_RATES,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LR,
    BACKBONE_DEPTH,
)

METHOD_REGISTRY = {
    "baseline": run_baseline_fold,
    # "elr":   run_elr_fold,
    "sce":   run_sce_fold,
    "asyco": run_asyco_fold,
}

NOISE_TYPE_TO_CV_DIR = {
    "standard_idn":       "cv_standard",
    "normalized_idn":     "cv_normalized",
    "feature_driven_idn": "cv_feature_driven",
}


def get_fold_paths(cv_root: Path, tau: float, fold_id: int) -> tuple[Path, Path]:
    if "feature_driven" in str(cv_root):
        folder = "clean" if tau == 0.0 else f"idn_feature_tau{int(tau * 100):02d}"
    else:
        folder = "clean" if tau == 0.0 else f"idn_tau{int(tau * 100):02d}"
    fold_dir = cv_root / folder / f"fold_{fold_id:02d}"
    return fold_dir / "train_noisy.csv", fold_dir / "test_clean.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",       type=int, required=True,
                        help="Fold index (0-indexed)")
    parser.add_argument("--noise_type", type=str, required=True,
                        choices=list(NOISE_TYPE_TO_CV_DIR.keys()),
                        help="Noise type determining which CV directory to use")
    parser.add_argument("--method",     type=str, default="baseline",
                        choices=list(METHOD_REGISTRY.keys()),
                        help="Training method to use")
    args = parser.parse_args()

    if not (0 <= args.fold < FOLDS):
        raise ValueError(f"--fold must be in [0, {FOLDS - 1}], got {args.fold}")

    seed_everything(SEED)

    root         = project_root()
    ham_root     = root / "data" / "processed" / "HAM10000"
    images_dir   = ham_root / "one_image_per_lesion" / "images"
    cv_root      = ham_root / NOISE_TYPE_TO_CV_DIR[args.noise_type]
    results_root = root / "results" / "HAM10000"

    run_fold_fn = METHOD_REGISTRY[args.method]

    print(f"\n{'='*60}")
    print(f"Classification CV")
    print(f"method={args.method} | noise={args.noise_type} | fold={args.fold}")
    print(f"Backbone: resnet{BACKBONE_DEPTH} | Epochs: {EPOCHS} | LR: {LR}")
    print(f"{'='*60}\n")

    for tau in NOISE_RATES:
        train_noisy_path, test_clean_path = get_fold_paths(cv_root, tau, args.fold)

        if not train_noisy_path.exists() or not test_clean_path.exists():
            print(f"  Skipping tau={tau:.2f} — fold data not found at {train_noisy_path}")
            continue

        # Skip already completed runs to allow safe resubmission
        out_dir = make_output_dir(results_root, args.method, tau, args.fold, args.noise_type)
        if (out_dir / "test_metrics.json").exists():
            print(f"  Skipping tau={tau:.2f} fold={args.fold} — already completed")
            continue

        train_noisy_df = pd.read_csv(train_noisy_path)
        test_clean_df  = pd.read_csv(test_clean_path)

        run_fold_fn(
            train_noisy_df=train_noisy_df,
            test_clean_df=test_clean_df,
            images_dir=images_dir,
            results_root=results_root,
            tau=tau,
            outer_fold=args.fold,
            seed=SEED * 10_000 + args.fold,
            noise_type=args.noise_type,
            backbone_depth=BACKBONE_DEPTH,
            image_size=IMAGE_SIZE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            num_workers=NUM_WORKERS,
        )

    print(f"\n{'='*60}")
    print(f"Fold {args.fold} complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()