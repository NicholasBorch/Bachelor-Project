"""
run_balanced_classification_cv.py

Runner script for balanced-dataset noise-robust classification experiments.
This is the balanced-experiment analogue of run_classification_cv.py.

Key differences from the imbalanced runner:
  1. NOISE_TYPE_TO_CV_DIR maps to cv_balanced_{normalized,feature_driven}
  2. run_*_fold() is called with use_weighted_sampler=False
  3. Accepts an optional --epochs CLI override (avoids editing config files)
  4. Results save to separate dirs via noise_type strings (no path changes needed)

All changes are NEW — this script does not modify run_classification_cv.py.

Usage:
    python -m src.utils.run_balanced_classification_cv \\
        --fold 0 \\
        --noise_type balanced_normalized_idn \\
        --method baseline \\
        --epochs 25
"""

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.common.io import project_root
from src.common.logging import make_output_dir
from src.common.seed import seed_everything

# ── Configs ───────────────────────────────────────────────────────────────────
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

# ── Method runners ────────────────────────────────────────────────────────────
from src.methods.baseline import run_baseline_fold
from src.methods.sce      import run_sce_fold
from src.methods.elr      import run_elr_fold
from src.methods.asyco    import run_asyco_fold

# ── Constants ─────────────────────────────────────────────────────────────────
METHOD_REGISTRY = {
    "baseline": run_baseline_fold,
    "sce":      run_sce_fold,
    "elr":      run_elr_fold,
    "asyco":    run_asyco_fold,
}

# Maps balanced noise-type names → CV data directories
NOISE_TYPE_TO_CV_DIR = {
    "balanced_normalized_idn":     "cv_balanced_normalized",
    "balanced_feature_driven_idn": "cv_balanced_feature_driven",
}


def get_fold_paths(cv_root: Path, tau: float, fold_id: int) -> tuple:
    """Return (train_noisy_path, test_clean_path) for a given tau and fold."""
    if tau == 0.0:
        folder = "clean"
    elif "feature_driven" in str(cv_root):
        folder = f"idn_tau{int(tau * 100):02d}"
    else:
        folder = f"idn_tau{int(tau * 100):02d}"
    fold_dir = cv_root / folder / f"fold_{fold_id:02d}"
    return fold_dir / "train_noisy.csv", fold_dir / "test_clean.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run balanced dataset classification CV experiment."
    )
    parser.add_argument("--fold",       type=int, required=True,
                        help="Fold index (0-indexed)")
    parser.add_argument("--noise_type", type=str, required=True,
                        choices=list(NOISE_TYPE_TO_CV_DIR.keys()),
                        help="Balanced noise type string")
    parser.add_argument("--method",     type=str, default="baseline",
                        choices=list(METHOD_REGISTRY.keys()),
                        help="Training method to use")
    parser.add_argument("--epochs",     type=int, default=None,
                        help="Override default epoch count from config")
    args = parser.parse_args()

    if not (0 <= args.fold < FOLDS):
        raise ValueError(f"--fold must be in [0, {FOLDS - 1}], got {args.fold}")

    seed_everything(SEED)

    epochs = args.epochs if args.epochs is not None else EPOCHS

    root         = project_root()
    ham_root     = root / "data" / "processed" / "HAM10000"
    images_dir   = ham_root / "one_image_per_lesion" / "images"
    cv_root      = ham_root / NOISE_TYPE_TO_CV_DIR[args.noise_type]
    results_root = root / "results" / "HAM10000"

    run_fold_fn = METHOD_REGISTRY[args.method]

    print(f"\n{'='*60}")
    print(f"Balanced Classification CV")
    print(f"method={args.method} | noise={args.noise_type} | fold={args.fold}")
    print(f"Backbone: resnet{BACKBONE_DEPTH} | Epochs: {epochs} | LR: {LR}")
    print(f"Sampler: shuffle (no weighted sampler — balanced data)")
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

        # Method hyperparameters are imported from config files inside each
        # run_*_fold function — we do NOT pass them here. The only difference
        # from the imbalanced runner is use_weighted_sampler=False.
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
            epochs=epochs,
            batch_size=BATCH_SIZE,
            lr=LR,
            num_workers=NUM_WORKERS,
            use_weighted_sampler=False,
        )

    print(f"\n{'='*60}")
    print(f"Fold {args.fold} complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()