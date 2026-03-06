# runs/run_classification_cv.py
# Entry point for running classification experiments across all outer folds and tau levels.
# Each method trains for a fixed number of epochs with no early stopping.
# Test evaluation happens once at the end of training for each fold.
# Run from repo root: python -m runs.run_classification_cv

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.common.io import project_root
from src.common.seed import seed_everything
from src.methods.baseline import run_baseline_fold
from configs.classification_default import (
    SEED,
    OUTER_FOLDS,
    NOISE_RATES,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LR,
    BACKBONE_DEPTH,
)

# =============================================================
# CONFIG — edit before running
# =============================================================
METHOD     = "baseline"       # "baseline" | "elr" | "sce" | "asyco"
NOISE_TYPE = "standard_idn"   # "standard_idn" | "feature_driven_idn"

# Subset of NOISE_RATES to run — set to NOISE_RATES to run all
TAU_VALUES = NOISE_RATES
# =============================================================


METHOD_REGISTRY = {
    "baseline": run_baseline_fold,
    # "elr":   run_elr_fold,    # uncomment when implemented
    # "sce":   run_sce_fold,    # uncomment when implemented
    # "asyco": run_asyco_fold,  # uncomment when implemented
}

NOISE_TYPE_TO_CV_DIR = {
    "standard_idn":       "cv",
    "feature_driven_idn": "cv_feature_driven",
}


def get_fold_paths(cv_root: Path, tau: float, fold_id: int) -> tuple[Path, Path]:
    # Returns paths to train_noisy and test_clean CSVs for one fold
    if "feature_driven" in str(cv_root):
        tau_folder = "clean" if tau == 0.0 else f"idn_feature_tau{int(tau * 100):02d}"
    else:
        tau_folder = "clean" if tau == 0.0 else f"idn_tau{int(tau * 100):02d}"

    fold_dir    = cv_root / tau_folder / f"fold_{fold_id:02d}"
    train_noisy = fold_dir / "train_noisy.csv"
    test_clean  = fold_dir / "test_clean.csv"
    return train_noisy, test_clean


def main() -> None:
    seed_everything(SEED)

    root         = project_root()
    ham_root     = root / "data" / "processed" / "HAM10000"
    images_dir   = ham_root / "one_image_per_lesion" / "images"
    cv_root      = ham_root / NOISE_TYPE_TO_CV_DIR[NOISE_TYPE]
    results_root = root / "results" / "HAM10000"

    if METHOD not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{METHOD}'. Available: {list(METHOD_REGISTRY.keys())}"
        )

    run_fold_fn = METHOD_REGISTRY[METHOD]

    print("\n" + "=" * 60)
    print(f"Classification CV — method={METHOD} | noise={NOISE_TYPE}")
    print(f"Folds: {OUTER_FOLDS} | Tau values: {TAU_VALUES}")
    print(f"Backbone: resnet{BACKBONE_DEPTH} | Epochs: {EPOCHS} | LR: {LR}")
    print("=" * 60 + "\n")

    completed, skipped, failed = 0, 0, 0

    for tau in tqdm(TAU_VALUES, desc="Tau levels"):
        for fold_id in range(OUTER_FOLDS):
            train_noisy_path, test_clean_path = get_fold_paths(cv_root, tau, fold_id)

            # Skip if fold data does not exist yet
            if not train_noisy_path.exists() or not test_clean_path.exists():
                print(f"  Skipping tau={tau:.2f} fold={fold_id} — fold data not found")
                skipped += 1
                continue

            # Skip completed runs to allow safe resubmission on HPC
            from src.common.logging import make_output_dir
            out_dir = make_output_dir(results_root, METHOD, tau, fold_id, NOISE_TYPE)
            if (out_dir / "test_metrics.json").exists():
                print(f"  Skipping tau={tau:.2f} fold={fold_id} — already completed")
                skipped += 1
                continue

            try:
                train_noisy_df = pd.read_csv(train_noisy_path)
                test_clean_df  = pd.read_csv(test_clean_path)

                run_fold_fn(
                    train_noisy_df=train_noisy_df,
                    test_clean_df=test_clean_df,
                    images_dir=images_dir,
                    results_root=results_root,
                    tau=tau,
                    outer_fold=fold_id,
                    seed=SEED * 10_000 + fold_id,
                    noise_type=NOISE_TYPE,
                    backbone_depth=BACKBONE_DEPTH,
                    image_size=IMAGE_SIZE,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    lr=LR,
                    num_workers=NUM_WORKERS,
                )
                completed += 1

            except Exception as e:
                print(f"  ERROR — tau={tau:.2f} fold={fold_id}: {e}")
                failed += 1
                continue

    print(f"\n{'='*60}")
    print(f"Done. Completed={completed} | Skipped={skipped} | Failed={failed}")
    print(f"Results: {(results_root / METHOD / NOISE_TYPE).resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()