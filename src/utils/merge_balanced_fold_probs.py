"""
merge_balanced_fold_probs.py

Merges per-fold OOF softmax probability files produced by collect_balanced_fold_probs.py
into a single (N_balanced, 7) array indexed by balanced dataset row position.

Must be run after ALL 10 fold jobs of collect_balanced_fold_probs.py have completed.

Usage:
    python -m src.utils.merge_balanced_fold_probs
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import project_root

# ── Config ────────────────────────────────────────────────────────────────────
FOLDS       = 10
NUM_CLASSES = 7

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = project_root()
METADATA_IN = ROOT / "data/processed/HAM10000/one_image_per_lesion/metadata_balanced.csv"
PROBS_DIR   = ROOT / "data/processed/HAM10000/fold_probs_balanced"
OUTPUT_FILE = PROBS_DIR / "fold_probs_full.npy"


def merge_fold_probs() -> np.ndarray:
    df = pd.read_csv(METADATA_IN)
    n_samples = len(df)
    print(f"Balanced dataset: {n_samples} samples")

    # Check all fold files exist before starting
    missing = []
    for fold_id in range(FOLDS):
        probs_file   = PROBS_DIR / f"fold_{fold_id:02d}_probs.npy"
        indices_file = PROBS_DIR / f"fold_{fold_id:02d}_indices.npy"
        if not probs_file.exists():
            missing.append(str(probs_file))
        if not indices_file.exists():
            missing.append(str(indices_file))

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} file(s). Run collect_balanced_fold_probs.py for all folds first.\n"
            + "\n".join(f"  {f}" for f in missing)
        )

    # Initialise full array with NaN so we can detect any samples that were missed
    full_probs = np.full((n_samples, NUM_CLASSES), fill_value=np.nan, dtype=np.float32)
    covered    = np.zeros(n_samples, dtype=bool)

    for fold_id in range(FOLDS):
        probs   = np.load(PROBS_DIR / f"fold_{fold_id:02d}_probs.npy")    # (n_val, 7)
        indices = np.load(PROBS_DIR / f"fold_{fold_id:02d}_indices.npy")  # (n_val,)

        print(f"  Fold {fold_id:02d}: probs {probs.shape}, indices {indices.shape}")

        if probs.shape[1] != NUM_CLASSES:
            raise ValueError(
                f"Fold {fold_id}: expected {NUM_CLASSES} classes, got {probs.shape[1]}"
            )
        if np.any(covered[indices]):
            overlap = np.sum(covered[indices])
            raise RuntimeError(
                f"Fold {fold_id}: {overlap} samples appear in multiple fold val sets. "
                "Fold assignments must be non-overlapping."
            )

        full_probs[indices] = probs
        covered[indices]    = True

    # Validate coverage
    uncovered = int(np.sum(~covered))
    if uncovered > 0:
        raise RuntimeError(
            f"{uncovered} samples have no OOF probability. "
            "Check that fold indices cover the full balanced dataset."
        )

    nan_count = int(np.isnan(full_probs).sum())
    if nan_count > 0:
        raise RuntimeError(f"Merged array still has {nan_count} NaN entries.")

    # Verify each row sums to ~1 (softmax outputs)
    row_sums = full_probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        bad = int(np.sum(~np.isclose(row_sums, 1.0, atol=1e-4)))
        raise RuntimeError(
            f"{bad} rows do not sum to 1.0 (max deviation: {np.abs(row_sums - 1.0).max():.6f})."
        )

    print(f"\nMerged array shape: {full_probs.shape}")
    print(f"All {n_samples} samples covered. Row sums ✓")

    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_FILE, full_probs)
    print(f"Saved → {OUTPUT_FILE}")

    return full_probs


if __name__ == "__main__":
    merge_fold_probs()
