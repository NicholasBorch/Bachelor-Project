# src/utils/merge_oof_probs.py
#
# Merges per-fold OOF probability files produced by collect_oof_probs.py into
# a single aligned array over the full dataset.
#
# Run this AFTER all 5 collect_oof_probs jobs have completed:
#   python -m src.utils.merge_oof_probs
#
# Input:  data/processed/HAM10000/oof_probs/fold_0X_probs.npy   (N_fold, C)
#         data/processed/HAM10000/oof_probs/fold_0X_indices.npy  (N_fold,)
# Output: data/processed/HAM10000/oof_probs/oof_probs_full.npy   (N_total, C)
#
# After this step, prepare_classification_cv_feature_driven.py can load
# oof_probs_full.npy and slice out the training-fold rows for noise injection.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import project_root, class_mapping
from configs.classification_default import OUTER_FOLDS, SEED


def main() -> None:
    root      = project_root()
    ham_one   = root / "data" / "processed" / "HAM10000" / "one_image_per_lesion"
    meta_path = ham_one / "HAM10000_metadata_one_per_lesion.csv"
    oof_dir   = root / "data" / "processed" / "HAM10000" / "oof_probs"

    df = pd.read_csv(meta_path)
    df["dx"] = df["dx"].astype(str)
    c2i, _   = class_mapping(df["dx"].tolist())
    num_classes = len(c2i)
    N           = len(df)

    print(f"Merging OOF probs | N={N} | C={num_classes} | OUTER_FOLDS={OUTER_FOLDS}")

    # Initialise full array with NaNs so we can detect missing folds
    oof_probs_full = np.full((N, num_classes), np.nan, dtype=np.float32)

    for fold_id in range(OUTER_FOLDS):
        probs_path   = oof_dir / f"fold_{fold_id:02d}_probs.npy"
        indices_path = oof_dir / f"fold_{fold_id:02d}_indices.npy"

        if not probs_path.exists() or not indices_path.exists():
            raise FileNotFoundError(
                f"Missing OOF files for fold {fold_id}. "
                f"Run collect_oof_probs.py --fold {fold_id} first."
            )

        probs   = np.load(probs_path)    # (N_fold, C)
        indices = np.load(indices_path)  # (N_fold,)

        # Place each fold's probs at the correct positions in the full array
        oof_probs_full[indices] = probs
        print(f"  Fold {fold_id}: placed {len(indices)} rows | "
              f"probs shape={probs.shape}")

    # Verify no gaps remain — every sample must have received probs
    n_missing = np.isnan(oof_probs_full).any(axis=1).sum()
    if n_missing > 0:
        raise RuntimeError(
            f"{n_missing} samples have no OOF probs. "
            "Check that all fold jobs completed successfully."
        )

    out_path = oof_dir / "oof_probs_full.npy"
    np.save(out_path, oof_probs_full)
    print(f"\nSaved full OOF probs: {out_path}  shape={oof_probs_full.shape}")
    print("Merge complete — ready for prepare_classification_cv_feature_driven.py")


if __name__ == "__main__":
    main()