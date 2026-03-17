# src/utils/merge_fold_probs.py
#
# Merges per-fold probability files produced by collect_fold_probs.py into
# a single aligned array over the full dataset.
#
# Run AFTER all collect_fold_probs jobs have completed:
#   python -m src.utils.merge_fold_probs
#
# Input:  data/processed/HAM10000/fold_probs/fold_0X_probs.npy    (N_fold, C)
#         data/processed/HAM10000/fold_probs/fold_0X_indices.npy   (N_fold,)
# Output: data/processed/HAM10000/fold_probs/fold_probs_full.npy   (N_total, C)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import project_root, class_mapping
from configs.classification_default import OUTER_FOLDS


def main() -> None:
    root      = project_root()
    ham_one   = root / "data" / "processed" / "HAM10000" / "one_image_per_lesion"
    meta_path = ham_one / "HAM10000_metadata_one_per_lesion.csv"
    fold_dir  = root / "data" / "processed" / "HAM10000" / "fold_probs"

    df = pd.read_csv(meta_path)
    df["dx"] = df["dx"].astype(str)
    c2i, _      = class_mapping(df["dx"].tolist())
    num_classes = len(c2i)
    N           = len(df)

    print(f"Merging fold probs | N={N} | C={num_classes} | OUTER_FOLDS={OUTER_FOLDS}")

    # Initialise with NaNs — any unfilled row reveals a missing fold job
    fold_probs_full = np.full((N, num_classes), np.nan, dtype=np.float32)

    for fold_id in range(OUTER_FOLDS):
        probs_path   = fold_dir / f"fold_{fold_id:02d}_probs.npy"
        indices_path = fold_dir / f"fold_{fold_id:02d}_indices.npy"

        if not probs_path.exists() or not indices_path.exists():
            raise FileNotFoundError(
                f"Missing files for fold {fold_id}. "
                f"Run collect_fold_probs.py --fold {fold_id} first."
            )

        probs   = np.load(probs_path)    # (N_fold, C)
        indices = np.load(indices_path)  # (N_fold,)

        fold_probs_full[indices] = probs
        print(f"  Fold {fold_id}: placed {len(indices)} rows | shape={probs.shape}")

    n_missing = np.isnan(fold_probs_full).any(axis=1).sum()
    if n_missing > 0:
        raise RuntimeError(
            f"{n_missing} samples have no fold probs. "
            "Check that all fold jobs completed successfully."
        )

    out_path = fold_dir / "fold_probs_full.npy"
    np.save(out_path, fold_probs_full)
    print(f"\nSaved: {out_path}  shape={fold_probs_full.shape}")
    print("Merge complete — ready for prepare_classification_cv_feature_driven.py")


if __name__ == "__main__":
    main()