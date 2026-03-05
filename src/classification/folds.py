# Fold construction utilities for HAM10000 cross-validation.

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_outer_folds_lesion_stratified(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    # Stratified outer folds split on unique lesion_id to prevent leakage across folds
    df = df.copy().sort_values(["lesion_id", "image_id"]).reset_index(drop=True)
    lesion_df = df.drop_duplicates(subset=["lesion_id"]).sort_values("lesion_id").reset_index(drop=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    lesion_fold = np.full(len(lesion_df), -1, dtype=int)

    for fold_id, (_, test_idx) in enumerate(skf.split(np.arange(len(lesion_df)), lesion_df["dx"].values)):
        lesion_fold[test_idx] = fold_id

    lesion_df["outer_fold"] = lesion_fold
    lesion_to_fold = dict(zip(lesion_df["lesion_id"].astype(str), lesion_df["outer_fold"].astype(int)))
    df["outer_fold"] = df["lesion_id"].astype(str).map(lesion_to_fold)

    return df