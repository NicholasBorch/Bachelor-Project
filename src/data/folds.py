"""
Stratified 10-fold CV assignments.
 
Created once in Stage 1a and consumed by every downstream stage, so fold N's test
set is identical across all noise types, taus, methods, and conditions. Stage 0
dedups to one image per lesion, so plain StratifiedKFold is leak-safe.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_fold_assignments(
    metadata: pd.DataFrame,
    n_splits: int = 10,
    seed: int = 10,
    label_col: str = "dx",
) -> pd.DataFrame:
    """Create a DataFrame with columns [image_id, dx, fold] via StratifiedKFold.

    The input `metadata` must contain `image_id` and `label_col` columns. All
    other columns are dropped from the output to keep fold assignments lean.
    """
    if label_col not in metadata.columns:
        raise ValueError(f"label_col '{label_col}' not found in metadata columns: {list(metadata.columns)}")
    if "image_id" not in metadata.columns:
        raise ValueError("metadata must have an 'image_id' column")

    df = metadata[["image_id", label_col]].reset_index(drop=True).copy()
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (_, test_idx) in enumerate(skf.split(df["image_id"], df[label_col])):
        df.loc[test_idx, "fold"] = fold_idx

    assert (df["fold"] >= 0).all(), "some samples were not assigned a fold"
    return df


def load_fold_assignments(path: str | Path) -> pd.DataFrame:
    """Load fold_assignments.csv. Returns DataFrame with image_id, dx, fold."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fold assignments not found at {path}. "
            f"Run stage1a_create_folds.py first."
        )
    df = pd.read_csv(path)
    expected = {"image_id", "dx", "fold"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{path} missing columns. Found: {list(df.columns)}")
    return df


def split_train_test_by_fold(
    metadata: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    test_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Given full metadata and fold assignments, return (train_df, test_df).

    train_df: rows with fold != test_fold
    test_df:  rows with fold == test_fold

    The returned DataFrames preserve all columns from the input metadata and
    merge in the fold column.
    """
    if "fold" not in metadata.columns:
        merged = metadata.merge(fold_assignments[["image_id", "fold"]], on="image_id", how="inner")
    else:
        merged = metadata.copy()

    train_df = merged[merged["fold"] != test_fold].reset_index(drop=True)
    test_df = merged[merged["fold"] == test_fold].reset_index(drop=True)
    return train_df, test_df
