"""Verify fold assignment properties:
    - every sample in exactly one fold
    - no image_id appears twice
    - stratified: each fold's class distribution is close to global distribution
    - reproducible: same seed → same assignments
"""
from __future__ import annotations

import pandas as pd

from src.data.folds import create_fold_assignments, split_train_test_by_fold
from src.data.ham10000 import CLASS_NAMES


def _dummy_metadata(n_per_class: int = 20) -> pd.DataFrame:
    rows = []
    for cls in CLASS_NAMES:
        for k in range(n_per_class):
            rows.append({"image_id": f"{cls}_{k:03d}", "dx": cls})
    return pd.DataFrame(rows)


def test_fold_coverage() -> None:
    md = _dummy_metadata()
    folds = create_fold_assignments(md, n_splits=10, seed=10)
    # exactly one fold per sample
    assert len(folds) == len(md)
    assert set(folds["fold"].unique()) == set(range(10))
    assert folds["image_id"].nunique() == len(folds)
    print("[test] fold coverage PASS")


def test_fold_stratification() -> None:
    md = _dummy_metadata(n_per_class=50)
    folds = create_fold_assignments(md, n_splits=10, seed=10)
    for cls in CLASS_NAMES:
        per_fold = folds[folds["dx"] == cls].groupby("fold").size()
        # Each fold has roughly 50/10 = 5 samples of each class; allow ±2.
        assert per_fold.min() >= 3 and per_fold.max() <= 7, (
            f"class {cls} unevenly distributed: {per_fold.to_dict()}"
        )
    print("[test] fold stratification PASS")


def test_reproducibility() -> None:
    md = _dummy_metadata()
    f1 = create_fold_assignments(md, n_splits=10, seed=10)
    f2 = create_fold_assignments(md, n_splits=10, seed=10)
    assert (f1["fold"].values == f2["fold"].values).all()
    f3 = create_fold_assignments(md, n_splits=10, seed=11)
    # Different seed should produce at least some different assignments.
    assert not (f1["fold"].values == f3["fold"].values).all()
    print("[test] fold reproducibility PASS")


def test_split_partition() -> None:
    md = _dummy_metadata()
    folds = create_fold_assignments(md, n_splits=10, seed=10)
    for test_fold in range(10):
        train_df, test_df = split_train_test_by_fold(md, folds, test_fold=test_fold)
        # No overlap
        assert set(train_df["image_id"]).isdisjoint(set(test_df["image_id"]))
        # Sum to the full dataset
        assert len(train_df) + len(test_df) == len(md)
    print("[test] train/test partition PASS")


if __name__ == "__main__":
    test_fold_coverage()
    test_fold_stratification()
    test_reproducibility()
    test_split_partition()
    print("[test] ALL FOLD TESTS PASSED")
