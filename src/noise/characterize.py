"""
Noise characterization metrics: row-normalized confusion matrices, concentration,
TVD, and class distributions. Inputs are class indices or class names.
"""
from __future__ import annotations

import numpy as np

from src.data.ham10000 import CLASS_NAMES, NUM_CLASSES, class_to_index


def _to_indices(labels) -> np.ndarray:
    """Accept either class-name strings or integer indices; return indices."""
    arr = np.asarray(labels)
    if arr.dtype.kind in ("U", "O"):  # string dtype
        return np.array([class_to_index(str(l)) for l in arr], dtype=np.int64)
    return arr.astype(np.int64)


def confusion_matrix_from_labels(
    clean: np.ndarray | list,
    noisy: np.ndarray | list,
    normalize: str | None = "row",
) -> np.ndarray:
    """Confusion matrix M[i, j] = P(noisy=j | clean=i); row-normalized by default."""
    c = _to_indices(clean)
    n = _to_indices(noisy)
    if len(c) != len(n):
        raise ValueError(f"clean and noisy must have same length, got {len(c)} vs {len(n)}")

    M = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for ci, ni in zip(c, n):
        M[ci, ni] += 1

    if normalize == "row":
        row_sums = M.sum(axis=1, keepdims=True)
        # avoid division by zero for classes that are absent
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        M = M / row_sums
    return M


def concentration(confusion_row_normalized: np.ndarray) -> float:
    """Mean over rows of the max off-diagonal share (how concentrated noise is on one class)."""
    C = confusion_row_normalized.shape[0]
    vals = []
    for i in range(C):
        row = confusion_row_normalized[i].copy()
        row[i] = 0.0
        s = row.sum()
        if s == 0.0:
            vals.append(0.0)
        else:
            vals.append(float(row.max() / s))
    return float(np.mean(vals))


def class_distribution(labels) -> np.ndarray:
    """Return (C,) array of class frequencies summing to 1."""
    idx = _to_indices(labels)
    counts = np.bincount(idx, minlength=NUM_CLASSES).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """TVD between two probability vectors: 0.5 * ||p - q||_1."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return 0.5 * float(np.abs(p - q).sum())


def off_diagonal_mae(A: np.ndarray, B: np.ndarray) -> float:
    """Off-diagonal mean absolute error between two (C, C) matrices (human-comparison metric)."""
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: {A.shape} vs {B.shape}")
    C = A.shape[0]
    mask = ~np.eye(C, dtype=bool)
    return float(np.abs(A[mask] - B[mask]).mean())