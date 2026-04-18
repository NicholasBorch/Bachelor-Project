"""Weighted samplers and class-weight helpers."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from src.data.ham10000 import NUM_CLASSES


def make_weighted_sampler(labels_idx: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency weighted sampler for imbalanced datasets.

    Args:
        labels_idx: array of integer class indices.

    Returns:
        WeightedRandomSampler(replacement=True, num_samples=len(labels)).
    """
    counts = np.bincount(labels_idx, minlength=NUM_CLASSES).astype(np.float64)
    # Inverse frequency (with floor to avoid div-by-zero on missing classes)
    weights_per_class = 1.0 / np.maximum(counts, 1.0)
    sample_weights = weights_per_class[labels_idx]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels_idx),
        replacement=True,
    )


def compute_class_weights(labels_idx: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights normalized to mean 1.

    For use as the `weight` argument of nn.CrossEntropyLoss on imbalanced data.
    """
    counts = np.bincount(labels_idx, minlength=NUM_CLASSES).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w / w.mean()  # so the *expected* contribution matches unweighted CE
    return torch.as_tensor(w, dtype=torch.float32, device=device)
