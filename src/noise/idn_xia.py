"""
Instance-dependent label noise (Xia et al. 2020, Algorithm 2).

Two variants: standard (normalize=False; ToTensor in [0,1]) and normalized
(normalize=True; +ImageNet normalization, giving genuine dot-product cancellation
and less concentration bias).

CRITICAL: both short-circuit at tau=0.0 and return labels bitwise identical to the
input (verified by tests/test_noise_tau_zero.py; must not regress). Output keeps the
original dx as dx_clean, writes noisy labels to dx, and adds a boolean flipped column.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import truncnorm

from src.data.ham10000 import CLASS_NAMES, NUM_CLASSES, class_to_index, index_to_class
from src.data.transforms import get_noise_injection_transforms


@dataclass
class NoiseReport:
    """Summary statistics about a noise injection pass."""
    tau_requested: float
    empirical_rate: float
    n_total: int
    n_flipped: int
    per_class_flip_rate: dict[str, float]


def _sample_truncnorm_flip_rates(
    n: int,
    tau: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n per-instance flip rates from TruncNormal(tau, sigma^2) on [0,1]."""
    a = (0.0 - tau) / sigma
    b = (1.0 - tau) / sigma
    rvs = truncnorm.rvs(
        a, b, loc=tau, scale=sigma, size=n,
        random_state=rng,
    )
    return rvs.astype(np.float64)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _load_flat_images(
    image_ids: list[str],
    images_dir: Path,
    normalize: bool,
    image_size: int = 224,
) -> np.ndarray:
    """Load images, apply the chosen transforms, and flatten to (N, 3*image_size^2)."""
    transform = get_noise_injection_transforms(image_size=image_size, normalize=normalize)
    rows = []
    for iid in image_ids:
        img = Image.open(images_dir / f"{iid}.jpg").convert("RGB")
        tensor = transform(img)  # (3, H, W)
        rows.append(tensor.view(-1).numpy())
    return np.stack(rows, axis=0).astype(np.float64)


def generate_xia_idn(
    metadata: pd.DataFrame,
    images_dir: str | Path,
    tau: float,
    seed: int,
    normalize: bool = False,
    sigma: float = 0.1,
    image_size: int = 224,
) -> tuple[pd.DataFrame, NoiseReport]:
    """Apply Xia et al. 2020 Alg. 2 IDN; returns (noisy_df, NoiseReport). Short-circuits at tau=0."""
    # Short-circuit at tau=0: return bitwise identical data.
    if tau == 0.0:
        out = metadata.reset_index(drop=True).copy()
        out["dx_clean"] = out["dx"].values
        out["flipped"] = False
        report = NoiseReport(
            tau_requested=0.0,
            empirical_rate=0.0,
            n_total=len(out),
            n_flipped=0,
            per_class_flip_rate={c: 0.0 for c in CLASS_NAMES},
        )
        return out, report

    if tau < 0.0 or tau >= 1.0:
        raise ValueError(f"tau must be in [0, 1), got {tau}")

    metadata = metadata.reset_index(drop=True).copy()
    n = len(metadata)
    rng = np.random.default_rng(seed)

    # Step 1: per-instance flip rates.
    q = _sample_truncnorm_flip_rates(n, tau, sigma, rng)

    # Step 2: per-class projection matrices w_y of shape (d, C).
    # We generate them on demand to avoid a (C, d, C) tensor in memory.
    d = 3 * image_size * image_size
    W = rng.standard_normal(size=(NUM_CLASSES, d, NUM_CLASSES))

    # Load images as flat vectors.
    images_dir = Path(images_dir)
    X = _load_flat_images(metadata["image_id"].tolist(), images_dir,
                          normalize=normalize, image_size=image_size)  # (n, d)

    # Step 3: per-sample, compute transition row.
    clean_labels = np.array(
        [class_to_index(c) for c in metadata["dx"].tolist()],
        dtype=np.int64,
    )
    noisy_labels = np.empty_like(clean_labels)

    for i in range(n):
        y = int(clean_labels[i])
        p = X[i] @ W[y]            # (C,)
        p[y] = -np.inf              # mask true class
        p = _softmax(p)             # sum over off-diagonals = 1
        p = q[i] * p                # scale to make off-diagonal sum = q_i
        p[y] = 1.0 - q[i]           # diagonal = 1 - q_i

        # Numerical safety: renormalize in case of tiny drift.
        p = np.clip(p, 0.0, None)
        p = p / p.sum()

        noisy_labels[i] = rng.choice(NUM_CLASSES, p=p)

    noisy_names = np.array([index_to_class(int(l)) for l in noisy_labels])
    clean_names = metadata["dx"].values.copy()

    out = metadata.copy()
    out["dx_clean"] = clean_names
    out["dx"] = noisy_names
    out["flipped"] = out["dx"] != out["dx_clean"]

    # Build report.
    per_class = {}
    for c in CLASS_NAMES:
        mask = clean_names == c
        if mask.sum() == 0:
            per_class[c] = 0.0
        else:
            per_class[c] = float(out.loc[mask, "flipped"].mean())
    report = NoiseReport(
        tau_requested=tau,
        empirical_rate=float(out["flipped"].mean()),
        n_total=n,
        n_flipped=int(out["flipped"].sum()),
        per_class_flip_rate=per_class,
    )
    return out, report