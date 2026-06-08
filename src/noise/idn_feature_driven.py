"""
Feature-driven instance-dependent label noise.

Uses out-of-fold softmax probabilities from a clean-label ResNet-18 (Stage 1b) as
the flip-target distribution, replacing Xia et al.'s random Gaussian projection.
Per sample: zero the true-class entry of the OOF row, renormalize, scale by the
per-instance flip rate q_i, set the true class to 1 - q_i, and sample the noisy
label. Short-circuits at tau=0.0 (labels unchanged).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from src.data.ham10000 import CLASS_NAMES, NUM_CLASSES, class_to_index, index_to_class


@dataclass
class NoiseReport:
    tau_requested: float
    empirical_rate: float
    n_total: int
    n_flipped: int
    per_class_flip_rate: dict[str, float]


def _sample_truncnorm_flip_rates(
    n: int, tau: float, sigma: float, rng: np.random.Generator,
) -> np.ndarray:
    a = (0.0 - tau) / sigma
    b = (1.0 - tau) / sigma
    rvs = truncnorm.rvs(a, b, loc=tau, scale=sigma, size=n, random_state=rng)
    return rvs.astype(np.float64)


def generate_feature_driven_idn(
    metadata: pd.DataFrame,
    oof_probs: np.ndarray,
    tau: float,
    seed: int,
    sigma: float = 0.1,
) -> tuple[pd.DataFrame, NoiseReport]:
    """Flip labels using feature-driven IDN; returns (noisy_df, NoiseReport). See idn_xia for the df schema."""
    if oof_probs.shape[0] != len(metadata):
        raise ValueError(
            f"oof_probs rows ({oof_probs.shape[0]}) must match metadata rows ({len(metadata)})"
        )
    if oof_probs.shape[1] != NUM_CLASSES:
        raise ValueError(f"oof_probs must have {NUM_CLASSES} columns, got {oof_probs.shape[1]}")

    # tau=0 short-circuit: no noise, return clean labels unchanged.
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

    q = _sample_truncnorm_flip_rates(n, tau, sigma, rng)

    clean_labels = np.array(
        [class_to_index(c) for c in metadata["dx"].tolist()],
        dtype=np.int64,
    )
    noisy_labels = np.empty_like(clean_labels)

    for i in range(n):
        y = int(clean_labels[i])
        p = oof_probs[i].astype(np.float64).copy()
        p[y] = 0.0

        s = p.sum()
        if s <= 0.0:
            # OOF model gave all mass to the true class; fall back to uniform
            # over off-diagonals. This is a rare edge case for high-confidence
            # easy samples.
            p = np.ones(NUM_CLASSES) / (NUM_CLASSES - 1)
            p[y] = 0.0
        else:
            p = p / s

        p = q[i] * p
        p[y] = 1.0 - q[i]

        p = np.clip(p, 0.0, None)
        p = p / p.sum()

        noisy_labels[i] = rng.choice(NUM_CLASSES, p=p)

    noisy_names = np.array([index_to_class(int(l)) for l in noisy_labels])
    clean_names = metadata["dx"].values.copy()

    out = metadata.copy()
    out["dx_clean"] = clean_names
    out["dx"] = noisy_names
    out["flipped"] = out["dx"] != out["dx_clean"]

    per_class = {}
    for c in CLASS_NAMES:
        mask = clean_names == c
        per_class[c] = float(out.loc[mask, "flipped"].mean()) if mask.sum() else 0.0

    report = NoiseReport(
        tau_requested=tau,
        empirical_rate=float(out["flipped"].mean()),
        n_total=n,
        n_flipped=int(out["flipped"].sum()),
        per_class_flip_rate=per_class,
    )
    return out, report