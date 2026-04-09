# src/classification/noise_idn_feature_driven_v2.py
#
# Feature-driven IDN (argmax variant) for label corruption.
#
# Drop-in alternative to noise_idn_feature_driven.py with one change:
#
#   v1 (softmax sampling): when a sample flips, the target class is SAMPLED
#       from the renormalized OOF softmax distribution over non-true classes.
#       Different samples from the same class can flip to different targets.
#
#   v2 (argmax):           when a sample flips, the target class is the ARGMAX
#       of the OOF softmax over non-true classes — always the single most
#       probable incorrect class. This produces deterministic, concentrated
#       flips where every flipped sample within a visual-confusion cluster
#       goes to the same target.
#
# Everything else is identical: truncated normal flip rate draw, τ
# parameterization, seed handling, output format, and function signature.

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.common.io import class_mapping
from src.classification.noise_idn import NoiseReport


def generate_feature_driven_noisy_labels_v2(
    df: pd.DataFrame,
    tau: float,
    seed: int,
    oof_probs: np.ndarray,
    *,
    norm_std: float = 0.1,
) -> Tuple[pd.DataFrame, NoiseReport]:
    """
    Applies feature-driven IDN corruption (argmax variant).

    For each sample, the flip decision is stochastic (Bernoulli with rate q_i),
    but the flip TARGET is deterministic: always the single most-probable
    non-true class according to the OOF softmax probabilities.

    Parameters
    ----------
    df        : DataFrame with columns [image_id, lesion_id, dx].
                Must be the same subset and ordering as when oof_probs was built.
    tau       : Target flip rate (centre of truncated normal)
    seed      : RNG seed for reproducibility
    oof_probs : (N, C) array of softmax probabilities; row i corresponds to df row i
    norm_std  : Std of truncated normal for per-instance flip rates
    """
    df = df.copy().reset_index(drop=True)
    df["image_id"]  = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"]        = df["dx"].astype(str)

    c2i, i2c    = class_mapping(df["dx"].tolist())
    num_classes = len(c2i)
    n           = len(df)

    # ── Short-circuit: tau=0.0 produces no flips ──────────────────────────
    if tau == 0.0:
        df_out = df.copy()
        df_out["dx_clean"] = df_out["dx"]
        df_out["dx_noisy"] = df_out["dx"]
        return df_out, NoiseReport(
            seed=int(seed), tau=0.0, norm_std=float(norm_std),
            normalize=False, num_classes=int(num_classes), feature_size=0,
            n_train=int(n), n_flipped=0,
            class_counts_clean=df_out["dx_clean"].value_counts().to_dict(),
            class_counts_noisy=df_out["dx_noisy"].value_counts().to_dict(),
            flip_confusion={}, flip_rate_min=0.0,
            flip_rate_median=0.0, flip_rate_max=0.0,
        )

    np.random.seed(int(seed))

    # ── Sample per-instance flip rates from truncated normal centred on tau
    flip_dist = stats.truncnorm(
        (0.0 - tau) / norm_std,
        (1.0 - tau) / norm_std,
        loc=tau, scale=norm_std,
    )
    flip_rate = flip_dist.rvs(n).astype(np.float32)

    # ── Determine per-sample argmax flip target ───────────────────────────
    # Copy OOF probs, zero out the true class, take argmax over remaining
    probs = oof_probs.copy()  # (N, C)
    labels = np.array([c2i[dx] for dx in df["dx"]], dtype=np.int64)

    # Zero out true class so argmax selects from non-true classes only
    probs[np.arange(n), labels] = 0.0

    # Argmax over non-true classes — deterministic flip target per sample
    argmax_targets = probs.argmax(axis=1)  # (N,)

    # ── Stochastic flip decision (Bernoulli with rate q_i) ────────────────
    # Draw uniform [0, 1) for each sample; flip if draw < q_i
    flip_coin = np.random.uniform(0.0, 1.0, size=n).astype(np.float32)
    flipped = flip_coin < flip_rate  # boolean mask

    # Assemble new labels: keep original if not flipped, use argmax target if flipped
    new_label_idx = labels.copy()
    new_label_idx[flipped] = argmax_targets[flipped]

    # ── Build flip confusion dict ─────────────────────────────────────────
    flip_confusion: Dict[str, Dict[str, int]] = {}
    for yi, ytilde in zip(labels, new_label_idx):
        if ytilde == yi:
            continue
        true_str  = i2c[int(yi)]
        noisy_str = i2c[int(ytilde)]
        flip_confusion.setdefault(true_str, {})
        flip_confusion[true_str][noisy_str] = (
            flip_confusion[true_str].get(noisy_str, 0) + 1
        )

    df_out = df.copy()
    df_out["dx_clean"] = df_out["dx"]
    df_out["dx_noisy"] = [i2c[int(i)] for i in new_label_idx]

    return df_out, NoiseReport(
        seed=int(seed),
        tau=float(tau),
        norm_std=float(norm_std),
        normalize=False,
        num_classes=int(num_classes),
        feature_size=0,
        n_train=int(n),
        n_flipped=int(flipped.sum()),
        class_counts_clean=df_out["dx_clean"].value_counts().to_dict(),
        class_counts_noisy=df_out["dx_noisy"].value_counts().to_dict(),
        flip_confusion=flip_confusion,
        flip_rate_min=float(np.min(flip_rate)),
        flip_rate_median=float(np.median(flip_rate)),
        flip_rate_max=float(np.max(flip_rate)),
    )
