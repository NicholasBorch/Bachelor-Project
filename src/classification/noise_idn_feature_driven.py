# src/classification/noise_idn_feature_driven.py
# Feature-driven IDN label corruption extending Xia et al. (2020).
#
# Replaces the random projection W with out-of-fold (OOF) softmax probabilities
# from a class-balanced ResNet-18, grounding flip targets in learned visual
# similarity rather than random geometry.
#
# This file handles noise injection only.
# OOF probability collection is handled separately in:
#   src/utils/collect_oof_probs.py   (one job per fold, parallelisable)
#   src/utils/merge_oof_probs.py     (assembles per-fold files into one array)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import torch
import torch.nn.functional as F

from src.common.io import class_mapping
# Import shared dataclasses from noise_idn to avoid duplication
from src.classification.noise_idn import NoiseReport, FoldOutputs


def generate_feature_driven_noisy_labels(
    df: pd.DataFrame,
    tau: float,
    seed: int,
    oof_probs: np.ndarray,
    *,
    norm_std: float = 0.1,
) -> Tuple[pd.DataFrame, NoiseReport]:
    """
    Applies feature-driven IDN corruption using pre-computed OOF softmax probs.

    The OOF probs replace the random W projection in standard IDN: each sample's
    flip target distribution is shaped by what a ResNet-18 (trained without that
    sample) found visually confusable — producing clinically plausible confusions.

    Parameters
    ----------
    df        : DataFrame with columns [image_id, lesion_id, dx].
                Must be the same subset and ordering as when oof_probs was built.
    tau       : Target flip rate (centre of truncated normal)
    seed      : RNG seed for reproducibility
    oof_probs : (N, C) array of softmax probabilities; row i corresponds to df row i
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
            normalize=False,  # not applicable for feature-driven
            num_classes=int(num_classes), feature_size=0,
            n_train=int(n), n_flipped=0,
            class_counts_clean=df_out["dx_clean"].value_counts().to_dict(),
            class_counts_noisy=df_out["dx_noisy"].value_counts().to_dict(),
            flip_confusion={}, flip_rate_min=0.0,
            flip_rate_median=0.0, flip_rate_max=0.0,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Fix seeds for reproducible flip rate sampling ─────────────────────
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    # ── Sample per-instance flip rates from truncated normal centred on tau
    flip_dist = stats.truncnorm(
        (0.0 - tau) / norm_std,
        (1.0 - tau) / norm_std,
        loc=tau, scale=norm_std,
    )
    flip_rate = flip_dist.rvs(n).astype(np.float32)

    # ── Build per-instance transition matrix using OOF probs ──────────────
    probs  = torch.from_numpy(oof_probs).float().to(device)
    labels = torch.tensor(
        [c2i[dx] for dx in df["dx"]], dtype=torch.long, device=device
    )

    # Zero out the true class so flips cannot land on the correct label,
    # then renormalise remaining mass over the wrong classes
    probs[torch.arange(n, device=device), labels] = 0.0
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

    # Row i of P: prob (1-q_i) of keeping own label + q_i * OOF confusion probs
    q = torch.from_numpy(flip_rate).float().to(device).view(-1, 1)
    P = q * probs
    P[torch.arange(n, device=device), labels] += (1.0 - q.squeeze(1))

    # ── Sample noisy labels ────────────────────────────────────────────────
    new_label_idx = torch.multinomial(P, num_samples=1).squeeze(1).cpu().numpy().astype(np.int64)
    labels_cpu    = labels.cpu().numpy().astype(np.int64)

    # Track flips for the noise report confusion matrix
    flip_confusion: Dict[str, Dict[str, int]] = {}
    for yi, ytilde in zip(labels_cpu, new_label_idx):
        if ytilde == yi:
            continue
        true_str  = i2c[int(yi)]
        noisy_str = i2c[int(ytilde)]
        flip_confusion.setdefault(true_str, {})
        flip_confusion[true_str][noisy_str] = flip_confusion[true_str].get(noisy_str, 0) + 1

    df_out = df.copy()
    df_out["dx_clean"] = df_out["dx"]
    df_out["dx_noisy"] = [i2c[int(i)] for i in new_label_idx]

    return df_out, NoiseReport(
        seed=int(seed),
        tau=float(tau),
        norm_std=float(norm_std),
        normalize=False,  # not applicable for feature-driven
        num_classes=int(num_classes),
        feature_size=0,   # not applicable for feature-driven
        n_train=int(n),
        n_flipped=int((df_out["dx_clean"] != df_out["dx_noisy"]).sum()),
        class_counts_clean=df_out["dx_clean"].value_counts().to_dict(),
        class_counts_noisy=df_out["dx_noisy"].value_counts().to_dict(),
        flip_confusion=flip_confusion,
        flip_rate_min=float(np.min(flip_rate)),
        flip_rate_median=float(np.median(flip_rate)),
        flip_rate_max=float(np.max(flip_rate)),
    )