# src/classification/noise_idn_feature_driven_v2.py
# Feature-driven IDN v2 — argmax variant.
#
# Identical to noise_idn_feature_driven.py except for the flip-target rule:
#
#   v1 (noise_idn_feature_driven.py):
#       For each sample, flip with per-instance probability q = flip_rate[i],
#       and when a flip occurs, draw the new label by multinomial sampling
#       from the masked-and-renormalised OOF softmax distribution.
#
#   v2 (this file):
#       Flip decision is still stochastic (Bernoulli with prob q = flip_rate[i]),
#       but when a flip occurs, the new label is *deterministic*: the argmax
#       over the masked-and-renormalised OOF softmax distribution (i.e. the
#       wrong class the reference model considers most visually similar).
#
# Everything else — truncated-normal flip rate sampling, masking of the true
# class, renormalisation, reproducibility — is identical to v1.
#
# OOF probability collection is still handled by the existing scripts:
#   src/utils/collect_fold_probs.py   (one job per fold)
#   src/utils/merge_fold_probs.py     (assembles per-fold files into one array)
# v2 reuses fold_probs_full.npy produced by those scripts — no recollection.

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import torch

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
    Applies feature-driven IDN v2 corruption using pre-computed OOF softmax probs.

    Flip decision: Bernoulli(flip_rate[i]) per instance, with flip_rate drawn
    from a truncated normal centred on tau (same as v1).
    Flip target: argmax over masked-and-renormalised OOF probs (the most
    visually-confusable wrong class, per the reference model).

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
            normalize=False, num_classes=int(num_classes), feature_size=0,
            n_train=int(n), n_flipped=0,
            class_counts_clean=df_out["dx_clean"].value_counts().to_dict(),
            class_counts_noisy=df_out["dx_noisy"].value_counts().to_dict(),
            flip_confusion={}, flip_rate_min=0.0,
            flip_rate_median=0.0, flip_rate_max=0.0,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # ── Mask true class from OOF probs, renormalise over wrong classes ────
    probs  = torch.from_numpy(oof_probs).float().to(device)
    labels = torch.tensor(
        [c2i[dx] for dx in df["dx"]], dtype=torch.long, device=device
    )

    probs[torch.arange(n, device=device), labels] = 0.0
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

    # ── v2: deterministic flip target = argmax over masked probs ──────────
    argmax_targets = probs.argmax(dim=1).cpu().numpy().astype(np.int64)
    labels_cpu     = labels.cpu().numpy().astype(np.int64)

    # ── Bernoulli flip decision per instance ──────────────────────────────
    u = np.random.rand(n).astype(np.float32)
    flip_mask = u < flip_rate  # True → flip to argmax_targets[i]

    new_label_idx = np.where(flip_mask, argmax_targets, labels_cpu).astype(np.int64)

    # ── Build flip confusion matrix ───────────────────────────────────────
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
        normalize=False,   # not applicable for feature-driven
        num_classes=int(num_classes),
        feature_size=0,    # not applicable for feature-driven
        n_train=int(n),
        n_flipped=int((df_out["dx_clean"] != df_out["dx_noisy"]).sum()),
        class_counts_clean=df_out["dx_clean"].value_counts().to_dict(),
        class_counts_noisy=df_out["dx_noisy"].value_counts().to_dict(),
        flip_confusion=flip_confusion,
        flip_rate_min=float(np.min(flip_rate)),
        flip_rate_median=float(np.median(flip_rate)),
        flip_rate_max=float(np.max(flip_rate)),
    )
