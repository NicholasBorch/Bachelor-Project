# src/classification/noise_idn.py
# IDN label corruption following Algorithm 2 from Xia et al. (2020).
#
# Supports two modes via the `normalize` flag:
#
#   normalize=False  →  standard IDN
#       Pixel values in [0, 1] after ToTensor only.
#       Original Xia et al. formulation.
#
#   normalize=True   →  normalized IDN
#       Pixel values standardised to ~N(0, 1) per channel using ImageNet stats.
#       Introduces genuine cancellation in the dot product x @ W_y, reducing
#       the concentration bias caused by class imbalance in the projection space.
#
# Everything else — W construction, flip rate sampling, masking, softmax,
# transition row construction, multinomial sampling — is identical in both modes.

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.common.io import class_mapping
from src.classification.dataset import HamTensorDataset
from src.classification.folds import make_outer_folds_lesion_stratified


# ── Shared dataclasses ────────────────────────────────────────────────────────

@dataclass
class NoiseReport:
    """Summary statistics for one noise injection run."""
    outer_fold:         int
    seed:               int
    tau:                float
    norm_std:           float
    normalize:          bool          # whether ImageNet normalization was applied
    num_classes:        int
    feature_size:       int
    n_train:            int
    n_flipped:          int
    class_counts_clean: Dict[str, int]
    class_counts_noisy: Dict[str, int]
    flip_confusion:     Dict[str, Dict[str, int]]
    flip_rate_min:      float
    flip_rate_median:   float
    flip_rate_max:      float


@dataclass
class FoldOutputs:
    """All artifacts produced for one outer CV fold at one tau level."""
    train_clean: pd.DataFrame
    train_noisy: pd.DataFrame
    test_clean:  pd.DataFrame
    report:      NoiseReport


# ── Core noise generation ─────────────────────────────────────────────────────

@torch.no_grad()
def generate_instance_dependent_noisy_labels(
    df: pd.DataFrame,
    images_dir: Path,
    tau: float,
    seed: int,
    *,
    normalize: bool = False,
    image_size: int = 224,
    norm_std: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[pd.DataFrame, NoiseReport]:
    """
    Applies IDN corruption to df following Xia et al. (2020) Algorithm 2.

    Parameters
    ----------
    df          : DataFrame with columns [image_id, lesion_id, dx]
    images_dir  : Directory containing the image files
    tau         : Target flip rate (centre of truncated normal)
    seed        : RNG seed for reproducibility
    normalize   : If True, applies ImageNet channel normalisation before
                  projection — reduces concentration bias from class imbalance
    """
    df = df.copy().reset_index(drop=True)
    df["image_id"]  = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"]        = df["dx"].astype(str)

    c2i, i2c    = class_mapping(df["dx"].tolist())
    num_classes = len(c2i)
    feature_size = 3 * image_size * image_size
    n = len(df)

    # ── Short-circuit for clean baseline — no flipping occurs ─────────────
    if tau == 0.0:
        df_out = df.copy()
        df_out["dx_clean"] = df_out["dx"]
        df_out["dx_noisy"] = df_out["dx"]
        return df_out, NoiseReport(
            outer_fold=-1, seed=int(seed), tau=0.0, norm_std=float(norm_std),
            normalize=normalize, num_classes=int(num_classes),
            feature_size=int(feature_size), n_train=int(n), n_flipped=0,
            class_counts_clean=df_out["dx_clean"].value_counts().to_dict(),
            class_counts_noisy=df_out["dx_noisy"].value_counts().to_dict(),
            flip_confusion={}, flip_rate_min=0.0,
            flip_rate_median=0.0, flip_rate_max=0.0,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Image transform — only difference between the two modes ──────────
    tfm_steps = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # → [0, 1]
    ]
    if normalize:
        # Centres pixel values around zero, ~half negative.
        # Introduces genuine cancellation in x @ W_y so that score ordering
        # becomes sensitive to image content rather than sign structure of W.
        tfm_steps.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ))
    tfm = transforms.Compose(tfm_steps)

    ds = HamTensorDataset(df=df, images_dir=images_dir, c2i=c2i, tfm=tfm)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=(pin_memory and device.startswith("cuda")),
    )

    # ── Fix seeds for reproducibility ─────────────────────────────────────
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

    # ── Sample class-conditional random weight matrices W ~ N(0, I) ───────
    # Shape: (C, d, C) where C = num_classes, d = feature_size
    W = torch.FloatTensor(
        np.random.randn(num_classes, feature_size, num_classes).astype(np.float32)
    ).to(device)

    new_label_idx = np.empty(n, dtype=np.int64)
    flip_confusion: Dict[str, Dict[str, int]] = {}
    cursor = 0

    mode_label = "normalised" if normalize else "standard"
    for x, y, _ in tqdm(dl, desc=f"IDN corruption ({mode_label})", leave=True):
        b = x.size(0)
        x = x.to(device)
        y = y.to(device).long()

        # Project pixel vectors through class-conditional W (Xia et al. Eq. 1)
        # x_flat: (B, d)  W_y: (B, d, C)  A: (B, C)
        x_flat = x.view(b, -1)
        W_y = W[y]
        A = torch.bmm(x_flat.unsqueeze(1), W_y).squeeze(1)

        # Mask true class so it cannot be sampled as its own flip target
        A[torch.arange(b, device=device), y] = -inf

        # Scale softmax probs by per-instance flip rate; restore true class mass
        q = torch.from_numpy(flip_rate[cursor:cursor + b]).to(device).view(-1, 1)
        P = q * F.softmax(A, dim=1)
        P[torch.arange(b, device=device), y] += (1.0 - q.squeeze(1))

        # Sample noisy labels from the per-instance transition distribution
        sampled     = torch.multinomial(P, num_samples=1).squeeze(1)
        sampled_cpu = sampled.cpu().numpy().astype(np.int64)
        y_cpu       = y.cpu().numpy().astype(np.int64)
        new_label_idx[cursor:cursor + b] = sampled_cpu

        # Track flips for the noise report confusion matrix
        for yi, ytilde in zip(y_cpu, sampled_cpu):
            if ytilde == yi:
                continue
            true_str  = i2c[int(yi)]
            noisy_str = i2c[int(ytilde)]
            flip_confusion.setdefault(true_str, {})
            flip_confusion[true_str][noisy_str] = (
                flip_confusion[true_str].get(noisy_str, 0) + 1
            )
        cursor += b

    df_out = df.copy()
    df_out["dx_clean"] = df_out["dx"]
    df_out["dx_noisy"] = [i2c[int(i)] for i in new_label_idx]

    return df_out, NoiseReport(
        outer_fold=-1,
        seed=int(seed),
        tau=float(tau),
        norm_std=float(norm_std),
        normalize=normalize,
        num_classes=int(num_classes),
        feature_size=int(feature_size),
        n_train=int(n),
        n_flipped=int((df_out["dx_clean"] != df_out["dx_noisy"]).sum()),
        class_counts_clean=df_out["dx_clean"].value_counts().to_dict(),
        class_counts_noisy=df_out["dx_noisy"].value_counts().to_dict(),
        flip_confusion=flip_confusion,
        flip_rate_min=float(np.min(flip_rate)),
        flip_rate_median=float(np.median(flip_rate)),
        flip_rate_max=float(np.max(flip_rate)),
    )