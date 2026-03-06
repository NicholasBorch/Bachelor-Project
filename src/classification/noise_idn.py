# IDN label corruption following Algorithm 2 from Xia et al. (2020).
# Adapted for HAM10000 with outer cross-validation fold support.

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Dict, Tuple

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


@dataclass
class NoiseReport:
    # Summary statistics for one noise injection run
    outer_fold: int
    seed: int
    tau: float
    norm_std: float
    num_classes: int
    feature_size: int
    n_train: int
    n_flipped: int
    class_counts_clean: Dict[str, int]
    class_counts_noisy: Dict[str, int]
    flip_confusion: Dict[str, Dict[str, int]]
    flip_rate_min: float
    flip_rate_median: float
    flip_rate_max: float


@dataclass
class FoldOutputs:
    train_clean: pd.DataFrame
    train_noisy: pd.DataFrame
    test_clean: pd.DataFrame
    report: NoiseReport


@dataclass
class IDNOutputs:
    fold_assignments: pd.DataFrame
    folds: Dict[int, FoldOutputs]


@torch.no_grad()
def generate_instance_dependent_noisy_labels(
    df: pd.DataFrame,
    images_dir: Path,
    tau: float,
    seed: int,
    *,
    image_size: int = 224,
    norm_std: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[pd.DataFrame, NoiseReport]:
    
    # Applies IDN corruption to a dataframe split, returns noisy labels and a report
    df = df.copy().reset_index(drop=True)
    df["image_id"] = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"] = df["dx"].astype(str)

    c2i, i2c = class_mapping(df["dx"].tolist())
    num_classes = len(c2i)
    feature_size = 3 * image_size * image_size
    n = len(df)
    
    # Short-circuit for clean baseline — no flipping should occur
    if tau == 0.0:
        df_out = df.copy()
        df_out["dx_clean"] = df_out["dx"]
        df_out["dx_noisy"] = df_out["dx"]
        report = NoiseReport(
            outer_fold=-1,
            seed=int(seed),
            tau=0.0,
            norm_std=float(norm_std),
            num_classes=int(num_classes),
            feature_size=int(3 * image_size * image_size),
            n_train=int(n),
            n_flipped=0,
            class_counts_clean=df_out["dx_clean"].value_counts().to_dict(),
            class_counts_noisy=df_out["dx_noisy"].value_counts().to_dict(),
            flip_confusion={},
            flip_rate_min=0.0,
            flip_rate_median=0.0,
            flip_rate_max=0.0,
        )
        return df_out, report

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    ds = HamTensorDataset(df=df, images_dir=images_dir, c2i=c2i, tfm=tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(pin_memory and device.startswith("cuda")),
    )

    # Fix random seeds for reproducibility
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    # Sample per-instance flip rates from truncated normal centred on tau
    flip_dist = stats.truncnorm(
        (0.0 - tau) / norm_std,
        (1.0 - tau) / norm_std,
        loc=tau,
        scale=norm_std,
    )
    flip_rate = flip_dist.rvs(n).astype(np.float32)

    # Sample class-conditional random weight matrices W ~ N(0, I), shape (C, d, C)
    W = torch.FloatTensor(
        np.random.randn(num_classes, feature_size, num_classes).astype(np.float32)
    ).to(device)

    new_label_idx = np.empty(n, dtype=np.int64)
    flip_confusion: Dict[str, Dict[str, int]] = {}
    cursor = 0

    for x, y, _ in tqdm(dl, desc="IDN corruption", leave=True):
        b = x.size(0)
        x = x.to(device)
        y = y.to(device).long()

        # Flatten images to pixel vectors and project through W_y (Xia et al. Eq. 1)
        x_flat = x.view(b, -1)
        W_y = W[y]
        A = torch.bmm(x_flat.unsqueeze(1), W_y).squeeze(1)

        # Mask true class so it cannot be a flip target
        A[torch.arange(b, device=device), y] = -inf

        # Scale softmax probabilities by per-instance flip rate, restore true class mass
        q = torch.from_numpy(flip_rate[cursor:cursor + b]).to(device).view(-1, 1)
        P = q * F.softmax(A, dim=1)
        P[torch.arange(b, device=device), y] += (1.0 - q.squeeze(1))

        # Sample noisy labels from the per-instance transition distribution
        sampled = torch.multinomial(P, num_samples=1).squeeze(1)
        sampled_cpu = sampled.cpu().numpy().astype(np.int64)
        y_cpu = y.cpu().numpy().astype(np.int64)

        new_label_idx[cursor:cursor + b] = sampled_cpu

        # Track which labels were flipped and to which class
        for yi, ytilde in zip(y_cpu, sampled_cpu):
            if ytilde == yi:
                continue
            true_str, noisy_str = i2c[int(yi)], i2c[int(ytilde)]
            flip_confusion.setdefault(true_str, {})
            flip_confusion[true_str][noisy_str] = flip_confusion[true_str].get(noisy_str, 0) + 1

        cursor += b

    df_out = df.copy()
    df_out["dx_clean"] = df_out["dx"]
    df_out["dx_noisy"] = [i2c[int(i)] for i in new_label_idx]

    report = NoiseReport(
        outer_fold=-1,
        seed=int(seed),
        tau=float(tau),
        norm_std=float(norm_std),
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

    return df_out, report


def generate_idn_outercv(
    df: pd.DataFrame,
    images_dir: Path,
    outer_folds: int,
    seed: int,
    tau: float,
    *,
    image_size: int = 224,
    norm_std: float = 0.1,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> IDNOutputs:
    # Generates all outer CV folds with clean test splits and IDN-corrupted train splits
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"] = df["dx"].astype(str)

    df_folds = make_outer_folds_lesion_stratified(df, n_splits=outer_folds, seed=seed)
    fold_assignments = df_folds[["image_id", "lesion_id", "dx", "outer_fold"]].copy()
    folds_out: Dict[int, FoldOutputs] = {}

    for fold_id in tqdm(range(outer_folds), desc="Outer folds", leave=True):
        test_df = df_folds[df_folds["outer_fold"] == fold_id].copy().reset_index(drop=True)
        train_df = df_folds[df_folds["outer_fold"] != fold_id].copy().reset_index(drop=True)

        df_corrupted, report = generate_instance_dependent_noisy_labels(
            df=train_df[["image_id", "lesion_id", "dx"]].copy(),
            images_dir=images_dir,
            tau=tau,
            seed=(seed * 10_000 + fold_id),
            image_size=image_size,
            norm_std=norm_std,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        report.outer_fold = int(fold_id)

        keep_cols = ["image_id", "lesion_id", "dx", "dx_clean", "dx_noisy"]

        train_clean = df_corrupted.copy()
        train_clean["dx"] = train_clean["dx_clean"]
        train_clean = train_clean[[c for c in keep_cols if c in train_clean.columns]]

        train_noisy = df_corrupted.copy()
        train_noisy["dx"] = train_noisy["dx_noisy"]
        train_noisy = train_noisy[[c for c in keep_cols if c in train_noisy.columns]]

        folds_out[int(fold_id)] = FoldOutputs(
            train_clean=train_clean,
            train_noisy=train_noisy,
            test_clean=test_df[["image_id", "lesion_id", "dx"]].copy(),
            report=report,
        )

    return IDNOutputs(fold_assignments=fold_assignments, folds=folds_out)