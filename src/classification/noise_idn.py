# src/classification/noise_idn.py
"""
Instance-Dependent Noise (IDN) label corruption — Synthetic generator (Algorithm 2).

This module implements the *standard* synthetic IDN generation procedure used in
the noisy-label learning literature, matching Algorithm 2 from the referenced paper
and the widely-circulated reference implementation pattern:

  - Sample per-instance flip rate q_i ~ TruncatedNormal(tau, sigma^2) clipped to [0, 1]
  - Sample class-conditional random matrices W_y ~ N(0, I)
  - For each sample (x_i, y_i):
        a = x_i^T W_{y_i}
        set a[y_i] = -inf
        p = q_i * softmax(a)
        p[y_i] += 1 - q_i
        y_tilde ~ Categorical(p)

Why this exists in our repo:
  - We generate *controlled*, reproducible IDN-corrupted training folds for HAM10000
    to evaluate noise-robust training methods in our bachelor project.

Industry-style provenance / attribution:
  - The algorithmic steps correspond to Algorithm 2 in the project’s referenced IDN paper.
  - This implementation is a batched, device-agnostic adaptation of the common reference
    code pattern (including the snippet we discussed) so it runs on CPU-only machines
    (e.g., Intel Mac) as well as CUDA GPUs (Windows, DTU HPC).

Notes:
  - This is a *data corruption* utility. It does not train models and does not require
    teacher predictions (unlike our previous OOF/teacher-driven pipeline).

Author: Bachelor Project team
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class NoiseReport:
    """
    Summary of noise injection for one outer fold.

    flip_confusion maps:
        true_label_str -> {noisy_label_str: count}
    """
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


# ----------------------------
# Helpers
# ----------------------------
def class_mapping(classes: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Stable mapping between string labels and integer indices."""
    classes_sorted = sorted(list(set(classes)))
    c2i = {c: i for i, c in enumerate(classes_sorted)}
    i2c = {i: c for c, i in c2i.items()}
    return c2i, i2c


# ----------------------------
# Dataset
# ----------------------------
class HamTensorDataset(Dataset):
    """
    Minimal dataset for IDN generation.
    Returns (x_tensor, y_index, image_id_str).
    """
    def __init__(self, df: pd.DataFrame, images_dir: Path, c2i: Dict[str, int], tfm):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.c2i = c2i
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        y_str = str(row["dx"])
        y = int(self.c2i[y_str])

        img_path = self.images_dir / f"{image_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        x = self.tfm(img)

        return x, y, image_id


# ----------------------------
# Core IDN generator (Algorithm 2)
# ----------------------------
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
    """
    Apply *standard synthetic IDN* corruption to a dataframe split.

    Parameters
    ----------
    df:
        DataFrame containing columns: ["image_id", "lesion_id", "dx"].
        dx is the *clean* label string.
    images_dir:
        Directory holding images named "<image_id>.jpg".
    tau:
        Mean corruption rate (paper uses τ). This is the *expected* flip probability.
    seed:
        RNG seed for reproducibility.
    image_size:
        Images are resized to (image_size, image_size) before flattening.
    norm_std:
        Std dev for truncated normal sampling of per-instance flip rates.
        (Paper default is typically 0.1).
    batch_size, num_workers, pin_memory:
        DataLoader settings.

    Returns
    -------
    df_out:
        Original rows plus ["dx_clean", "dx_noisy"].
    report:
        NoiseReport with confusion summary and flip statistics.

    Implementation notes:
      - Uses transforms.Resize + ToTensor (no ImageNet normalization) to keep "pixel-vector"
        semantics close to the original synthetic IDN generator setting.
      - Batched computation:
            logits = x_flat @ W_y
        where W_y is selected per-sample by its clean label y.
    """
    df = df.copy().reset_index(drop=True)
    df["image_id"] = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"] = df["dx"].astype(str)

    c2i, i2c = class_mapping(df["dx"].tolist())
    num_classes = len(c2i)

    # Device selection: CPU on Intel Mac, CUDA on Windows/HPC if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms: keep close to "raw feature vector" setting.
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0,1]
    ])

    ds = HamTensorDataset(df=df, images_dir=images_dir, c2i=c2i, tfm=tfm)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(pin_memory and device.startswith("cuda")),
    )

    # Feature size d = 3 * H * W
    feature_size = 3 * image_size * image_size

    # ----------------------------
    # Reproducible RNG
    # ----------------------------
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

    # ----------------------------
    # Sample per-instance flip rates q_i ~ TruncNorm(tau, norm_std^2, [0,1])
    # ----------------------------
    n = len(df)
    flip_dist = stats.truncnorm(
        (0.0 - tau) / norm_std,
        (1.0 - tau) / norm_std,
        loc=tau,
        scale=norm_std,
    )
    flip_rates = flip_dist.rvs(n).astype(np.float32)

    # ----------------------------
    # Sample class-conditional random matrices W_y ~ N(0, I)
    # Shape: (C, d, C)
    # ----------------------------
    W = np.random.randn(num_classes, feature_size, num_classes).astype(np.float32)
    W = torch.from_numpy(W).to(device)

    # Output arrays
    new_label_idx = np.empty(n, dtype=np.int64)

    # Confusion bookkeeping on string labels
    flip_confusion: Dict[str, Dict[str, int]] = {}

    # Iterate in order, so we can align flip_rates with dataset indices
    cursor = 0
    pbar = tqdm(dl, desc="IDN corruption (Algorithm 2)", leave=True)

    for x, y, _image_ids in pbar:
        b = x.size(0)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        # Flatten to (B, d)
        x_flat = x.view(b, -1)

        # Select W_y per sample: (B, d, C)
        W_y = W[y]

        # logits: (B, C) via batched matrix multiply
        # (B,1,d) @ (B,d,C) -> (B,1,C) -> (B,C)
        logits = torch.bmm(x_flat.unsqueeze(1), W_y).squeeze(1)

        # Set true-class logit to -inf so softmax mass is only on wrong classes
        logits[torch.arange(b, device=device), y] = -inf

        # softmax over wrong classes
        off_diag = F.softmax(logits, dim=1)

        # Apply per-instance flip rates
        q = torch.from_numpy(flip_rates[cursor:cursor + b]).to(device).view(-1, 1)  # (B,1)
        P = q * off_diag
        P[torch.arange(b, device=device), y] += (1.0 - q.squeeze(1))

        # Sample new labels from categorical distribution P
        sampled = torch.multinomial(P, num_samples=1).squeeze(1)  # (B,)
        sampled_cpu = sampled.detach().cpu().numpy().astype(np.int64)
        y_cpu = y.detach().cpu().numpy().astype(np.int64)

        new_label_idx[cursor:cursor + b] = sampled_cpu

        # Update confusion counts
        for yi, ytilde in zip(y_cpu, sampled_cpu):
            if ytilde == yi:
                continue
            true_str = i2c[int(yi)]
            noisy_str = i2c[int(ytilde)]
            flip_confusion.setdefault(true_str, {})
            flip_confusion[true_str][noisy_str] = flip_confusion[true_str].get(noisy_str, 0) + 1

        cursor += b

    # Build output dataframe
    df_out = df.copy()
    df_out["dx_clean"] = df_out["dx"]
    df_out["dx_noisy"] = [i2c[int(i)] for i in new_label_idx]

    n_flipped = int((df_out["dx_clean"] != df_out["dx_noisy"]).sum())

    # Flip-rate summary
    fr_min = float(np.min(flip_rates)) if len(flip_rates) else float("nan")
    fr_med = float(np.median(flip_rates)) if len(flip_rates) else float("nan")
    fr_max = float(np.max(flip_rates)) if len(flip_rates) else float("nan")

    report = NoiseReport(
        outer_fold=-1,  # filled by caller if used per fold
        seed=int(seed),
        tau=float(tau),
        norm_std=float(norm_std),
        num_classes=int(num_classes),
        feature_size=int(feature_size),
        n_train=int(n),
        n_flipped=int(n_flipped),
        class_counts_clean=df_out["dx_clean"].value_counts().to_dict(),
        class_counts_noisy=df_out["dx_noisy"].value_counts().to_dict(),
        flip_confusion=flip_confusion,
        flip_rate_min=fr_min,
        flip_rate_median=fr_med,
        flip_rate_max=fr_max,
    )

    return df_out, report


# ----------------------------
# Outer-fold pipeline helpers
# ----------------------------
def make_outer_folds_lesion_stratified(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """
    Create outer folds on unique lesion_id with stratification on dx.

    This matches our evaluation protocol: the test fold remains clean; only the
    outer-train split is corrupted.
    """
    from sklearn.model_selection import StratifiedKFold

    df = df.copy().sort_values(["lesion_id", "image_id"]).reset_index(drop=True)

    lesion_df = df.drop_duplicates(subset=["lesion_id"]).copy()
    lesion_df = lesion_df.sort_values(["lesion_id"]).reset_index(drop=True)

    y = lesion_df["dx"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    lesion_fold = np.full(len(lesion_df), -1, dtype=int)
    for fold_id, (_, test_idx) in enumerate(skf.split(np.arange(len(lesion_df)), y)):
        lesion_fold[test_idx] = fold_id

    lesion_df["outer_fold"] = lesion_fold
    lesion_to_fold = dict(zip(lesion_df["lesion_id"].astype(str), lesion_df["outer_fold"].astype(int)))
    df["outer_fold"] = df["lesion_id"].astype(str).map(lesion_to_fold)

    return df


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
    """
    Generate outer CV folds where:
      - test is clean
      - train is corrupted using standard synthetic IDN (Algorithm 2)

    Returns:
      - fold_assignments (shared)
      - per-fold train_clean / train_noisy / test_clean
      - per-fold NoiseReport
    """
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"] = df["dx"].astype(str)

    df_folds = make_outer_folds_lesion_stratified(df, n_splits=outer_folds, seed=seed)
    fold_assignments = df_folds[["image_id", "lesion_id", "dx", "outer_fold"]].copy()

    folds_out: Dict[int, FoldOutputs] = {}

    for fold_id in tqdm(range(outer_folds), desc="Outer folds (apply IDN)", leave=True):
        test_df = df_folds[df_folds["outer_fold"] == fold_id].copy().reset_index(drop=True)
        train_df = df_folds[df_folds["outer_fold"] != fold_id].copy().reset_index(drop=True)

        # Corrupt only the train split
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

        train_clean = df_corrupted.copy()
        train_clean["dx"] = train_clean["dx_clean"]

        train_noisy = df_corrupted.copy()
        train_noisy["dx"] = train_noisy["dx_noisy"]

        # Keep columns consistent with the rest of the repo
        keep_cols = ["image_id", "lesion_id", "dx", "dx_clean", "dx_noisy"]
        train_clean = train_clean[[c for c in keep_cols if c in train_clean.columns]]
        train_noisy = train_noisy[[c for c in keep_cols if c in train_noisy.columns]]
        test_clean = test_df[["image_id", "lesion_id", "dx"]].copy()

        folds_out[int(fold_id)] = FoldOutputs(
            train_clean=train_clean,
            train_noisy=train_noisy,
            test_clean=test_clean,
            report=report,
        )

    return IDNOutputs(fold_assignments=fold_assignments, folds=folds_out)