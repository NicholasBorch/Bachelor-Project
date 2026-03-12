# src/classification/noise_idn_feature_driven.py
# Feature-driven IDN label corruption extending Xia et al. (2020).
# Replaces random projections with out-of-fold softmax probabilities from a
# class-balanced ResNet, grounding flip targets in learned visual similarity.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common.io import class_mapping
from src.classification.dataset import HamTensorDataset
from src.classification.folds import make_outer_folds_lesion_stratified, make_inner_folds_lesion_stratified
from src.classification.models import build_resnet
from src.classification.train import (
    get_transforms,
    make_weighted_sampler,
    compute_class_weights,
    train_one_epoch,
)


@dataclass
class NoiseReport:
    # Summary statistics for one noise injection run
    outer_fold: int
    seed: int
    tau: float
    norm_std: float
    num_classes: int
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


def _fit_baseline(
    model: nn.Module,
    train_df: pd.DataFrame,
    images_dir: Path,
    c2i: dict,
    *,
    image_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    device: torch.device,
) -> nn.Module:
    # Trains a class-balanced ResNet for OOF probability collection only
    # Fixed epoch count with no early stopping — inner test split never influences training
    model = model.to(device)
    num_classes  = len(c2i)
    train_labels = [c2i[str(dx)] for dx in train_df["dx"]]

    train_ds = HamTensorDataset(train_df, images_dir, c2i, get_transforms(image_size, augment=True))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=make_weighted_sampler(train_labels),
        num_workers=num_workers, pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_labels, num_classes, device))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        scheduler.step()
        print(f"    Epoch {epoch+1:03d}/{epochs} | train={train_loss:.4f}")

    return model


def collect_oof_probabilities(
    train_df: pd.DataFrame,
    images_dir: Path,
    c2i: dict,
    inner_folds: int,
    seed: int,
    outer_fold_id: int,
    *,
    image_size: int = 224,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    num_workers: int = 2,
) -> np.ndarray:
    # Trains one ResNet per inner fold and collects out-of-fold softmax probabilities
    # so every training sample's confusion scores come from a model that never saw it
    num_classes = len(c2i)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n           = len(train_df)

    oof_probs = np.zeros((n, num_classes), dtype=np.float32)
    df_inner  = make_inner_folds_lesion_stratified(train_df, n_splits=inner_folds, seed=seed)

    for inner_fold_id in range(inner_folds):
        print(f"  Outer fold {outer_fold_id} | Inner fold {inner_fold_id + 1}/{inner_folds}")

        inner_train_df = df_inner[df_inner["inner_fold"] != inner_fold_id].copy().reset_index(drop=True)
        inner_val_df   = df_inner[df_inner["inner_fold"] == inner_fold_id].copy().reset_index(drop=True)
        val_indices    = df_inner[df_inner["inner_fold"] == inner_fold_id].index.tolist()

        # Train on inner training split only — inner val split never touches training
        model = _fit_baseline(
            model=build_resnet(num_classes=num_classes, pretrained=True, depth=18),
            train_df=inner_train_df,
            images_dir=images_dir,
            c2i=c2i,
            image_size=image_size,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            num_workers=num_workers,
            device=device,
        )

        # Collect softmax probabilities on inner val split — inference only
        model.eval()
        val_ds     = HamTensorDataset(inner_val_df, images_dir, c2i, get_transforms(image_size, augment=False))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

        with torch.no_grad():
            val_probs = np.concatenate([
                F.softmax(model(x.to(device)), dim=1).cpu().numpy()
                for x, _, _ in val_loader
            ], axis=0)

        # Place probabilities back at their correct positions in the full training array
        for local_idx, global_idx in enumerate(val_indices):
            oof_probs[global_idx] = val_probs[local_idx]

    return oof_probs


def generate_feature_driven_noisy_labels(
    df: pd.DataFrame,
    tau: float,
    seed: int,
    oof_probs: np.ndarray,
    *,
    norm_std: float = 0.1,
) -> Tuple[pd.DataFrame, NoiseReport]:
    
    # Applies feature-driven IDN corruption using pre-computed OOF softmax probabilities
    df = df.copy().reset_index(drop=True)
    df["image_id"] = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"]       = df["dx"].astype(str)

    c2i, i2c    = class_mapping(df["dx"].tolist())
    num_classes = len(c2i)
    n           = len(df)
    
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
    
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fix seeds for reproducible flip rate sampling
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

    probs  = torch.from_numpy(oof_probs).float().to(device)
    labels = torch.tensor([c2i[dx] for dx in df["dx"]], dtype=torch.long, device=device)

    # Mask true class and renormalise remaining probability mass over wrong classes
    probs[torch.arange(n, device=device), labels] = 0.0
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

    # Build per-instance transition rows using feature-driven confusion
    q = torch.from_numpy(flip_rate).float().to(device).view(-1, 1)
    P = q * probs
    P[torch.arange(n, device=device), labels] += (1.0 - q.squeeze(1))

    # Sample noisy labels from per-instance transition distribution
    new_label_idx = torch.multinomial(P, num_samples=1).squeeze(1).cpu().numpy().astype(np.int64)
    labels_cpu    = labels.cpu().numpy().astype(np.int64)

    # Track which labels were flipped and to which class
    flip_confusion: Dict[str, Dict[str, int]] = {}
    for yi, ytilde in zip(labels_cpu, new_label_idx):
        if ytilde == yi:
            continue
        true_str, noisy_str = i2c[int(yi)], i2c[int(ytilde)]
        flip_confusion.setdefault(true_str, {})
        flip_confusion[true_str][noisy_str] = flip_confusion[true_str].get(noisy_str, 0) + 1

    df_out = df.copy()
    df_out["dx_clean"] = df_out["dx"]
    df_out["dx_noisy"] = [i2c[int(i)] for i in new_label_idx]

    report = NoiseReport(
        outer_fold=-1,
        seed=int(seed),
        tau=float(tau),
        norm_std=float(norm_std),
        num_classes=int(num_classes),
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


def generate_feature_driven_idn_outercv(
    df: pd.DataFrame,
    images_dir: Path,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    tau_values: List[float],
    *,
    image_size: int = 224,
    norm_std: float = 0.1,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    num_workers: int = 2,
) -> Dict[float, IDNOutputs]:
    # Outer loop: collects OOF probs once per outer fold then applies IDN at all tau values
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"]       = df["dx"].astype(str)

    c2i, i2c    = class_mapping(df["dx"].tolist())
    df_folds    = make_outer_folds_lesion_stratified(df, n_splits=outer_folds, seed=seed)
    fold_assignments = df_folds[["image_id", "lesion_id", "dx", "outer_fold"]].copy()
    all_outputs: Dict[float, Dict[int, FoldOutputs]] = {tau: {} for tau in tau_values}

    for outer_fold_id in tqdm(range(outer_folds), desc="Outer folds", leave=True):
        test_df  = df_folds[df_folds["outer_fold"] == outer_fold_id].copy().reset_index(drop=True)
        train_df = df_folds[df_folds["outer_fold"] != outer_fold_id].copy().reset_index(drop=True)

        # OOF collection is expensive — done once per outer fold and reused across all tau values
        print(f"\nCollecting OOF probabilities for outer fold {outer_fold_id}...")
        oof_probs = collect_oof_probabilities(
            train_df=train_df,
            images_dir=images_dir,
            c2i=c2i,
            inner_folds=inner_folds,
            seed=seed,
            outer_fold_id=outer_fold_id,
            image_size=image_size,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            num_workers=num_workers,
        )

        # Apply feature-driven IDN at each tau using the same OOF probs
        for tau in tau_values:
            df_corrupted, report = generate_feature_driven_noisy_labels(
                df=train_df[["image_id", "lesion_id", "dx"]].copy(),
                tau=tau,
                seed=(seed * 10_000 + outer_fold_id),
                oof_probs=oof_probs.copy(),
                norm_std=norm_std,
            )
            report.outer_fold = int(outer_fold_id)

            keep_cols   = ["image_id", "lesion_id", "dx", "dx_clean", "dx_noisy"]
            train_clean = df_corrupted.copy()
            train_clean["dx"] = train_clean["dx_clean"]
            train_clean = train_clean[[c for c in keep_cols if c in train_clean.columns]]

            train_noisy = df_corrupted.copy()
            train_noisy["dx"] = train_noisy["dx_noisy"]
            train_noisy = train_noisy[[c for c in keep_cols if c in train_noisy.columns]]

            all_outputs[tau][int(outer_fold_id)] = FoldOutputs(
                train_clean=train_clean,
                train_noisy=train_noisy,
                test_clean=test_df[["image_id", "lesion_id", "dx"]].copy(),
                report=report,
            )

    return {
        tau: IDNOutputs(fold_assignments=fold_assignments, folds=folds)
        for tau, folds in all_outputs.items()
    }