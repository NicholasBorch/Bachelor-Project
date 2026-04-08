# src/methods/sce.py
#
# Symmetric Cross Entropy (SCE) for HAM10000 classification.
# Wang et al. (2019): https://arxiv.org/abs/1908.06112
#
# Structure mirrors baseline.py.
# The only difference from baseline is the loss function:
#
#   baseline : nn.CrossEntropyLoss (class-weighted)
#   SCE      : SCELoss (alpha * weighted_CE + beta * RCE)
#
# No additional model components, no sample filtering, no auxiliary networks.
# SCE is a drop-in loss replacement — the training loop is identical to baseline.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.classification.dataset import HamTensorDataset
from src.classification.models import build_resnet
from src.classification.train import (
    compute_class_weights,
    get_transforms,
    make_weighted_sampler,
    train_one_epoch,
)
from src.common.io import class_mapping
from src.common.logging import ResultsLogger, RunConfig, make_output_dir
from src.common.metrics import compute_metrics, print_metrics
from src.common.seed import seed_everything
from src.methods.sce_loss import SCELoss
from configs.classification_default import PIN_MEMORY
from configs.classification_sce import (
    SCE_ALPHA,
    SCE_BETA,
    SCE_A,
)


def run_sce_fold(
    train_noisy_df: pd.DataFrame,
    test_clean_df: pd.DataFrame,
    images_dir: Path,
    results_root: Path,
    *,
    tau: float,
    outer_fold: int,
    seed: int,
    noise_type: str,
    backbone_depth: int = 50,
    image_size: int = 224,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    num_workers: int = 2,
    device: Optional[torch.device] = None,
    use_weighted_sampler: bool = True,
) -> dict:
    """
    Trains SCE for a fixed number of epochs on one noisy training fold
    and evaluates once on the clean test fold after the final epoch.

    SCE hyperparameters are imported from configs/classification_sce.py
    """
    seed_everything(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Label mappings ─────────────────────────────────────────────────────
    # Build from union of train and test to guarantee consistent class ordering
    all_labels  = pd.concat([train_noisy_df["dx"], test_clean_df["dx"]]).tolist()
    c2i, i2c    = class_mapping(all_labels)
    num_classes = len(c2i)
    class_names = [i2c[i] for i in range(num_classes)]

    train_labels = [c2i[str(dx)] for dx in train_noisy_df["dx"]]

    # ── Datasets and loaders ───────────────────────────────────────────────
    train_ds = HamTensorDataset(
        train_noisy_df, images_dir, c2i, get_transforms(image_size, augment=True)
    )
    test_ds = HamTensorDataset(
        test_clean_df, images_dir, c2i, get_transforms(image_size, augment=False)
    )

    sampler = make_weighted_sampler(train_labels) if use_weighted_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),   # shuffle=True only when no sampler — DataLoader
                                     # raises an error if both sampler and shuffle=True
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
    )

    print(f"    Sampler: {'weighted (replacement=True)' if use_weighted_sampler else 'shuffle=True (no sampler)'}")

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_resnet(
        num_classes=num_classes, pretrained=True, depth=backbone_depth
    ).to(device)

    # ── SCE loss ───────────────────────────────────────────────────────────
    # Class weights applied to CE component only — matches baseline weighting
    # for fair comparison. RCE operates on softmax predictions and is left
    # unweighted since it already has a bounded, per-sample noise suppression
    # effect that does not require explicit class rebalancing.
    class_weights = compute_class_weights(train_labels, num_classes, device)
    criterion = SCELoss(
        alpha=SCE_ALPHA,
        beta=SCE_BETA,
        num_classes=num_classes,
        A=SCE_A,
        class_weights=class_weights,
    )

    # ── Optimiser and scheduler ────────────────────────────────────────────
    # Identical to baseline — SCE is a loss-only change, not an optimisation change
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    # ── Logging ────────────────────────────────────────────────────────────
    out_dir = make_output_dir(results_root, "sce", tau, outer_fold, noise_type)
    config  = RunConfig(
        method="sce",
        tau=tau,
        outer_fold=outer_fold,
        seed=seed,
        backbone=f"resnet{backbone_depth}",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size,
        noise_type=noise_type,
        extra={
            "alpha": SCE_ALPHA,
            "beta":  SCE_BETA,
            "A":     SCE_A,
        },
    )
    logger = ResultsLogger(out_dir, config)

    print(f"\n--- SCE | noise={noise_type} | tau={tau:.2f} | fold={outer_fold} ---")
    print(f"    alpha={SCE_ALPHA} | beta={SCE_BETA} | A={SCE_A}")
    print(f"    Train samples : {len(train_ds)}")
    print(f"    Test samples  : {len(test_ds)}")
    print(f"    Device        : {device} | Backbone: resnet{backbone_depth}")
    print(f"    Epochs        : {epochs}")

    # ── Training loop ──────────────────────────────────────────────────────
    # Identical to baseline — train_one_epoch from train.py handles the loop.
    # SCE loss replaces CE as a drop-in — no other changes needed.
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        scheduler.step()
        logger.log_epoch(epoch + 1, train_loss)
        print(f"    Epoch {epoch+1:03d}/{epochs} | train_loss={train_loss:.4f}")

    # ── Single test evaluation after all epochs complete ───────────────────
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for x, y, _ in test_loader:
            x      = x.to(device)
            logits = model(x)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_prob.append(probs)
            all_pred.append(preds)
            all_true.append(y.numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)

    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    logger.log_test_metrics(metrics)

    print(f"\n    Test results (fold={outer_fold}, tau={tau:.2f}):")
    print_metrics(metrics, prefix="    ")

    return metrics