# src/methods/elr.py
#
# Early-Learning Regularization (ELR) for HAM10000 classification.
# Liu et al. (2020): https://arxiv.org/abs/2007.00151
#
# Structure mirrors baseline.py and sce.py.
# The only differences from baseline are:
#   1. Loss is ELRLoss instead of CrossEntropyLoss
#   2. Training loop passes sample indices to the loss for target updates
#   3. dataset.py returns integer sample index as third element (used here)

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.classification.dataset import HamTensorDataset
from src.classification.models import build_resnet
from src.classification.train import (
    compute_class_weights,
    get_transforms,
    make_weighted_sampler,
)
from src.common.io import class_mapping
from src.common.logging import ResultsLogger, RunConfig, make_output_dir
from src.common.metrics import compute_metrics, print_metrics
from src.common.seed import seed_everything
from src.methods.elr_loss import ELRLoss
from configs.classification_default import PIN_MEMORY
from configs.classification_elr import ELR_BETA, ELR_LAMBDA


def run_elr_fold(
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
) -> dict:
    """
    Trains ELR for a fixed number of epochs on one noisy training fold
    and evaluates once on the clean test fold after the final epoch.
    """
    seed_everything(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Label mappings ─────────────────────────────────────────────────────
    all_labels  = pd.concat([train_noisy_df["dx"], test_clean_df["dx"]]).tolist()
    c2i, i2c    = class_mapping(all_labels)
    num_classes = len(c2i)
    class_names = [i2c[i] for i in range(num_classes)]

    train_labels = [c2i[str(dx)] for dx in train_noisy_df["dx"]]
    n_train      = len(train_noisy_df)

    # ── Datasets and loaders ───────────────────────────────────────────────
    train_ds = HamTensorDataset(
        train_noisy_df, images_dir, c2i, get_transforms(image_size, augment=True)
    )
    test_ds = HamTensorDataset(
        test_clean_df, images_dir, c2i, get_transforms(image_size, augment=False)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=make_weighted_sampler(train_labels),
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
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

    # ── ELR loss ───────────────────────────────────────────────────────────
    # Class weights applied to CE component only — matches baseline weighting.
    # The ELR regularisation term operates on softmax predictions and does
    # not require explicit class rebalancing.
    class_weights = compute_class_weights(train_labels, num_classes, device)
    criterion = ELRLoss(
        num_examp=n_train,
        num_classes=num_classes,
        elr_lambda=ELR_LAMBDA,
        beta=ELR_BETA,
        device=device,
        class_weights=class_weights,
    )

    # ── Optimiser and scheduler ────────────────────────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    # ── Logging ────────────────────────────────────────────────────────────
    out_dir = make_output_dir(results_root, "elr", tau, outer_fold, noise_type)
    config  = RunConfig(
        method="elr",
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
            "beta":       ELR_BETA,
            "elr_lambda": ELR_LAMBDA,
        },
    )
    logger = ResultsLogger(out_dir, config)

    print(f"\n--- ELR | noise={noise_type} | tau={tau:.2f} | fold={outer_fold} ---")
    print(f"    beta={ELR_BETA} | lambda={ELR_LAMBDA}")
    print(f"    Train samples : {len(train_ds)}")
    print(f"    Test samples  : {len(test_ds)}")
    print(f"    Device        : {device} | Backbone: resnet{backbone_depth}")
    print(f"    Epochs        : {epochs}")

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0

        for x, y, idx in train_loader:
            x   = x.to(device)
            y   = y.to(device)
            idx = idx.to(device)

            optimiser.zero_grad()
            logits = model(x)
            loss   = criterion(idx, logits, y)
            loss.backward()
            optimiser.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = total_loss / n
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