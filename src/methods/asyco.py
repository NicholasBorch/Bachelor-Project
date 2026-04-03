# src/methods/asyco.py
#
# Asymmetric Co-teaching (AsyCo) for HAM10000 classification.
# Implements Liu et al. (2023) "Asymmetric Co-teaching with Multi-view
# Consensus for Noisy Label Learning".
#
# Fixes applied relative to the original implementation:
#
#   1. BatchNorm running statistics are frozen during subset forward passes
#      in asyco_epoch. Without this, each subset pass (clean-only, noisy-only)
#      corrupts the running mean/variance with biased subset statistics,
#      degrading test-time performance — especially for the majority class.
#
#   2. The double forward pass for noisy pseudo-labels is consolidated into
#      a single pass that computes both the pseudo-label (detached) and the
#      predictions for the consistency loss from the same logits.
#
#   3. Warmup ratio is controlled via configs/classification_asyco.py.
#      With a 25-epoch budget, warmup must be short (3-5 epochs) to give
#      the selection mechanism sufficient post-warmup training time.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from configs.classification_default import PIN_MEMORY
from configs.classification_asyco import (
    WARMUP_EPOCHS,
    K_TOPLABEL,
    LAMBDA_U,
    TEMPERATURE,
)


# ── BatchNorm stat freezing utilities ─────────────────────────────────────
# These allow subset forward passes without corrupting the running mean and
# variance that will be used at test time. Setting momentum=0 makes the
# running stats ignore the current batch entirely. The model still uses
# per-batch statistics for normalisation during training (train mode), so
# gradients are unaffected — only the running buffer update is suppressed.

def freeze_bn_running_stats(model: nn.Module) -> None:
    """Set momentum=0 on all BN layers to prevent running stat updates."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = 0


def restore_bn_running_stats(model: nn.Module, momentum: float = 0.1) -> None:
    """Restore default momentum on all BN layers."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = momentum


def sharpen(probs: torch.Tensor, T: float) -> torch.Tensor:
    sharp = probs.pow(1.0 / T)
    return sharp / sharp.sum(dim=1, keepdim=True)


def compute_sample_weights(
    y_noisy: torch.Tensor,
    y_pred_n: torch.Tensor,
    y_pred_r: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    N      = y_noisy.size(0)
    device = y_noisy.device

    y_oh  = F.one_hot(y_noisy,  num_classes).float()
    yn_oh = F.one_hot(y_pred_n, num_classes).float()
    yr    = y_pred_r.float()

    agree_y_yn  = (y_oh  * yn_oh).sum(dim=1)
    agree_yn_yr = (yn_oh * yr   ).sum(dim=1)
    agree_y_yr  = (y_oh  * yr   ).sum(dim=1)

    ag = agree_y_yn + agree_yn_yr + agree_y_yr

    w = torch.full((N,), -1, dtype=torch.long, device=device)
    matched = ag > 0
    w[matched & (agree_y_yr == 1)] = 1
    w[matched & (agree_y_yr == 0)] = 0

    return w


def compute_relabels(
    y_noisy: torch.Tensor,
    y_pred_n: torch.Tensor,
    y_pred_r: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    y_oh  = F.one_hot(y_noisy,  num_classes).float()
    yn_oh = F.one_hot(y_pred_n, num_classes).float()
    yr    = y_pred_r.float()

    agree_y_yn  = (y_oh  * yn_oh).sum(dim=1)
    agree_yn_yr = (yn_oh * yr   ).sum(dim=1)
    agree_y_yr  = (y_oh  * yr   ).sum(dim=1)

    is_sidecore = (agree_y_yn == 0) & (agree_yn_yr == 1) & (agree_y_yr == 1)
    is_nr       = (agree_y_yn == 0) & (agree_yn_yr == 1) & (agree_y_yr == 0)

    y_hat = y_oh.clone()
    y_hat[is_sidecore] = yn_oh[is_sidecore]
    y_hat[is_nr]       = (y_oh[is_nr] + yn_oh[is_nr]).clamp(max=1.0)

    return y_hat


def warmup_epoch(
    clf_net: nn.Module,
    ref_net: nn.Module,
    loader: DataLoader,
    clf_criterion: nn.Module,
    clf_optimiser: torch.optim.Optimizer,
    ref_optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    clf_net.train()
    ref_net.train()

    clf_total, ref_total = 0.0, 0.0
    n = 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        b    = x.size(0)

        clf_optimiser.zero_grad()
        logits_n = clf_net(x)
        loss_clf = clf_criterion(logits_n, y)
        loss_clf.backward()
        clf_optimiser.step()

        ref_optimiser.zero_grad()
        logits_r = ref_net(x)
        y_oh     = F.one_hot(y, num_classes=logits_r.size(1)).float()
        loss_ref = F.binary_cross_entropy_with_logits(logits_r, y_oh)
        loss_ref.backward()
        ref_optimiser.step()

        clf_total += loss_clf.item() * b
        ref_total += loss_ref.item() * b
        n += b

    return clf_total / n, ref_total / n


def asyco_epoch(
    clf_net: nn.Module,
    ref_net: nn.Module,
    loader: DataLoader,
    clf_criterion: nn.Module,
    clf_optimiser: torch.optim.Optimizer,
    ref_optimiser: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    lambda_u: float,
    temperature: float,
    k_toplabel: int,
) -> tuple[float, float]:
    clf_net.train()
    ref_net.train()

    clf_total, ref_total = 0.0, 0.0
    n_clf, n_ref = 0, 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        b    = x.size(0)

        # ── Step 1: Full-batch inference for sample selection ─────────
        # Both networks run on the full batch under no_grad.
        # BN running stats update here is fine — this is a full batch.
        with torch.no_grad():
            logits_n = clf_net(x)
            logits_r = ref_net(x)

            y_pred_n = logits_n.argmax(dim=1)

            probs_r         = torch.sigmoid(logits_r)
            _, topk_idx     = probs_r.topk(k_toplabel, dim=1)
            y_pred_r        = torch.zeros_like(probs_r)
            y_pred_r.scatter_(1, topk_idx, 1.0)

        w     = compute_sample_weights(y, y_pred_n, y_pred_r, num_classes)
        y_hat = compute_relabels(y, y_pred_n, y_pred_r, num_classes)

        clean_mask = w == 1
        noisy_mask = w == 0

        # ── Step 2: Train clf_net on selected subsets ─────────────────
        # CRITICAL: Freeze BN running stats before subset forward passes.
        # Without this, passing only clean or only noisy subsets through
        # the network updates the running mean/variance with biased
        # statistics (e.g., a batch of only minority-class samples).
        # Over many iterations this corrupts the BN state used at test
        # time, causing systematic prediction failures.
        #
        # Setting momentum=0 prevents running stat updates while still
        # using per-batch statistics for normalisation (gradients are
        # unaffected — only the persistent buffer is protected).
        clf_optimiser.zero_grad()
        loss_clf = torch.tensor(0.0, device=device)
        has_clf_loss = False

        freeze_bn_running_stats(clf_net)

        if clean_mask.sum() > 0:
            logits_clean = clf_net(x[clean_mask])
            loss_clf = loss_clf + clf_criterion(logits_clean, y[clean_mask])
            has_clf_loss = True

        if noisy_mask.sum() > 0:
            # Single forward pass for both pseudo-label and loss.
            # Previously this was two separate passes — wasteful and
            # causing double BN corruption on the noisy subset.
            logits_noisy = clf_net(x[noisy_mask])
            probs_noisy  = torch.softmax(logits_noisy, dim=1)
            pseudo_label = sharpen(probs_noisy, temperature).detach()
            loss_clf     = loss_clf + lambda_u * F.mse_loss(probs_noisy, pseudo_label)
            has_clf_loss = True

        restore_bn_running_stats(clf_net)

        if has_clf_loss:
            loss_clf.backward()
            clf_optimiser.step()
            clf_total += loss_clf.item() * b
            n_clf     += b

        # ── Step 3: Train ref_net on relabeled data ───────────────────
        # ref_net always trains on the full batch with relabeled targets,
        # so BN stats are fine here — no subsetting.
        ref_optimiser.zero_grad()
        loss_ref = F.binary_cross_entropy_with_logits(ref_net(x), y_hat)
        loss_ref.backward()
        ref_optimiser.step()
        ref_total += loss_ref.item() * b
        n_ref     += b

    return clf_total / max(n_clf, 1), ref_total / max(n_ref, 1)


def run_asyco_fold(
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
    seed_everything(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_labels  = pd.concat([train_noisy_df["dx"], test_clean_df["dx"]]).tolist()
    c2i, i2c    = class_mapping(all_labels)
    num_classes = len(c2i)
    class_names = [i2c[i] for i in range(num_classes)]

    train_labels = [c2i[str(dx)] for dx in train_noisy_df["dx"]]

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
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
    )

    clf_net = build_resnet(
        num_classes=num_classes, pretrained=True, depth=backbone_depth
    ).to(device)
    ref_net = build_resnet(
        num_classes=num_classes, pretrained=True, depth=backbone_depth
    ).to(device)

    clf_criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_labels, num_classes, device)
    )
    clf_optimiser = torch.optim.Adam(clf_net.parameters(), lr=lr)
    ref_optimiser = torch.optim.Adam(ref_net.parameters(), lr=lr)
    clf_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        clf_optimiser, T_max=epochs
    )
    ref_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        ref_optimiser, T_max=epochs
    )

    out_dir = make_output_dir(results_root, "asyco", tau, outer_fold, noise_type)
    config  = RunConfig(
        method="asyco",
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
            "warmup_epochs": WARMUP_EPOCHS,
            "k_toplabel":    K_TOPLABEL,
            "lambda_u":      LAMBDA_U,
            "temperature":   TEMPERATURE,
        },
    )
    logger = ResultsLogger(out_dir, config)

    print(f"\n--- AsyCo | noise={noise_type} | tau={tau:.2f} | fold={outer_fold} ---")
    print(f"    Train samples : {len(train_ds)}")
    print(f"    Test samples  : {len(test_ds)}")
    print(f"    Device        : {device} | Backbone: resnet{backbone_depth}")
    print(f"    Epochs        : {epochs} | Warmup: {WARMUP_EPOCHS}")
    print(f"    K={K_TOPLABEL} | lambda_u={LAMBDA_U} | T={TEMPERATURE}")

    for epoch in range(epochs):
        if epoch < WARMUP_EPOCHS:
            clf_loss, ref_loss = warmup_epoch(
                clf_net, ref_net, train_loader,
                clf_criterion, clf_optimiser, ref_optimiser, device,
            )
            print(f"    [Warmup] Epoch {epoch+1:03d}/{epochs} | "
                  f"clf_loss={clf_loss:.4f} | ref_loss={ref_loss:.4f}")
        else:
            clf_loss, ref_loss = asyco_epoch(
                clf_net, ref_net, train_loader,
                clf_criterion, clf_optimiser, ref_optimiser,
                device, num_classes, LAMBDA_U, TEMPERATURE, K_TOPLABEL,
            )
            print(f"    Epoch {epoch+1:03d}/{epochs} | "
                  f"clf_loss={clf_loss:.4f} | ref_loss={ref_loss:.4f}")

        logger.log_epoch(epoch + 1, clf_loss)
        clf_scheduler.step()
        ref_scheduler.step()

    clf_net.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for x, y, _ in test_loader:
            x      = x.to(device)
            logits = clf_net(x)
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