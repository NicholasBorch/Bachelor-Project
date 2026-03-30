# src/utils/find_epoch_budget.py
#
# Preliminary experiment to determine the fixed epoch budget.
# Runs the clean Baseline (tau=0.0) on ONE CV fold with a temporary
# stratified validation split (15%) carved from the training partition.
# Designed to run as a parallel HPC job — submit one job per fold.
#
# After all 10 fold jobs complete, run aggregate_epoch_budget.py to
# combine results and produce the averaged curve and plot.
#
# Usage (from repo root):
#   python -m src.utils.find_epoch_budget --fold 0 [--epochs 100] [--val_frac 0.15]

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from src.classification.dataset import HamTensorDataset
from src.classification.models import build_resnet
from src.classification.train import (
    compute_class_weights,
    get_transforms,
    make_weighted_sampler,
    train_one_epoch,
)
from src.common.io import class_mapping, project_root
from src.common.seed import seed_everything
from src.classification.folds import make_folds_lesion_stratified
from configs.classification_default import (
    SEED,
    FOLDS,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    LR,
    BACKBONE_DEPTH,
)


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            total_loss   += loss.item() * x.size(0)
            total_samples += x.size(0)
    return total_loss / max(total_samples, 1)


def run_fold(
    fold_id: int,
    df: pd.DataFrame,
    images_dir: Path,
    val_frac: float,
    epochs: int,
    device: torch.device,
    out_dir: Path,
) -> None:
    seed_everything(SEED * 10_000 + fold_id)

    df_folds = make_folds_lesion_stratified(df, n_splits=FOLDS, seed=SEED)
    train_df = df_folds[df_folds["fold"] != fold_id].copy().reset_index(drop=True)

    all_labels  = df["dx"].unique().tolist()
    c2i, _      = class_mapping(all_labels)
    num_classes = len(c2i)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=SEED * 10_000 + fold_id
    )
    train_idx, val_idx = next(splitter.split(train_df, train_df["dx"].values))

    train_sub_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_sub_df   = train_df.iloc[val_idx].reset_index(drop=True)

    print(f"\nFold {fold_id} | train={len(train_sub_df)} | val={len(val_sub_df)}")

    train_ds = HamTensorDataset(
        train_sub_df, images_dir, c2i, get_transforms(IMAGE_SIZE, augment=True)
    )
    val_ds = HamTensorDataset(
        val_sub_df, images_dir, c2i, get_transforms(IMAGE_SIZE, augment=False)
    )

    train_labels_int = [c2i[str(dx)] for dx in train_sub_df["dx"]]

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=make_weighted_sampler(train_labels_int),
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = build_resnet(
        num_classes=num_classes, pretrained=True, depth=BACKBONE_DEPTH
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_labels_int, num_classes, device)
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    records = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_loss   = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        records.append({
            "fold":       fold_id,
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
        })
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d}/{epochs} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    # Save this fold's curves immediately
    fold_path = out_dir / f"fold_{fold_id:02d}_curves.csv"
    pd.DataFrame(records).to_csv(fold_path, index=False)
    print(f"\n  Saved: {fold_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",     type=int,   required=True,
                        help="Fold index (0-indexed)")
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    if not (0 <= args.fold < FOLDS):
        raise ValueError(f"--fold must be in [0, {FOLDS - 1}], got {args.fold}")

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root       = project_root()
    ham_one    = root / "data" / "processed" / "HAM10000" / "one_image_per_lesion"
    meta_path  = ham_one / "HAM10000_metadata_one_per_lesion.csv"
    images_dir = ham_one / "images"
    out_dir    = root / "results" / "HAM10000" / "epoch_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Epoch Budget Selection — Fold {args.fold} ===")
    print(f"Epochs: {args.epochs} | Val fraction: {args.val_frac}")
    print(f"Backbone: resnet{BACKBONE_DEPTH} | LR: {LR} | Device: {device}")

    df = pd.read_csv(meta_path)
    df["image_id"]  = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"]        = df["dx"].astype(str)

    run_fold(
        fold_id=args.fold,
        df=df,
        images_dir=images_dir,
        val_frac=args.val_frac,
        epochs=args.epochs,
        device=device,
        out_dir=out_dir,
    )

    print(f"\nDone — fold {args.fold} written to {out_dir}")


if __name__ == "__main__":
    main()