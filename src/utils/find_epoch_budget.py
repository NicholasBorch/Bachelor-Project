# src/utils/find_epoch_budget.py
#
# Preliminary experiment to determine the fixed epoch budget.
# Runs the clean Baseline (tau=0.0) on all 10 CV folds with a temporary
# stratified validation split (15%) carved from the training partition.
# Logs training loss and validation loss per epoch, then produces an
# averaged curve across folds to identify the convergence point.
#
# This script is NOT part of the main evaluation pipeline — it is a
# one-off hyperparameter selection step. The chosen epoch count is then
# used for the real experiments with the original unmodified fold splits.
#
# Usage (from repo root):
#   python -m src.utils.find_epoch_budget [--epochs 100] [--val_frac 0.15]
#
# Outputs:
#   results/HAM10000/epoch_selection/
#     curves_per_fold.csv          — per-fold, per-epoch train/val loss
#     curves_averaged.csv          — mean ± std across folds
#     epoch_selection_curves.png   — training + validation loss plot

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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
    """Compute average validation loss over one full pass."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    return total_loss / max(total_samples, 1)


def run_fold(
    fold_id: int,
    df: pd.DataFrame,
    images_dir: Path,
    val_frac: float,
    epochs: int,
    device: torch.device,
) -> pd.DataFrame:
    """Train clean Baseline on one fold with a temporary val split."""

    # Reproduce the exact same fold assignments as the main pipeline
    df_folds = make_folds_lesion_stratified(df, n_splits=FOLDS, seed=SEED)

    train_df = df_folds[df_folds["fold"] != fold_id].copy().reset_index(drop=True)

    # Use clean labels (dx column is already clean at tau=0.0)
    all_labels = df["dx"].unique().tolist()
    c2i, i2c = class_mapping(all_labels)
    num_classes = len(c2i)

    # --- Stratified train/val split within the training partition ---
    train_labels_str = train_df["dx"].values
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=SEED * 10_000 + fold_id
    )
    train_idx, val_idx = next(splitter.split(train_df, train_labels_str))

    train_sub_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_sub_df = train_df.iloc[val_idx].reset_index(drop=True)

    print(f"\n  Fold {fold_id} | train={len(train_sub_df)} | val={len(val_sub_df)}")

    # --- Datasets and loaders ---
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

    # --- Model, loss, optimizer (mirrors baseline.py exactly) ---
    model = build_resnet(
        num_classes=num_classes, pretrained=True, depth=BACKBONE_DEPTH
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_labels_int, num_classes, device)
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    # --- Training loop ---
    records = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        records.append({
            "fold": fold_id,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d}/{epochs} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    return pd.DataFrame(records)


def plot_curves(avg_df: pd.DataFrame, out_path: Path) -> None:
    """Plot averaged training + validation loss with std bands."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = avg_df["epoch"].values

    # Training loss
    ax.plot(epochs, avg_df["train_loss_mean"], label="Training Loss", color="#1f77b4", linewidth=2)
    ax.fill_between(
        epochs,
        avg_df["train_loss_mean"] - avg_df["train_loss_std"],
        avg_df["train_loss_mean"] + avg_df["train_loss_std"],
        alpha=0.2, color="#1f77b4",
    )

    # Validation loss
    ax.plot(epochs, avg_df["val_loss_mean"], label="Validation Loss", color="#d62728", linewidth=2)
    ax.fill_between(
        epochs,
        avg_df["val_loss_mean"] - avg_df["val_loss_std"],
        avg_df["val_loss_mean"] + avg_df["val_loss_std"],
        alpha=0.2, color="#d62728",
    )

    # Mark the epoch with minimum validation loss
    best_epoch = avg_df.loc[avg_df["val_loss_mean"].idxmin(), "epoch"]
    best_val = avg_df["val_loss_mean"].min()
    ax.axvline(x=best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.scatter([best_epoch], [best_val], color="#2ca02c", s=80, zorder=5)
    ax.annotate(
        f"Min val loss: epoch {best_epoch}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 3, best_val + 0.05),
        fontsize=10,
        color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(
        "Epoch Selection — Clean Baseline (mean ± std across 10 folds)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the optimal fixed epoch budget using clean Baseline + val split"
    )
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs to train (default: 100)")
    parser.add_argument("--val_frac", type=float, default=0.15,
                        help="Fraction of training data to hold out for validation (default: 0.15)")
    args = parser.parse_args()

    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = project_root()
    ham_one = root / "data" / "processed" / "HAM10000" / "one_image_per_lesion"
    meta_path = ham_one / "HAM10000_metadata_one_per_lesion.csv"
    images_dir = ham_one / "images"
    out_dir = root / "results" / "HAM10000" / "epoch_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Epoch Budget Selection — Clean Baseline with Validation Split")
    print(f"Epochs: {args.epochs} | Val fraction: {args.val_frac}")
    print(f"Backbone: resnet{BACKBONE_DEPTH} | LR: {LR} | Batch: {BATCH_SIZE}")
    print(f"Device: {device}")
    print("=" * 60)

    df = pd.read_csv(meta_path)
    df["image_id"] = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"] = df["dx"].astype(str)

    # Run all folds
    all_fold_dfs = []
    for fold_id in range(FOLDS):
        fold_df = run_fold(
            fold_id=fold_id,
            df=df,
            images_dir=images_dir,
            val_frac=args.val_frac,
            epochs=args.epochs,
            device=device,
        )
        all_fold_dfs.append(fold_df)

    # Combine and aggregate
    curves_df = pd.concat(all_fold_dfs, ignore_index=True)
    curves_df.to_csv(out_dir / "curves_per_fold.csv", index=False)

    avg_df = (
        curves_df
        .groupby("epoch")
        .agg(
            train_loss_mean=("train_loss", "mean"),
            train_loss_std=("train_loss", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_std=("val_loss", "std"),
        )
        .reset_index()
    )
    avg_df.to_csv(out_dir / "curves_averaged.csv", index=False)

    # Report
    best_epoch = int(avg_df.loc[avg_df["val_loss_mean"].idxmin(), "epoch"])
    best_val = avg_df["val_loss_mean"].min()

    print(f"\n{'=' * 60}")
    print(f"Results")
    print(f"  Best epoch (min avg val loss): {best_epoch}")
    print(f"  Val loss at best epoch:        {best_val:.4f}")
    print(f"  Curves saved: {out_dir / 'curves_per_fold.csv'}")
    print(f"  Averaged:     {out_dir / 'curves_averaged.csv'}")
    print(f"{'=' * 60}")

    # Plot
    plot_curves(avg_df, out_dir / "epoch_selection_curves.png")


if __name__ == "__main__":
    main()