# src/utils/find_epoch_budget.py
#
# Epoch budget selection for ONE fold.
# Loads the clean training split directly from cv_normalized/clean/fold_XX/
# to guarantee identical fold assignments as the main classification pipeline.
# Carves a stratified 15% validation split from train_clean.csv, then trains
# a clean Baseline tracking both loss and balanced accuracy per epoch.
#
# Designed to run as a parallel HPC job — submit one job per fold.
# After all 10 fold jobs complete, run aggregate_epoch_budget.py locally.
#
# Usage:
#   python -m src.utils.find_epoch_budget --fold 0 [--epochs 100] [--val_frac 0.15]

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

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
from configs.classification_default import (
    SEED,
    FOLDS,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    LR,
    BACKBONE_DEPTH,
    PIN_MEMORY,
)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (mean_loss, balanced_accuracy) on the given loader."""
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_true, all_pred = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            total_loss    += loss.item() * x.size(0)
            total_samples += x.size(0)
            all_true.extend(y.cpu().numpy())
            all_pred.extend(logits.argmax(dim=1).cpu().numpy())

    mean_loss = total_loss / max(total_samples, 1)
    bal_acc   = float(balanced_accuracy_score(all_true, all_pred))
    return mean_loss, bal_acc


def run_fold(
    fold_id: int,
    val_frac: float,
    epochs: int,
    device: torch.device,
    out_dir: Path,
) -> None:
    seed_everything(SEED * 10_000 + fold_id)

    # ── Load clean fold data from cv_normalized (same splits as main pipeline)
    root      = project_root()
    fold_dir  = (root / "data" / "processed" / "HAM10000"
                 / "cv_normalized" / "clean" / f"fold_{fold_id:02d}")
    train_path = fold_dir / "train_clean.csv"
    test_path  = fold_dir / "test_clean.csv"
    images_dir = root / "data" / "processed" / "HAM10000" / "one_image_per_lesion" / "images"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Clean fold data not found at {train_path}.\n"
            "Make sure cv_normalized has been prepared."
        )

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Use dx_clean so we are always training on clean labels
    train_df["dx"] = train_df["dx_clean"] if "dx_clean" in train_df.columns else train_df["dx"]

    all_labels  = pd.concat([train_df["dx"], test_df["dx"]]).tolist()
    c2i, _      = class_mapping(all_labels)
    num_classes = len(c2i)

    # ── Stratified val split from training data ────────────────────────────
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=SEED * 10_000 + fold_id
    )
    train_idx, val_idx = next(splitter.split(train_df, train_df["dx"].values))

    train_sub = train_df.iloc[train_idx].reset_index(drop=True)
    val_sub   = train_df.iloc[val_idx].reset_index(drop=True)

    print(f"\nFold {fold_id} | train={len(train_sub)} | val={len(val_sub)} "
          f"| test={len(test_df)}")

    # ── Datasets and loaders ───────────────────────────────────────────────
    train_labels_int = [c2i[str(dx)] for dx in train_sub["dx"]]

    train_ds = HamTensorDataset(
        train_sub, images_dir, c2i, get_transforms(IMAGE_SIZE, augment=True)
    )
    val_ds = HamTensorDataset(
        val_sub, images_dir, c2i, get_transforms(IMAGE_SIZE, augment=False)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=make_weighted_sampler(train_labels_int),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # ── Model, loss, optimiser ─────────────────────────────────────────────
    model = build_resnet(
        num_classes=num_classes, pretrained=True, depth=BACKBONE_DEPTH
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_labels_int, num_classes, device)
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    # ── Training loop ──────────────────────────────────────────────────────
    records = []
    for epoch in range(1, epochs + 1):
        train_loss              = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_loss, val_bal_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        records.append({
            "fold":        fold_id,
            "epoch":       epoch,
            "train_loss":  train_loss,
            "val_loss":    val_loss,
            "val_bal_acc": val_bal_acc,
        })

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:03d}/{epochs} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_bal_acc={val_bal_acc:.4f}")

    # ── Save — always overwrite ────────────────────────────────────────────
    out_path = out_dir / f"fold_{fold_id:02d}_curves.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",     type=int,   required=True)
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--val_frac", type=float, default=0.15)
    args = parser.parse_args()

    if not (0 <= args.fold < FOLDS):
        raise ValueError(f"--fold must be in [0, {FOLDS - 1}], got {args.fold}")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = project_root() / "results" / "HAM10000" / "epoch_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Epoch Budget Selection — Fold {args.fold} ===")
    print(f"Epochs: {args.epochs} | Val fraction: {args.val_frac}")
    print(f"Backbone: resnet{BACKBONE_DEPTH} | LR: {LR} | Device: {device}")
    print(f"Loading from: cv_normalized/clean/fold_{args.fold:02d}/")

    run_fold(
        fold_id=args.fold,
        val_frac=args.val_frac,
        epochs=args.epochs,
        device=device,
        out_dir=out_dir,
    )

    print(f"\nDone — fold {args.fold} written to {out_dir}")


if __name__ == "__main__":
    main()