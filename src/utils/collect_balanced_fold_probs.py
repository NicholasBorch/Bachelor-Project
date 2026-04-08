"""
collect_balanced_fold_probs.py

Trains a ResNet-18 on each fold's balanced training split and collects softmax
probabilities on the held-out fold. These OOF probabilities are used as the basis
for feature-driven IDN on the balanced dataset.

This is the balanced-dataset analogue of collect_fold_probs.py.

Key differences from the imbalanced version:
  - Loads metadata_balanced.csv (~511 samples) instead of the full metadata
  - Uses StratifiedKFold on the balanced data (new, independent fold assignments)
  - NO WeightedRandomSampler — classes are already equal
  - Saves to data/processed/HAM10000/fold_probs_balanced/

Usage (one job per fold on HPC):
    python -m src.utils.collect_balanced_fold_probs --fold 0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from src.common.io import project_root, class_mapping
from src.common.seed import seed_everything
from src.classification.dataset import HamTensorDataset
from src.classification.models import build_resnet
from src.classification.train import get_transforms

# ── Config ────────────────────────────────────────────────────────────────────
SEED         = 10
FOLDS        = 10
CLASS_COL    = "dx"
CLASS_NAMES  = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMAGE_SIZE   = 224
BATCH_SIZE   = 32       # smaller — only ~460 training samples on balanced data
NUM_WORKERS  = 2
PIN_MEMORY   = True
EPOCHS_OOF   = 30       # ResNet-18 training epochs for OOF collection
LR_OOF       = 1e-4

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = project_root()
METADATA_IN = ROOT / "data/processed/HAM10000/one_image_per_lesion/metadata_balanced.csv"
IMAGES_DIR  = ROOT / "data/processed/HAM10000/one_image_per_lesion/images"
PROBS_DIR   = ROOT / "data/processed/HAM10000/fold_probs_balanced"


def collect_fold_probs(fold_id: int, device: torch.device) -> None:
    fold_seed = SEED * 10_000 + fold_id
    seed_everything(fold_seed)

    df = pd.read_csv(METADATA_IN)
    c2i, _ = class_mapping(CLASS_NAMES)
    num_classes = len(CLASS_NAMES)

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(df, df[CLASS_COL]))

    train_idx, val_idx = splits[fold_id]
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df   = df.iloc[val_idx].copy().reset_index(drop=True)

    print(f"Fold {fold_id:02d}: {len(train_df)} train / {len(val_df)} val samples")

    # ── Transforms ────────────────────────────────────────────────────────────
    train_transform = get_transforms(IMAGE_SIZE, augment=True)
    val_transform   = get_transforms(IMAGE_SIZE, augment=False)

    # ── Datasets ──────────────────────────────────────────────────────────────
    # HamTensorDataset takes positional args: (df, images_dir, c2i, tfm)
    train_ds = HamTensorDataset(train_df, IMAGES_DIR, c2i, train_transform)
    val_ds   = HamTensorDataset(val_df,   IMAGES_DIR, c2i, val_transform)

    # Balanced data → plain shuffle, no WeightedRandomSampler
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_resnet(num_classes=num_classes, pretrained=True, depth=18)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_OOF)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_OOF)

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(EPOCHS_OOF):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = running_loss / len(train_loader)
            print(f"  Epoch {epoch + 1:3d}/{EPOCHS_OOF} | loss={avg_loss:.4f}")

    # ── Collect OOF probabilities ──────────────────────────────────────────────
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    probs_array = np.concatenate(all_probs, axis=0)   # (n_val, 7)

    # Map to global (balanced dataset) row positions.
    # val_idx from StratifiedKFold gives positional indices into the balanced df.
    # Since val_loader has shuffle=False, probs_array[i] corresponds to val_df.iloc[i]
    # which is df.iloc[val_idx[i]]. So val_idx IS the global index array.
    global_indices = np.array(val_idx, dtype=np.int64)

    print(f"  Collected probs: {probs_array.shape}  indices: {global_indices.shape}")
    assert probs_array.shape == (len(val_df), num_classes), \
        f"Unexpected probs shape: {probs_array.shape}"

    # ── Save ──────────────────────────────────────────────────────────────────
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROBS_DIR / f"fold_{fold_id:02d}_probs.npy",   probs_array)
    np.save(PROBS_DIR / f"fold_{fold_id:02d}_indices.npy", global_indices)
    print(f"  Saved → {PROBS_DIR / f'fold_{fold_id:02d}_probs.npy'}")
    print(f"  Saved → {PROBS_DIR / f'fold_{fold_id:02d}_indices.npy'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect OOF softmax probabilities on balanced HAM10000."
    )
    parser.add_argument("--fold", type=int, required=True,
                        help="Fold index (0-9)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g. 'cuda' or 'cpu'. Auto-detected if omitted.")
    args = parser.parse_args()

    assert 0 <= args.fold < FOLDS, f"fold must be 0–{FOLDS - 1}, got {args.fold}"

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Running OOF collection for fold {args.fold} on balanced dataset.")

    collect_fold_probs(fold_id=args.fold, device=device)