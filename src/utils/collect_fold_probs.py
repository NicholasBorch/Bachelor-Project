# src/utils/collect_fold_probs.py
#
# Collects fold softmax probabilities for ONE fold of a 10-fold split.
# Designed to run as a parallel HPC job — submit 10 jobs each with --fold 0..9.
#
# For fold F: trains ResNet-18 on all folds except F, then collects softmax
# probabilities on fold F (model never saw these samples during training).
# After all 10 jobs complete, merge_fold_probs.py assembles the full array.
#
# Usage (from repo root):
#   python -m src.utils.collect_fold_probs --fold 0

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common.io import project_root, class_mapping
from src.common.seed import seed_everything
from src.classification.dataset import HamTensorDataset
from src.classification.folds import make_folds_lesion_stratified
from src.classification.models import build_resnet
from src.classification.train import (
    get_transforms,
    make_weighted_sampler,
    compute_class_weights,
    train_one_epoch,
)
from configs.classification_default import (
    SEED,
    FOLDS,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)

# ResNet-18 is sufficient here — its role is to capture visual confusion
# patterns for noise generation, not to achieve peak classification accuracy
FOLD_PROB_EPOCHS = 30
FOLD_PROB_LR     = 1e-4


def collect_probs_for_fold(
    fold_id: int,
    df: pd.DataFrame,
    images_dir: Path,
    out_dir: Path,
) -> None:
    """
    Trains ResNet-18 on all folds except fold_id, collects softmax probs on fold_id.
    Saves probs and original dataframe indices to out_dir.
    """
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    c2i, _ = class_mapping(df["dx"].tolist())
    num_classes = len(c2i)

    # Same fold split used everywhere — guarantees leakage-free prob collection
    df_folds = make_folds_lesion_stratified(df, n_splits=FOLDS, seed=SEED)

    train_df = (
        df_folds[df_folds["fold"] != fold_id]
        .copy()
        .reset_index(drop=True)
    )

    val_mask    = df_folds["fold"] == fold_id
    val_df      = df_folds[val_mask].copy().reset_index(drop=True)
    val_indices = df_folds[val_mask].index.tolist()

    print(f"\nFold prob collection | fold {fold_id} | "
          f"train={len(train_df)} | val={len(val_df)} | device={device}")

    # ── Train ResNet-18 with class balancing ──────────────────────────────
    train_labels = [c2i[str(dx)] for dx in train_df["dx"]]
    train_ds = HamTensorDataset(
        train_df, images_dir, c2i,
        get_transforms(IMAGE_SIZE, augment=True),
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=make_weighted_sampler(train_labels),
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    model     = build_resnet(num_classes=num_classes, pretrained=True, depth=18).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=compute_class_weights(train_labels, num_classes, device)
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=FOLD_PROB_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=FOLD_PROB_EPOCHS)

    # Fixed epoch count — no early stopping so that the val split has zero
    # influence on training
    for epoch in range(FOLD_PROB_EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        scheduler.step()
        print(f"  Epoch {epoch+1:02d}/{FOLD_PROB_EPOCHS} | train_loss={loss:.4f}")

    # ── Collect softmax probabilities on the held-out fold ────────────────
    model.eval()
    val_ds = HamTensorDataset(
        val_df, images_dir, c2i,
        get_transforms(IMAGE_SIZE, augment=False),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    with torch.no_grad():
        probs = np.concatenate([
            F.softmax(model(x.to(device)), dim=1).cpu().numpy()
            for x, _, _ in val_loader
        ], axis=0)  # shape: (N_fold, C)

    out_dir.mkdir(parents=True, exist_ok=True)
    probs_path   = out_dir / f"fold_{fold_id:02d}_probs.npy"
    indices_path = out_dir / f"fold_{fold_id:02d}_indices.npy"

    np.save(probs_path,   probs)
    np.save(indices_path, np.array(val_indices, dtype=np.int64))

    print(f"\nSaved fold probs  : {probs_path}  shape={probs.shape}")
    print(f"Saved fold indices: {indices_path} n={len(val_indices)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect fold softmax probs for one fold (run in parallel on HPC)."
    )
    parser.add_argument("--fold", type=int, required=True,
                        help="Fold index to hold out (0-indexed)")
    args = parser.parse_args()

    if not (0 <= args.fold < FOLDS):
        raise ValueError(f"--fold must be in [0, {FOLDS - 1}], got {args.fold}")

    root       = project_root()
    ham_one    = root / "data" / "processed" / "HAM10000" / "one_image_per_lesion"
    meta_path  = ham_one / "HAM10000_metadata_one_per_lesion.csv"
    images_dir = ham_one / "images"
    out_dir    = root / "data" / "processed" / "HAM10000" / "fold_probs"

    print(f"=== Fold Probability Collection — Fold {args.fold} ===")
    print(f"SEED={SEED} | FOLDS={FOLDS} | EPOCHS={FOLD_PROB_EPOCHS}")

    df = pd.read_csv(meta_path)
    df["image_id"]  = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"]        = df["dx"].astype(str)

    collect_probs_for_fold(
        fold_id=args.fold,
        df=df,
        images_dir=images_dir,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()