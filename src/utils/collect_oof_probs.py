# src/utils/collect_oof_probs.py
#
# Collects out-of-fold (OOF) softmax probabilities for ONE fold of a 5-fold split.
# Designed to run as a parallel HPC job — submit 5 jobs each with --fold 0..4.
#
# Design rationale:
#   Standard IDN uses random projections to determine flip targets.
#   On imbalanced datasets like HAM10000, class size distorts the projection space
#   and produces near-deterministic, tau-invariant flip targets.
#   Feature-driven IDN instead uses per-sample softmax probabilities from a model
#   that has never seen that sample — grounding flip targets in visual similarity.
#
#   The OOF design ensures leakage-free probability collection:
#     - For fold F: train ResNet-18 on all folds except F
#     - Collect softmax probs on fold F (model never saw these samples)
#     - After all 5 jobs complete, merge_oof_probs.py assembles the full array
#
#   IMPORTANT: uses the same fold split (seed + function) as the evaluation CV,
#   so every training sample in eval fold F has probs from a model trained
#   without it — the leakage guarantee holds end-to-end.
#
# Usage (from repo root):
#   python -m src.utils.collect_oof_probs --fold 0
#
# Output:
#   data/processed/HAM10000/oof_probs/fold_00_probs.npy    shape (N_fold, C)
#   data/processed/HAM10000/oof_probs/fold_00_indices.npy  original df indices

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
from src.classification.folds import make_outer_folds_lesion_stratified
from src.classification.models import build_resnet
from src.classification.train import (
    get_transforms,
    make_weighted_sampler,
    compute_class_weights,
    train_one_epoch,
)
from configs.classification_default import (
    SEED,
    OUTER_FOLDS,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)

# ResNet-18 is used here (lighter than ResNet-50) — its role is to capture
# visual confusion patterns, not to achieve peak classification accuracy
OOF_EPOCHS = 30
OOF_LR     = 1e-4


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

    # ── Build the same fold split used everywhere in the pipeline ─────────
    # Using identical seed + function guarantees the OOF split matches the
    # evaluation CV split, so leakage cannot occur between the two pipelines.
    df_folds = make_outer_folds_lesion_stratified(df, n_splits=OUTER_FOLDS, seed=SEED)

    # Training data: all folds except fold_id
    train_df = (
        df_folds[df_folds["outer_fold"] != fold_id]
        .copy()
        .reset_index(drop=True)
    )

    # Validation data: fold_id only — model never sees these during training
    val_mask    = df_folds["outer_fold"] == fold_id
    val_df      = df_folds[val_mask].copy().reset_index(drop=True)
    val_indices = df_folds[val_mask].index.tolist()  # positions in original df

    print(f"\nOOF collection | fold {fold_id} | "
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
    optimiser = torch.optim.Adam(model.parameters(), lr=OOF_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=OOF_EPOCHS)

    # Fixed epoch count — no early stopping so that the val split has zero
    # influence on training. Inner test split is purely for prob collection.
    for epoch in range(OOF_EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimiser, device)
        scheduler.step()
        print(f"  Epoch {epoch+1:02d}/{OOF_EPOCHS} | train_loss={loss:.4f}")

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

    # ── Save probs and their original indices ─────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    probs_path   = out_dir / f"fold_{fold_id:02d}_probs.npy"
    indices_path = out_dir / f"fold_{fold_id:02d}_indices.npy"

    np.save(probs_path,   probs)
    np.save(indices_path, np.array(val_indices, dtype=np.int64))

    print(f"\nSaved OOF probs  : {probs_path}  shape={probs.shape}")
    print(f"Saved OOF indices: {indices_path} n={len(val_indices)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect OOF softmax probs for one fold (run 5 in parallel on HPC)."
    )
    parser.add_argument("--fold", type=int, required=True,
                        help="Fold index to hold out (0-indexed, 0 to OUTER_FOLDS-1)")
    args = parser.parse_args()

    root       = project_root()
    ham_one    = root / "data" / "processed" / "HAM10000" / "one_image_per_lesion"
    meta_path  = ham_one / "HAM10000_metadata_one_per_lesion.csv"
    images_dir = ham_one / "images"
    out_dir    = root / "data" / "processed" / "HAM10000" / "oof_probs"

    print(f"=== OOF Probability Collection — Fold {args.fold} ===")
    print(f"SEED={SEED} | OUTER_FOLDS={OUTER_FOLDS} | OOF_EPOCHS={OOF_EPOCHS}")

    df = pd.read_csv(meta_path)
    df["image_id"]  = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"]        = df["dx"].astype(str)

    if not (0 <= args.fold < OUTER_FOLDS):
        raise ValueError(f"--fold must be in [0, {OUTER_FOLDS - 1}], got {args.fold}")

    collect_probs_for_fold(
        fold_id=args.fold,
        df=df,
        images_dir=images_dir,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()