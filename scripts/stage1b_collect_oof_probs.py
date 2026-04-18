"""Stage 1b: collect out-of-fold softmax probabilities for feature-driven IDN.

For a given fold N: train ResNet-18 on folds != N using CLEAN labels only,
then run inference on fold N to get (n_held_out, 7) softmax predictions.

OOF training protocol (PROJECT_DOCUMENTATION §6 Stage 1b / §9):
    - Backbone: ResNet-18 pretrained on ImageNet
    - Optimizer: Adam, lr=1e-4 (no momentum betas override, no weight decay)
    - Schedule: CosineAnnealingLR over 30 epochs
    - Epoch budget: 30 for both balanced and imbalanced
    - Sampling: WeightedRandomSampler + class-weighted CE for imbalanced;
                standard shuffle + unweighted CE for balanced
    - Seed: per-fold `global_seed * 10_000 + fold_id`

This protocol is a fixed preprocessing step — it is NOT tuned for HAM10000.
The point of Stage 1b is to produce reasonable OOF confusion signals for
feature-driven IDN, not to produce a state-of-the-art classifier.

Output: data/processed/HAM10000/cv_folds/{dataset}/oof_probs/fold_{NN}.npy
        data/processed/HAM10000/cv_folds/{dataset}/oof_probs/fold_{NN}_ids.csv

This script is parallelizable across folds via HPC. Run once per fold.
After all 10 folds complete, run stage1b_merge_oof_probs.py.

Run: python -m scripts.stage1b_collect_oof_probs --dataset imbalanced --fold 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.folds import load_fold_assignments, split_train_test_by_fold
from src.data.ham10000 import NUM_CLASSES, HamDataset, class_to_index
from src.data.transforms import get_test_transforms, get_train_transforms
from src.models.resnet import build_resnet
from src.utils.io import ensure_dir, load_config, project_root
from src.utils.manifest import write_manifest
from src.utils.seed import fold_seed, seed_everything

# ── OOF training protocol — FIXED, see module docstring ──────────────────────
OOF_EPOCHS = 30
OOF_LR = 1e-4


def _make_sampler(labels_idx: np.ndarray, dataset_kind: str):
    """WeightedRandomSampler for imbalanced, None (→ shuffle) for balanced."""
    if dataset_kind == "balanced":
        return None
    counts = np.bincount(labels_idx, minlength=NUM_CLASSES).astype(np.float64)
    weights_per_class = 1.0 / np.maximum(counts, 1.0)
    sample_weights = weights_per_class[labels_idx]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels_idx),
        replacement=True,
    )


def _class_weights(labels_idx: np.ndarray, dataset_kind: str, device) -> torch.Tensor | None:
    """Inverse-frequency class weights (mean=1) for imbalanced, None for balanced."""
    if dataset_kind == "balanced":
        return None
    counts = np.bincount(labels_idx, minlength=NUM_CLASSES).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w / w.mean()  # normalize so mean weight = 1
    return torch.as_tensor(w, dtype=torch.float32, device=device)


def main(args: argparse.Namespace) -> int:
    cfg = load_config("base.yaml", f"data/{args.dataset}.yaml")
    root = project_root()

    # ------- Prerequisites ---------
    fa_path = root / cfg["paths"]["cv_folds"] / args.dataset / "fold_assignments.csv"
    folds_df = load_fold_assignments(fa_path)

    metadata_path = (
        root / cfg["paths"]["data_processed"]
        / "one_image_per_lesion"
        / cfg["data"]["metadata_file"]
    )
    metadata = pd.read_csv(metadata_path)
    images_dir = root / cfg["paths"]["images"]

    train_df, test_df = split_train_test_by_fold(metadata, folds_df, test_fold=args.fold)
    print(f"[stage1b] fold {args.fold}: train={len(train_df)}, held-out={len(test_df)}")

    # ------- Seed ---------
    seed_everything(fold_seed(cfg["seed"], args.fold), deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    # ------- Datasets / Loaders ---------
    train_ds = HamDataset(
        train_df, images_dir=images_dir, transform=get_train_transforms(cfg["image_size"]),
    )
    heldout_ds = HamDataset(
        test_df, images_dir=images_dir, transform=get_test_transforms(cfg["image_size"]),
    )
    batch_size = cfg["data"]["batch_size"]
    train_labels_idx = np.array([class_to_index(c) for c in train_df["dx"]])
    sampler = _make_sampler(train_labels_idx, cfg["data"]["name"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        persistent_workers=cfg["num_workers"] > 0,
    )
    heldout_loader = DataLoader(
        heldout_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        persistent_workers=cfg["num_workers"] > 0,
    )

    # ------- Model / Optimizer (ResNet-18, Adam, cosine, 30 epochs) ----------
    # NOTE: This is the locked OOF protocol — see module docstring.
    # The OOF model is a preprocessing step for feature-driven IDN, not a
    # production classifier. Do not tune these hyperparameters; any change
    # changes the injected noise and therefore the whole experiment.
    model = build_resnet(num_classes=NUM_CLASSES, depth=18, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=OOF_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=OOF_EPOCHS)
    class_weights = _class_weights(train_labels_idx, cfg["data"]["name"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cfg.get("mixed_precision", True) and device.type == "cuda",
    )

    # ------- Train ---------
    print(
        f"[stage1b] training ResNet-18 with Adam(lr={OOF_LR}) for "
        f"{OOF_EPOCHS} epochs on fold {args.fold}"
    )
    for epoch in range(1, OOF_EPOCHS + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1 or epoch == OOF_EPOCHS:
            print(
                f"[stage1b] epoch {epoch:3d}/{OOF_EPOCHS}  "
                f"avg_loss={total_loss / max(n_batches, 1):.4f}"
            )

    # ------- Held-out inference ---------
    print(f"[stage1b] running inference on held-out fold {args.fold}")
    model.eval()
    probs_chunks = []
    ids_chunks = []
    with torch.no_grad():
        for images, _, idx in heldout_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_chunks.append(probs)
            ids_chunks.extend([test_df.iloc[int(i)]["image_id"] for i in idx.tolist()])

    all_probs = np.concatenate(probs_chunks, axis=0)
    assert all_probs.shape == (len(test_df), NUM_CLASSES)

    # ------- Save ---------
    out_dir = ensure_dir(root / cfg["paths"]["cv_folds"] / args.dataset / "oof_probs")
    npy_path = out_dir / f"fold_{args.fold:02d}.npy"
    ids_path = out_dir / f"fold_{args.fold:02d}_ids.csv"
    np.save(npy_path, all_probs.astype(np.float32))
    pd.DataFrame({"image_id": ids_chunks}).to_csv(ids_path, index=False)
    print(f"[stage1b] wrote {npy_path} (shape {all_probs.shape})")
    print(f"[stage1b] wrote {ids_path}")

    # ------- Manifest ---------
    manifest_path = root / cfg["paths"]["manifests"] / f"stage1b_{args.dataset}_fold{args.fold:02d}.json"
    write_manifest(
        manifest_path,
        stage="stage1b",
        params={
            "dataset": args.dataset,
            "fold": args.fold,
            "epochs": OOF_EPOCHS,
            "optimizer": "adam",
            "lr": OOF_LR,
        },
        outputs=[
            str(npy_path.relative_to(root)),
            str(ids_path.relative_to(root)),
        ],
        extra={"n_held_out": len(test_df)},
    )
    print("[stage1b] DONE")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1b: OOF probability collection (per fold)")
    p.add_argument("--dataset", required=True, choices=["balanced", "imbalanced"])
    p.add_argument("--fold", type=int, required=True, help="fold id (0..9)")
    sys.exit(main(p.parse_args()))
