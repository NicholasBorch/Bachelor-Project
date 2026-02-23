#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HAM10000 — Confidence-based Instance-Dependent Noise (IDN) using nested CV (OOF uncertainty)
========================================================================================

Purpose (for our bachelor project):
----------------------------------
We want a *realistic* instance-dependent label-noise process for HAM10000:
- For each outer 10-fold CV split (used later for model evaluation):
    * Keep the outer test fold CLEAN (never altered).
    * Inject noise ONLY into the outer training fold (90%).
- Noise is "instance-dependent" by construction:
    * Train a teacher model and measure uncertainty.
    * Flip the TOP `NOISE_RATE` most uncertain training samples.
    * Flip each selected sample to the model's best alternative class (highest probability
      among classes excluding the current/clean label).

Why nested CV (OOF scoring)?
----------------------------
If we train the teacher on the entire outer-train split and score uncertainty on the same data,
the model can be overconfident due to memorization. This makes uncertainty less reliable.
Instead, we do *inner CV inside the outer training split* and compute *out-of-fold (OOF)*
uncertainty for every training sample:
- Each sample's uncertainty is produced by a teacher model that did NOT train on that sample.
This yields more honest "hard/ambiguous" ranking -> more defensible IDN simulation.

What this script produces:
--------------------------
OUT_ROOT/
  fold_assignments.csv                    # fixed outer fold id per sample (reproducible)
  fold_00/
    train_clean.csv                       # dx = clean labels (training part of fold)
    train_noisy.csv                       # dx = noisy labels + dx_clean/dx_noisy/uncertainty
    test_clean.csv                        # dx = clean labels (outer test)
    noise_report.json                     # summary + flip confusion
  fold_01/
    ...
  ...

Repository usage:
-----------------
- Put this in e.g. `scripts/make_ham_idn_nestedcv.py`
- Set variables in the CONFIG section below.
- Run directly in VS Code (no CLI required).
- Progress will show in the terminal via tqdm.

Assumptions about the dataset:
------------------------------
DATA_ROOT/
  images/                               # <image_id>.jpg
  HAM10000_metadata_modified.csv         # columns: image_id, lesion_id, dx

Leakage safety (lesion-level grouping):
---------------------------------------
Even if you *think* you have 1 image per lesion, this script is group-safe by default:
- Outer folds are created on unique lesion_id (stratified by dx at lesion-level).
- This prevents the same lesion appearing in both train and test within a fold.

Notes / Practical considerations:
---------------------------------
- This runs many teacher trainings:
    OUTER_FOLDS * INNER_FOLDS trainings (e.g. 10*10 = 100).
  Keep TEACHER_EPOCHS small (e.g. 2–5) and consider ResNet34/50 depending on GPU budget.
- We use ImageNet pretrained weights by default (recommended).
- For speed/stability, we use AMP mixed precision on CUDA by default.

"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


# ============================================================
# CONFIG — edit these variables and run the script in VS Code
# ============================================================

# --- Paths ---
DATA_ROOT = Path("Data/HAM10000_modified")  # folder containing images/ and metadata CSV
METADATA_CSV = "HAM10000_metadata_modified.csv"
IMAGES_DIR = "images"

OUT_ROOT = Path("Data/HAM10000_noisy_confidence_idn_nestedcv")

# --- Cross-validation design ---
OUTER_FOLDS = 10                # evaluation folds
INNER_FOLDS = 10                # OOF scoring folds inside outer-train

SEED = 42

# --- Noise design ---
NOISE_RATE = 0.20               # flip top 20% most uncertain in outer-train
SCORE_TYPE = "p_true"           # {"p_true", "max_softmax", "entropy"}

# Flip rule:
# For each selected sample, new label = teacher argmax over classes excluding the true label.
# (This is the "most likely alternative" under the teacher.)

# --- Teacher model & training ---
ARCH = "resnet50"               # {"resnet18","resnet34","resnet50","resnet101"}
PRETRAINED = True               # ImageNet weights

TEACHER_EPOCHS = 3              # keep small: this script trains many teachers
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4

# --- Runtime ---
NUM_WORKERS = 2                 # increase if your machine can handle it
USE_AMP = True                  # mixed precision on CUDA
PIN_MEMORY = True

# --- Optional: set to True to dump per-fold teacher OOF score caches (debugging) ---
SAVE_OOF_DEBUG = False


# ============================================================
# Utilities
# ============================================================

def seed_everything(seed: int) -> None:
    """Make randomness reproducible across python / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_mapping(classes: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Map string labels to indices and back (stable ordering)."""
    classes = sorted(list(classes))
    c2i = {c: i for i, c in enumerate(classes)}
    i2c = {i: c for c, i in c2i.items()}
    return c2i, i2c


# ============================================================
# Dataset
# ============================================================

class HamDataset(Dataset):
    """
    Torch dataset returning (image_tensor, label_idx, image_id_str).
    """
    def __init__(self, df: pd.DataFrame, images_dir: Path, c2i: Dict[str, int], transform):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.c2i = c2i
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        label_str = row["dx"]
        y = self.c2i[label_str]

        img_path = self.images_dir / f"{image_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, y, image_id


# ============================================================
# Model building
# ============================================================

def build_resnet(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    arch = arch.lower().strip()

    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif arch == "resnet101":
        m = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unknown ARCH={arch}")

    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


# ============================================================
# Training + scoring
# ============================================================

def train_teacher(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    desc: str,
) -> None:
    """
    Simple warmup training for the teacher.
    We keep it lightweight (few epochs) since we train many teachers in nested CV.
    """
    model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.startswith("cuda")))

    for ep in range(1, epochs + 1):
        running_loss = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"{desc} | ep {ep}/{epochs}", leave=False)
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += float(loss.item()) * x.size(0)
            n += x.size(0)

            pbar.set_postfix(loss=(running_loss / max(n, 1)))

    model.eval()


@torch.no_grad()
def score_uncertainty_and_altlabel(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    score_type: str,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    For each image in loader:
      - uncertainty score (higher = more uncertain)
      - alt label target = model argmax among classes excluding the true label
    """
    model.eval()
    scores: Dict[str, float] = {}
    alt_targets: Dict[str, int] = {}

    for x, y, image_ids in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        # --- Uncertainty ---
        if score_type == "p_true":
            p_true = probs.gather(1, y.view(-1, 1)).squeeze(1)
            uncertainty = 1.0 - p_true
        elif score_type == "max_softmax":
            p_max, _ = probs.max(dim=1)
            uncertainty = 1.0 - p_max
        elif score_type == "entropy":
            eps = 1e-8
            uncertainty = -(probs * (probs + eps).log()).sum(dim=1)
        else:
            raise ValueError(f"Unknown SCORE_TYPE={score_type}")

        # --- Best alternative label (exclude true) ---
        probs_alt = probs.clone()
        probs_alt.scatter_(1, y.view(-1, 1), float("-inf"))
        alt = probs_alt.argmax(dim=1)

        for img_id, s, a in zip(image_ids, uncertainty.detach().cpu().numpy(), alt.detach().cpu().numpy()):
            scores[str(img_id)] = float(s)
            alt_targets[str(img_id)] = int(a)

    return scores, alt_targets


# ============================================================
# Noise injection bookkeeping
# ============================================================

@dataclass
class NoiseReport:
    outer_fold: int
    seed: int
    arch: str
    score_type: str
    noise_rate: float
    n_train: int
    n_flipped: int
    class_counts_clean: Dict[str, int]
    class_counts_noisy: Dict[str, int]
    flip_confusion: Dict[str, Dict[str, int]]  # true -> noisy counts


def inject_noise_top_uncertain(
    train_df: pd.DataFrame,
    oof_scores: Dict[str, float],
    oof_alt_targets: Dict[str, int],
    i2c: Dict[int, str],
    noise_rate: float,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Flip the top noise_rate most uncertain samples in train_df using OOF scores/targets.

    Output df contains:
      dx_clean (original),
      dx_noisy (possibly flipped),
      uncertainty (OOF uncertainty score)
    """
    df = train_df.copy().reset_index(drop=True)
    df["dx_clean"] = df["dx"]
    df["dx_noisy"] = df["dx"]

    # attach OOF uncertainty
    df["uncertainty"] = df["image_id"].astype(str).map(oof_scores)
    if df["uncertainty"].isna().any():
        missing = df[df["uncertainty"].isna()]["image_id"].head(10).tolist()
        raise RuntimeError(f"Missing OOF uncertainty for some images. Example: {missing}")

    n = len(df)
    n_flip = int(round(noise_rate * n))

    # sort desc by uncertainty
    df_sorted = df.sort_values("uncertainty", ascending=False).reset_index(drop=True)
    flip_rows = df_sorted.head(n_flip).copy()

    flip_conf: Dict[str, Dict[str, int]] = {}

    for idx in flip_rows.index:
        img_id = str(flip_rows.loc[idx, "image_id"])
        true_label = flip_rows.loc[idx, "dx_clean"]

        alt_idx = oof_alt_targets.get(img_id, None)
        if alt_idx is None:
            new_label = true_label
        else:
            new_label = i2c[int(alt_idx)]

        flip_rows.loc[idx, "dx_noisy"] = new_label

        if new_label != true_label:
            flip_conf.setdefault(true_label, {})
            flip_conf[true_label][new_label] = flip_conf[true_label].get(new_label, 0) + 1

    df_sorted.loc[flip_rows.index, "dx_noisy"] = flip_rows["dx_noisy"].values

    # restore original order (not essential, but keeps things stable)
    df_out = df_sorted.sort_index().copy()
    return df_out, flip_conf


# ============================================================
# Fold creation (lesion-level stratified folds)
# ============================================================

def make_outer_folds_lesion_stratified(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """
    Create outer folds on UNIQUE lesion_id with stratification on dx.
    Then propagate fold assignment back to all rows.

    Returns df with an added column: outer_fold ∈ [0..n_splits-1]
    """
    # Always sort deterministically so results are stable across reruns
    df = df.copy().sort_values(["lesion_id", "image_id"]).reset_index(drop=True)

    lesion_df = df.drop_duplicates(subset=["lesion_id"]).copy()
    lesion_df = lesion_df.sort_values(["lesion_id"]).reset_index(drop=True)

    y = lesion_df["dx"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    lesion_fold = np.full(len(lesion_df), -1, dtype=int)
    for fold_id, (_, test_idx) in enumerate(skf.split(np.arange(len(lesion_df)), y)):
        lesion_fold[test_idx] = fold_id

    lesion_df["outer_fold"] = lesion_fold
    lesion_to_fold = dict(zip(lesion_df["lesion_id"].astype(str), lesion_df["outer_fold"].astype(int)))

    df["outer_fold"] = df["lesion_id"].astype(str).map(lesion_to_fold)
    if df["outer_fold"].isna().any():
        raise RuntimeError("Some rows did not get an outer_fold assignment.")

    return df


def make_inner_folds_lesion_stratified(train_df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """
    Create inner folds on UNIQUE lesion_id within the OUTER TRAIN split (90%).
    Returns train_df with added column inner_fold.
    """
    train_df = train_df.copy().sort_values(["lesion_id", "image_id"]).reset_index(drop=True)

    lesion_df = train_df.drop_duplicates(subset=["lesion_id"]).copy()
    lesion_df = lesion_df.sort_values(["lesion_id"]).reset_index(drop=True)

    y = lesion_df["dx"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    lesion_fold = np.full(len(lesion_df), -1, dtype=int)
    for fold_id, (_, val_idx) in enumerate(skf.split(np.arange(len(lesion_df)), y)):
        lesion_fold[val_idx] = fold_id

    lesion_df["inner_fold"] = lesion_fold
    lesion_to_fold = dict(zip(lesion_df["lesion_id"].astype(str), lesion_df["inner_fold"].astype(int)))

    train_df["inner_fold"] = train_df["lesion_id"].astype(str).map(lesion_to_fold)
    if train_df["inner_fold"].isna().any():
        raise RuntimeError("Some train rows did not get an inner_fold assignment.")

    return train_df


# ============================================================
# Main routine
# ============================================================

def main() -> None:
    seed_everything(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    meta_path = DATA_ROOT / METADATA_CSV
    images_dir = DATA_ROOT / IMAGES_DIR

    df = pd.read_csv(meta_path)
    required = {"image_id", "lesion_id", "dx"}
    if not required.issubset(df.columns):
        raise ValueError(f"Metadata must contain columns {required}, got {set(df.columns)}")

    # Stable sort once at load time
    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"] = df["dx"].astype(str)
    df = df.sort_values(["lesion_id", "image_id"]).reset_index(drop=True)

    # Class mapping
    c2i, i2c = class_mapping(df["dx"].unique().tolist())
    num_classes = len(c2i)

    print("\n======================")
    print("HAM10000 IDN Generator")
    print("======================")
    print(f"Samples: {len(df)} | Classes: {num_classes} -> {c2i}")
    print(f"Outer folds: {OUTER_FOLDS} | Inner folds (OOF): {INNER_FOLDS}")
    print(f"Teacher: {ARCH} | pretrained={PRETRAINED} | epochs={TEACHER_EPOCHS}")
    print(f"Noise: rate={NOISE_RATE} | score_type={SCORE_TYPE}")
    print(f"Seed: {SEED}")
    print("======================\n")

    # Transforms
    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = USE_AMP and device.startswith("cuda")
    print(f"Device: {device} | AMP: {use_amp}\n")

    # ------------------------------------------------------------------
    # 1) Create and SAVE outer fold assignments (reproducibility anchor)
    # ------------------------------------------------------------------
    df_folds = make_outer_folds_lesion_stratified(df, n_splits=OUTER_FOLDS, seed=SEED)

    fold_assign_path = OUT_ROOT / "fold_assignments.csv"
    df_folds[["image_id", "lesion_id", "dx", "outer_fold"]].to_csv(fold_assign_path, index=False)
    print(f"Saved outer fold assignments -> {fold_assign_path.resolve()}\n")

    # ------------------------------------------------------------------
    # 2) For each outer fold: generate noisy train + clean test
    # ------------------------------------------------------------------
    outer_iter = tqdm(range(OUTER_FOLDS), desc="Outer folds", leave=True)

    for outer_fold in outer_iter:
        fold_dir = OUT_ROOT / f"fold_{outer_fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        test_df = df_folds[df_folds["outer_fold"] == outer_fold].copy().reset_index(drop=True)
        train_df = df_folds[df_folds["outer_fold"] != outer_fold].copy().reset_index(drop=True)

        # Inner folds inside outer train for OOF scoring
        # Use a fold-specific seed for determinism but different inner splits per outer fold
        inner_seed = SEED * 10_000 + outer_fold
        train_df = make_inner_folds_lesion_stratified(train_df, n_splits=INNER_FOLDS, seed=inner_seed)

        # Collect OOF scores for the entire outer-train split
        oof_scores: Dict[str, float] = {}
        oof_alt_targets: Dict[str, int] = {}

        inner_iter = tqdm(range(INNER_FOLDS), desc=f"  Inner folds (outer={outer_fold:02d})", leave=False)
        for inner_fold in inner_iter:
            inner_train = train_df[train_df["inner_fold"] != inner_fold].copy().reset_index(drop=True)
            inner_val = train_df[train_df["inner_fold"] == inner_fold].copy().reset_index(drop=True)

            ds_tr = HamDataset(inner_train, images_dir, c2i, transform=tf_train)
            ds_val = HamDataset(inner_val, images_dir, c2i, transform=tf_eval)

            dl_tr = DataLoader(
                ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
            )
            dl_val = DataLoader(
                ds_val, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
            )

            # Build + train teacher
            teacher = build_resnet(ARCH, num_classes, pretrained=PRETRAINED)
            train_teacher(
                model=teacher,
                train_loader=dl_tr,
                device=device,
                epochs=TEACHER_EPOCHS,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                use_amp=use_amp,
                desc=f"outer {outer_fold:02d} / inner {inner_fold:02d} ({ARCH})",
            )

            # Score on held-out inner fold (OOF)
            fold_scores, fold_alt = score_uncertainty_and_altlabel(
                model=teacher,
                loader=dl_val,
                device=device,
                score_type=SCORE_TYPE,
            )

            # Merge into OOF dictionaries
            overlap = set(oof_scores.keys()).intersection(fold_scores.keys())
            if overlap:
                raise RuntimeError(f"OOF overlap detected (should not happen). Example: {list(overlap)[:5]}")
            oof_scores.update(fold_scores)
            oof_alt_targets.update(fold_alt)

            # Free memory ASAP (important on GPU)
            del teacher
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        # Sanity: every outer-train sample must have an OOF score
        if len(oof_scores) != len(train_df):
            raise RuntimeError(
                f"OOF scoring incomplete for outer fold {outer_fold}: "
                f"{len(oof_scores)} scores for {len(train_df)} samples."
            )

        if SAVE_OOF_DEBUG:
            pd.DataFrame({
                "image_id": list(oof_scores.keys()),
                "uncertainty": list(oof_scores.values()),
                "alt_target_idx": [oof_alt_targets[k] for k in oof_scores.keys()],
            }).to_csv(fold_dir / "oof_debug.csv", index=False)

        # Inject noise into outer-train using OOF ranking
        train_out, flip_conf = inject_noise_top_uncertain(
            train_df=train_df.drop(columns=["inner_fold"], errors="ignore"),
            oof_scores=oof_scores,
            oof_alt_targets=oof_alt_targets,
            i2c=i2c,
            noise_rate=NOISE_RATE,
        )

        # Prepare outputs
        train_clean = train_out.copy()
        train_clean["dx"] = train_clean["dx_clean"]

        train_noisy = train_out.copy()
        train_noisy["dx"] = train_noisy["dx_noisy"]

        # Keep a clean, consistent schema
        keep_cols = ["image_id", "lesion_id", "dx", "dx_clean", "dx_noisy", "uncertainty"]
        train_clean = train_clean[[c for c in keep_cols if c in train_clean.columns]]
        train_noisy = train_noisy[[c for c in keep_cols if c in train_noisy.columns]]

        test_clean = test_df.copy()[["image_id", "lesion_id", "dx"]]

        # Save CSVs
        train_clean.to_csv(fold_dir / "train_clean.csv", index=False)
        train_noisy.to_csv(fold_dir / "train_noisy.csv", index=False)
        test_clean.to_csv(fold_dir / "test_clean.csv", index=False)

        # Fold report
        n_train = len(train_out)
        n_flipped = int((train_out["dx_clean"] != train_out["dx_noisy"]).sum())
        report = NoiseReport(
            outer_fold=outer_fold,
            seed=SEED,
            arch=ARCH,
            score_type=SCORE_TYPE,
            noise_rate=NOISE_RATE,
            n_train=n_train,
            n_flipped=n_flipped,
            class_counts_clean=train_out["dx_clean"].value_counts().to_dict(),
            class_counts_noisy=train_out["dx_noisy"].value_counts().to_dict(),
            flip_confusion=flip_conf,
        )
        with open(fold_dir / "noise_report.json", "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)

        outer_iter.set_postfix(
            fold=f"{outer_fold:02d}",
            flipped=f"{n_flipped}/{n_train} ({(n_flipped/max(n_train,1))*100:.1f}%)"
        )

    print("\nDone.")
    print(f"Outputs written to: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()