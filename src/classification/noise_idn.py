from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.common.seed import seed_everything


# Data structures
@dataclass
class NoiseReport:
    """
    Summary of noise injection for one outer fold.
    flip_confusion maps: true_label -> {noisy_label: count}
    """
    outer_fold: int
    seed: int
    arch: str
    score_type: str
    noise_rate: float
    n_train: int
    n_flipped: int
    class_counts_clean: Dict[str, int]
    class_counts_noisy: Dict[str, int]
    flip_confusion: Dict[str, Dict[str, int]]
    flipped_uncertainty_min: float
    flipped_uncertainty_median: float
    flipped_uncertainty_max: float


@dataclass
class FoldOutputs:
    """
    Output bundle for one outer fold.
    """
    train_clean: pd.DataFrame
    train_noisy: pd.DataFrame
    test_clean: pd.DataFrame
    report: NoiseReport


@dataclass
class IDNOutputs:
    """
    Output bundle for all folds.
    """
    fold_assignments: pd.DataFrame
    folds: Dict[int, FoldOutputs]


@dataclass
class OOFFoldOutputs:
    """
    Output bundle for one outer fold (OOF stage).
    """
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    oof_scores: Dict[str, float]
    oof_probs: Dict[str, np.ndarray]


@dataclass
class OOFOutputs:
    """
    Output bundle for all folds (OOF stage).
    """
    fold_assignments: pd.DataFrame
    folds: Dict[int, OOFFoldOutputs]


# Helpers
def class_mapping(classes: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Stable mapping between string labels and integer indices."""
    classes_sorted = sorted(list(set(classes)))
    c2i = {c: i for i, c in enumerate(classes_sorted)}
    i2c = {i: c for c, i in c2i.items()}
    return c2i, i2c


# Dataset
class HamDataset(Dataset):
    """
    Torch dataset returning (image_tensor, label_idx, image_id_str).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        c2i: Dict[str, int],
        transform: transforms.Compose,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.c2i = c2i
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        label_str = str(row["dx"])
        y = self.c2i[label_str]

        img_path = self.images_dir / f"{image_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, y, image_id


# Model building
def build_resnet(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    """
    Build ResNet teacher with classification head.
    """
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
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


# Training + OOF scoring
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
    Lightweight teacher training (few epochs) since nested CV trains many teachers.
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
def score_uncertainty_and_probs(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    score_type: str,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    For each sample:
      - uncertainty score (higher = more uncertain)
      - full softmax probability vector (used for probabilistic flip sampling)
    """
    model.eval()
    scores: Dict[str, float] = {}
    probs_dict: Dict[str, np.ndarray] = {}

    for x, y, image_ids in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        if score_type == "p_true":
            p_true = probs.gather(1, y.view(-1, 1)).squeeze(1)
            uncertainty = 1.0 - p_true
        elif score_type == "max_softmax":
            p_max, _ = probs.max(dim=1)
            uncertainty = 1.0 - p_max
        else:  # "entropy"
            eps = 1e-8
            uncertainty = -(probs * (probs + eps).log()).sum(dim=1)

        probs_np = probs.detach().cpu().numpy().astype(np.float32)
        unc_np = uncertainty.detach().cpu().numpy().astype(np.float32)

        for img_id, u, p in zip(image_ids, unc_np, probs_np):
            scores[str(img_id)] = float(u)
            probs_dict[str(img_id)] = p

    return scores, probs_dict


# Fold creation (lesion-level stratified)
def make_outer_folds_lesion_stratified(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """
    Outer folds are created on unique lesion_id with stratification on dx.
    """
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

    return df


def make_inner_folds_lesion_stratified(train_df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """
    Inner folds are created on unique lesion_id within the outer-train split.
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

    return train_df


# Noise injection
def inject_noise_idn_with_caps(
    train_df: pd.DataFrame,
    oof_scores: Dict[str, float],
    oof_probs: Dict[str, np.ndarray],
    c2i: Dict[str, int],
    i2c: Dict[int, str],
    noise_rate: float,
    eta_max: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Instance-dependent noise injection (IDN):
      1) Rank by OOF uncertainty (difficulty)
      2) Select top r*N to flip, with per-class flip caps (eta_max * N_c)
      3) For each selected sample, flip label by sampling from teacher probabilities
         after removing the true class and renormalizing.

    Returns:
      df_out: original rows + dx_clean, dx_noisy, uncertainty
      flip_confusion: true_label -> {noisy_label: count}
    """
    df = train_df.copy().reset_index(drop=True)
    df["dx_clean"] = df["dx"]
    df["dx_noisy"] = df["dx"]
    df["uncertainty"] = df["image_id"].astype(str).map(oof_scores)

    n = len(df)
    n_flip_target = int(round(noise_rate * n))

    class_counts = df["dx_clean"].value_counts().to_dict()
    class_cap = {cls: int(np.floor(eta_max * cnt)) for cls, cnt in class_counts.items()}
    flipped_per_class = {cls: 0 for cls in class_counts.keys()}

    df_sorted = df.sort_values("uncertainty", ascending=False).reset_index(drop=True)

    selected_idx: List[int] = []
    for idx in df_sorted.index:
        if len(selected_idx) >= n_flip_target:
            break

        true_label = str(df_sorted.loc[idx, "dx_clean"])
        if flipped_per_class[true_label] >= class_cap[true_label]:
            continue

        selected_idx.append(int(idx))
        flipped_per_class[true_label] += 1

    flip_conf: Dict[str, Dict[str, int]] = {}

    for idx in selected_idx:
        img_id = str(df_sorted.loc[idx, "image_id"])
        true_label = str(df_sorted.loc[idx, "dx_clean"])
        true_idx = c2i[true_label]

        p = oof_probs[img_id].astype(np.float64, copy=True)
        p[true_idx] = 0.0
        s = float(p.sum())

        if s <= 0.0:
            continue

        p = p / s
        new_idx = int(rng.choice(np.arange(len(p)), p=p))
        new_label = i2c[new_idx]

        df_sorted.loc[idx, "dx_noisy"] = new_label

        if new_label != true_label:
            flip_conf.setdefault(true_label, {})
            flip_conf[true_label][new_label] = flip_conf[true_label].get(new_label, 0) + 1

    df_out = df_sorted.sort_index().copy()
    return df_out, flip_conf


def apply_idn_from_oof(
    train_df: pd.DataFrame,
    oof_scores: Dict[str, float],
    oof_probs: Dict[str, np.ndarray],
    outer_fold: int,
    seed: int,
    noise_rate: float,
    eta_max: float,
    score_type: str,
    arch: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, NoiseReport]:
    """
    Apply IDN noise to one outer-train split using cached OOF probabilities.
    Returns train_clean, train_noisy, and a NoiseReport.
    """
    rng = np.random.default_rng(seed * 10_000 + outer_fold)

    c2i, i2c = class_mapping(train_df["dx"].astype(str).tolist())

    train_out, flip_conf = inject_noise_idn_with_caps(
        train_df=train_df,
        oof_scores=oof_scores,
        oof_probs=oof_probs,
        c2i=c2i,
        i2c=i2c,
        noise_rate=noise_rate,
        eta_max=eta_max,
        rng=rng,
    )

    train_clean = train_out.copy()
    train_clean["dx"] = train_clean["dx_clean"]

    train_noisy = train_out.copy()
    train_noisy["dx"] = train_noisy["dx_noisy"]

    keep_cols = ["image_id", "lesion_id", "dx", "dx_clean", "dx_noisy", "uncertainty"]
    train_clean = train_clean[[c for c in keep_cols if c in train_clean.columns]]
    train_noisy = train_noisy[[c for c in keep_cols if c in train_noisy.columns]]

    n_train = len(train_out)
    n_flipped = int((train_out["dx_clean"] != train_out["dx_noisy"]).sum())

    flipped_mask = train_out["dx_clean"] != train_out["dx_noisy"]
    flipped_unc = train_out.loc[flipped_mask, "uncertainty"].astype(float)

    if len(flipped_unc) == 0:
        u_min = u_med = u_max = float("nan")
    else:
        u_min = float(flipped_unc.min())
        u_med = float(flipped_unc.median())
        u_max = float(flipped_unc.max())

    report = NoiseReport(
        outer_fold=outer_fold,
        seed=seed,
        arch=arch,
        score_type=score_type,
        noise_rate=noise_rate,
        n_train=n_train,
        n_flipped=n_flipped,
        class_counts_clean=train_out["dx_clean"].value_counts().to_dict(),
        class_counts_noisy=train_out["dx_noisy"].value_counts().to_dict(),
        flip_confusion=flip_conf,
        flipped_uncertainty_min=u_min,
        flipped_uncertainty_median=u_med,
        flipped_uncertainty_max=u_max,
    )

    return train_clean, train_noisy, report


def generate_oof_nestedcv(
    df: pd.DataFrame,
    images_dir: Path,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    score_type: str,
    arch: str,
    pretrained: bool,
    teacher_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    use_amp: bool,
    pin_memory: bool,
) -> OOFOutputs:
    """
    Nested CV OOF generator:
      - outer folds for evaluation (test stays clean)
      - inner folds for out-of-fold uncertainty scoring
      - returns OOF probabilities and uncertainty scores per outer fold
    """
    seed_everything(seed)

    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"] = df["dx"].astype(str)
    df = df.sort_values(["lesion_id", "image_id"]).reset_index(drop=True)

    c2i, _ = class_mapping(df["dx"].tolist())
    num_classes = len(c2i)

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
    amp_enabled = use_amp and device.startswith("cuda")

    df_folds = make_outer_folds_lesion_stratified(df, n_splits=outer_folds, seed=seed)

    fold_outputs: Dict[int, OOFFoldOutputs] = {}
    outer_iter = tqdm(range(outer_folds), desc="Outer folds (OOF)", leave=True)

    for outer_fold in outer_iter:
        test_df = df_folds[df_folds["outer_fold"] == outer_fold].copy().reset_index(drop=True)
        train_df = df_folds[df_folds["outer_fold"] != outer_fold].copy().reset_index(drop=True)

        inner_seed = seed * 10_000 + outer_fold
        train_df = make_inner_folds_lesion_stratified(train_df, n_splits=inner_folds, seed=inner_seed)

        oof_scores: Dict[str, float] = {}
        oof_probs: Dict[str, np.ndarray] = {}

        inner_iter = tqdm(range(inner_folds), desc=f"  Inner folds (outer={outer_fold:02d})", leave=False)
        for inner_fold in inner_iter:
            inner_train = train_df[train_df["inner_fold"] != inner_fold].copy().reset_index(drop=True)
            inner_val = train_df[train_df["inner_fold"] == inner_fold].copy().reset_index(drop=True)

            ds_tr = HamDataset(inner_train, images_dir=images_dir, c2i=c2i, transform=tf_train)
            ds_val = HamDataset(inner_val, images_dir=images_dir, c2i=c2i, transform=tf_eval)

            dl_tr = DataLoader(
                ds_tr, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory
            )
            dl_val = DataLoader(
                ds_val, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory
            )

            teacher = build_resnet(arch=arch, num_classes=num_classes, pretrained=pretrained)

            train_teacher(
                model=teacher,
                train_loader=dl_tr,
                device=device,
                epochs=teacher_epochs,
                lr=lr,
                weight_decay=weight_decay,
                use_amp=amp_enabled,
                desc=f"outer {outer_fold:02d} / inner {inner_fold:02d} ({arch})",
            )

            fold_scores, fold_probs = score_uncertainty_and_probs(
                model=teacher,
                loader=dl_val,
                device=device,
                score_type=score_type,
            )

            oof_scores.update(fold_scores)
            oof_probs.update(fold_probs)

            del teacher
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        train_df_out = train_df.drop(columns=["inner_fold"], errors="ignore").copy()

        fold_outputs[outer_fold] = OOFFoldOutputs(
            train_df=train_df_out,
            test_df=test_df[["image_id", "lesion_id", "dx"]].copy(),
            oof_scores=oof_scores,
            oof_probs=oof_probs,
        )

    fold_assignments = df_folds[["image_id", "lesion_id", "dx", "outer_fold"]].copy()
    return OOFOutputs(fold_assignments=fold_assignments, folds=fold_outputs)


def generate_idn_nestedcv(
    df: pd.DataFrame,
    images_dir: Path,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    noise_rate: float,
    eta_max: float,
    score_type: str,
    arch: str,
    pretrained: bool,
    teacher_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    use_amp: bool,
    pin_memory: bool,
) -> IDNOutputs:
    """
    Nested CV IDN generator:
      - outer folds for evaluation (test stays clean)
      - inner folds for out-of-fold uncertainty scoring
      - flip top noise_rate uncertain samples in outer train split
      - enforce per-class flip cap eta_max
      - assign new labels probabilistically from teacher distribution

    Returns fold assignments + per-fold clean/noisy dataframes + reports.
    """
    oof_outputs = generate_oof_nestedcv(
        df=df,
        images_dir=images_dir,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
        score_type=score_type,
        arch=arch,
        pretrained=pretrained,
        teacher_epochs=teacher_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        use_amp=use_amp,
        pin_memory=pin_memory,
    )

    fold_outputs: Dict[int, FoldOutputs] = {}
    outer_iter = tqdm(range(outer_folds), desc="Apply IDN per outer fold", leave=True)

    for outer_fold in outer_iter:
        fold = oof_outputs.folds[outer_fold]

        train_clean, train_noisy, report = apply_idn_from_oof(
            train_df=fold.train_df,
            oof_scores=fold.oof_scores,
            oof_probs=fold.oof_probs,
            outer_fold=outer_fold,
            seed=seed,
            noise_rate=noise_rate,
            eta_max=eta_max,
            score_type=score_type,
            arch=arch,
        )

        test_clean = fold.test_df.copy()

        fold_outputs[outer_fold] = FoldOutputs(
            train_clean=train_clean,
            train_noisy=train_noisy,
            test_clean=test_clean,
            report=report,
        )

        outer_iter.set_postfix(
            fold=f"{outer_fold:02d}",
            flipped=f"{report.n_flipped}/{report.n_train} ({(report.n_flipped/max(report.n_train,1))*100:.1f}%)"
        )

    return IDNOutputs(fold_assignments=oof_outputs.fold_assignments, folds=fold_outputs)