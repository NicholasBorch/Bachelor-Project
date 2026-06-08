"""
Top-level training orchestration.

run_training trains one model on one fold end-to-end. Stage 2 passes val_df (per-
epoch validation logged) and test_df=None (the test fold is never touched). Stage 3
passes test_df (final clean-test eval + training-set NTA/LNMR) and val_df=None.

HamDataset yields (image, label, sample_index); the index is required by ELR.
Methods with requires_two_views=True (only asyco_divmix) get the TRAIN set wrapped
in TwoViewHamDataset; test/val stay single-view. The imbalanced dataset uses a
WeightedRandomSampler and class-weighted CE.
Mixed precision is uniform and off on CPU. NTA/LNMR need a second test-time pass
over the training set. All artifacts are written to output_dir.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from src.data.ham10000 import CLASS_NAMES, NUM_CLASSES, HamDataset
from src.data.transforms import get_test_transforms, get_train_transforms
from src.data.two_view import TwoViewHamDataset
from src.methods import build_method
from src.models.resnet import build_resnet
from src.training.metrics import compute_metrics, compute_noise_label_interaction
from src.training.samplers import compute_class_weights, make_weighted_sampler
from src.utils.io import save_yaml
from src.utils.seed import seed_everything


class StopTraining(Exception):
    """Raised in an epoch_callback to request graceful early termination (used for Optuna pruning)."""
    pass


def _labels_to_indices(labels: pd.Series) -> np.ndarray:
    """Map a column of class names ('mel', 'nv', ...) to integer indices."""
    mapping = {c: i for i, c in enumerate(CLASS_NAMES)}
    out = labels.map(mapping).to_numpy()
    if pd.isna(out).any():
        unknown = set(labels) - set(CLASS_NAMES)
        raise ValueError(f"Unknown class labels in train_df: {unknown}")
    return out.astype(np.int64)


def _build_model_builder(cfg: dict):
    """Return a zero-arg callable that constructs a fresh ResNet."""
    model_cfg = cfg["model"]
    num_classes = int(cfg.get("num_classes", NUM_CLASSES))
    depth = int(model_cfg["depth"])
    pretrained = bool(model_cfg.get("pretrained", False))
    weights_name = model_cfg.get("weights", "IMAGENET1K_V1") or "IMAGENET1K_V1"

    def _builder():
        return build_resnet(
            num_classes=num_classes,
            depth=depth,
            pretrained=pretrained,
            weights_name=weights_name,
        )

    return _builder


def _build_loaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    images_dir: Path,
    cfg: dict,
    val_df: pd.DataFrame | None,
    requires_two_views: bool = False,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, np.ndarray]:
    """Construct train / optional test / optional val DataLoaders (train wrapped two-view if required)."""
    image_size = int(cfg["image_size"])
    resize_size = int(round(image_size * 256 / 224))
    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg.get("num_workers", 4))
    pin_memory = bool(cfg.get("pin_memory", True))

    train_tf = get_train_transforms(image_size=image_size)
    test_tf = get_test_transforms(image_size=image_size, resize_size=resize_size)

    if requires_two_views:
        # Build the base PIL-cache once with transform=None, then wrap.
        # The wrapper applies train_tf twice per __getitem__ on the cached PIL.
        base_train = HamDataset(train_df, images_dir=images_dir, transform=None)
        train_ds: torch.utils.data.Dataset = TwoViewHamDataset(base_train, transform=train_tf)
    else:
        train_ds = HamDataset(train_df, images_dir=images_dir, transform=train_tf)

    test_ds = (
        HamDataset(test_df, images_dir=images_dir, transform=test_tf)
        if test_df is not None else None
    )
    val_ds = (
        HamDataset(val_df, images_dir=images_dir, transform=test_tf)
        if val_df is not None else None
    )

    train_labels_idx = _labels_to_indices(train_df["dx"])

    sampler_kind = cfg["data"].get("sampler", "shuffle")
    if sampler_kind == "weighted_random":
        sampler = make_weighted_sampler(train_labels_idx)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    pw = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=pw,
        drop_last=False,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=pw,
        )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=pw,
        )
    return train_loader, test_loader, val_loader, train_labels_idx


def _evaluate(method, loader, device: torch.device) -> dict[str, Any]:
    """Run inference and compute the standard metric suite."""
    y_true, y_pred, y_prob = method.predict(loader, device=device)
    return compute_metrics(y_true, y_pred, y_prob)


def _predict_on_train_set(
    method,
    train_df: pd.DataFrame,
    images_dir: Path,
    cfg: dict,
    device: torch.device,
) -> np.ndarray:
    """Run the trained model over the training set with test-time transforms; return argmax preds in train_df order."""
    image_size = int(cfg["image_size"])
    resize_size = int(round(image_size * 256 / 224))
    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg.get("num_workers", 4))
    pin_memory = bool(cfg.get("pin_memory", True))
    test_tf = get_test_transforms(image_size=image_size, resize_size=resize_size)

    eval_ds = HamDataset(train_df, images_dir=images_dir, transform=test_tf)
    pw = num_workers > 0
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=pw,
    )
    _, y_pred_train, _ = method.predict(eval_loader, device=device)
    return np.asarray(y_pred_train).astype(np.int64)


def _jsonl_append(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, default=float) + "\n")


def _compute_per_class_noise_diagnostics(
    y_pred_train: np.ndarray,
    y_noisy: np.ndarray,
    y_clean: np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    """Per-class NTA/LNMR on flipped samples, conditioned on clean class (*_by_clean) and noisy class (*_by_noisy); None for empty buckets."""
    flipped = y_noisy != y_clean
    per_class_nta_by_clean: list[float | None] = []
    per_class_lnmr_by_clean: list[float | None] = []
    per_class_nta_by_noisy: list[float | None] = []
    per_class_lnmr_by_noisy: list[float | None] = []
    n_by_clean: list[int] = []
    n_by_noisy: list[int] = []

    for c in range(int(num_classes)):
        # Conditioning on CLEAN class
        mask_clean = flipped & (y_clean == c)
        n_clean = int(mask_clean.sum())
        n_by_clean.append(n_clean)
        if n_clean > 0:
            preds_c = y_pred_train[mask_clean]
            noisy_c = y_noisy[mask_clean]
            per_class_nta_by_clean.append(float((preds_c == c).mean()))
            per_class_lnmr_by_clean.append(float((preds_c == noisy_c).mean()))
        else:
            per_class_nta_by_clean.append(None)
            per_class_lnmr_by_clean.append(None)

        # Conditioning on NOISY class
        mask_noisy = flipped & (y_noisy == c)
        n_noisy = int(mask_noisy.sum())
        n_by_noisy.append(n_noisy)
        if n_noisy > 0:
            preds_n = y_pred_train[mask_noisy]
            clean_n = y_clean[mask_noisy]
            per_class_nta_by_noisy.append(float((preds_n == clean_n).mean()))
            per_class_lnmr_by_noisy.append(float((preds_n == c).mean()))
        else:
            per_class_nta_by_noisy.append(None)
            per_class_lnmr_by_noisy.append(None)

    return {
        "per_class_nta_by_clean": per_class_nta_by_clean,
        "per_class_lnmr_by_clean": per_class_lnmr_by_clean,
        "per_class_nta_by_noisy": per_class_nta_by_noisy,
        "per_class_lnmr_by_noisy": per_class_lnmr_by_noisy,
        "n_flipped_by_clean": n_by_clean,
        "n_flipped_by_noisy": n_by_noisy,
    }


def run_training(
    cfg: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    images_dir: Path,
    method_name: str,
    total_epochs: int,
    output_dir: Path,
    val_df: pd.DataFrame | None = None,
    seed: int | None = None,
    epoch_callback=None,
    track_train_diagnostics_every: int | None = None,
) -> dict[str, Any]:
    """Train one model on one fold; returns final test metrics (with NTA/LNMR) in Stage 3, else {}."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        seed_everything(int(seed), deterministic=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if device.type == "cuda" and bool(cfg.get("cudnn_benchmark", True)):
        cudnn.benchmark = True

    resolved = dict(cfg)
    resolved["_resolved"] = {
        "method_name": method_name,
        "total_epochs": int(total_epochs),
        "seed_used": int(seed) if seed is not None else None,
        "has_val_df": val_df is not None,
        "has_test_df": test_df is not None,
        "device": str(device),
    }
    save_yaml(resolved, output_dir / "config.yaml")

    # Probe the method class for its two-view requirement BEFORE building
    # loaders. Importing here keeps the dependency local to this function
    # and avoids circular imports.
    from src.methods import build_method as _bm  # noqa: F401 (already imported above)
    from src.methods.base import Method as _MethodCls  # noqa: F401
    # The flag lives on the class; query via the registry.
    requires_two_views = _method_requires_two_views(method_name)

    train_loader, test_loader, val_loader, train_labels_idx = _build_loaders(
        train_df=train_df, test_df=test_df, images_dir=images_dir,
        cfg=cfg, val_df=val_df, requires_two_views=requires_two_views,
    )

    class_weights = None
    if bool(cfg["data"].get("class_weighted_loss", False)):
        class_weights = compute_class_weights(train_labels_idx, device=device)

    method = build_method(
        method_name=method_name,
        cfg=cfg,
        num_train_samples=len(train_df),
        num_classes=int(cfg.get("num_classes", NUM_CLASSES)),
        device=device,
        class_weights=class_weights,
    )
    method.build(
        total_epochs=int(total_epochs),
        model_builder=_build_model_builder(cfg),
    )

    amp_enabled = bool(cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    log_path = output_dir / "training_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    stopped_early = False

    # cached clean/noisy label arrays for per-epoch diagnostics (computed once)
    _diag_enabled = (
        track_train_diagnostics_every is not None
        and int(track_train_diagnostics_every) > 0
        and "dx_clean" in train_df.columns
    )
    if _diag_enabled:
        _diag_every = int(track_train_diagnostics_every)
        _diag_y_clean = _labels_to_indices(train_df["dx_clean"])
        _diag_y_noisy = _labels_to_indices(train_df["dx"])
        _diag_num_classes = int(cfg.get("num_classes", NUM_CLASSES))
    else:
        _diag_every = 0
        _diag_y_clean = None
        _diag_y_noisy = None
        _diag_num_classes = 0

    for epoch in range(int(total_epochs)):
        t0 = time.time()
        loss_sum = 0.0
        n_samples = 0
        component_sums: dict[str, float] = {}

        for batch in train_loader:
            out = method.train_step(batch, epoch, scaler)
            bs = int(out.batch_size)
            loss_sum += float(out.loss_total) * bs
            n_samples += bs
            for k, v in out.loss_components.items():
                component_sums[k] = component_sums.get(k, 0.0) + float(v) * bs

        method.scheduler_step()

        train_loss = loss_sum / max(n_samples, 1)
        components = {k: v / max(n_samples, 1) for k, v in component_sums.items()}

        try:
            lr_now = float(method._all_schedulers()[0].get_last_lr()[0])
        except Exception:
            lr_now = float("nan")

        record: dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "loss_components": components,
            "lr": lr_now,
            "epoch_time_s": float(time.time() - t0),
        }

        if val_loader is not None:
            val_metrics = _evaluate(method, val_loader, device=device)
            record["val_balanced_accuracy"] = float(val_metrics["balanced_accuracy"])
            record["val_macro_f1"] = float(val_metrics["macro_f1"])
            record["val_weighted_f1"] = float(val_metrics["weighted_f1"])
            record["val_macro_auc"] = float(val_metrics["macro_auc"])

        # per-epoch training-set NTA/LNMR (every _diag_every epochs + final), stored as train_diagnostics
        is_last_epoch = (epoch == int(total_epochs) - 1)
        if _diag_enabled and (epoch % _diag_every == 0 or is_last_epoch):
            _diag_y_pred = _predict_on_train_set(
                method=method, train_df=train_df, images_dir=images_dir,
                cfg=cfg, device=device,
            )
            _diag_scalar = compute_noise_label_interaction(
                y_pred_train=_diag_y_pred,
                y_noisy=_diag_y_noisy,
                y_clean=_diag_y_clean,
            )
            _diag_per_class = _compute_per_class_noise_diagnostics(
                y_pred_train=_diag_y_pred,
                y_noisy=_diag_y_noisy,
                y_clean=_diag_y_clean,
                num_classes=_diag_num_classes,
            )
            record["train_diagnostics"] = {**_diag_scalar, **_diag_per_class}

        _jsonl_append(log_path, record)

        # optional epoch-end hook (e.g. Optuna pruning); StopTraining exits the loop cleanly
        if epoch_callback is not None:
            try:
                epoch_callback(int(epoch), record)
            except StopTraining:
                stopped_early = True
                break

    # stage 2 (no test set) or stopped early: skip final test + NTA/LNMR
    if test_loader is None or stopped_early:
        return {}

    # Stage 3 mode: final test evaluation + training-set noise-label diagnostics.
    test_metrics = _evaluate(method, test_loader, device=device)
    test_metrics["method"] = method_name
    test_metrics["total_epochs"] = int(total_epochs)
    test_metrics["dataset"] = cfg["data"].get("name")

    # NTA/LNMR require both dx (possibly noisy) and dx_clean on train_df.
    # Stage 1c always writes both columns. If `dx_clean` is absent for any
    # reason, emit nulls rather than crashing so the result is still readable.
    if "dx_clean" in train_df.columns:
        y_pred_train = _predict_on_train_set(
            method=method, train_df=train_df, images_dir=images_dir,
            cfg=cfg, device=device,
        )
        y_noisy = _labels_to_indices(train_df["dx"])
        y_clean = _labels_to_indices(train_df["dx_clean"])
        noise_metrics = compute_noise_label_interaction(
            y_pred_train=y_pred_train, y_noisy=y_noisy, y_clean=y_clean,
        )
        test_metrics.update(noise_metrics)
        # Per-class breakdowns alongside the scalars. These are stored
        # under the same flat keys as the scalars to keep the JSON shape
        # uniform with the per-epoch `train_diagnostics` sub-dict.
        per_class_metrics = _compute_per_class_noise_diagnostics(
            y_pred_train=y_pred_train, y_noisy=y_noisy, y_clean=y_clean,
            num_classes=int(cfg.get("num_classes", NUM_CLASSES)),
        )
        test_metrics.update(per_class_metrics)
    else:
        test_metrics.update({
            "nta": None,
            "lnmr": None,
            "n_flipped": None,
            "n_train": int(len(train_df)),
            "empirical_flip_rate": None,
        })

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2, default=float)

    return test_metrics


def _method_requires_two_views(method_name: str) -> bool:
    """Look up requires_two_views on the method class without instantiating it."""
    from src.methods.baseline import BaselineMethod
    from src.methods.sce import SCEMethod
    from src.methods.elr import ELRMethod
    from src.methods.asyco_divmix import AsyCoDivMixMethod

    name = method_name.lower()
    cls_map = {
        "baseline": BaselineMethod,
        "sce": SCEMethod,
        "elr": ELRMethod,
        "asyco_divmix": AsyCoDivMixMethod,
    }
    if name not in cls_map:
        raise ValueError(f"Unknown method: {method_name}")
    return bool(getattr(cls_map[name], "requires_two_views", False))