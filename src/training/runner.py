"""Top-level training orchestration.

One entry point — `run_training` — takes a fully-resolved config and trains
one model on one fold end-to-end. The same function is used by:

- Stage 2 (epoch budget selection): called with `val_df` set so per-epoch
  validation balanced accuracy is logged alongside training loss.
  `test_df` is `None` in Stage 2 — the fold's test set must not be touched
  (PROJECT_DOCUMENTATION §10, item 4).
- Stage 3 (final method training): called with `test_df` set and
  `val_df=None`; both the final clean test evaluation and the
  training-set noise-label interaction diagnostics (NTA, LNMR) are
  computed and saved.

Design constraints enforced here:

- HamDataset returns (image, label, sample_index) — the sample_index is
  required by ELR's target buffer and must flow through every loader.
- Methods declaring ``requires_two_views = True`` (currently only
  ``asyco_divmix``) get their TRAIN dataset wrapped in
  ``TwoViewHamDataset`` so each batch is (img1, img2, label, idx). The
  TEST and VAL loaders are unaffected — both still produce single-view
  batches for ``predict()``.
- For the imbalanced dataset, a WeightedRandomSampler is used and CE losses
  are class-weighted via inverse frequency. For the balanced dataset, both
  are disabled and we rely on shuffled sampling + unweighted CE (consistent
  with PROJECT_DOCUMENTATION §2.3.6).
- Mixed precision is uniform across all methods and toggled off on CPU.
- The fold's test set is loaded from `test_df` only when `test_df is not
  None`; when it is None (Stage 2), we never build a test loader and never
  evaluate on it.
- NTA/LNMR require a SECOND pass over the training set with test-time
  transforms (no augmentation). This runs only when `test_df is not
  None` and `dx_clean` exists on `train_df`. Cost is one forward pass over
  the training set at the end of training.
- All artifacts — the resolved config, the per-epoch log, and (when a test
  set is provided) the final test metrics enriched with NTA/LNMR — are
  written to `output_dir` so downstream analysis has a single place to look.
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
    """Raised inside an epoch_callback to request graceful early termination.

    Caught at the per-epoch loop boundary in run_training. The post-epoch
    record is already written to training_log.jsonl before the exception
    propagates, and final test evaluation (Stage 3 mode) does NOT run when
    training is stopped this way — the partial run is treated as a
    legitimately-terminated training, not a completed one.

    The Optuna search uses this to honor pruning decisions without leaking
    Optuna concepts into runner.py. Other callers can use it for any
    reasonable early-stopping criterion.
    """
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
    """Construct train / (optional test) / (optional val) DataLoaders.

    When ``requires_two_views`` is True, the TRAIN dataset is wrapped in
    ``TwoViewHamDataset`` so each item is (img1, img2, label, idx) instead
    of (img, label, idx). Test and val loaders are unchanged regardless.
    """
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
    """Run the trained model over the training set with TEST-TIME transforms
    and return argmax predictions (one int per sample, aligned with
    train_df row order).

    Using test-time transforms (no augmentation, deterministic
    resize+centercrop) is essential: NTA/LNMR should reflect what the
    trained model thinks about each image, not what it thinks about a
    single random augmented view of it. Always uses the standard
    single-view ``HamDataset`` regardless of the method's training-time
    two-view requirement.
    """
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
) -> dict[str, Any]:
    """Train one model on one fold end-to-end.

    Args:
        cfg: the fully-merged config (base + data + model + optim + method + noise).
        train_df: DataFrame with columns `image_id`, `dx` (possibly noisy),
            and typically `dx_clean` for bookkeeping and NTA/LNMR. Training
            uses `dx`.
        test_df: clean-labeled test set, or `None`. If `None`, no final
            test evaluation is performed and no `test_metrics.json` is
            written — Stage 2 mode.
        images_dir: directory with the JPEGs referenced by `image_id`.
        method_name: one of "baseline", "sce", "elr", "asyco", "asyco_divmix".
        total_epochs: the epoch budget (cosine-annealing T_max).
        output_dir: where to write `config.yaml`, `training_log.jsonl`,
            and (Stage 3) `test_metrics.json`.
        val_df: optional validation DataFrame for per-epoch monitoring.
        seed: optional per-fold seed.
        epoch_callback: optional callable ``(epoch_idx, record) -> None``
            invoked after each epoch's record is written to the JSONL log.
            ``record`` is the same dict that was just appended (epoch,
            train_loss, loss_components, lr, epoch_time_s, and val_*
            metrics if val_df was provided). The callback may raise
            ``StopTraining`` to request early termination of training;
            in that case Stage 3 final test evaluation is skipped and an
            empty dict is returned (same as Stage 2 mode). Any other
            exception propagates normally. Callbacks see post-write,
            post-validation state — they should not mutate ``record``.

    Returns:
        The final test metrics dict (with NTA/LNMR merged in when
        `dx_clean` is available on `train_df`) in Stage 3; empty dict in
        Stage 2 or when training was stopped early via epoch_callback.
    """
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

        _jsonl_append(log_path, record)

        # Optional epoch-end hook (e.g. for Optuna pruning). The callback
        # sees the same record we just appended. If it raises StopTraining
        # we exit the training loop cleanly; any other exception propagates.
        if epoch_callback is not None:
            try:
                epoch_callback(int(epoch), record)
            except StopTraining:
                stopped_early = True
                break

    # Stage 2 mode: no test set, no NTA/LNMR, just return.
    # Also short-circuit when training was stopped early via the callback —
    # an interrupted training is not a completed one, so we don't run the
    # final test/NTA/LNMR pass.
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
    """Look up the ``requires_two_views`` flag on the method class without
    instantiating the method (which needs a built cfg + device).

    The mapping mirrors ``build_method`` in ``src.methods.__init__``.
    Keeping this lookup local to the runner avoids modifying the
    factory's signature and keeps the method classes the single source
    of truth for the flag value.
    """
    from src.methods.baseline import BaselineMethod
    from src.methods.sce import SCEMethod
    from src.methods.elr import ELRMethod
    from src.methods.asyco import AsyCoMethod
    from src.methods.asyco_divmix import AsyCoDivMixMethod

    name = method_name.lower()
    cls_map = {
        "baseline": BaselineMethod,
        "sce": SCEMethod,
        "elr": ELRMethod,
        "asyco": AsyCoMethod,
        "asyco_divmix": AsyCoDivMixMethod,
    }
    if name not in cls_map:
        raise ValueError(f"Unknown method: {method_name}")
    return bool(getattr(cls_map[name], "requires_two_views", False))
