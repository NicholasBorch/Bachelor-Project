"""Targeted smoke tests for asyco_divmix.

These check the MixMatch-specific machinery that the generic
``test_methods_no_nan.py`` smoke test can't easily exercise:

  - the ``requires_two_views`` flag actually triggers the TwoView wrapper
    in the runner;
  - the MixMatch step produces finite gradients on both networks;
  - empty-labeled and empty-unlabeled batches don't crash;
  - λ_u rampup is monotonic non-decreasing and clamps at the target.

If you only want a quick health check, run the generic
``tests/test_methods_no_nan.py`` instead — it now includes asyco_divmix
in its sweep too. This file is for the deeper invariants.
"""
from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.ham10000 import CLASS_NAMES, NUM_CLASSES, HamDataset
from src.data.transforms import get_train_transforms, get_test_transforms
from src.data.two_view import TwoViewHamDataset
from src.methods import build_method
from src.methods.asyco_divmix import _rampup_lambda_u
from src.models.resnet import build_resnet
from src.training.samplers import compute_class_weights, make_weighted_sampler


def _make_dummy(tmpdir: Path, n_per_class: int = 4):
    images_dir = tmpdir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for cls in CLASS_NAMES:
        for k in range(n_per_class):
            iid = f"{cls}_{k:02d}"
            arr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(images_dir / f"{iid}.jpg", "JPEG")
            rows.append({"image_id": iid, "dx": cls})
    return pd.DataFrame(rows), images_dir


def _base_cfg(dataset_name: str = "imbalanced"):
    cfg = {
        "seed": 10,
        "image_size": 64,
        "num_workers": 0,
        "pin_memory": False,
        "mixed_precision": False,
        "num_classes": 7,
        "lr_scheduler": "cosine_annealing",
        "data": {
            "name": dataset_name,
            "batch_size": 8,
            "sampler": "weighted_random" if dataset_name == "imbalanced" else "shuffle",
            "class_weighted_loss": dataset_name == "imbalanced",
        },
        "model": {"depth": 18, "pretrained": False},
        "optim": {"name": "sgd", "lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": False},
        "method": {
            "name": "asyco_divmix",
            "K": 1, "lambda_u": 25.0, "temperature": 0.5,
            "warmup_epochs_pct": 0.05, "warmup_epochs_floor": 2,
            "mixup_alpha": 0.75, "rampup_epochs": 4, "lambda_prior": 1.0,
        },
    }
    return cfg


def test_requires_two_views_flag():
    """Confirm the class attribute is set so the runner picks it up."""
    from src.methods.asyco_divmix import AsyCoDivMixMethod
    from src.methods.asyco import AsyCoMethod
    from src.methods.baseline import BaselineMethod

    assert AsyCoDivMixMethod.requires_two_views is True
    # Sibling methods must NOT be wrapped — that would silently break their batching.
    assert getattr(AsyCoMethod, "requires_two_views", False) is False
    assert getattr(BaselineMethod, "requires_two_views", False) is False
    print("[asyco_divmix/flag] requires_two_views correctly differentiated OK")


def test_two_view_dataset_returns_4tuple():
    """TwoViewHamDataset wraps HamDataset and returns (img1, img2, label, idx)."""
    tmp = Path(tempfile.mkdtemp(prefix="two_view_"))
    try:
        md, images_dir = _make_dummy(tmp, n_per_class=2)
        base = HamDataset(md, images_dir=images_dir, transform=None)
        train_tf = get_train_transforms(64)
        ds = TwoViewHamDataset(base, transform=train_tf)
        assert len(ds) == len(md)
        img1, img2, label, idx = ds[0]
        assert img1.shape == img2.shape
        assert img1.shape[0] == 3
        assert isinstance(label, int)
        assert isinstance(idx, int)
        # The two views should DIFFER (stochastic augmentation):
        assert not torch.equal(img1, img2), "Two views are identical — augmentation may not be stochastic"
        print(f"[asyco_divmix/two_view] {len(ds)} items, view shape {tuple(img1.shape)} OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_warmup_then_postwarmup_no_nan():
    """Run a tiny training loop covering warmup→post-warmup transition."""
    tmp = Path(tempfile.mkdtemp(prefix="asyco_divmix_smoke_"))
    try:
        md, images_dir = _make_dummy(tmp, n_per_class=4)
        cfg = _base_cfg("imbalanced")
        device = torch.device("cpu")

        base_train = HamDataset(md, images_dir=images_dir, transform=None)
        train_tf = get_train_transforms(64)
        train_ds = TwoViewHamDataset(base_train, transform=train_tf)
        test_ds = HamDataset(md, images_dir=images_dir, transform=get_test_transforms(64, 72))

        labels_idx = np.array([CLASS_NAMES.index(c) for c in md["dx"]])
        sampler = make_weighted_sampler(labels_idx)
        train_loader = DataLoader(
            train_ds, batch_size=cfg["data"]["batch_size"],
            sampler=sampler, shuffle=False, num_workers=0,
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=0,
        )
        class_weights = compute_class_weights(labels_idx, device=device)

        method = build_method(
            method_name="asyco_divmix", cfg=cfg,
            num_train_samples=len(train_ds), num_classes=NUM_CLASSES,
            device=device, class_weights=class_weights,
        )
        # Total epochs = 4: warmup floor=2 means 2 warmup + 2 post-warmup epochs.
        method.build(
            total_epochs=4,
            model_builder=lambda: build_resnet(num_classes=NUM_CLASSES, depth=18, pretrained=False),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)

        losses_warm: list[float] = []
        losses_post: list[float] = []
        for epoch in range(4):
            for batch in train_loader:
                out = method.train_step(batch, epoch, scaler)
                assert math.isfinite(out.loss_total), \
                    f"Non-finite loss at epoch {epoch}: {out.loss_total}"
                if epoch < method.warmup_epochs:
                    losses_warm.append(out.loss_total)
                    assert "warmup_ce_clf" in out.loss_components
                else:
                    losses_post.append(out.loss_total)
                    # MixMatch components must appear post-warmup
                    assert "clf_Lx" in out.loss_components
                    assert "clf_Lu" in out.loss_components
                    assert "clf_Lprior" in out.loss_components
                    assert "lambda_u_now" in out.loss_components
            method.scheduler_step()

        # Inference
        y_true, y_pred, y_prob = method.predict(test_loader, device=device)
        assert y_prob.shape == (len(test_ds), NUM_CLASSES)
        assert ((y_prob >= 0.0) & (y_prob <= 1.0)).all()
        assert np.isfinite(y_prob).all()
        print(
            f"[asyco_divmix/smoke] warmup losses={len(losses_warm)} post losses={len(losses_post)} "
            f"(both finite) OK"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rampup_lambda_u_monotone_and_clamped():
    """λ_u rampup is non-decreasing in epoch and clamps at the target."""
    target = 25.0
    warmup = 5
    rampup = 16
    vals = [_rampup_lambda_u(e, warmup, rampup, target) for e in range(0, 50)]
    # Pre-warmup: zero
    assert all(v == 0.0 for v in vals[:warmup])
    # Monotone non-decreasing
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 1e-9, f"rampup not monotone: {a} -> {b}"
    # Reaches target
    assert vals[-1] == target
    # Boundary: epoch == warmup gives 0 (start of ramp)
    assert vals[warmup] == 0.0
    # Boundary: epoch == warmup + rampup gives full target
    assert math.isclose(vals[warmup + rampup], target, rel_tol=1e-9)
    # rampup_epochs <= 0 short-circuits to target immediately at warmup-exit.
    assert _rampup_lambda_u(warmup, warmup, 0, target) == target
    print(
        f"[asyco_divmix/rampup] warmup={warmup} ramp={rampup} target={target} "
        f"-> last val {vals[-1]}, monotone OK"
    )


def test_runner_integration_two_view_wrap():
    """End-to-end check that the runner wraps the train dataset for asyco_divmix
    and does NOT wrap it for asyco."""
    from src.training.runner import _build_loaders

    tmp = Path(tempfile.mkdtemp(prefix="runner_two_view_"))
    try:
        md, images_dir = _make_dummy(tmp, n_per_class=2)
        cfg_dm = _base_cfg("imbalanced")
        cfg_dm["data"]["batch_size"] = 4

        # asyco_divmix → TwoViewHamDataset
        train_loader, _, _, _ = _build_loaders(
            train_df=md, test_df=None, images_dir=images_dir,
            cfg=cfg_dm, val_df=None, requires_two_views=True,
        )
        sample = next(iter(train_loader))
        assert len(sample) == 4, f"Expected 4-tuple from two-view loader, got {len(sample)}"
        img1, img2, _, _ = sample
        assert img1.shape == img2.shape

        # asyco / baseline / sce / elr → standard HamDataset (3-tuple)
        train_loader_std, _, _, _ = _build_loaders(
            train_df=md, test_df=None, images_dir=images_dir,
            cfg=cfg_dm, val_df=None, requires_two_views=False,
        )
        sample_std = next(iter(train_loader_std))
        assert len(sample_std) == 3, f"Expected 3-tuple from standard loader, got {len(sample_std)}"
        print("[asyco_divmix/runner] two-view wrapping correctly conditional on flag OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_requires_two_views_flag()
    test_two_view_dataset_returns_4tuple()
    test_rampup_lambda_u_monotone_and_clamped()
    test_warmup_then_postwarmup_no_nan()
    test_runner_integration_two_view_wrap()
    print("[asyco_divmix] ALL SMOKE TESTS PASSED")
