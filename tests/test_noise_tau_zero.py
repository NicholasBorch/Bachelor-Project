"""Verify that τ=0 short-circuits in BOTH noise generation functions and
produces output bitwise identical to the input.

Run: python -m tests.test_noise_tau_zero
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.ham10000 import CLASS_NAMES, NUM_CLASSES
from src.noise.idn_feature_driven import generate_feature_driven_idn
from src.noise.idn_xia import generate_xia_idn


def _make_dummy_images(tmpdir: Path, n_per_class: int = 2) -> pd.DataFrame:
    images_dir = tmpdir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(0)
    for cls in CLASS_NAMES:
        for k in range(n_per_class):
            iid = f"{cls}_{k:02d}"
            arr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(images_dir / f"{iid}.jpg", "JPEG")
            rows.append({"image_id": iid, "dx": cls})
    return pd.DataFrame(rows)


def test_xia_standard_tau_zero() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="noise_tau0_"))
    try:
        md = _make_dummy_images(tmp)
        out, rep = generate_xia_idn(
            md, images_dir=tmp / "images", tau=0.0, seed=42,
            normalize=False, image_size=32,
        )
        assert rep.n_flipped == 0, f"expected 0 flips, got {rep.n_flipped}"
        assert rep.empirical_rate == 0.0
        assert (out["dx"].values == md["dx"].values).all(), "labels changed at tau=0"
        assert (out["dx_clean"].values == md["dx"].values).all()
        assert (~out["flipped"]).all()
        print("[test] xia-standard tau=0 PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_xia_normalized_tau_zero() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="noise_tau0_"))
    try:
        md = _make_dummy_images(tmp)
        out, rep = generate_xia_idn(
            md, images_dir=tmp / "images", tau=0.0, seed=42,
            normalize=True, image_size=32,
        )
        assert rep.n_flipped == 0
        assert (out["dx"].values == md["dx"].values).all()
        print("[test] xia-normalized tau=0 PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_feature_driven_tau_zero() -> None:
    n = 14
    md = pd.DataFrame({
        "image_id": [f"img_{i:03d}" for i in range(n)],
        "dx": [CLASS_NAMES[i % NUM_CLASSES] for i in range(n)],
    })
    # random oof_probs; for tau=0 they must be IGNORED.
    rng = np.random.default_rng(0)
    oof = rng.random((n, NUM_CLASSES)).astype(np.float32)
    oof = oof / oof.sum(axis=1, keepdims=True)

    out, rep = generate_feature_driven_idn(md, oof_probs=oof, tau=0.0, seed=123)
    assert rep.n_flipped == 0
    assert (out["dx"].values == md["dx"].values).all()
    assert (~out["flipped"]).all()
    print("[test] feature-driven tau=0 PASS")


def test_feature_driven_nonzero_tau_produces_flips() -> None:
    # Sanity counter-check: at tau=0.3 we DO get flips.
    n = 200
    md = pd.DataFrame({
        "image_id": [f"img_{i:04d}" for i in range(n)],
        "dx": [CLASS_NAMES[i % NUM_CLASSES] for i in range(n)],
    })
    rng = np.random.default_rng(0)
    oof = rng.random((n, NUM_CLASSES)).astype(np.float32)
    oof = oof / oof.sum(axis=1, keepdims=True)

    out, rep = generate_feature_driven_idn(md, oof_probs=oof, tau=0.3, seed=123)
    # Empirical rate should be within reasonable range of 0.3 given truncnorm with sigma=0.1
    assert 0.15 < rep.empirical_rate < 0.45, f"unexpected empirical rate: {rep.empirical_rate}"
    print(f"[test] feature-driven tau=0.3 empirical={rep.empirical_rate:.3f} PASS")


def test_reproducibility() -> None:
    # Same seed → same labels.
    n = 100
    md = pd.DataFrame({
        "image_id": [f"i_{i}" for i in range(n)],
        "dx": [CLASS_NAMES[i % NUM_CLASSES] for i in range(n)],
    })
    rng = np.random.default_rng(0)
    oof = rng.random((n, NUM_CLASSES)).astype(np.float32)
    oof = oof / oof.sum(axis=1, keepdims=True)

    a, _ = generate_feature_driven_idn(md, oof_probs=oof, tau=0.2, seed=777)
    b, _ = generate_feature_driven_idn(md, oof_probs=oof, tau=0.2, seed=777)
    assert (a["dx"].values == b["dx"].values).all(), "same seed gave different labels"
    print("[test] reproducibility PASS")


if __name__ == "__main__":
    test_xia_standard_tau_zero()
    test_xia_normalized_tau_zero()
    test_feature_driven_tau_zero()
    test_feature_driven_nonzero_tau_produces_flips()
    test_reproducibility()
    print("[test] ALL NOISE TESTS PASSED")
