"""Smoke test for the data and model layer.

Run with: python -m tests.test_data_and_model

This test does NOT require the real dataset — it synthesizes a tiny dummy
metadata + image set so you can verify the code loads without errors before
running Stage 0.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.folds import create_fold_assignments, split_train_test_by_fold
from src.data.ham10000 import CLASS_NAMES, NUM_CLASSES, HamDataset
from src.data.transforms import get_test_transforms, get_train_transforms
from src.models.resnet import build_resnet


def _make_dummy(tmpdir: Path, n_per_class: int = 3) -> pd.DataFrame:
    """Create n_per_class * 7 = 21 dummy images with known class labels."""
    images_dir = tmpdir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cls in CLASS_NAMES:
        for k in range(n_per_class):
            iid = f"{cls}_{k:02d}"
            arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            Image.fromarray(arr).save(images_dir / f"{iid}.jpg", "JPEG")
            rows.append({"image_id": iid, "dx": cls})
    return pd.DataFrame(rows)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ham_smoke_"))
    try:
        print(f"[smoke] working dir: {tmp}")
        metadata = _make_dummy(tmp)
        print(f"[smoke] built dummy metadata, {len(metadata)} rows")

        # Fold assignment
        folds = create_fold_assignments(metadata, n_splits=3, seed=10)
        assert set(folds["fold"].unique()) == {0, 1, 2}
        train_df, test_df = split_train_test_by_fold(metadata, folds, test_fold=0)
        assert len(train_df) + len(test_df) == len(metadata)
        print(f"[smoke] fold 0 split: train={len(train_df)}, test={len(test_df)}")

        # Dataset + transforms
        train_ds = HamDataset(train_df, images_dir=tmp / "images",
                              transform=get_train_transforms(image_size=64))
        test_ds = HamDataset(test_df, images_dir=tmp / "images",
                             transform=get_test_transforms(image_size=64, resize_size=72))
        img, label, idx = train_ds[0]
        assert isinstance(img, torch.Tensor) and img.shape == (3, 64, 64)
        assert 0 <= label < NUM_CLASSES
        assert isinstance(idx, int)
        print(f"[smoke] sample shape={tuple(img.shape)}, label={label}, idx={idx}")

        # Model
        model = build_resnet(num_classes=NUM_CLASSES, depth=18, pretrained=False)
        model.eval()
        with torch.no_grad():
            logits = model(img.unsqueeze(0))
        assert logits.shape == (1, NUM_CLASSES)
        print(f"[smoke] resnet18 logits shape={tuple(logits.shape)}")

        print("[smoke] ALL CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
