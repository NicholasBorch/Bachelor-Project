"""HAM10000 dataset.

The class index mapping is FIXED and ALPHABETICAL. Every metric, plot, and
model head in the project relies on this ordering. Do not change it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Fixed alphabetical ordering — DO NOT REORDER.
CLASS_NAMES: list[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES: int = len(CLASS_NAMES)
_CLASS_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
_INDEX_TO_CLASS: dict[int, str] = {i: c for i, c in enumerate(CLASS_NAMES)}


def class_to_index(name: str) -> int:
    return _CLASS_TO_INDEX[name]


def index_to_class(idx: int) -> str:
    return _INDEX_TO_CLASS[idx]


class HamDataset(Dataset):
    """In-memory HAM10000 dataset.

    Reads a metadata CSV (columns must include `image_id` and `dx`, and
    optionally `fold`). Loads all images into memory at init for fast training
    (the full 7,470-image imbalanced set is ~4.5 GB in uint8 form, which fits
    comfortably in V100 RAM).

    Returns (image_tensor, label_index, sample_index). The sample_index is the
    row index into the metadata DataFrame and is REQUIRED by ELR for target
    buffer updates. Other methods can ignore it.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        images_dir: str | Path,
        transform: Callable | None = None,
        label_col: str = "dx",
        preload: bool = True,
    ):
        self.metadata = metadata.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.label_col = label_col

        # Map string labels to indices
        self.labels = self.metadata[label_col].map(_CLASS_TO_INDEX).to_numpy()
        if (self.labels < 0).any() or pd.isna(self.labels).any():
            unknown = set(self.metadata[label_col]) - set(CLASS_NAMES)
            raise ValueError(f"Unknown class labels in metadata: {unknown}")

        self.image_ids = self.metadata["image_id"].tolist()

        # Preload images as PIL (transforms applied on-the-fly per __getitem__)
        self._images: list[Image.Image] | None = None
        if preload:
            self._images = []
            for iid in self.image_ids:
                img_path = self.images_dir / f"{iid}.jpg"
                img = Image.open(img_path).convert("RGB")
                img.load()  # force read into memory
                self._images.append(img)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        if self._images is not None:
            img = self._images[idx]
        else:
            img_path = self.images_dir / f"{self.image_ids[idx]}.jpg"
            img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            # Fallback: just convert to tensor without normalization
            import torchvision.transforms.functional as TF
            img_tensor = TF.to_tensor(img)

        label = int(self.labels[idx])
        return img_tensor, label, idx
