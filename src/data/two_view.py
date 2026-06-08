"""
TwoViewHamDataset: returns two stochastic augmentations per image.

Required by MixMatch/DivideMix-style training (AsyCoDivMix). Wraps a built
HamDataset so the PIL cache is shared and only the train transform is re-applied a
second time (no extra image I/O). Other methods set requires_two_views = False and
keep single-view batches.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import Dataset

from src.data.ham10000 import HamDataset


class TwoViewHamDataset(Dataset):
    """Wraps HamDataset to return (img1, img2, label, idx); both views from the same cached PIL."""

    def __init__(self, base: HamDataset, transform: Callable):
        if not isinstance(base, HamDataset):
            raise TypeError(
                "TwoViewHamDataset requires a HamDataset; got "
                f"{type(base).__name__}"
            )
        if transform is None:
            raise ValueError(
                "TwoViewHamDataset requires a non-None train transform "
                "to apply twice per item."
            )
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        # Pull the source PIL image directly from the base dataset's cache
        # (or from disk if the base wasn't constructed with preload=True).
        if self.base._images is not None:
            img = self.base._images[idx]
        else:
            from PIL import Image
            img_path = self.base.images_dir / f"{self.base.image_ids[idx]}.jpg"
            img = Image.open(img_path).convert("RGB")

        img1 = self.transform(img)
        img2 = self.transform(img)
        label = int(self.base.labels[idx])
        return img1, img2, label, idx