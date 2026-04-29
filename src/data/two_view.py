"""TwoViewHamDataset — returns two stochastic augmentations per image.

Required by MixMatch / DivideMix-style training, where label co-guessing
averages predictions over two augmented views and MixUp blends across
labeled and unlabeled samples regardless of view.

This wrapper is constructed AFTER the base ``HamDataset`` so the in-memory
PIL cache is shared; we only re-apply the (stochastic) train transform a
second time. Cost is one extra augmentation pipeline per ``__getitem__``,
no extra image I/O.

Currently used only by ``AsyCoDivMixMethod``. Other methods do not opt in
(via ``Method.requires_two_views = False``) and continue to receive
single-view batches from the standard ``HamDataset``.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import Dataset

from src.data.ham10000 import HamDataset


class TwoViewHamDataset(Dataset):
    """Wraps ``HamDataset`` to return ``(img1, img2, label, idx)`` per item.

    Both views come from independent invocations of the same stochastic
    ``transform`` callable applied to the same source PIL image (which is
    held in the wrapped dataset's in-memory cache, so there is no extra
    file I/O).

    The returned ``idx`` is the same row index used by the wrapped
    dataset, preserving compatibility with ELR-style index-keyed buffers
    if they are ever combined with two-view training in the future.
    """

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
