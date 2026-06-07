#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the per-channel (RGB) mean and standard deviation of the
one_image_per_lesion HAM10000 split.

The statistics are computed on pixel values scaled to [0, 1] (i.e. the same
scale ToTensor produces), so they are directly comparable to the ImageNet
channel statistics mu = (0.485, 0.456, 0.406), sigma = (0.229, 0.224, 0.225)
referenced in the methods.

Streaming accumulation (sum and sum-of-squares over all pixels) is used so the
whole dataset is never held in memory; std is the population std over every
pixel of every image.

Usage:
    python compute_dataset_mean_std.py
    python compute_dataset_mean_std.py --image-dir /path/to/images
    python compute_dataset_mean_std.py --resize 224      # match training resize
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# Default location from the project tree:
# data/processed/HAM10000/one_image_per_lesion/images
DEFAULT_IMAGE_DIR = Path(
    "data/processed/HAM10000/one_image_per_lesion/images"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_image_paths(image_dir: Path):
    for p in sorted(image_dir.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


def compute_mean_std(image_dir: Path, resize: int | None = None):
    """Return (mean, std, n_images, n_pixels) over RGB channels in [0, 1].

    Uses running sums so memory stays flat regardless of dataset size:
        mean_c   = sum(x_c) / N
        std_c    = sqrt( sum(x_c^2)/N - mean_c^2 )
    where N is the total pixel count per channel and x in [0, 1].
    """
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    n_pixels = 0          # per-channel pixel count (same for all 3 channels)
    n_images = 0

    for path in iter_image_paths(image_dir):
        with Image.open(path) as im:
            im = im.convert("RGB")
            if resize is not None:
                im = im.resize((resize, resize), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float64) / 255.0   # [H, W, 3] in [0,1]

        pixels = arr.reshape(-1, 3)            # [H*W, 3]
        channel_sum += pixels.sum(axis=0)
        channel_sum_sq += (pixels ** 2).sum(axis=0)
        n_pixels += pixels.shape[0]
        n_images += 1

    if n_images == 0:
        raise FileNotFoundError(
            f"No images found under {image_dir!s}. "
            f"Check the path or pass --image-dir."
        )

    mean = channel_sum / n_pixels
    var = channel_sum_sq / n_pixels - mean ** 2
    var = np.clip(var, 0.0, None)             # guard tiny negative from FP error
    std = np.sqrt(var)
    return mean, std, n_images, n_pixels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR,
                    help=f"Folder of images (default: {DEFAULT_IMAGE_DIR}).")
    ap.add_argument("--resize", type=int, default=None,
                    help="Optional square resize before accumulating, e.g. 224 "
                         "to match the training transform. Omit to use native "
                         "resolution.")
    args = ap.parse_args()

    mean, std, n_images, n_pixels = compute_mean_std(args.image_dir, args.resize)

    print(f"images        : {n_images}")
    print(f"pixels/channel : {n_pixels:,}")
    print(f"resize        : {args.resize if args.resize else 'native'}")
    print()
    print("Per-channel statistics on [0, 1] (R, G, B):")
    print(f"  mean = ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f})")
    print(f"  std  = ({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f})")
    print()
    print("ImageNet reference for comparison:")
    print("  mean = (0.4850, 0.4560, 0.4060)")
    print("  std  = (0.2290, 0.2240, 0.2250)")


if __name__ == "__main__":
    main()