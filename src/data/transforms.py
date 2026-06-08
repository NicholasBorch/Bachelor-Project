"""
Image transforms. Train and test pipelines are explicit and separate;
normalization uses ImageNet statistics (applied even for from-scratch init, for
fairness across the init axis).
"""
from __future__ import annotations

import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224):
    """Training augmentation pipeline (vertical flip is valid for dermatoscopic images)."""
    return T.Compose([
        T.RandomResizedCrop(size=image_size, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transforms(image_size: int = 224, resize_size: int = 256):
    """Test-time transforms: resize + center crop + normalize. No augmentation."""
    return T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_noise_injection_transforms(image_size: int = 224, normalize: bool = False):
    """Transforms for IDN noise injection (raw [0,1], or +ImageNet norm); not used in training."""
    ops = [T.Resize((image_size, image_size)), T.ToTensor()]
    if normalize:
        ops.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(ops)