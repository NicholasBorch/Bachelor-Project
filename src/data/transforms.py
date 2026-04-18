"""Image transforms.

Training and test transforms are kept separate and explicit. Normalization
uses ImageNet statistics because the backbone is ImageNet-pretrained (and we
apply the same normalization even for from-scratch initialization, for
fairness across the init axis).
"""
from __future__ import annotations

import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224):
    """Training augmentation pipeline.

    Matches the augmentations documented in PROJECT_DOCUMENTATION.md §9.
    Medical dermatoscopic images: vertical flip is valid because lesions have
    no canonical orientation.
    """
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
    """Transforms used during IDN noise injection (NOT during training).

    The standard Xia IDN algorithm uses raw [0, 1] ToTensor images; the
    normalized variant additionally applies ImageNet normalization. This is
    independent of the training-time transforms.
    """
    ops = [T.Resize((image_size, image_size)), T.ToTensor()]
    if normalize:
        ops.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(ops)
