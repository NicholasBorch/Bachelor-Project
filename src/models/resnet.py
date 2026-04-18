"""ResNet builder.

Supports ResNet-18 (used only for OOF probability collection in Stage 1b) and
ResNet-34 (used for all main training in Stage 2 and Stage 3, and for both
clf_net and ref_net in AsyCo — see PROJECT_DOCUMENTATION.md §8.4).
"""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm


_WEIGHTS = {
    (18, "IMAGENET1K_V1"): tvm.ResNet18_Weights.IMAGENET1K_V1,
    (34, "IMAGENET1K_V1"): tvm.ResNet34_Weights.IMAGENET1K_V1,
}


def build_resnet(
    num_classes: int,
    depth: int = 34,
    pretrained: bool = True,
    weights_name: str = "IMAGENET1K_V1",
) -> nn.Module:
    """Build a ResNet with a replaced final FC layer.

    Args:
        num_classes: output dimensionality (7 for HAM10000).
        depth: 18 or 34.
        pretrained: if True, load ImageNet weights.
        weights_name: which ImageNet weight set to use. Only IMAGENET1K_V1 is
            supported for the depths we use.
    """
    if depth not in (18, 34):
        raise ValueError(f"Only depths 18 and 34 are supported, got {depth}")

    weights = None
    if pretrained:
        key = (depth, weights_name)
        if key not in _WEIGHTS:
            raise ValueError(f"Unsupported (depth, weights): {key}")
        weights = _WEIGHTS[key]

    if depth == 18:
        model = tvm.resnet18(weights=weights)
    else:  # depth == 34
        model = tvm.resnet34(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
