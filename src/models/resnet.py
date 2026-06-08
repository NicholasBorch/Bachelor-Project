"""
ResNet builder. ResNet-18 for OOF probability collection (Stage 1b); ResNet-34 for
all main training (Stages 2-3, and both clf_net and ref_net in AsyCo).
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
    """Build a ResNet-18/34 with a replaced final FC; optionally ImageNet-pretrained."""
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