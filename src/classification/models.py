# Model definitions for HAM10000 classification experiments.
import torch.nn as nn
from torchvision import models


def build_resnet(num_classes: int, pretrained: bool = True, depth: int = 50) -> nn.Module:
    # Builds a ResNet with replaced classification head for num_classes outputs
    # depth=18: used for IDN inner fold baseline models
    # depth=50: used for all main classification experiments
    if depth == 18:
        model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    elif depth == 50:
        model = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
    else:
        raise ValueError(f"Unsupported depth: {depth}. Choose 18 or 50.")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model