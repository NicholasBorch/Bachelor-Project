# Model definitions for HAM10000 classification experiments.

import torch.nn as nn
from torchvision import models


def build_resnet(num_classes: int, pretrained: bool = True) -> nn.Module:
    # ResNet-18 backbone with replaced classification head for num_classes outputs
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model