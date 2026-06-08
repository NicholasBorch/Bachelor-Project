"""
Baseline: standard cross-entropy training — the reference curve every robust
method is compared against. On the imbalanced arm CE is class-weighted.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.methods.base import Method, MethodOutput
from src.training.optim import build_optimizer, build_scheduler


class BaselineMethod(Method):
    """Plain cross-entropy. Serves as the reference curve."""

    def __init__(self, cfg: dict, device: torch.device, class_weights: torch.Tensor | None = None):
        super().__init__(cfg, device)
        self.class_weights = class_weights
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.criterion: nn.Module | None = None

    def build(self, total_epochs: int, model_builder) -> None:
        self.model = model_builder().to(self.device)
        self.optimizer = build_optimizer(self.model.parameters(), self.cfg["optim"])
        self.scheduler = build_scheduler(
            self.optimizer, total_epochs=total_epochs, name=self.cfg["lr_scheduler"],
        )
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self._built = True

    def train_step(self, batch, epoch, scaler) -> MethodOutput:
        images, labels, _idx = batch
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            logits = self.model(images)
            loss = self.criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()

        return MethodOutput(
            loss_total=float(loss.item()),
            loss_components={"ce": float(loss.item())},
            batch_size=int(images.size(0)),
        )

    def _all_schedulers(self):
        return [self.scheduler]

    def inference_model(self) -> nn.Module:
        return self.model