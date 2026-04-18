"""SCE: Symmetric Cross-Entropy (Wang et al. 2019).

L = alpha * CE + beta * RCE

RCE is the "reverse" cross-entropy where ground-truth is treated as the
prediction and predictions are treated as ground-truth. Since q(y|x) = 1 and
q(k|x) = 0 for k != y, log(0) is replaced by a clipping constant A.

Per-sample derivation:
    RCE = -Σ_k p(k|x) * log q(k|x)
        = -p(y|x) * log(1) - Σ_{k != y} p(k|x) * A
        = -A * (1 - p(y|x))

CRITICAL (per paper §5.2): class weights apply to CE term only, not RCE. On
the balanced dataset we pass class_weights=None so this is moot; on imbalanced
the distinction matters.

Hyperparameters from CIFAR-10 settings: alpha=0.1, beta=1.0, A=-4.0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.ham10000 import NUM_CLASSES
from src.methods.base import Method, MethodOutput
from src.training.optim import build_optimizer, build_scheduler


class SCELoss(nn.Module):
    """alpha*CE + beta*RCE."""

    def __init__(self, alpha: float, beta: float, A: float, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.A = float(A)
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        # Standard CE (optionally class-weighted)
        ce = F.cross_entropy(logits, labels, weight=self.class_weights, reduction="mean")

        # RCE (never class-weighted, per paper)
        probs = F.softmax(logits, dim=-1)
        # p(y|x) per sample
        p_y = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        # RCE = -A * (1 - p_y); A is negative so -A > 0.
        rce_per_sample = (-self.A) * (1.0 - p_y)
        rce = rce_per_sample.mean()

        total = self.alpha * ce + self.beta * rce
        return {"total": total, "ce": ce.detach(), "rce": rce.detach()}


class SCEMethod(Method):
    def __init__(self, cfg: dict, device: torch.device, class_weights: torch.Tensor | None = None):
        super().__init__(cfg, device)
        self.class_weights = class_weights
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.criterion: SCELoss | None = None

    def build(self, total_epochs: int, model_builder) -> None:
        self.model = model_builder().to(self.device)
        self.optimizer = build_optimizer(self.model.parameters(), self.cfg["optim"])
        self.scheduler = build_scheduler(
            self.optimizer, total_epochs=total_epochs, name=self.cfg["lr_scheduler"],
        )
        m = self.cfg["method"]
        self.criterion = SCELoss(
            alpha=m["alpha"], beta=m["beta"], A=m["A"],
            class_weights=self.class_weights,
        ).to(self.device)
        self._built = True

    def train_step(self, batch, epoch, scaler) -> MethodOutput:
        images, labels, _idx = batch
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            logits = self.model(images)
            parts = self.criterion(logits, labels)
            loss = parts["total"]

        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()

        return MethodOutput(
            loss_total=float(loss.item()),
            loss_components={
                "ce": float(parts["ce"].item()),
                "rce": float(parts["rce"].item()),
            },
            batch_size=int(images.size(0)),
        )

    def _all_schedulers(self):
        return [self.scheduler]

    def inference_model(self) -> nn.Module:
        return self.model
