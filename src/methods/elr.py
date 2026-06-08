"""
ELR: Early-Learning Regularization (Liu et al. 2020).

Deep nets fit clean patterns early and memorize noisy labels later; ELR keeps a
running estimate of those early predictions and pulls the model toward it,
resisting the later drift.

    L_ELR = CE + lambda * mean_i log(1 - <p_i, t_i>)

t_i is a per-sample EMA of the softmax prediction, held in a non-trainable
(N, C) buffer that is zeroed per fold:

    t_i(k) = beta * t_i(k-1) + (1 - beta) * p_i(k).

<p_i, t_i> grows as the prediction aligns with the target and log(1 - .) falls,
so the term rewards alignment; the inner product is clamped at 1 - 1e-4 to avoid
log 0.

Class weights (imbalanced arm) apply to the CE term only, matching the baseline.
lambda, beta are tuned per protocol via Optuna; the paper's CIFAR-10 reference
values are lambda=3.0, beta=0.7.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.base import Method, MethodOutput
from src.training.optim import build_optimizer, build_scheduler


class ELRLoss(nn.Module):
    """ELR loss with per-sample target buffer."""

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        lambda_elr: float,
        beta: float,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.lambda_elr = float(lambda_elr)
        self.beta = float(beta)
        self.num_classes = int(num_classes)
        self.class_weights = class_weights

        # Target buffer, registered as a buffer so it moves with .to(device)
        # but is NOT trainable.
        self.register_buffer(
            "target",
            torch.zeros(num_samples, num_classes, dtype=torch.float32),
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # CE term
        ce = F.cross_entropy(logits, labels, weight=self.class_weights, reduction="mean")

        # Softmax WITH gradients for the regularization term
        probs = F.softmax(logits, dim=1)  # has gradients

        # Target buffer update uses detached probs
        with torch.no_grad():
            probs_detached = probs.detach()
            # Update rows indicated by `indices`.
            self.target[indices] = (
                self.beta * self.target[indices]
                + (1.0 - self.beta) * probs_detached.to(self.target.dtype)
            )

        # regularization term: <p_i, t_i> with grads on p, against the detached target buffer
        t = self.target[indices].to(probs.dtype)  # shape (B, C), no grad
        inner = (probs * t).sum(dim=1)            # shape (B,), has grad via probs
        # Clamp for numerical stability around <p, t> ≈ 1.
        inner = torch.clamp(inner, max=1.0 - 1e-4)
        reg = torch.log(1.0 - inner).mean()       # negative scalar

        total = ce + self.lambda_elr * reg

        return {"total": total, "ce": ce.detach(), "reg": reg.detach()}


class ELRMethod(Method):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        num_train_samples: int,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(cfg, device)
        self.num_train_samples = num_train_samples
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.criterion: ELRLoss | None = None

    def build(self, total_epochs: int, model_builder) -> None:
        self.model = model_builder().to(self.device)
        self.optimizer = build_optimizer(self.model.parameters(), self.cfg["optim"])
        self.scheduler = build_scheduler(
            self.optimizer, total_epochs=total_epochs, name=self.cfg["lr_scheduler"],
        )
        m = self.cfg["method"]
        self.criterion = ELRLoss(
            num_samples=self.num_train_samples,
            num_classes=self.num_classes,
            lambda_elr=m["lambda"],
            beta=m["beta"],
            class_weights=self.class_weights,
        ).to(self.device)
        self._built = True

    def train_step(self, batch, epoch, scaler) -> MethodOutput:
        images, labels, indices = batch
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        indices = indices.to(self.device, non_blocking=True)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            logits = self.model(images)
            parts = self.criterion(logits, labels, indices)
            loss = parts["total"]

        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()

        return MethodOutput(
            loss_total=float(loss.item()),
            loss_components={
                "ce": float(parts["ce"].item()),
                "reg": float(parts["reg"].item()),
            },
            batch_size=int(images.size(0)),
        )

    def _all_schedulers(self):
        return [self.scheduler]

    def inference_model(self) -> nn.Module:
        return self.model