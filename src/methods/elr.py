"""ELR: Early-Learning Regularization (Liu et al. 2020).

L_ELR = L_CE + lambda * mean[log(1 - <p_i, t_i>)]

The target vector t_i is a per-sample temporal moving average of softmax
predictions:
    t_i(k) = beta * t_i(k-1) + (1 - beta) * p_i(k)

CRITICAL (the ELR detach bug this entire project was built around):
  - The inner product <p_i, t_i> in the REGULARIZATION term uses
    probs WITH gradients (created from logits via softmax in the forward pass).
  - The target BUFFER UPDATE uses probs_detached (we don't want gradients
    leaking into the stored buffer across iterations).
  - The buffer t_i itself is a plain tensor without gradients; it acts like
    a "target" that p_i is being pulled toward.

If you detach BOTH sides, the regularization term becomes a gradient-free
constant and ELR degenerates into pure CE. That is THE bug; the fix is below.

Hyperparameters from CIFAR-10 settings: lambda=3.0, beta=0.7.
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
        # ----- CE term -----
        ce = F.cross_entropy(logits, labels, weight=self.class_weights, reduction="mean")

        # ----- Softmax WITH gradients for the regularization term -----
        probs = F.softmax(logits, dim=1)  # has gradients

        # ----- Target buffer update uses detached probs -----
        with torch.no_grad():
            probs_detached = probs.detach()
            # Update rows indicated by `indices`. Note that rare-sample indices
            # may not appear in this batch — that's fine, their buffer rows
            # just stay as they were.
            self.target[indices] = (
                self.beta * self.target[indices]
                + (1.0 - self.beta) * probs_detached.to(self.target.dtype)
            )

        # ----- Regularization term uses probs (with grads) and the current
        # (detached-by-construction) target buffer -----
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
