"""
Abstract Method base class.

Every training strategy (Baseline, SCE, ELR, AsyCo+DivMix) subclasses this.
Contract: build(total_epochs, model_builder); per epoch, train_step(batch, epoch,
scaler) -> MethodOutput then scheduler_step(); predict(loader) -> (y_true, y_pred,
y_prob). Methods owning multiple networks step them all inside train_step, and a
shared GradScaler keeps mixed precision uniform.

Subclasses needing MixMatch-style two-view batches set requires_two_views = True;
the runner then wraps the train dataset in TwoViewHamDataset. Validation/test
always use single-view data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MethodOutput:
    """Per-batch training output: loss_total, loss_components (named sub-losses), batch_size."""
    loss_total: float
    loss_components: dict[str, float]
    batch_size: int


class Method(ABC):
    """Base class for all training methods."""

    # Subclasses set this to True if they need TwoViewHamDataset batches
    # (img1, img2, label, idx) for MixMatch-style training. The runner reads
    # this attribute to decide whether to wrap the train dataset.
    requires_two_views: bool = False

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._built = False

    @abstractmethod
    def build(self, total_epochs: int, model_builder) -> None:
        """Construct model(s), optimizer(s), scheduler(s); model_builder is a zero-arg net factory."""

    @abstractmethod
    def train_step(
        self,
        batch: tuple,
        epoch: int,
        scaler: torch.amp.GradScaler,
    ) -> MethodOutput:
        """One gradient step (forward, loss, backward, optimizer step) for all internal networks."""

    def scheduler_step(self) -> None:
        """Step every scheduler the method owns (once per epoch)."""
        for sched in self._all_schedulers():
            sched.step()

    @abstractmethod
    def _all_schedulers(self) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """Return all schedulers owned by this method."""

    @abstractmethod
    def inference_model(self) -> nn.Module:
        """Return the network used for evaluation."""

    @torch.no_grad()
    def predict(
        self, loader, device: torch.device | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate in loader order (no shuffle); returns (y_true, y_pred, y_prob)."""
        if device is None:
            device = self.device
        model = self.inference_model()
        model.eval()
        trues, preds, probs = [], [], []
        for images, labels, _idx in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                p = F.softmax(logits.float(), dim=1)
            trues.append(labels.numpy())
            preds.append(p.argmax(dim=1).cpu().numpy())
            probs.append(p.cpu().numpy())
        return np.concatenate(trues), np.concatenate(preds), np.concatenate(probs)