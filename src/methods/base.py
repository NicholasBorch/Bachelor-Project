"""Abstract Method base class.

Every training strategy (Baseline, SCE, ELR, AsyCo, AsyCo+DivMix) is a
subclass. The contract is intentionally narrow:

    method.build(total_epochs)          # attach optimizers/schedulers
    for epoch in range(total_epochs):
        for batch in loader:
            out = method.train_step(batch, epoch, scaler)   # returns MethodOutput
        method.scheduler_step()

    preds = method.predict(test_loader)     # returns (y_true, y_pred, y_prob)

Methods with multiple networks (AsyCo, AsyCo+DivMix) own multiple optimizers
and step them all internally inside `train_step`. The external runner doesn't
know or care.

A shared GradScaler is passed into train_step so that mixed precision is
uniform across methods.

Two-view batching
-----------------
Most methods operate on (image, label, idx) batches from ``HamDataset``.
AsyCo+DivMix needs MixMatch-style two-augmentation batches; subclasses opt
in by setting the class attribute ``requires_two_views = True``. The
runner inspects this flag and wraps the train dataset in
``TwoViewHamDataset`` when set, producing (img1, img2, label, idx) batches.
The flag has NO effect on validation/test loaders — all methods evaluate
on standard single-view test data via ``predict()``.
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
    """Per-batch training output returned by Method.train_step.

    Attributes:
        loss_total: scalar float for logging.
        loss_components: dict of named sub-losses for logging (e.g. CE, RCE).
        batch_size: number of samples in the batch (after any filtering).
    """
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
        """Construct model(s), optimizer(s), scheduler(s).

        Args:
            total_epochs: the epoch budget (for cosine annealing T_max).
            model_builder: a zero-arg callable returning a freshly-initialized
                network. Methods that need N networks call it N times.
        """

    @abstractmethod
    def train_step(
        self,
        batch: tuple,
        epoch: int,
        scaler: torch.amp.GradScaler,
    ) -> MethodOutput:
        """One gradient step. Handles forward, loss, backward, and optimizer
        step(s) for all internal networks.

        ``batch`` is (image, label, idx) for normal methods and
        (image1, image2, label, idx) when ``requires_two_views`` is True.
        """

    def scheduler_step(self) -> None:
        """Called once per epoch after all batches. Default: step every
        scheduler the method owns."""
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
        """Standard evaluation loop. Returns (y_true, y_pred, y_prob).

        y_true: (N,) true labels.
        y_pred: (N,) argmax predictions.
        y_prob: (N, C) softmax probabilities.

        Samples are returned in LOADER ORDER (so loader must not shuffle).
        Test/val loaders always provide single-view (img, label, idx)
        batches regardless of ``requires_two_views``.
        """
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
