# src/methods/elr_loss.py
#
# ELR loss module following the official implementation of Liu et al. (2020).
# https://github.com/shengliu66/ELR
#
# The loss module owns the target buffer internally.
# The dataset must return sample indices as the third element so the correct
# target row can be updated each iteration.
#
# Full loss:
#   L_ELR = L_CE + lambda * mean_i[ log(1 - <p_i, t_i>) ]
#
# Target update per iteration:
#   t_i(k) = beta * t_i(k-1) + (1 - beta) * p_i(k)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ELRLoss(nn.Module):
    def __init__(
        self,
        num_examp: int,
        num_classes: int = 7,
        elr_lambda: float = 3.0,
        beta: float = 0.7,
        device: torch.device = None,
        class_weights: torch.Tensor = None,
    ) -> None:
        """
        Parameters
        ----------
        num_examp     : Total number of training examples N.
        num_classes   : Number of output classes C.
        elr_lambda    : Regularisation coefficient lambda.
        beta          : Temporal ensembling momentum for target update.
        device        : Device to store target buffer on.
        class_weights : Optional inverse-frequency weights for CE term only.
        """
        super().__init__()
        self.num_classes   = num_classes
        self.elr_lambda    = elr_lambda
        self.beta          = beta
        self.class_weights = class_weights

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Target buffer — shape (N, C), initialised to zeros following
        # the official implementation. Stored on the training device.
        self.register_buffer(
            "target",
            torch.zeros(num_examp, num_classes, device=device)
        )

    def forward(
        self,
        index: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        index  : (B,) integer sample indices into the target buffer
        logits : (B, C) raw model outputs
        labels : (B,) integer class labels (may be noisy)
        """
        # ── CE term ───────────────────────────────────────────────────────
        loss_ce = F.cross_entropy(logits, labels, weight=self.class_weights)

        # ── Softmax predictions (detached — target update is not trained) ─
        probs = torch.softmax(logits.detach(), dim=1)

        # ── Update targets with temporal ensembling ───────────────────────
        # t_i(k) = beta * t_i(k-1) + (1 - beta) * p_i(k)
        self.target[index] = (
            self.beta * self.target[index] + (1.0 - self.beta) * probs
        )

        # ── ELR regularisation term ───────────────────────────────────────
        # -log(1 - <p_i, t_i>) — clamp inner product away from 1 to avoid
        # log(0). Use the freshly updated targets for consistency with the
        # official implementation.
        t     = self.target[index]
        inner = (probs * t).sum(dim=1).clamp(max=1.0 - 1e-4)
        loss_elr = -torch.log(1.0 - inner).mean()

        return loss_ce + self.elr_lambda * loss_elr