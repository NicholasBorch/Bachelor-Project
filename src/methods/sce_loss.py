# src/methods/sce_loss.py
#
# Symmetric Cross Entropy loss following Wang et al. (2019).
# arXiv:1908.06112
#
# Addresses two failure modes of standard CE under noisy labels:
#
#   1. Overfitting — CE produces large unbounded gradients on confidently
#      wrong predictions, pulling the model strongly toward corrupted labels.
#
#   2. Under-learning — hard classes with overlapping representations never
#      receive sufficient gradient signal and converge below clean performance.
#
# SCE combines CE with a Reverse Cross Entropy (RCE) term:
#
#   L_SCE = alpha * L_CE + beta * L_RCE
#
# where:
#   L_CE  = -sum_k q(k|x) log p(k|x)   standard cross entropy
#   L_RCE = -sum_k p(k|x) log q(k|x)   reverse cross entropy
#
# RCE is noise tolerant by construction. With one-hot labels, q(k|x) = 1 for
# the correct class and 0 elsewhere. log(0) is replaced by a small negative
# clipping constant A. When a label is corrupted, q assigns mass to the wrong
# class — but the bounded clipping at A limits the gradient contribution of
# that sample, making RCE inherently resistant to label noise.
#
# CE provides the convergence drive on clean samples. RCE suppresses gradient
# damage from corrupted ones. Together they address both failure modes.
#
# Default hyperparameters from the paper (ResNet, moderate noise):
#   alpha = 0.1   small CE weight limits overfitting while keeping convergence
#   beta  = 1.0   full RCE weight for maximum noise suppression
#   A     = -4.0  clipping constant — method is insensitive to A when alpha is small

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCELoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 1.0,
        num_classes: int = 7,
        A: float = -4.0,
        class_weights: torch.Tensor = None,
    ) -> None:
        """
        Parameters
        ----------
        alpha         : Weight on CE term. Small alpha (0.1) limits overfitting
                        while retaining convergence signal on clean samples.
        beta          : Weight on RCE term. beta=1.0 is the paper's default.
        num_classes   : Number of output classes.
        A             : log(0) clipping constant — must be negative. Controls
                        RCE gradient scale. Paper default is -4.0. The method
                        is not sensitive to A when alpha is small.
        class_weights : Optional inverse-frequency weights applied to the CE
                        term only, matching the weighted CE used in baseline.
                        RCE operates on predicted probabilities and is left
                        unweighted.
        """
        super().__init__()
        self.alpha        = alpha
        self.beta         = beta
        self.num_classes  = num_classes
        self.A            = A
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C) raw model outputs before softmax
        targets : (B,)  integer class labels (may be noisy)
        """
        # ── CE term ───────────────────────────────────────────────────────
        # Weighted cross entropy — identical weighting to baseline for fair
        # comparison. Drives learning on clean samples but is not noise tolerant.
        loss_ce = F.cross_entropy(logits, targets, weight=self.class_weights)

        # ── RCE term ──────────────────────────────────────────────────────
        # RCE = -sum_k p(k|x) * log(q(k|x))
        #
        # With one-hot q:
        #   log(q_y) = log(1) = 0      for the correct class position
        #   log(q_k) = log(0) = A      for all other positions
        #
        # This means RCE only penalises probability mass assigned to wrong
        # classes, and that penalty is bounded by A — making it noise tolerant.
        # When a corrupted label sets q_y=1 at the wrong position, the gradient
        # contribution of that sample is still bounded by A.

        # Convert logits to softmax predictions p(k|x)
        pred = F.softmax(logits, dim=1)  # (B, C)

        # Build log(q): fill with A everywhere, then set the correct class
        # position to log(1) = 0
        log_q = torch.full_like(pred, self.A)         # (B, C) filled with A
        log_q.scatter_(1, targets.unsqueeze(1), 0.0)  # log(q_y) = 0

        # Average RCE over the batch
        loss_rce = -torch.mean(torch.sum(pred * log_q, dim=1))

        # ── Combined SCE loss ──────────────────────────────────────────────
        return self.alpha * loss_ce + self.beta * loss_rce