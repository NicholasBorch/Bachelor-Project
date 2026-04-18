"""AsyCo: Asymmetric Co-teaching with Multi-view Consensus (Liu et al. 2023).

Two networks with the SAME architecture (ResNet-34 in our setup) but ASYMMETRIC
training objectives:

    clf_net:  single-label multi-class (cross-entropy)
    ref_net:  multi-label (binary cross-entropy, K-hot targets)

Key concepts from the paper (§3):

1. Warmup phase (first `warmup_epochs` epochs):
    - clf_net: CE on noisy labels
    - ref_net: BCE on noisy labels (treated as 1-hot multi-label)

2. Multi-view consensus (post-warmup). For every sample i, three label views:
    y_tilde:    the noisy training label (1-hot)
    y_n_tilde:  clf_net's argmax prediction (1-hot)
    y_r_tilde:  ref_net's top-K prediction (K-hot multi-label)
    K=1 for CIFAR-10-sized label spaces (our setting for HAM10000).

3. Per-sample weight for clf_net training (Eq. 4):
       w = +1 (clean)     if AG > 0 AND y_tilde ∈ y_r_tilde  (subsets C, SC, RY)
       w =  0 (noisy)     if AG > 0 AND y_tilde ∉ y_r_tilde  (subsets NY, NR)
       w = -1 (discard)   if AG == 0                         (subset U)

   where AG = y_tilde·y_n_tilde + y_n_tilde·y_r_tilde + y_tilde·y_r_tilde.

4. clf_net loss (Eq. 5):
       L_clf = Σ_{w=+1} CE(y_tilde, p_clf) + λ_u · Σ_{w=0} MSE(sharpen(p_clf, T), p_clf)
   The MSE is a self-consistency term pushing noisy-sample predictions toward
   their own sharpened version (encouraging confidence). Discarded samples
   (w=-1) contribute nothing.

5. Re-labeled targets for ref_net (Eq. 6):
       y_hat = y_n_tilde                      if sample in SideCore (SC)
       y_hat = y_tilde + y_n_tilde             if sample in NR (multi-label)
       y_hat = y_tilde                         otherwise

6. ref_net loss: BCE(y_hat, sigmoid(ref_logits)).

BatchNorm protection: instead of doing subset-specific forward passes (which
corrupt BN running stats with biased subset statistics, per the AsyCo BN bug
discovered in the original codebase), we do a SINGLE forward pass per network
per batch and mask the per-sample losses by subset. This is both cleaner and
faster than the BN-momentum-zero trick mentioned in PROJECT_DOCUMENTATION §8.4,
and is mathematically equivalent for everything except the BN running stats
— which is exactly what we want to protect.

Hyperparameters from CIFAR-10 settings: K=1, lambda_u=25.0, T=0.5.
Warmup: max(5, round(0.05 * total_epochs)) — see PROJECT_DOCUMENTATION §2.3.3.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.ham10000 import NUM_CLASSES
from src.methods.base import Method, MethodOutput
from src.training.optim import build_optimizer, build_scheduler


def _sharpen(p: torch.Tensor, T: float) -> torch.Tensor:
    """Temperature sharpening: p^(1/T) / sum(p^(1/T)). T < 1 sharpens."""
    p_sharp = p.pow(1.0 / T)
    return p_sharp / p_sharp.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _compute_warmup_epochs(total_epochs: int, pct: float, floor: int) -> int:
    return max(floor, int(round(pct * total_epochs)))


class AsyCoMethod(Method):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(cfg, device)
        self.class_weights = class_weights
        m = cfg["method"]
        self.K = int(m["K"])
        self.lambda_u = float(m["lambda_u"])
        self.T = float(m["temperature"])
        self.warmup_pct = float(m["warmup_epochs_pct"])
        self.warmup_floor = int(m["warmup_epochs_floor"])
        self.total_epochs: int | None = None
        self.warmup_epochs: int | None = None

        self.clf_net: nn.Module | None = None
        self.ref_net: nn.Module | None = None
        self.clf_opt: torch.optim.Optimizer | None = None
        self.ref_opt: torch.optim.Optimizer | None = None
        self.clf_sched: torch.optim.lr_scheduler.LRScheduler | None = None
        self.ref_sched: torch.optim.lr_scheduler.LRScheduler | None = None

    def build(self, total_epochs: int, model_builder) -> None:
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = _compute_warmup_epochs(
            total_epochs=total_epochs, pct=self.warmup_pct, floor=self.warmup_floor,
        )
        # Two independent models, SAME architecture.
        self.clf_net = model_builder().to(self.device)
        self.ref_net = model_builder().to(self.device)

        # Two optimizers, identical config. Two schedulers.
        self.clf_opt = build_optimizer(self.clf_net.parameters(), self.cfg["optim"])
        self.ref_opt = build_optimizer(self.ref_net.parameters(), self.cfg["optim"])
        self.clf_sched = build_scheduler(
            self.clf_opt, total_epochs=total_epochs, name=self.cfg["lr_scheduler"],
        )
        self.ref_sched = build_scheduler(
            self.ref_opt, total_epochs=total_epochs, name=self.cfg["lr_scheduler"],
        )
        self._built = True

    # ------------------------------------------------------------------
    # Warmup-phase loss builders
    # ------------------------------------------------------------------
    def _warmup_clf_loss(self, clf_logits, labels):
        """Standard CE (class-weighted if imbalanced)."""
        return F.cross_entropy(clf_logits, labels, weight=self.class_weights)

    def _warmup_ref_loss(self, ref_logits, labels):
        """Multi-label BCE with 1-hot targets (noisy label treated as one-hot).
        No class weights on ref_net (BCE is per-class independent)."""
        onehot = F.one_hot(labels, num_classes=NUM_CLASSES).float()
        return F.binary_cross_entropy_with_logits(ref_logits, onehot)

    # ------------------------------------------------------------------
    # Post-warmup: multi-view consensus and asymmetric losses
    # ------------------------------------------------------------------
    def _compute_views_and_subsets(
        self,
        noisy_labels: torch.Tensor,         # (B,) long
        clf_probs_detached: torch.Tensor,   # (B, C)
        ref_sigmoid_detached: torch.Tensor, # (B, C)
    ):
        """Compute the three label views (as multi-hot 0/1 tensors) and the
        per-sample agreement sets. Returns:
            y_tilde:   (B, C) one-hot
            y_n_tilde: (B, C) one-hot from clf argmax
            y_r_tilde: (B, C) K-hot from ref top-K
        """
        B = noisy_labels.size(0)
        y_tilde = F.one_hot(noisy_labels, num_classes=NUM_CLASSES).float()

        n_argmax = clf_probs_detached.argmax(dim=1)
        y_n_tilde = F.one_hot(n_argmax, num_classes=NUM_CLASSES).float()

        topk_idx = ref_sigmoid_detached.topk(self.K, dim=1).indices  # (B, K)
        y_r_tilde = torch.zeros_like(y_tilde)
        y_r_tilde.scatter_(1, topk_idx, 1.0)
        return y_tilde, y_n_tilde, y_r_tilde

    @staticmethod
    def _compute_weights(
        y_tilde: torch.Tensor,
        y_n_tilde: torch.Tensor,
        y_r_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample weight in {+1, 0, -1} per Eq. 4."""
        # Inner products (per-sample indicators 0/1 because these are disjoint masks)
        t_n = (y_tilde * y_n_tilde).sum(dim=1)
        n_r = (y_n_tilde * y_r_tilde).sum(dim=1)
        t_r = (y_tilde * y_r_tilde).sum(dim=1)
        AG = t_n + n_r + t_r
        # Start with discard (-1), upgrade to noisy (0) if AG>0, clean (+1) if also y_tilde matches ref top-K.
        w = torch.full_like(AG, -1.0)
        w = torch.where(AG > 0, torch.zeros_like(w), w)
        w = torch.where((AG > 0) & (t_r > 0), torch.ones_like(w), w)
        return w

    @staticmethod
    def _compute_relabels(
        y_tilde: torch.Tensor,
        y_n_tilde: torch.Tensor,
        y_r_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample re-labeled multi-hot target y_hat for ref_net (Eq. 6).

        SideCore: y_tilde != y_n_tilde, y_n_tilde in y_r_tilde, y_tilde in y_r_tilde
                  → y_hat = y_n_tilde
        NR:       y_tilde != y_n_tilde, y_n_tilde in y_r_tilde, y_tilde not in y_r_tilde
                  → y_hat = y_tilde + y_n_tilde   (multi-label)
        otherwise → y_hat = y_tilde
        """
        t_n = (y_tilde * y_n_tilde).sum(dim=1)       # 1 if y_tilde == y_n_tilde else 0
        n_r = (y_n_tilde * y_r_tilde).sum(dim=1)
        t_r = (y_tilde * y_r_tilde).sum(dim=1)

        is_SC = (t_n == 0) & (n_r > 0) & (t_r > 0)
        is_NR = (t_n == 0) & (n_r > 0) & (t_r == 0)

        y_hat = y_tilde.clone()
        # NR first: additive multi-label
        y_hat = torch.where(is_NR.unsqueeze(1), y_tilde + y_n_tilde, y_hat)
        # SideCore: replace with y_n_tilde
        y_hat = torch.where(is_SC.unsqueeze(1), y_n_tilde, y_hat)
        # Clamp to 0/1 in case of future combinations.
        return y_hat.clamp(0.0, 1.0)

    def _postwarmup_clf_loss(
        self,
        clf_logits: torch.Tensor,
        noisy_labels: torch.Tensor,
        w: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """CE on clean-selected samples + λ_u * MSE consistency on noisy
        samples. Discarded samples (w=-1) contribute nothing."""
        probs = F.softmax(clf_logits, dim=1)  # gradients flow through this
        # CE per sample (raw, no reduction)
        ce_per = F.cross_entropy(
            clf_logits, noisy_labels, weight=self.class_weights, reduction="none",
        )
        clean_mask = (w == 1).float()
        n_clean = clean_mask.sum().clamp_min(1.0)
        L_clean = (ce_per * clean_mask).sum() / n_clean

        # MSE consistency on noisy samples: ((sharpen(p, T).detach() - p) ** 2).sum(-1)
        with torch.no_grad():
            sharp = _sharpen(probs.detach(), self.T)
        mse_per = ((sharp - probs) ** 2).sum(dim=1)
        noisy_mask = (w == 0).float()
        n_noisy = noisy_mask.sum().clamp_min(1.0)
        L_noisy = (mse_per * noisy_mask).sum() / n_noisy

        total = L_clean + self.lambda_u * L_noisy
        return total, {"clean_ce": L_clean.detach(), "mse": L_noisy.detach()}

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------
    def train_step(self, batch, epoch, scaler) -> MethodOutput:
        images, labels, _idx = batch
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        self.clf_net.train()
        self.ref_net.train()
        self.clf_opt.zero_grad(set_to_none=True)
        self.ref_opt.zero_grad(set_to_none=True)

        in_warmup = epoch < int(self.warmup_epochs)

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            # Full-batch forward through BOTH networks (protects BN).
            clf_logits = self.clf_net(images)
            ref_logits = self.ref_net(images)

            if in_warmup:
                L_clf = self._warmup_clf_loss(clf_logits, labels)
                L_ref = self._warmup_ref_loss(ref_logits, labels)
                components = {
                    "warmup_ce_clf": L_clf.detach(),
                    "warmup_bce_ref": L_ref.detach(),
                }
            else:
                with torch.no_grad():
                    clf_probs = F.softmax(clf_logits.detach().float(), dim=1)
                    ref_sig = torch.sigmoid(ref_logits.detach().float())
                    y_tilde, y_n_tilde, y_r_tilde = self._compute_views_and_subsets(
                        labels, clf_probs, ref_sig,
                    )
                    w = self._compute_weights(y_tilde, y_n_tilde, y_r_tilde)
                    y_hat = self._compute_relabels(y_tilde, y_n_tilde, y_r_tilde)

                L_clf, clf_parts = self._postwarmup_clf_loss(clf_logits, labels, w)
                # ref_net loss: BCE with re-labeled multi-hot targets.
                L_ref = F.binary_cross_entropy_with_logits(ref_logits, y_hat)

                components = {
                    "clf_clean_ce": clf_parts["clean_ce"].detach(),
                    "clf_mse": clf_parts["mse"].detach(),
                    "ref_bce": L_ref.detach(),
                    "n_clean": (w == 1).float().sum().detach(),
                    "n_noisy": (w == 0).float().sum().detach(),
                    "n_discard": (w == -1).float().sum().detach(),
                }

        # Backward + step for each network independently.
        scaler.scale(L_clf).backward()
        scaler.scale(L_ref).backward()
        scaler.step(self.clf_opt)
        scaler.step(self.ref_opt)
        scaler.update()

        # Scalar totals
        comp_scalar = {k: float(v.item()) if hasattr(v, "item") else float(v) for k, v in components.items()}

        return MethodOutput(
            loss_total=float((L_clf + L_ref).item()),
            loss_components=comp_scalar,
            batch_size=int(images.size(0)),
        )

    def _all_schedulers(self):
        return [self.clf_sched, self.ref_sched]

    def inference_model(self) -> nn.Module:
        """Evaluation uses clf_net only (per AsyCo §3.3)."""
        return self.clf_net
