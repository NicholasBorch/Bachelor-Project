"""AsyCo + DivideMix MixMatch (the precise implementation the AsyCo paper
actually uses for its experimental results, see Liu et al. 2023 §4.2).

Relationship to ``asyco.py`` in this codebase
---------------------------------------------
``asyco.py`` implements AsyCo's Eq. (5) literally: CE on clean-selected
samples + λ * MSE(sharpen(p, T), p) self-consistency on noisy-selected
samples. That matches the equation in the paper, but Section 4.2 says:

    "For the semi-supervised training of n_θ(.), we use MixMatch [2]
     from DivideMix [16]."

i.e. the published numbers come from wrapping AsyCo's sample-selection
inside MixMatch, not from the simple Eq. (5). This module provides that
wrapped variant. Both methods coexist; pick at runtime via the
``--method`` flag (``asyco`` vs ``asyco_divmix``).

Pipeline (post-warmup, per training batch)
------------------------------------------
Inputs: two stochastic augmentations of each image — ``x1, x2`` — plus
the noisy labels ``y_tilde`` (provided by ``TwoViewHamDataset``).

1. Multi-view sample selection (no_grad) — exactly the same logic as
   ``asyco.py``. Three label views (training label, clf argmax,
   ref top-K) are compared on view 1 to compute:
     w_i ∈ {+1, 0, -1}    (clean / noisy / discard)
     ŷ_i ∈ {0,1}^C         (re-labeled multi-hot target for ref_net)

2. Label construction (no_grad):
     - ``w_i = +1`` (labeled): target = one_hot(y_tilde_i).
       (No "co-refinement" with a per-sample blend weight; AsyCo's
       discrete selection variable replaces DivideMix's continuous
       GMM clean-probability.)
     - ``w_i = 0`` (unlabeled): co-guessing — average softmax(clf_net)
       over views x1 and x2, then temperature-sharpen.
     - ``w_i = -1`` (discard): drop, contribute nothing.

3. MixMatch:
     all_inputs  = cat([x1_lab, x2_lab, x1_unl, x2_unl])
     all_targets = cat([t_lab,  t_lab,  t_unl,  t_unl])
     l = max(Beta(α, α), 1 - Beta(α, α))
     perm = random
     mixed_input  = l * all_inputs  + (1-l) * all_inputs[perm]
     mixed_target = l * all_targets + (1-l) * all_targets[perm]
     logits = clf_net(mixed_input)        # ONE forward pass with grad

4. clf_net loss:
     L_x      = -mean(sum(weighted_target * log_softmax(logits_lab))) [1]
     L_u      = mean(sum((softmax(logits_unl) - target_unl)^2))
     L_prior  = KL(uniform || mean_softmax(logits))
     L_total  = L_x + ramp(λ_u, epoch) * L_u + λ_prior * L_prior

   [1] Class-weighted soft CE: per-class weights from the imbalanced
       dataset are broadcast into the soft-target log-prob product.
       Reduces to plain softmax-CE on balanced runs (class_weights=None).

5. ref_net loss: BCE on view 1 with the multi-hot ŷ from step 1
   (identical to ``asyco.py`` — the reference net is unaffected by
   the MixMatch wrapper).

Why this is more expensive than ``asyco.py``
--------------------------------------------
Per train_step, this method does FOUR forward passes through clf_net
(view1 no_grad for selection, view2 no_grad for co-guessing, mixed_input
with grad — and the mixed_input batch is up to 2× the original batch
size because both views of every kept sample are concatenated). Plus one
forward through ref_net (same as ``asyco.py``). Total ~2-2.5× the
compute of ``asyco.py``. Memory peaks when the labeled+unlabeled fraction
is large; samples in subset U (w=-1) are dropped before MixUp.

Hyperparameter notes
--------------------
- ``mixup_alpha``: DivideMix uses α=4 on CIFAR. For HAM10000 (small
  medical dataset, fine-grained classes, less aggressive aug appropriate)
  we default to α=0.75. Adjust per dataset.
- ``rampup_epochs``: linear ramp of λ_u from 0 to its target value over
  this many epochs after warmup. DivideMix uses 16. For very short
  epoch budgets, λ_u may not reach full strength — this is by design,
  matching DivideMix's behavior.
- ``lambda_prior``: weight of the KL-to-uniform prior penalty. DivideMix
  uses 1.0; we keep that default.

Edge cases handled
------------------
- Empty labeled subset within a batch (n_lab == 0): L_x = 0, only L_u
  and L_prior contribute. Common at warmup-exit when sample selection
  is still noisy.
- Empty unlabeled subset (n_unl == 0): L_u = 0, only L_x and L_prior
  contribute. Common on clean (τ=0) data — basically reduces to plain CE.
- Both empty: skip clf_net backward entirely; still train ref_net.
- WeightedRandomSampler + class_weighted_loss: both flags are honored
  exactly as in ``asyco.py``. The class weights are applied to the soft
  CE term L_x; L_u is unweighted (matches DivideMix's design).
"""
from __future__ import annotations

import math

import numpy as np
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


def _rampup_lambda_u(
    epoch: int,
    warmup_epochs: int,
    rampup_epochs: int,
    target_lambda_u: float,
) -> float:
    """Linear ramp of λ_u from 0 (at warmup-exit) to target over rampup_epochs.

    Matches DivideMix's ``linear_rampup``: clamps at the target value
    once ``rampup_epochs`` post-warmup epochs have passed. Returns 0
    during warmup itself (caller usually doesn't call this during warmup
    anyway — defensive default).
    """
    if epoch < warmup_epochs:
        return 0.0
    if rampup_epochs <= 0:
        return float(target_lambda_u)
    frac = (epoch - warmup_epochs) / float(rampup_epochs)
    frac = max(0.0, min(1.0, frac))
    return float(target_lambda_u) * frac


class AsyCoDivMixMethod(Method):
    """AsyCo with the full DivideMix MixMatch SSL pipeline wrapping its
    multi-view sample selection. See module docstring for the full design.
    """

    requires_two_views: bool = True

    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__(cfg, device)
        self.class_weights = class_weights
        m = cfg["method"]

        # Multi-view consensus (same as asyco.py)
        self.K = int(m["K"])
        self.lambda_u = float(m["lambda_u"])
        self.T = float(m["temperature"])
        self.warmup_pct = float(m["warmup_epochs_pct"])
        self.warmup_floor = int(m["warmup_epochs_floor"])

        # MixMatch additions
        self.mixup_alpha = float(m.get("mixup_alpha", 0.75))
        self.rampup_epochs = int(m.get("rampup_epochs", 16))
        self.lambda_prior = float(m.get("lambda_prior", 1.0))

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
            total_epochs=total_epochs,
            pct=self.warmup_pct,
            floor=self.warmup_floor,
        )
        # Two independent models, SAME architecture as asyco.py.
        self.clf_net = model_builder().to(self.device)
        self.ref_net = model_builder().to(self.device)

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
    # Multi-view consensus — verbatim from asyco.py for parity.
    # (Kept private to this module rather than imported, so a future edit
    # to one method's selection logic does not silently affect the other.)
    # ------------------------------------------------------------------
    def _compute_views(
        self,
        noisy_labels: torch.Tensor,
        clf_probs_v1: torch.Tensor,
        ref_sigmoid_v1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_tilde = F.one_hot(noisy_labels, num_classes=NUM_CLASSES).float()
        n_argmax = clf_probs_v1.argmax(dim=1)
        y_n_tilde = F.one_hot(n_argmax, num_classes=NUM_CLASSES).float()
        topk_idx = ref_sigmoid_v1.topk(self.K, dim=1).indices
        y_r_tilde = torch.zeros_like(y_tilde)
        y_r_tilde.scatter_(1, topk_idx, 1.0)
        return y_tilde, y_n_tilde, y_r_tilde

    @staticmethod
    def _compute_weights(
        y_tilde: torch.Tensor,
        y_n_tilde: torch.Tensor,
        y_r_tilde: torch.Tensor,
    ) -> torch.Tensor:
        t_n = (y_tilde * y_n_tilde).sum(dim=1)
        n_r = (y_n_tilde * y_r_tilde).sum(dim=1)
        t_r = (y_tilde * y_r_tilde).sum(dim=1)
        AG = t_n + n_r + t_r
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
        t_n = (y_tilde * y_n_tilde).sum(dim=1)
        n_r = (y_n_tilde * y_r_tilde).sum(dim=1)
        t_r = (y_tilde * y_r_tilde).sum(dim=1)
        is_SC = (t_n == 0) & (n_r > 0) & (t_r > 0)
        is_NR = (t_n == 0) & (n_r > 0) & (t_r == 0)
        y_hat = y_tilde.clone()
        y_hat = torch.where(is_NR.unsqueeze(1), y_tilde + y_n_tilde, y_hat)
        y_hat = torch.where(is_SC.unsqueeze(1), y_n_tilde, y_hat)
        return y_hat.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Warmup loss (single view; view2 is computed but ignored).
    # ------------------------------------------------------------------
    def _warmup_clf_loss(self, clf_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(clf_logits, labels, weight=self.class_weights)

    def _warmup_ref_loss(self, ref_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(labels, num_classes=NUM_CLASSES).float()
        return F.binary_cross_entropy_with_logits(ref_logits, onehot)

    # ------------------------------------------------------------------
    # Soft-target cross-entropy (with optional per-class weighting).
    # ------------------------------------------------------------------
    def _soft_ce(
        self,
        logits: torch.Tensor,        # (N, C)
        soft_targets: torch.Tensor,  # (N, C) — non-negative, rows can sum to >=1 (multi-hot allowed)
    ) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.new_zeros(())
        log_p = F.log_softmax(logits, dim=-1)
        if self.class_weights is not None:
            log_p = log_p * self.class_weights.view(1, -1)
        return -(soft_targets * log_p).sum(dim=-1).mean()

    # ------------------------------------------------------------------
    # MixMatch core (post-warmup).
    # ------------------------------------------------------------------
    def _mixmatch_step(
        self,
        x1: torch.Tensor,                 # (B, 3, H, W)
        x2: torch.Tensor,                 # (B, 3, H, W)
        labels: torch.Tensor,             # (B,) long
        ref_logits_v1: torch.Tensor,      # (B, C) — needs grads for L_ref
        epoch: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute (L_clf, L_ref, components) for one post-warmup batch."""
        B = x1.size(0)
        components: dict[str, torch.Tensor] = {}

        # ---- 1. Sample selection (no_grad) on view 1 -------------------
        with torch.no_grad():
            clf_logits_v1 = self.clf_net(x1)
            clf_probs_v1 = F.softmax(clf_logits_v1.float(), dim=1)
            ref_sig_v1 = torch.sigmoid(ref_logits_v1.detach().float())
            y_tilde_oh, y_n_tilde, y_r_tilde = self._compute_views(
                labels, clf_probs_v1, ref_sig_v1,
            )
            w = self._compute_weights(y_tilde_oh, y_n_tilde, y_r_tilde)
            y_hat = self._compute_relabels(y_tilde_oh, y_n_tilde, y_r_tilde)

        # ---- 2. ref_net BCE loss (uses the WITH-grad ref_logits) ------
        L_ref = F.binary_cross_entropy_with_logits(ref_logits_v1, y_hat)

        # Indicator masks
        is_lab = (w == 1)
        is_unl = (w == 0)
        n_lab = int(is_lab.sum().item())
        n_unl = int(is_unl.sum().item())

        components["n_clean"] = is_lab.float().sum().detach()
        components["n_noisy"] = is_unl.float().sum().detach()
        components["n_discard"] = (w == -1).float().sum().detach()

        # ---- 3. Build targets ----------------------------------------
        if n_lab + n_unl == 0:
            # Everything discarded — only train ref_net this batch.
            L_clf = ref_logits_v1.new_zeros(())
            components["clf_Lx"] = L_clf.detach()
            components["clf_Lu"] = L_clf.detach()
            components["clf_Lprior"] = L_clf.detach()
            components["lambda_u_now"] = ref_logits_v1.new_tensor(0.0)
            components["ref_bce"] = L_ref.detach()
            return L_clf, L_ref, components

        with torch.no_grad():
            # Co-guess for unlabeled samples: average softmax over the two views.
            if n_unl > 0:
                clf_logits_v2_unl = self.clf_net(x2[is_unl])
                clf_probs_v2_unl = F.softmax(clf_logits_v2_unl.float(), dim=1)
                clf_probs_v1_unl = clf_probs_v1[is_unl]
                p_avg = 0.5 * (clf_probs_v1_unl + clf_probs_v2_unl)
                t_unl = _sharpen(p_avg, self.T).to(x1.dtype)
            else:
                t_unl = x1.new_zeros((0, NUM_CLASSES))

            # Labeled targets: one-hot of the noisy label (no co-refinement;
            # AsyCo's discrete w_i = +1 says "trust the noisy label fully").
            t_lab = y_tilde_oh[is_lab].to(x1.dtype) if n_lab > 0 else x1.new_zeros((0, NUM_CLASSES))

        # ---- 4. Concat both views & MixUp -----------------------------
        # Order: [x1_lab, x2_lab, x1_unl, x2_unl]
        x1_lab = x1[is_lab]
        x2_lab = x2[is_lab]
        x1_unl = x1[is_unl]
        x2_unl = x2[is_unl]
        all_inputs = torch.cat([x1_lab, x2_lab, x1_unl, x2_unl], dim=0)
        all_targets = torch.cat([t_lab, t_lab, t_unl, t_unl], dim=0)

        # MixUp coefficient: max(λ, 1-λ) keeps the mixed sample closer to
        # the original of the pair, matching DivideMix's convention.
        if self.mixup_alpha > 0:
            l = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            l = max(l, 1.0 - l)
        else:
            l = 1.0
        perm = torch.randperm(all_inputs.size(0), device=all_inputs.device)
        mixed_input = l * all_inputs + (1.0 - l) * all_inputs[perm]
        mixed_target = l * all_targets + (1.0 - l) * all_targets[perm]

        # ---- 5. Forward (with grad) on the full mixed batch -----------
        mixed_logits = self.clf_net(mixed_input)
        n_lab_x2 = 2 * n_lab
        logits_x = mixed_logits[:n_lab_x2]
        logits_u = mixed_logits[n_lab_x2:]
        target_x_mix = mixed_target[:n_lab_x2]
        target_u_mix = mixed_target[n_lab_x2:]

        # ---- 6. Losses ------------------------------------------------
        L_x = self._soft_ce(logits_x, target_x_mix)
        if logits_u.numel() > 0:
            p_u = F.softmax(logits_u, dim=-1)
            L_u = ((p_u - target_u_mix) ** 2).sum(dim=-1).mean()
        else:
            L_u = mixed_logits.new_zeros(())

        # KL(uniform || mean_p): encourages diverse predictions across the batch.
        # Computed in fp32 for stability.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            mean_p = F.softmax(mixed_logits.float(), dim=-1).mean(dim=0).clamp_min(1e-8)
            uniform = mean_p.new_full((NUM_CLASSES,), 1.0 / NUM_CLASSES)
            L_prior = (uniform * (uniform.log() - mean_p.log())).sum()

        lambda_u_now = _rampup_lambda_u(
            epoch=epoch,
            warmup_epochs=int(self.warmup_epochs),
            rampup_epochs=self.rampup_epochs,
            target_lambda_u=self.lambda_u,
        )
        L_clf = L_x + lambda_u_now * L_u + self.lambda_prior * L_prior

        components["clf_Lx"] = L_x.detach()
        components["clf_Lu"] = L_u.detach()
        components["clf_Lprior"] = L_prior.detach()
        components["lambda_u_now"] = mixed_logits.new_tensor(lambda_u_now)
        components["ref_bce"] = L_ref.detach()
        return L_clf, L_ref, components

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------
    def train_step(self, batch, epoch, scaler) -> MethodOutput:
        # batch from TwoViewHamDataset = (img1, img2, label, idx)
        if len(batch) != 4:
            raise RuntimeError(
                "asyco_divmix expects TwoViewHamDataset batches "
                f"(img1, img2, label, idx), got {len(batch)} items. "
                "Check Method.requires_two_views and runner._build_loaders."
            )
        x1, x2, labels, _idx = batch
        x1 = x1.to(self.device, non_blocking=True)
        x2 = x2.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        self.clf_net.train()
        self.ref_net.train()
        self.clf_opt.zero_grad(set_to_none=True)
        self.ref_opt.zero_grad(set_to_none=True)

        in_warmup = epoch < int(self.warmup_epochs)

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            # ref_net forward on view 1 — used in both warmup and post-warmup
            # (for warmup BCE, or for sample-selection + post-warmup BCE).
            ref_logits_v1 = self.ref_net(x1)

            if in_warmup:
                clf_logits_v1 = self.clf_net(x1)
                L_clf = self._warmup_clf_loss(clf_logits_v1, labels)
                L_ref = self._warmup_ref_loss(ref_logits_v1, labels)
                components: dict[str, torch.Tensor] = {
                    "warmup_ce_clf": L_clf.detach(),
                    "warmup_bce_ref": L_ref.detach(),
                }
            else:
                L_clf, L_ref, components = self._mixmatch_step(
                    x1=x1, x2=x2, labels=labels,
                    ref_logits_v1=ref_logits_v1, epoch=epoch,
                )

        # Backward + step. Scale ref always (we always have a non-zero L_ref).
        scaler.scale(L_ref).backward()

        # If everything in this batch was discarded post-warmup, L_clf is
        # a freshly-allocated zero tensor with no graph. Skip its backward.
        if (not in_warmup) and (L_clf.requires_grad is False):
            scaler.step(self.ref_opt)
        else:
            scaler.scale(L_clf).backward()
            scaler.step(self.clf_opt)
            scaler.step(self.ref_opt)
        scaler.update()

        comp_scalar = {
            k: float(v.item()) if hasattr(v, "item") else float(v)
            for k, v in components.items()
        }
        return MethodOutput(
            loss_total=float((L_clf + L_ref).item()),
            loss_components=comp_scalar,
            batch_size=int(x1.size(0)),
        )

    def _all_schedulers(self):
        return [self.clf_sched, self.ref_sched]

    def inference_model(self) -> nn.Module:
        """Evaluation uses clf_net only (per AsyCo §3.3)."""
        return self.clf_net
