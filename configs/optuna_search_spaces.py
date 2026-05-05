"""Optuna search spaces for noisy-label method hyperparameter selection.

This module defines, for each tunable method, the function that samples a
fresh hyperparameter config from a single Optuna trial. The returned dict
is a partial config that the search runner merges into the base method
config (configs/method/<method>.yaml).

Why pure Python rather than YAML? Optuna's distribution API (suggest_float,
suggest_int, suggest_categorical) is the part of the interface we care
about, and these are best expressed as code. Putting them in YAML would
require a custom DSL that adds zero value over just calling the API.

Bounds are documented inline. The general philosophy is "broad but
realistic": cover the published settings of the original paper and
reasonable extensions, but not absurd values that we already know fail.
For asyco_divmix specifically the upper bound on lambda_u is set to 12,
deliberately below the CIFAR-style 25 that we already empirically observed
to cause training collapse on HAM10000.
"""
from __future__ import annotations

from typing import Any

import optuna


def sample_elr(trial: optuna.Trial) -> dict[str, Any]:
    """ELR search space.

    Two parameters:
        lambda    — regularization strength on the ELR term
        beta      — temporal-ensemble momentum on the per-sample target buffer

    Bounds rationale:
        lambda in [0.05, 10] covers the paper's reported range {1, 3, 5, 7, 10}
        plus a small extension on each side. Log-uniform because the
        interesting variation is in the order of magnitude.

        beta in [0.5, 0.95] covers responsive (low momentum, fast tracking
        but noisy) to smooth (high momentum, slow tracking but stable).
        Paper uses 0.7. Uniform because the effect is roughly linear in
        the range.
    """
    return {
        "lambda": trial.suggest_float("lambda", 0.05, 10.0, log=True),
        "beta": trial.suggest_float("beta", 0.5, 0.95),
    }


def sample_asyco_divmix(trial: optuna.Trial) -> dict[str, Any]:
    """AsyCo (paper version, with MixMatch wrapper) search space.

    Eight parameters. See module docstring for the high-level rationale;
    per-parameter notes below.
    """
    return {
        # Core: weight on the unsupervised MSE consistency loss.
        # 0 (Clothing1M paper setting) up to 25 (CIFAR-10's 25). Above 25 we already empirically saw collapse on
        # HAM10000, so we cap there.
        # Uniform (not log) because we want 0 to be a reachable value.
        "lambda_u": trial.suggest_float("lambda_u", 0.0, 25.0),

        # Weight on the KL-to-uniform anti-collapse prior.
        # 0 (off) to 1.5 (slight overshoot of the paper's 1.0).
        "lambda_prior": trial.suggest_float("lambda_prior", 0.0, 1.5),

        # Warmup duration, expressed as a fraction of total epochs and an
        # absolute floor. Effective warmup = max(floor, pct * total_epochs).
        "warmup_epochs_pct": trial.suggest_float("warmup_epochs_pct", 0.03, 0.20),
        "warmup_epochs_floor": trial.suggest_int("warmup_epochs_floor", 5, 25),

        # Reference net's top-K. Paper uses K=1 for CIFAR-10, K=2 for
        # CIFAR-100. K=3 is worth trying on our 7-class imbalanced problem
        # where multi-view consensus often fails too aggressively.
        "K": trial.suggest_categorical("K", [1, 2, 3, 4, 5]),

        # MixUp Beta(alpha, alpha) parameter. 0.2 (very gentle) to 4.0
        # (DivideMix-CIFAR aggressive). Paper uses 0.75. Log-uniform because
        # the effect is multiplicative on the resulting blend distribution.
        "mixup_alpha": trial.suggest_float("mixup_alpha", 0.2, 4.0, log=True),

        # Temperature for sharpening the co-guessed pseudo-label. T < 1
        # sharpens; T = 1 is no-op. Lower = more confident pseudo-labels
        # (more useful when correct, more harmful when wrong).
        "temperature": trial.suggest_float("temperature", 0.25, 1.0),

        # Linear rampup duration for lambda_u, in epochs after warmup.
        # Paper uses 16. Longer rampup gives the model more time to develop
        # reliable pseudo-labels before the unsupervised loss takes hold.
        "rampup_epochs": trial.suggest_int("rampup_epochs", 8, 32),
    }


SAMPLERS = {
    "elr": sample_elr,
    "asyco_divmix": sample_asyco_divmix,
}


def sample(method: str, trial: optuna.Trial) -> dict[str, Any]:
    if method not in SAMPLERS:
        raise ValueError(
            f"No Optuna search space defined for method '{method}'. "
            f"Available: {sorted(SAMPLERS)}"
        )
    return SAMPLERS[method](trial)
