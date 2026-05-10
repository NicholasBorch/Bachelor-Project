"""Optuna search spaces for the FINAL hyperparameter search.

Distinct from configs/optuna_search_spaces.py to keep the final run cleanly
separable from previous experimental runs. Three methods are supported:

  ELR (Liu et al. 2020) — 2 parameters
    lambda  in loguniform(0.5, 15.0)   -- paper sweeps {1, 3, 5, 7, 10}
    beta    in uniform(0.5, 0.95)      -- paper uses 0.7

  SCE (Wang et al. 2019) — 3 parameters
    alpha   in loguniform(0.01, 10.0)  -- paper tested [0.01, 1] on CIFAR-10;
                                          CIFAR-100 used 6.0; we span both
    beta    in uniform(0.1, 2.0)       -- paper tested [0.1, 1]; small extension
    A       in uniform(-8.0, -2.0)     -- paper tested [-8, -2] step 2

  AsyCo+DivMix (Liu et al. 2023) — 4 SEARCHED + 3 FIXED parameters
    Searched (the AsyCo paper's tunable hyperparameters):
      lambda_u      in uniform(0.0, 25.0)   -- paper: 0 (Clothing1M) to 100 (CIFAR-100)
      K             in {1, 2, 3, 4, 5}       -- paper: 1 (CIFAR-10), 3 (CIFAR-100)
      temperature   in uniform(0.25, 1.0)   -- paper inherits T=0.5 from DivideMix
      warmup_epochs in int(5, 25)            -- paper: 10 for CIFAR/Animal-10N

    Fixed at DivideMix paper defaults (NOT searched):
      mixup_alpha   = 4.0   -- DivideMix paper (CIFAR-10/100)
      lambda_prior  = 1.0   -- DivideMix paper
      rampup_epochs = 16    -- DivideMix paper

    Rationale for fixing these three: the AsyCo paper's stated hyperparameter
    list contains lambda_u, K, T, warmup_epochs only. The MixMatch components
    (mixup_alpha, lambda_prior, rampup_epochs) are inherited unchanged from
    DivideMix. Tuning them would go beyond the paper's methodology; fixing
    them at DivideMix defaults preserves the paper's intended algorithm.

The sample functions return a dict that is merged over the base method config.
For AsyCo, the dict includes both searched values and fixed defaults so the
final config is independent of any contrary base-YAML values.
"""
from __future__ import annotations

from typing import Any

import optuna


def sample_elr(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "lambda": trial.suggest_float("lambda", 0.5, 15.0, log=True),
        "beta":   trial.suggest_float("beta", 0.5, 0.95),
    }


def sample_sce(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
        "beta":  trial.suggest_float("beta", 0.1, 2.0),
        "A":     trial.suggest_float("A", -8.0, -2.0),
    }


def sample_asyco_divmix(trial: optuna.Trial) -> dict[str, Any]:
    return {
        # SEARCHED — AsyCo paper's stated hyperparameters
        "lambda_u":      trial.suggest_float("lambda_u", 0.0, 25.0),
        "K":             trial.suggest_categorical("K", [1, 2, 3, 4, 5]),
        "temperature":   trial.suggest_float("temperature", 0.25, 1.0),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 5, 25),
        # FIXED at DivideMix paper defaults — overrides base config to ensure
        # every trial uses the canonical DivideMix MixMatch configuration.
        "mixup_alpha":   4.0,
        "lambda_prior":  1.0,
        "rampup_epochs": 16,
    }


SAMPLERS = {
    "elr":          sample_elr,
    "sce":          sample_sce,
    "asyco_divmix": sample_asyco_divmix,
}


def sample(method: str, trial: optuna.Trial) -> dict[str, Any]:
    if method not in SAMPLERS:
        raise ValueError(
            f"No FINAL search space defined for method '{method}'. "
            f"Available: {sorted(SAMPLERS)}"
        )
    return SAMPLERS[method](trial)
