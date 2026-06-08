"""
Optuna search spaces for the FINAL hyperparameter search.

Separate from configs/optuna_search_spaces.py to keep the final run cleanly
isolated. Searched: ELR (lambda, beta); SCE (alpha, beta, A); AsyCo+DivMix
(lambda_u, K, temperature, warmup_epochs). For AsyCo+DivMix the three MixMatch
components (mixup_alpha, lambda_prior, rampup_epochs) are FIXED at DivideMix
defaults, not searched. Each sample
function returns a dict merged over the base method config; the AsyCo dict carries
the fixed defaults so the final config is independent of base-YAML values.
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
        # FIXED at DivideMix defaults (override base config)
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