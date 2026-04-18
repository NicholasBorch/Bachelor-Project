"""Global seeding for reproducibility."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, CUDA. When deterministic=True, also set
    cudnn flags for bit-identical runs across machines.

    Note: deterministic mode is slower. For Stage 3 we accept non-bitwise
    reproducibility across machines (both HPC nodes) but still set torch seed
    so within-run randomness is controlled.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def fold_seed(global_seed: int, fold_id: int) -> int:
    """Deterministic per-fold seed. Used for noise injection and per-fold
    stochastic operations."""
    return global_seed * 10_000 + fold_id
