from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Seed python, numpy and torch for reproducible experiments.

    Parameters
    ----------
    seed:
        Global seed controlling all RNG sources.
    deterministic:
        If True, sets PyTorch/CUDNN to deterministic mode.
        This can reduce speed but improves reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False