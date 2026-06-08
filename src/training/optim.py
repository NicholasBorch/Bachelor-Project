"""Optimizer and scheduler builders driven by config."""
from __future__ import annotations

from typing import Any, Iterable

import torch


def build_optimizer(params: Iterable, optim_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """Build an optimizer from configs/optim/<name>.yaml.

    Supported names: 'sgd', 'adam'.
    """
    name = optim_cfg["name"].lower()
    lr = float(optim_cfg["lr"])
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(optim_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(optim_cfg.get("nesterov", False)),
        )
    if name == "adam":
        betas = optim_cfg.get("betas", [0.9, 0.999])
        return torch.optim.Adam(
            params,
            lr=lr,
            betas=(float(betas[0]), float(betas[1])),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer name: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    name: str = "cosine_annealing",
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build an epoch-level LR scheduler."""
    if name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    raise ValueError(f"Unknown scheduler name: {name}")
