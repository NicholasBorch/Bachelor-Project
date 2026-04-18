from src.methods.base import Method, MethodOutput
from src.methods.baseline import BaselineMethod
from src.methods.sce import SCEMethod
from src.methods.elr import ELRMethod
from src.methods.asyco import AsyCoMethod


def build_method(
    method_name: str,
    cfg: dict,
    num_train_samples: int,
    num_classes: int,
    device,
    class_weights=None,
) -> Method:
    """Factory that returns a configured Method instance.

    Args:
        method_name: one of "baseline", "sce", "elr", "asyco".
        cfg: full merged config dict (base + data + method + optim + model + noise).
        num_train_samples: needed by ELR to size its target buffer.
        num_classes: 7 for HAM10000.
        device: torch.device.
        class_weights: (optional) tensor for CE weighting, used only by
            imbalanced-dataset runs.
    """
    name = method_name.lower()
    if name == "baseline":
        return BaselineMethod(cfg, device=device, class_weights=class_weights)
    if name == "sce":
        return SCEMethod(cfg, device=device, class_weights=class_weights)
    if name == "elr":
        return ELRMethod(
            cfg, device=device,
            num_train_samples=num_train_samples, num_classes=num_classes,
            class_weights=class_weights,
        )
    if name == "asyco":
        return AsyCoMethod(cfg, device=device, class_weights=class_weights)
    raise ValueError(f"Unknown method: {method_name}")


__all__ = [
    "Method",
    "MethodOutput",
    "BaselineMethod",
    "SCEMethod",
    "ELRMethod",
    "AsyCoMethod",
    "build_method",
]
