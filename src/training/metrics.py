"""
Test-set metrics and noise-label interaction diagnostics.

Test set (per Stage 3 run): balanced_accuracy (primary), macro_f1 (co-primary),
macro_auc (supportive), per_class_f1, confusion_matrix, weighted_f1. Noise-label
interaction (computed on the training set after training): nta, lnmr. See
PROJECT_DOCUMENTATION §2.4.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.data.ham10000 import CLASS_NAMES, NUM_CLASSES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute the standard test-set metric suite from (y_true, y_pred, y_prob)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    labels_all = list(range(NUM_CLASSES))

    # Macro AUC checkup
    try:
        macro_auc = float(roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro", labels=labels_all,
        ))
    except ValueError:
        macro_auc = float("nan")

    per_class_f1_arr = f1_score(
        y_true, y_pred, labels=labels_all, average=None, zero_division=0,
    )
    per_class_f1 = {CLASS_NAMES[i]: float(per_class_f1_arr[i]) for i in range(NUM_CLASSES)}

    cm = confusion_matrix(y_true, y_pred, labels=labels_all)

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels_all, average="weighted", zero_division=0)),
        "macro_auc": macro_auc,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }


def compute_noise_label_interaction(
    y_pred_train: np.ndarray,
    y_noisy: np.ndarray,
    y_clean: np.ndarray,
) -> dict[str, Any]:
    """NTA=P(pred==clean | flipped) and LNMR=P(pred==noisy | flipped) on the training set; NaN at tau=0. y_pred_train must use test-time transforms."""
    y_pred_train = np.asarray(y_pred_train).astype(np.int64)
    y_noisy = np.asarray(y_noisy).astype(np.int64)
    y_clean = np.asarray(y_clean).astype(np.int64)

    if not (len(y_pred_train) == len(y_noisy) == len(y_clean)):
        raise ValueError(
            f"length mismatch: y_pred_train={len(y_pred_train)}, "
            f"y_noisy={len(y_noisy)}, y_clean={len(y_clean)}"
        )

    flipped_mask = y_noisy != y_clean
    n_flipped = int(flipped_mask.sum())
    n_train = int(len(y_pred_train))

    if n_flipped == 0:
        # τ=0 case: no flipped samples, NTA/LNMR undefined.
        return {
            "nta": float("nan"),
            "lnmr": float("nan"),
            "n_flipped": 0,
            "n_train": n_train,
            "empirical_flip_rate": 0.0,
        }

    pred_flipped = y_pred_train[flipped_mask]
    clean_flipped = y_clean[flipped_mask]
    noisy_flipped = y_noisy[flipped_mask]

    nta = float((pred_flipped == clean_flipped).mean())
    lnmr = float((pred_flipped == noisy_flipped).mean())

    return {
        "nta": nta,
        "lnmr": lnmr,
        "n_flipped": n_flipped,
        "n_train": n_train,
        "empirical_flip_rate": float(flipped_mask.mean()),
    }


def aggregate_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-fold metrics: scalars to mean/std, per-class F1 mean/std, confusion matrices summed."""
    if not fold_metrics:
        return {}

    scalar_keys = ["balanced_accuracy", "macro_f1", "weighted_f1", "macro_auc"]
    out: dict[str, Any] = {}
    for k in scalar_keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=np.float64)
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            out[k] = {"mean": float("nan"), "std": float("nan"), "n_valid": 0}
        else:
            out[k] = {
                "mean": float(valid.mean()),
                "std": float(valid.std(ddof=0)),
                "n_valid": int(len(valid)),
            }

    per_class: dict[str, dict[str, float]] = {}
    for cls in CLASS_NAMES:
        vals = np.array([m["per_class_f1"][cls] for m in fold_metrics], dtype=np.float64)
        per_class[cls] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    out["per_class_f1"] = per_class

    cm_sum = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for m in fold_metrics:
        cm_sum += np.array(m["confusion_matrix"], dtype=np.int64)
    out["confusion_matrix_sum"] = cm_sum.tolist()
    out["n_folds"] = len(fold_metrics)
    out["n_samples_total"] = int(sum(m["n_samples"] for m in fold_metrics))
    return out