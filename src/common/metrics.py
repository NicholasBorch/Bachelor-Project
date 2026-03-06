# Evaluation metrics for HAM10000 classification.
# All functions operate on numpy arrays of true and predicted labels.
# Used identically across baseline, AsyCo, ELR, and SCE evaluation loops.

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict:
    # Computes the full suite of classification metrics for one evaluation run
    # y_true: integer ground truth labels, shape (N,)
    # y_pred: integer predicted labels, shape (N,)
    # y_prob: softmax probabilities, shape (N, C)
    # class_names: ordered list of class name strings matching label indices

    num_classes = len(class_names)

    accuracy          = float(np.mean(y_true == y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1          = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1       = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    kappa             = float(cohen_kappa_score(y_true, y_pred))

    # Per-class F1 scores keyed by class name
    per_class_f1_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1     = {class_names[i]: float(per_class_f1_arr[i]) for i in range(num_classes)}

    # Macro AUC using one-vs-rest strategy — requires probability estimates
    try:
        auc = float(roc_auc_score(
            y_true, y_prob,
            multi_class="ovr",
            average="macro",
        ))
    except ValueError:
        # Raised when a class is absent from y_true in this fold/split
        auc = float("nan")

    # Confusion matrix as nested list for JSON serialisation
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    return {
        "accuracy":          accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1":          macro_f1,
        "weighted_f1":       weighted_f1,
        "kappa":             kappa,
        "auc_macro_ovr":     auc,
        "per_class_f1":      per_class_f1,
        "confusion_matrix":  cm.tolist(),
        "class_names":       class_names,
        "n_samples":         int(len(y_true)),
    }


def aggregate_metrics(fold_metrics: List[Dict]) -> Dict:
    # Aggregates per-fold metric dicts into mean and std across folds
    # Returns a dict with keys like accuracy_mean, accuracy_std, etc.
    scalar_keys = [
        "accuracy", "balanced_accuracy", "macro_f1",
        "weighted_f1", "kappa", "auc_macro_ovr",
    ]
    result = {}

    for key in scalar_keys:
        values = [m[key] for m in fold_metrics if not np.isnan(m[key])]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"]  = float(np.std(values))

    # Aggregate per-class F1 across folds
    class_names = fold_metrics[0]["class_names"]
    per_class_agg = {}
    for cls in class_names:
        values = [m["per_class_f1"][cls] for m in fold_metrics]
        per_class_agg[cls] = {
            "mean": float(np.mean(values)),
            "std":  float(np.std(values)),
        }
    result["per_class_f1"] = per_class_agg
    result["n_folds"]      = len(fold_metrics)
    result["class_names"]  = class_names

    return result


def print_metrics(metrics: Dict, prefix: str = "") -> None:
    # Prints a compact summary of a single-fold or aggregated metrics dict to console
    is_aggregated = "accuracy_mean" in metrics

    if is_aggregated:
        print(f"{prefix}accuracy:          {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
        print(f"{prefix}balanced_accuracy: {metrics['balanced_accuracy_mean']:.4f} ± {metrics['balanced_accuracy_std']:.4f}")
        print(f"{prefix}macro_f1:          {metrics['macro_f1_mean']:.4f} ± {metrics['macro_f1_std']:.4f}")
        print(f"{prefix}auc_macro_ovr:     {metrics['auc_macro_ovr_mean']:.4f} ± {metrics['auc_macro_ovr_std']:.4f}")
        print(f"{prefix}kappa:             {metrics['kappa_mean']:.4f} ± {metrics['kappa_std']:.4f}")
    else:
        print(f"{prefix}accuracy:          {metrics['accuracy']:.4f}")
        print(f"{prefix}balanced_accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"{prefix}macro_f1:          {metrics['macro_f1']:.4f}")
        print(f"{prefix}auc_macro_ovr:     {metrics['auc_macro_ovr']:.4f}")
        print(f"{prefix}kappa:             {metrics['kappa']:.4f}")