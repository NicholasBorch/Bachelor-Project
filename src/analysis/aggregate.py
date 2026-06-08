"""
Walk the Stage 3 results tree and build tidy DataFrames for Stage 4.

Reads every results/training/.../test_metrics.json and every
results/epoch_selection/.../selected_budget.json. Resilient to partial results
(missing or malformed files are logged and skipped), since Stage 4 may run while
Stage 3 jobs are still in flight. NTA/LNMR are NaN at tau=0.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from src.data.ham10000 import CLASS_NAMES

logger = logging.getLogger(__name__)

# path parsing

# results/training/{method}/{dataset}/{init}_{optim}/tau_NN/fold_NN/test_metrics.json
_INIT_OPTIM_RE = re.compile(r"^(?P<init>[a-z0-9]+)_(?P<optim>[a-z0-9]+)$")
_TAU_RE = re.compile(r"^tau_(?P<tau>\d{2})$")
_FOLD_RE = re.compile(r"^fold_(?P<fold>\d{2})$")

# Scalar metric columns loaded from test_metrics.json, in the order they are
# written into the output DataFrame.
_SCALAR_TEST_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_auc",
    "n_samples",
)

# Noise-label interaction diagnostics, computed on the training set with
# test-time transforms. NTA and LNMR are NaN at τ=0.
_NOISE_INTERACTION_METRICS = (
    "nta",
    "lnmr",
    "n_flipped",
    "n_train",
    "empirical_flip_rate",
)


def _parse_result_path(metrics_path: Path, training_root: Path) -> dict | None:
    """Parse a test_metrics.json path into (method, dataset, init, optim, tau, fold)."""
    try:
        rel = metrics_path.relative_to(training_root)
    except ValueError:
        logger.warning("Path %s is not under %s, skipping.", metrics_path, training_root)
        return None

    parts = rel.parts
    if len(parts) != 6 or parts[-1] != "test_metrics.json":
        logger.warning("Unexpected path layout %s, skipping.", rel)
        return None

    method, dataset, init_optim, tau_str, fold_str, _ = parts

    m_io = _INIT_OPTIM_RE.match(init_optim)
    m_tau = _TAU_RE.match(tau_str)
    m_fold = _FOLD_RE.match(fold_str)
    if not (m_io and m_tau and m_fold):
        logger.warning("Could not parse axes from %s, skipping.", rel)
        return None

    return {
        "method": method,
        "dataset": dataset,
        "init": m_io.group("init"),
        "optim": m_io.group("optim"),
        "tau": int(m_tau.group("tau")) / 100.0,
        "fold": int(m_fold.group("fold")),
    }


def _read_metrics(metrics_path: Path) -> dict | None:
    """Load a ``test_metrics.json`` file, returning ``None`` on error."""
    try:
        with metrics_path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read %s: %s", metrics_path, exc)
        return None


# main loaders


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all training/.../test_metrics.json into a tidy DataFrame (empty with schema if none)."""
    results_dir = Path(results_dir)
    training_root = results_dir / "training"

    if not training_root.exists():
        logger.warning("No training root at %s; returning empty DataFrame.", training_root)
        return _empty_results_df()

    rows: list[dict] = []
    metric_files = sorted(training_root.glob("*/*/*/*/*/test_metrics.json"))
    logger.info("Found %d test_metrics.json files under %s.", len(metric_files), training_root)

    for metrics_path in metric_files:
        axes = _parse_result_path(metrics_path, training_root)
        if axes is None:
            continue
        metrics = _read_metrics(metrics_path)
        if metrics is None:
            continue

        row: dict = {**axes}
        # Core scalar test-set metrics (skip silently if missing).
        for key in _SCALAR_TEST_METRICS:
            row[key] = metrics.get(key)

        # Noise-label interaction diagnostics (training set, flipped subset).
        # These may be None or NaN at τ=0; that is expected behaviour.
        for key in _NOISE_INTERACTION_METRICS:
            row[key] = metrics.get(key)

        # Per-class F1 — flatten into columns, one per class.
        per_class = metrics.get("per_class_f1", {}) or {}
        for cls in CLASS_NAMES:
            row[f"per_class_f1_{cls}"] = per_class.get(cls)

        # Confusion matrix kept as nested list; downstream code can convert.
        row["confusion_matrix"] = metrics.get("confusion_matrix")

        rows.append(row)

    if not rows:
        return _empty_results_df()

    df = pd.DataFrame(rows)
    sort_cols = ["dataset", "init", "optim", "method", "tau", "fold"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def load_selected_budgets(results_dir: Path) -> pd.DataFrame:
    """Walk ``results_dir/epoch_selection/{dataset}/{method}/selected_budget.json``."""
    results_dir = Path(results_dir)
    selection_root = results_dir / "epoch_selection"

    empty_cols = [
        "dataset", "method", "selected_epochs",
        "per_fold_convergence", "median", "mean", "std",
    ]

    if not selection_root.exists():
        logger.warning(
            "No epoch_selection root at %s; returning empty DataFrame.", selection_root
        )
        return pd.DataFrame(columns=empty_cols)

    rows: list[dict] = []
    budget_files = sorted(selection_root.glob("*/*/selected_budget.json"))
    logger.info(
        "Found %d selected_budget.json files under %s.", len(budget_files), selection_root
    )

    for budget_path in budget_files:
        try:
            rel = budget_path.relative_to(selection_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) != 3 or parts[-1] != "selected_budget.json":
            logger.warning("Unexpected budget path layout %s, skipping.", rel)
            continue
        dataset, method, _ = parts

        try:
            with budget_path.open() as f:
                budget = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read %s: %s", budget_path, exc)
            continue

        rows.append({
            "dataset": dataset,
            "method": method,
            "selected_epochs": budget.get("selected_epochs"),
            "per_fold_convergence": json.dumps(budget.get("per_fold_convergence")),
            "median": budget.get("median"),
            "mean": budget.get("mean"),
            "std": budget.get("std"),
        })

    if not rows:
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)


# aggregation helpers


def aggregate_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the fold axis to per-metric mean/std (incl. NTA/LNMR, NaN at tau=0), plus n_folds."""
    if df.empty:
        return df.copy()

    metric_cols = [
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_auc",
        # Noise-label interaction diagnostics
        "nta",
        "lnmr",
        "empirical_flip_rate",
        # Per-class F1
        *[f"per_class_f1_{c}" for c in CLASS_NAMES],
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    group_cols = ["dataset", "init", "optim", "method", "tau"]
    grouped = df.groupby(group_cols, dropna=False)

    agg: dict[str, list[str]] = {c: ["mean", "std"] for c in metric_cols}
    out = grouped.agg(agg)
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    out["n_folds"] = grouped.size()
    out = out.reset_index()
    return out


# private helpers


def _empty_results_df() -> pd.DataFrame:
    """Return an empty DataFrame with the expected schema."""
    cols = [
        "method", "dataset", "init", "optim", "tau", "fold",
        *_SCALAR_TEST_METRICS,
        *_NOISE_INTERACTION_METRICS,
        *[f"per_class_f1_{c}" for c in CLASS_NAMES],
        "confusion_matrix",
    ]
    return pd.DataFrame(columns=cols)