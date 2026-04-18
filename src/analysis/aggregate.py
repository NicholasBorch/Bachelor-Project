"""Walk the Stage 3 results directory tree and build tidy pandas DataFrames.

This module is the entry point for Stage 4 analysis. It reads every
``test_metrics.json`` under ``results/training/`` and every
``selected_budget.json`` under ``results/epoch_selection/`` and returns
tidy DataFrames ready for plotting and statistical testing.

Path conventions (set by the runner in batch 4):

    results/training/{method}/{dataset}/{init}_{optim}/tau_NN/fold_NN/test_metrics.json
    results/epoch_selection/{dataset}/{method}/selected_budget.json

The loader is **resilient to partial results**: if a file is missing or
malformed it logs a warning and skips the row rather than crashing. This
matters because Stage 4 is run while Stage 3 jobs may still be in flight.

Each ``test_metrics.json`` contains the standard test-set metric suite
plus the training-set noise-label interaction diagnostics (nta, lnmr,
n_flipped, n_train, empirical_flip_rate) — see PROJECT_DOCUMENTATION §2.4.
NTA and LNMR are NaN at τ=0 (the flipped subset is empty).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from src.data.ham10000 import CLASS_NAMES

logger = logging.getLogger(__name__)

# ----- path parsing ---------------------------------------------------------

# results/training/{method}/{dataset}/{init}_{optim}/tau_NN/fold_NN/test_metrics.json
_INIT_OPTIM_RE = re.compile(r"^(?P<init>[a-z0-9]+)_(?P<optim>[a-z0-9]+)$")
_TAU_RE = re.compile(r"^tau_(?P<tau>\d{2})$")
_FOLD_RE = re.compile(r"^fold_(?P<fold>\d{2})$")

# Scalar metric columns loaded from test_metrics.json, in the order they are
# written into the output DataFrame. Cohen's kappa is intentionally NOT in
# this list (removed per PROJECT_DOCUMENTATION §2.4).
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


# ----- main loaders ---------------------------------------------------------


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Walk ``results_dir/training/*/*/*/*/*/test_metrics.json`` and build a tidy df.

    Returns:
        A tidy DataFrame with columns::

            method, dataset, init, optim, tau, fold,
            balanced_accuracy, macro_f1, weighted_f1, macro_auc, n_samples,
            nta, lnmr, n_flipped, n_train, empirical_flip_rate,
            per_class_f1_akiec, ..., per_class_f1_vasc,
            confusion_matrix (list-of-lists)

        Empty DataFrame (with correct columns) if no results are present.
    """
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


# ----- aggregation helpers --------------------------------------------------


def aggregate_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the fold axis by computing mean and std for every scalar metric.

    Includes NTA and LNMR in the aggregation. At τ=0, both will be aggregated
    over an all-NaN column and return NaN mean/std — this is the correct
    behaviour (the metrics are undefined when no samples were flipped).

    Args:
        df: Output of :func:`load_all_results`.

    Returns:
        DataFrame indexed by (dataset, init, optim, method, tau), with
        columns ``{metric}_mean`` and ``{metric}_std`` for every scalar
        metric, plus ``n_folds`` (how many folds contributed).
    """
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


# ----- private helpers ------------------------------------------------------


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
