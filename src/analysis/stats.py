"""
Paired Wilcoxon signed-rank tests and confidence intervals for Stage 4.

Two families, both paired by fold and non-parametric: (1) wilcoxon_vs_baseline tests
each robust method against the CE baseline at each (dataset, init, optim, tau); (2)
wilcoxon_vs_clean tests each method's tau>0 against its own tau=0.
apply_multiple_testing_corrections adds Bonferroni and (OLD version - NOT USED) Holm-Bonferroni
family-wise corrections. Raw p is encoded in significance_code (***/**/*/ns).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

MIN_FOLDS_FOR_WILCOXON = 5
"""Minimum number of paired folds required to run the signed-rank test.

Below this, the test has essentially no power and we skip the row rather
than emit a misleading p-value.
"""

_SIG_THRESHOLDS = [
    (0.001, "***"),
    (0.01, "**"),
    (0.05, "*"),
]


def _significance_code(p: float) -> str:
    """Convert a p-value to the conventional ``*/***/**/ns`` encoding."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "nan"
    for threshold, code in _SIG_THRESHOLDS:
        if p < threshold:
            return code
    return "ns"


def _wilcoxon_safe(diffs: np.ndarray) -> tuple[float, float]:
    """scipy wilcoxon with defensive handling: (nan, 1.0) for all-zero diffs, (nan, nan) on other errors."""
    if np.all(diffs == 0):
        return float("nan"), 1.0
    try:
        res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
    except ValueError as exc:
        logger.warning("Wilcoxon failed: %s", exc)
        return float("nan"), float("nan")
    return float(res.statistic), float(res.pvalue)


# Family 1: method vs baseline

def wilcoxon_vs_baseline(
    df: pd.DataFrame,
    metric: str = "balanced_accuracy",
    min_folds: int = MIN_FOLDS_FOR_WILCOXON,
) -> pd.DataFrame:
    """Paired Wilcoxon of each robust method vs the CE baseline, per (dataset, init, optim, tau)."""
    if df.empty or metric not in df.columns:
        return _empty_vs_baseline_df()

    rows: list[dict] = []
    group_cols = ["dataset", "init", "optim", "tau"]

    for keys, sub in df.groupby(group_cols, dropna=False):
        dataset, init, optim, tau = keys
        baseline = sub[sub["method"] == "baseline"][["fold", metric]].dropna()
        if baseline.empty:
            continue

        for method_name, method_df in sub.groupby("method", dropna=False):
            if method_name == "baseline":
                continue
            method_df = method_df[["fold", metric]].dropna()
            if method_df.empty:
                continue
            merged = baseline.merge(
                method_df, on="fold", how="inner", suffixes=("_baseline", "_method"),
            )
            if len(merged) < min_folds:
                continue
            diffs = (
                merged[f"{metric}_method"].to_numpy()
                - merged[f"{metric}_baseline"].to_numpy()
            )
            stat, p = _wilcoxon_safe(diffs)
            rows.append({
                "dataset": dataset,
                "init": init,
                "optim": optim,
                "tau": float(tau),
                "method": method_name,
                "n_pairs": int(len(merged)),
                "mean_diff": float(np.mean(diffs)),
                "median_diff": float(np.median(diffs)),
                "wilcoxon_statistic": stat,
                "p_value": p,
                "significance_code": _significance_code(p),
                "significant_at_05": bool(not np.isnan(p) and p < 0.05),
            })

    if not rows:
        return _empty_vs_baseline_df()
    return pd.DataFrame(rows).sort_values(
        ["dataset", "init", "optim", "tau", "method"]
    ).reset_index(drop=True)


# Family 2: noise sensitivity (tau=0 vs tau>0 per method)

def wilcoxon_vs_clean(
    df: pd.DataFrame,
    metric: str = "balanced_accuracy",
    min_folds: int = MIN_FOLDS_FOR_WILCOXON,
    clean_tau: float = 0.0,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Per-method noise sensitivity: each tau>0 vs the method's own tau=0, paired by fold."""
    if df.empty or metric not in df.columns:
        return _empty_vs_clean_df()

    rows: list[dict] = []
    group_cols = ["dataset", "init", "optim", "method"]

    for keys, sub in df.groupby(group_cols, dropna=False):
        dataset, init, optim, method_name = keys
        clean_sub = sub[np.isclose(sub["tau"], clean_tau, atol=tol)][["fold", metric]].dropna()
        if clean_sub.empty:
            continue

        for tau_value, tau_df in sub.groupby("tau", dropna=False):
            if np.isclose(tau_value, clean_tau, atol=tol):
                continue
            tau_df = tau_df[["fold", metric]].dropna()
            if tau_df.empty:
                continue
            merged = clean_sub.merge(
                tau_df, on="fold", how="inner", suffixes=("_clean", "_noisy"),
            )
            if len(merged) < min_folds:
                continue
            diffs = (
                merged[f"{metric}_noisy"].to_numpy()
                - merged[f"{metric}_clean"].to_numpy()
            )
            stat, p = _wilcoxon_safe(diffs)
            rows.append({
                "dataset": dataset,
                "init": init,
                "optim": optim,
                "method": method_name,
                "tau": float(tau_value),
                "n_pairs": int(len(merged)),
                "mean_diff": float(np.mean(diffs)),
                "median_diff": float(np.median(diffs)),
                "wilcoxon_statistic": stat,
                "p_value": p,
                "significance_code": _significance_code(p),
                "significant_at_05": bool(not np.isnan(p) and p < 0.05),
            })

    if not rows:
        return _empty_vs_clean_df()
    return pd.DataFrame(rows).sort_values(
        ["dataset", "init", "optim", "method", "tau"]
    ).reset_index(drop=True)


# Multiple-testing corrections

def apply_multiple_testing_corrections(
    df: pd.DataFrame,
    p_col: str = "p_value",
    family_cols: list[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add Bonferroni and Holm-Bonferroni adjusted p-values (and significance flags) within each family."""
    out = df.copy().reset_index(drop=True)
    out["p_value_bonferroni"] = np.nan
    out["significant_bonferroni"] = False
    out["p_value_holm"] = np.nan
    out["significant_holm"] = False
    out["family_size"] = 0

    if len(out) == 0:
        return out

    if family_cols is None:
        family_groups = [(None, out.index.to_list())]
    else:
        family_groups = [
            (keys, group.index.to_list())
            for keys, group in out.groupby(family_cols, dropna=False)
        ]

    for _family_key, idx_list in family_groups:
        # Only non-NaN p-values contribute to the family size and the
        # correction. NaN entries stay NaN on both adjusted columns.
        idx_valid = [i for i in idx_list if not np.isnan(out.loc[i, p_col])]
        n = len(idx_valid)
        out.loc[idx_list, "family_size"] = n
        if n == 0:
            continue

        ps = out.loc[idx_valid, p_col].to_numpy(dtype=float)

        # Bonferroni: p_adj = min(n * p, 1)
        p_bonf = np.minimum(ps * n, 1.0)

        # Holm-Bonferroni: sort ascending, multiply by (n - rank + 1),
        # enforce monotonicity by running max.
        order = np.argsort(ps, kind="stable")
        p_sorted = ps[order]
        coeffs = (n - np.arange(n)).astype(float)  # [n, n-1, ..., 1]
        raw_adj = coeffs * p_sorted
        p_holm_sorted = np.minimum(np.maximum.accumulate(raw_adj), 1.0)
        p_holm = np.empty_like(ps)
        p_holm[order] = p_holm_sorted

        for k, i in enumerate(idx_valid):
            out.at[i, "p_value_bonferroni"] = float(p_bonf[k])
            out.at[i, "significant_bonferroni"] = bool(p_bonf[k] < alpha)
            out.at[i, "p_value_holm"] = float(p_holm[k])
            out.at[i, "significant_holm"] = bool(p_holm[k] < alpha)

    return out


# Bootstrap CI

def bootstrap_ci(
    values: np.ndarray | list[float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean (n_bootstrap=2000); (nan, nan) if fewer than two valid values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(random_state)
    n = arr.size
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = sample.mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


# Helpers

def _empty_vs_baseline_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "dataset", "init", "optim", "tau", "method",
        "n_pairs", "mean_diff", "median_diff",
        "wilcoxon_statistic", "p_value",
        "significance_code", "significant_at_05",
    ])


def _empty_vs_clean_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "dataset", "init", "optim", "method", "tau",
        "n_pairs", "mean_diff", "median_diff",
        "wilcoxon_statistic", "p_value",
        "significance_code", "significant_at_05",
    ])