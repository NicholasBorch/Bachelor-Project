"""Statistical tests and confidence intervals for Stage 4.

Two families of paired Wilcoxon signed-rank tests are implemented. Both
use pairing by fold (each method sees the same 10 folds) and both are
non-parametric, appropriate for small n (10 folds) and non-normal metric
distributions.

1. Method-vs-Baseline (``wilcoxon_vs_baseline``):
   At each (dataset, init, optim, τ), test whether each robust method
   differs from the CE baseline. Quantifies the benefit (or lack of
   benefit) of the noise-handling mechanism.

2. Noise-Sensitivity (``wilcoxon_vs_clean``):
   At each (dataset, init, optim, method), test whether the method's
   performance at τ > 0 differs from its own performance at τ = 0.
   Quantifies each method's resilience to label noise.

Both families can be corrected for multiple testing via
``apply_multiple_testing_corrections``, which produces Bonferroni and
Holm-Bonferroni adjusted p-values. Holm is strictly more powerful than
Bonferroni while still controlling the family-wise error rate, and is
generally preferred; both are reported so the reader can decide.

Raw p-values are encoded into a ``significance_code`` column:
    ``***`` for p < 0.001, ``**`` for p < 0.01, ``*`` for p < 0.05,
    ``ns`` otherwise.
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
    """Run scipy.stats.wilcoxon with defensive handling.

    Returns (statistic, p_value). Returns (nan, 1.0) for all-zero diffs
    (scipy raises on this; we treat as unambiguously not significant) and
    (nan, nan) on any other ValueError.
    """
    if np.all(diffs == 0):
        return float("nan"), 1.0
    try:
        res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
    except ValueError as exc:
        logger.warning("Wilcoxon failed: %s", exc)
        return float("nan"), float("nan")
    return float(res.statistic), float(res.pvalue)


# ─── Family 1: method vs baseline ──────────────────────────────────────────

def wilcoxon_vs_baseline(
    df: pd.DataFrame,
    metric: str = "balanced_accuracy",
    min_folds: int = MIN_FOLDS_FOR_WILCOXON,
) -> pd.DataFrame:
    """Paired Wilcoxon tests of each robust method against the CE baseline.

    For each (dataset, init, optim, tau), compare each robust method's
    per-fold metric against the baseline's per-fold metric, paired by
    fold via an inner merge.

    Args:
        df: Tidy results DataFrame from
            :func:`~src.analysis.aggregate.load_all_results`.
        metric: Scalar metric column to test on.
        min_folds: Skip rows with fewer than this many paired folds.

    Returns:
        DataFrame with columns::

            dataset, init, optim, tau, method,
            n_pairs, mean_diff, median_diff,
            wilcoxon_statistic, p_value,
            significance_code, significant_at_05
    """
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


# ─── Family 2: noise sensitivity (tau=0 vs tau>0 per method) ──────────────

def wilcoxon_vs_clean(
    df: pd.DataFrame,
    metric: str = "balanced_accuracy",
    min_folds: int = MIN_FOLDS_FOR_WILCOXON,
    clean_tau: float = 0.0,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Per-method noise-sensitivity: each τ > 0 against the method's own τ = 0.

    For each (dataset, init, optim, method), compare metric at τ = clean_tau
    against metric at each τ ≠ clean_tau, paired by fold.

    Negative ``mean_diff`` indicates the method performs WORSE under noise
    (the expected direction for methods that are not fully noise-robust).
    Small-magnitude ``mean_diff`` together with non-significant p suggests
    the method maintains performance.

    Returns:
        DataFrame with columns::

            dataset, init, optim, method, tau,
            n_pairs, mean_diff, median_diff,
            wilcoxon_statistic, p_value,
            significance_code, significant_at_05
    """
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


# ─── Multiple-testing corrections ─────────────────────────────────────────

def apply_multiple_testing_corrections(
    df: pd.DataFrame,
    p_col: str = "p_value",
    family_cols: list[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply Bonferroni and Holm-Bonferroni corrections within each family.

    Both corrections control the family-wise error rate (probability of
    any false positive) at the specified ``alpha``. Holm-Bonferroni is
    uniformly more powerful than plain Bonferroni and should generally be
    preferred; both are reported so readers can decide.

    Args:
        df: Tidy DataFrame containing a column of raw p-values.
        p_col: Name of the p-value column.
        family_cols: Column names defining test families. Each unique
            combination is treated as an independent family and correction
            is applied within it. If ``None``, the entire DataFrame is
            treated as one family (global correction).
        alpha: Family-wise error-rate target.

    Returns:
        A copy of ``df`` with added columns::

            p_value_bonferroni       (min(p * n_family, 1.0))
            significant_bonferroni   (bool at alpha)
            p_value_holm             (Holm-adjusted)
            significant_holm         (bool at alpha)
            family_size              (int, tests in the family)

        NaN p-values are preserved as-is; they do not consume a slot in
        the family size.

    Holm-Bonferroni procedure:
        Sort p-values ascending: p_(1) ≤ p_(2) ≤ ... ≤ p_(n).
        Adjusted p_(i) = max_{j ≤ i} ((n - j + 1) * p_(j)), clipped to 1.0.
        Reject at alpha iff the Holm-adjusted p < alpha.
    """
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


# ─── Bootstrap CI ─────────────────────────────────────────────────────────

def bootstrap_ci(
    values: np.ndarray | list[float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int | None = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Default ``n_bootstrap=2000`` matches the thesis protocol.

    Args:
        values: Observations to resample with replacement.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (returns a (1-alpha) CI).
        random_state: Seed for reproducibility.

    Returns:
        (lower, upper) bounds of the percentile CI. Returns (nan, nan) if
        fewer than two non-NaN values are provided.
    """
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


# ─── Helpers ──────────────────────────────────────────────────────────────

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
