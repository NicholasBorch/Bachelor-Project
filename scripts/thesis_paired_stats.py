#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
thesis_paired_stats.py
======================
One statistics module shared by all three Results scripts (RQ1 human comparison,
Results.2 baseline degradation, Results.3 method comparison) so the whole thesis
uses identical machinery. Everything operates on a single per-fold paired
difference vector d (length n = number of folds), where

    d_k = score_A(fold k) - score_B(fold k)

and the test asks whether d is centred away from zero.

For every (cell) it returns, on the SAME d:

  delta            mean(d)                              effect size (raw units)
  delta_ci_lo/hi   bootstrap CI of mean(d)              effect-size uncertainty
  W, p_wilcoxon    exact paired Wilcoxon signed-rank    primary significance test
  p_perm           exact paired sign-flip permutation   confirmatory (assumption-light)
  r_rb             matched-pairs rank-biserial          standardized effect size
  direction        +1 / -1 / 0  (sign of delta)         "better" vs "worse"

Holm correction and a Wilcoxon-vs-permutation concordance flag are applied at the
family level by `holm` and `add_holm_and_flags`.

Design notes baked in:
  * n is small (typically 10). Wilcoxon and permutation are computed EXACTLY
    (full enumeration for permutation when n <= PERM_EXACT_MAX, else sampled).
    At n=10 the smallest two-sided p for either test is 2/1024 ~= 0.00195; this
    is the floor, not an artifact.
  * The bootstrap resamples the DIFFERENCES (pairing preserved). It is an
    effect-size tool, not a third significance test.
  * Direction is carried everywhere so "significantly worse" (delta<0) is
    reported distinctly from "significantly better" (delta>0).

This module has no project dependencies (only numpy/scipy) so all three scripts
can import it regardless of their own package layout.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from scipy import stats


# Enumerate all 2^n sign flips up to this n; above it, sample PERM_N_SAMPLES.
PERM_EXACT_MAX = 20
PERM_N_SAMPLES = 100_000

SIG_LEVELS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))
NS_SYMBOL = "n.s."


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
@dataclass
class PairedResult:
    n: int
    delta: float                 # mean(d)
    delta_ci_lo: float
    delta_ci_hi: float
    W: float
    p_wilcoxon: float            # raw (uncorrected)
    p_perm: float                # raw (uncorrected)
    r_rb: float                  # matched-pairs rank-biserial
    direction: int               # sign of delta: +1, -1, 0
    n_boot: int
    perm_exact: bool

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# the four core computations, all on one difference vector d
# ---------------------------------------------------------------------------
def _clean(d) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    return d[~np.isnan(d)]


def wilcoxon_exact(d) -> tuple[float, float]:
    """Exact two-sided paired Wilcoxon signed-rank on differences d.
    Returns (W, p). W is scipy's statistic (min of signed-rank sums)."""
    d = _clean(d)
    n = d.size
    if n == 0:
        return (np.nan, np.nan)
    if np.allclose(d, 0.0):
        return (0.0, 1.0)
    # scipy needs the two paired vectors or the differences; pass differences
    # against zero via the one-sample form (x = d).
    try:
        res = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox",
                             correction=False, mode="exact")
    except TypeError:
        # newer scipy renamed mode -> method
        res = stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox",
                             correction=False, method="exact")
    except ValueError:
        return (0.0, 1.0)
    return (float(res.statistic), float(res.pvalue))


def permutation_exact(d) -> tuple[float, bool]:
    """Exact two-sided paired sign-flip permutation p on the MEAN of d.
    Enumerates all 2^n sign assignments when n <= PERM_EXACT_MAX, else samples.
    Returns (p, exact_flag)."""
    d = _clean(d)
    n = d.size
    if n == 0:
        return (np.nan, True)
    if np.allclose(d, 0.0):
        return (1.0, True)
    t_obs = abs(d.mean())
    if n <= PERM_EXACT_MAX:
        # all 2^n sign vectors via the rows of a {-1,+1} matrix
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
        means = np.abs(signs @ d) / n
        p = float(np.mean(means >= t_obs - 1e-12))
        return (p, True)
    rng = np.random.default_rng(0)
    signs = rng.choice((-1.0, 1.0), size=(PERM_N_SAMPLES, n))
    means = np.abs(signs @ d) / n
    # +1 / (N+1) style guard so p is never exactly 0 under sampling
    p = float((np.sum(means >= t_obs - 1e-12) + 1) / (PERM_N_SAMPLES + 1))
    return (p, False)


def bootstrap_diff_ci(d, n_boot=10_000, alpha=0.05, seed=0) -> tuple[float, float]:
    """Percentile bootstrap CI of mean(d): resample the differences with
    replacement. Pairing is preserved because d is already the paired diff."""
    d = _clean(d)
    n = d.size
    if n == 0:
        return (np.nan, np.nan)
    if n == 1:
        return (float(d[0]), float(d[0]))
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return (lo, hi)


def rank_biserial(d, W=None) -> float:
    """Matched-pairs rank-biserial correlation as an effect size.
    r = 1 - 4W / (n(n+1)); sign re-attached from the mean so it reads as
    +1 (A>B unanimously) ... -1 (A<B unanimously). Excludes exact-zero diffs
    from n, matching the signed-rank treatment."""
    d = _clean(d)
    d = d[d != 0.0]
    n = d.size
    if n == 0:
        return np.nan
    if W is None:
        W, _ = wilcoxon_exact(d)
    mag = 1.0 - (4.0 * W) / (n * (n + 1))
    mag = abs(mag)
    return float(np.sign(d.mean()) * mag)


# ---------------------------------------------------------------------------
# one call computes everything for a single difference vector
# ---------------------------------------------------------------------------
def paired_compare(d, n_boot=10_000, boot_seed=0, alpha=0.05) -> PairedResult:
    """Run all four computations on one paired difference vector d."""
    d = _clean(d)
    n = d.size
    if n == 0:
        return PairedResult(0, *(np.nan,) * 7, 0, n_boot, True)
    delta = float(d.mean())
    W, p_w = wilcoxon_exact(d)
    p_perm, exact = permutation_exact(d)
    lo, hi = bootstrap_diff_ci(d, n_boot=n_boot, alpha=alpha, seed=boot_seed)
    r = rank_biserial(d, W=W)
    direction = int(np.sign(delta)) if not np.isclose(delta, 0.0) else 0
    return PairedResult(n, delta, lo, hi, W, p_w, p_perm, r, direction,
                        n_boot, exact)


def paired_compare_AB(a, b, **kw) -> PairedResult:
    """Convenience: difference two fold-aligned score vectors then compare.
    a, b are per-fold scores for method A and B (or method/baseline); the
    difference is a - b, so delta>0 means A scores higher."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    return paired_compare(a[m] - b[m], **kw)


# ---------------------------------------------------------------------------
# multiplicity + reporting helpers (applied across a family)
# ---------------------------------------------------------------------------
def holm(pvals) -> list[float]:
    """Holm step-down adjusted p-values, input order preserved. NaNs pass
    through and are excluded from the family size."""
    p = list(pvals)
    idx = [i for i, v in enumerate(p) if v is not None and not np.isnan(v)]
    m = len(idx)
    adj = [float("nan")] * len(p)
    if m == 0:
        return adj
    order = sorted(idx, key=lambda i: p[i])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def sig_code(p, ns=NS_SYMBOL) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ns
    for thr, sym in SIG_LEVELS:
        if p < thr:
            return sym
    return ns


def directional_code(p, direction, ns=NS_SYMBOL) -> str:
    """Significance stars, but a leading sign communicates direction:
    '+**' = significantly higher (A>B), '-**' = significantly lower (A<B)."""
    base = sig_code(p, ns=ns)
    if base == ns or direction == 0:
        return base
    return ("+" if direction > 0 else "-") + base


def add_holm_and_flags(results: list[dict], pkey_w="p_wilcoxon",
                       pkey_perm="p_perm", alpha=0.05) -> list[dict]:
    """Given a family (list of per-cell dicts each carrying raw p_wilcoxon and
    p_perm and a 'direction'), add Holm-corrected p for both tests, directional
    significance codes, and a concordance flag where the two tests disagree on
    significance at alpha. Mutates and returns the list."""
    pw = holm([r[pkey_w] for r in results])
    pp = holm([r[pkey_perm] for r in results])
    for r, hw, hp in zip(results, pw, pp):
        r["p_wilcoxon_holm"] = hw
        r["p_perm_holm"] = hp
        r["sig"] = directional_code(hw, r.get("direction", 0))
        r["sig_perm"] = directional_code(hp, r.get("direction", 0))
        sig_w = not (np.isnan(hw)) and hw < alpha
        sig_p = not (np.isnan(hp)) and hp < alpha
        r["concordant"] = (sig_w == sig_p)
        r["flag"] = "" if r["concordant"] else "!"
    return results


# small floor reminder other scripts can print once
def p_floor(n: int) -> float:
    """Smallest achievable two-sided p for the exact tests at sample size n."""
    return 2.0 / (2 ** n)


if __name__ == "__main__":
    # self-test on a couple of toy vectors
    rng = np.random.default_rng(1)
    print("floor at n=10:", p_floor(10))
    d_unanimous = np.array([0.10, 0.06, 0.14, 0.09, 0.11, 0.07, 0.13, 0.08, 0.12, 0.10])
    r = paired_compare(d_unanimous)
    print("unanimous:", {k: round(v, 5) if isinstance(v, float) else v
                          for k, v in r.as_dict().items()})
    d_mixed = np.array([0.03, -0.02, 0.04, 0.01, -0.01, 0.05, 0.02, -0.03, 0.06, 0.02])
    r2 = paired_compare(d_mixed)
    print("mixed:    ", {k: round(v, 5) if isinstance(v, float) else v
                         for k, v in r2.as_dict().items()})