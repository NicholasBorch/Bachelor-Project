#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Results Part 3 - Method comparison under label noise (RQ2) + associated appendix.

WHAT THIS PRODUCES
------------------
Body (Results.3):
  1. fig_results3_money_<P>.pdf/.png
     Twin-panel grouped bar chart: BA (left) and Macro F1 (right),
     4 methods x 6 tau, bootstrap-CI error bars, method-vs-baseline
     significance markers above each non-baseline bar.
  2. tab_results3_body_<P>.tex
     One combined table: tau as rows, method-grouped columns, two-line
     cells (mean on top, 95% bootstrap CI below), method-vs-baseline
     significance as superscript stars, best method per (tau, metric) bold.

Appendix:
  3. tab_app_auc_<P>.tex           - Macro AUC body-style table (point 2).
     fig_app_auc_<P>.pdf/.png      - optional Macro AUC grouped-bar figure.
  4. tab_app_mvb_wilcoxon_<P>.tex  - full method-vs-baseline Wilcoxon stats:
     per (metric, method, tau): mean delta, W, raw p, Holm p, sig (point 3).
  5. tab_app_noise_vs_clean_<P>.tex- noise-sensitivity (tau vs clean) for the
     three robust methods, supports the one-sentence body claim (optional pt 4).
  6. tab_app_method_vs_method_<P>.tex - pairwise method-vs-method Wilcoxon,
     per tau, Holm-corrected within tau (your decision 2: appendix only).

It also prints a "PROSE HELPER" block: the descriptive facts the body
commentary needs (first significant tau per method, whether the gap widens
with tau, BA-vs-MacroF1 divergence at the top tau, noise-sensitivity summary).

============================================================================
THE ONLY THING YOU EDIT IS THE CONFIG BLOCK BELOW.
============================================================================
Two assumptions you may need to change to match your actual repo:

  (a) WHERE the per-run metrics live and HOW the path encodes
      (protocol, method, tau, fold).  -> CONFIG.SOURCE / PATH_TEMPLATE / PATH_REGEX
      Alternatively point at a single tidy CSV.       -> CONFIG.SOURCE="csv"

  (b) The JSON KEY NAMES for BA / Macro F1 / Macro AUC. -> CONFIG.METRIC_KEYS
      Run once; if a key is missing the script prints the keys it *did*
      find in the first metrics file so you can map them.

Everything else (statistics, plotting, LaTeX) follows the two design notes
and needs no change.
"""

from __future__ import annotations

import json
import re
import sys
import hashlib
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================================
# CONFIG  -- edit this block, nothing else
# ============================================================================
@dataclass
class Config:
    # ---- where the data is -------------------------------------------------
    # SOURCE = "json": walk a directory tree of per-run test_metrics.json files
    # SOURCE = "csv" : read one tidy long CSV (columns named in CSV_COLUMNS)
    SOURCE: str = "json"

    DATA_ROOT: Path = Path("./results")  # root of the run tree (SOURCE="json")
    CSV_PATH: Path = Path("./results_long.csv")  # tidy CSV (SOURCE="csv")

    # JSON tree: how a metrics file path is built and parsed.
    # Default layout assumed:
    #   {DATA_ROOT}/{protocol}/{method}/tau{tau}/fold{fold}/test_metrics.json
    # The loader GLOBS for METRICS_FILENAME under DATA_ROOT and parses the
    # four fields out of the path with PATH_REGEX. If your tree differs, edit
    # PATH_REGEX (named groups: protocol, method, tau, fold). If the fields are
    # instead stored *inside* the JSON, set USE_JSON_FIELDS=True and name them
    # in JSON_FIELD_KEYS.
    METRICS_FILENAME: str = "test_metrics.json"
    PATH_REGEX: str = r"/(?P<protocol>[^/]+)/(?P<method>[^/]+)/tau(?P<tau>[0-9.]+)/fold(?P<fold>[0-9]+)/"
    USE_JSON_FIELDS: bool = False
    JSON_FIELD_KEYS: dict = field(default_factory=lambda: {
        "protocol": "protocol", "method": "method", "tau": "tau", "fold": "fold",
    })

    # CSV layout (SOURCE="csv"): logical-name -> column-name-in-your-CSV
    CSV_COLUMNS: dict = field(default_factory=lambda: {
        "protocol": "protocol", "method": "method", "tau": "tau", "fold": "fold",
        "BA": "balanced_accuracy", "MacroF1": "macro_f1", "MacroAUC": "macro_auc",
    })

    # metric logical-name -> JSON key, with fallback aliases tried in order
    METRIC_KEYS: dict = field(default_factory=lambda: {
        "BA":       ["balanced_accuracy", "bacc", "balanced_acc", "BA"],
        "MacroF1":  ["macro_f1", "f1_macro", "macro_F1", "f1macro"],
        "MacroAUC": ["macro_auc", "auc_macro", "macro_AUC", "roc_auc_macro"],
    })

    # ---- experimental design ----------------------------------------------
    METHODS: tuple = ("baseline", "SCE", "ELR", "AsyCo")  # plot/table order
    BASELINE: str = "baseline"
    METHOD_LABELS: dict = field(default_factory=lambda: {
        "baseline": "Baseline", "SCE": "SCE", "ELR": "ELR", "AsyCo": "AsyCo",
    })
    PROTOCOLS: tuple = ("S", "SP", "A", "AP")  # SGD/scratch, SGD/pretr, Adam/scratch, Adam/pretr
    PRIMARY_PROTOCOL: str = "AP"               # clinically practical setting
    TAUS: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    N_FOLDS: int = 10

    # The abandoned "balanced" sampling runs must never enter the analysis.
    # Any path/row whose protocol or method token contains this substring is
    # dropped (case-insensitive).
    EXCLUDE_BALANCED: bool = True
    BALANCED_TOKEN: str = "balanced"

    # ---- metrics shown -----------------------------------------------------
    # logical-name -> (display, axis label, y-min, y-max)
    BODY_METRICS: tuple = ("BA", "MacroF1")
    METRIC_DISPLAY: dict = field(default_factory=lambda: {
        "BA":       ("Balanced accuracy", "Balanced accuracy", 0.0, 1.0),
        "MacroF1":  ("Macro F1",          "Macro F1",          0.0, 1.0),
        "MacroAUC": ("Macro AUC",         "Macro AUC",         0.5, 1.0),
    })

    # ---- statistics --------------------------------------------------------
    N_BOOT: int = 10000
    CI: float = 0.95
    SEED: int = 20240501
    WILCOXON_ALT: str = "two-sided"   # "two-sided" | "greater" | "less"
    HOLM_ALPHA: float = 0.05

    # significance thresholds and symbols (matches the ***/**/*/n.s. scheme)
    SIG_LEVELS: tuple = ((0.001, "***"), (0.01, "**"), (0.05, "*"))
    NS_SYMBOL: str = "n.s."
    # Cleaner + consistent with the table (blank = n.s.); flip to True if you
    # want an explicit n.s. above every non-significant bar.
    SHOW_NS_IN_FIG: bool = False
    SIG_USES_CORRECTED: bool = True   # mark significance from Holm-adjusted p

    # ---- palette (baseline light-blue, SCE teal, ELR orange, AsyCo violet) -
    PALETTE: dict = field(default_factory=lambda: {
        "baseline": "#9ec9e2",
        "SCE":      "#2a9d8f",
        "ELR":      "#e07a3f",
        "AsyCo":    "#7b5cb8",
    })

    # ---- output ------------------------------------------------------------
    FIG_DIR: Path = Path("./outputs/figures")
    TAB_DIR: Path = Path("./outputs/tables")
    FIG_DPI: int = 200
    SAVE_PNG: bool = True
    SAVE_PDF: bool = True

    # combined body table can get wide; if True emit one table per metric too
    ALSO_EMIT_PER_METRIC_BODY_TABLES: bool = False


CFG = Config()


# ============================================================================
# small utilities
# ============================================================================
def _seed_for(*parts) -> int:
    """Deterministic per-cell seed so bootstrap CIs are reproducible."""
    h = hashlib.sha256(("|".join(map(str, parts))).encode()).hexdigest()
    return (CFG.SEED + int(h[:8], 16)) % (2**32 - 1)


def sig_symbol(p: float, ns: bool = True) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    for thr, sym in CFG.SIG_LEVELS:
        if p < thr:
            return sym
    return CFG.NS_SYMBOL if ns else ""


def fmt_metric(x: float, nd: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fmt_signed(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if abs(x) < 0.5 * 10 ** (-nd):   # avoid "-0.000"
        return f"{0.0:+.{nd}f}"
    return f"{x:+.{nd}f}"


def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 0.001:
        return r"$<0.001$"
    return f"{p:.3f}"


def fmt_W(w: float) -> str:
    if w is None or (isinstance(w, float) and np.isnan(w)):
        return "--"
    return f"{w:.0f}" if abs(w - round(w)) < 1e-9 else f"{w:.1f}"


# ============================================================================
# data loading  ->  tidy long DataFrame
# columns: protocol, method, tau, fold, BA, MacroF1, MacroAUC
# ============================================================================
def _read_metrics_json(fp: Path) -> dict:
    with open(fp, "r") as fh:
        return json.load(fh)


def _extract_metric(d: dict, aliases: list[str]) -> Optional[float]:
    for k in aliases:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                return None
    return None


def _sniff_and_warn(sample: dict, missing_logical: set[str]) -> None:
    print("\n[schema] Could not find these metric(s) in the first file: "
          f"{sorted(missing_logical)}")
    print("[schema] Keys present in that file were:")
    for k in sorted(sample.keys()):
        print(f"           {k!r}")
    print("[schema] -> map them in CONFIG.METRIC_KEYS and re-run.\n")


def load_long_df() -> pd.DataFrame:
    if CFG.SOURCE == "csv":
        df = pd.read_csv(CFG.CSV_PATH)
        ren = {v: k for k, v in CFG.CSV_COLUMNS.items()}
        df = df.rename(columns=ren)
        keep = ["protocol", "method", "tau", "fold", "BA", "MacroF1", "MacroAUC"]
        df = df[[c for c in keep if c in df.columns]].copy()
    else:
        rgx = re.compile(CFG.PATH_REGEX)
        rows, sample, missing = [], None, set()
        files = sorted(Path(CFG.DATA_ROOT).rglob(CFG.METRICS_FILENAME))
        if not files:
            raise FileNotFoundError(
                f"No '{CFG.METRICS_FILENAME}' under {CFG.DATA_ROOT}. "
                "Set CONFIG.DATA_ROOT / METRICS_FILENAME, or use SOURCE='csv'.")
        for fp in files:
            d = _read_metrics_json(fp)
            if sample is None:
                sample = d
            if CFG.USE_JSON_FIELDS:
                jk = CFG.JSON_FIELD_KEYS
                meta = {k: d.get(jk[k]) for k in ("protocol", "method", "tau", "fold")}
                if any(v is None for v in meta.values()):
                    continue
            else:
                m = rgx.search(str(fp).replace("\\", "/"))
                if not m:
                    continue
                meta = m.groupdict()
            rec = {
                "protocol": str(meta["protocol"]),
                "method": str(meta["method"]),
                "tau": float(meta["tau"]),
                "fold": int(meta["fold"]),
            }
            for logical, aliases in CFG.METRIC_KEYS.items():
                val = _extract_metric(d, aliases)
                if val is None:
                    missing.add(logical)
                rec[logical] = val
            rows.append(rec)
        if not rows:
            raise RuntimeError("Found metrics files but parsed 0 rows. "
                               "Check CONFIG.PATH_REGEX against your tree.")
        # if a body metric was missing everywhere, sniff and stop early
        df = pd.DataFrame(rows)
        for logical in ("BA", "MacroF1"):
            if logical in missing and df[logical].isna().all():
                _sniff_and_warn(sample, missing)
                raise SystemExit("Map metric keys in CONFIG.METRIC_KEYS, then re-run.")

    # ---- normalise + filter ------------------------------------------------
    if CFG.EXCLUDE_BALANCED:
        tok = CFG.BALANCED_TOKEN.lower()
        mask = (df["protocol"].str.lower().str.contains(tok)
                | df["method"].str.lower().str.contains(tok))
        if mask.any():
            print(f"[load] dropping {int(mask.sum())} 'balanced' rows "
                  "(EXCLUDE_BALANCED=True).")
        df = df[~mask].copy()

    # snap tau to the configured grid (guards 0.30000004-style float noise)
    grid = np.array(CFG.TAUS, float)
    df["tau"] = df["tau"].apply(lambda t: float(grid[np.argmin(np.abs(grid - t))]))
    df = df[df["method"].isin(CFG.METHODS)].copy()
    df = df.drop_duplicates(subset=["protocol", "method", "tau", "fold"])
    df = df.sort_values(["protocol", "method", "tau", "fold"]).reset_index(drop=True)
    return df


def completeness_report(df: pd.DataFrame, protocol: str) -> None:
    """Print missing (method, tau, fold) cells for the primary protocol.
    Directly serves the 'is AsyCo complete?' check from the design note."""
    print(f"\n[completeness] protocol = {protocol} "
          f"(expected {CFG.N_FOLDS} folds per method x tau)")
    sub = df[df["protocol"] == protocol]
    any_missing = False
    for method in CFG.METHODS:
        for tau in CFG.TAUS:
            cell = sub[(sub["method"] == method) & (np.isclose(sub["tau"], tau))]
            folds = sorted(cell["fold"].unique().tolist())
            if len(folds) != CFG.N_FOLDS:
                any_missing = True
                want = set(range(CFG.N_FOLDS))
                miss = sorted(want - set(folds))
                print(f"   ! {method:9s} tau={tau:.1f}: "
                      f"{len(folds)}/{CFG.N_FOLDS} folds (missing folds {miss})")
    if not any_missing:
        print("   OK - every method x tau has the full fold set.")
    print()


# ============================================================================
# statistics
# ============================================================================
def bootstrap_ci(values, n_boot=None, ci=None, seed=0):
    """Percentile bootstrap over fold values. Returns (mean, lo, hi).
    Swap in your own bootstrap_ci here if you prefer; the interface is
    (mean, lo, hi)."""
    n_boot = CFG.N_BOOT if n_boot is None else n_boot
    ci = CFG.CI if ci is None else ci
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    if v.size == 1:
        return (float(v[0]), float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    boot = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return (float(v.mean()), lo, hi)


def _wilcoxon_compat(a, b, alternative):
    """scipy renamed mode->method across versions; call defensively."""
    try:
        res = stats.wilcoxon(a, b, alternative=alternative,
                             zero_method="wilcox", correction=False)
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        # e.g. all differences zero
        return 0.0, 1.0


def wilcoxon_paired(a, b, alternative=None):
    """Paired Wilcoxon signed-rank on fold-aligned vectors a (e.g. method)
    vs b (e.g. baseline). Returns (W, p, n_pairs, mean_delta) where
    mean_delta = mean(a-b)."""
    alternative = CFG.WILCOXON_ALT if alternative is None else alternative
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    n = a.size
    if n == 0:
        return (np.nan, np.nan, 0, np.nan)
    d = a - b
    md = float(np.mean(d))
    if np.allclose(d, 0.0):
        return (0.0, 1.0, n, md)
    W, p = _wilcoxon_compat(a, b, alternative)
    return (W, p, n, md)


def holm(pvals):
    """Holm step-down correction. NaNs are ignored (stay NaN)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if idx.size == 0:
        return out
    pv = p[idx]
    order = np.argsort(pv)
    m = pv.size
    running = 0.0
    adj = np.empty(m)
    for rank, oi in enumerate(order):
        running = max(running, (m - rank) * pv[oi])
        adj[oi] = min(running, 1.0)
    out[idx] = adj
    return out


def _wide_on_fold(df, protocol, metric, tau):
    """Pivot to fold x method for one (protocol, metric, tau) so paired tests
    use matched folds."""
    sub = df[(df["protocol"] == protocol) & (np.isclose(df["tau"], tau))]
    return sub.pivot_table(index="fold", columns="method", values=metric)


def summarize(df, protocol):
    """Per (metric, method, tau): mean, lo, hi over folds."""
    recs = []
    for metric in CFG.METRIC_DISPLAY:
        if metric not in df.columns:
            continue
        for method in CFG.METHODS:
            for tau in CFG.TAUS:
                cell = df[(df["protocol"] == protocol)
                          & (df["method"] == method)
                          & (np.isclose(df["tau"], tau))][metric].values
                mean, lo, hi = bootstrap_ci(cell, seed=_seed_for(protocol, metric, method, tau))
                recs.append(dict(metric=metric, method=method, tau=tau,
                                 mean=mean, lo=lo, hi=hi, n=int(np.sum(~np.isnan(cell)))))
    return pd.DataFrame(recs)


def method_vs_baseline(df, protocol):
    """Paired Wilcoxon, each non-baseline method vs baseline, by fold, at each
    tau. Holm-corrected ACROSS tau within (metric, method)."""
    recs = []
    for metric in CFG.METRIC_DISPLAY:
        if metric not in df.columns:
            continue
        for method in [m for m in CFG.METHODS if m != CFG.BASELINE]:
            block = []
            for tau in CFG.TAUS:
                w = _wide_on_fold(df, protocol, metric, tau)
                if CFG.BASELINE in w.columns and method in w.columns:
                    W, p, n, md = wilcoxon_paired(w[method].values, w[CFG.BASELINE].values)
                else:
                    W, p, n, md = (np.nan, np.nan, 0, np.nan)
                block.append(dict(metric=metric, method=method, tau=tau,
                                  W=W, p_raw=p, n=n, mean_delta=md))
            padj = holm([b["p_raw"] for b in block])
            for b, pa in zip(block, padj):
                b["p_holm"] = pa
                b["sig"] = sig_symbol(pa if CFG.SIG_USES_CORRECTED else b["p_raw"])
            recs.extend(block)
    return pd.DataFrame(recs)


def noise_vs_clean(df, protocol, methods=None):
    """Paired Wilcoxon, each method at tau>0 vs the SAME method at tau=0, by
    fold. Holm-corrected across tau within (metric, method)."""
    methods = methods or [m for m in CFG.METHODS if m != CFG.BASELINE]
    clean = CFG.TAUS[0]
    recs = []
    for metric in CFG.METRIC_DISPLAY:
        if metric not in df.columns:
            continue
        for method in methods:
            block = []
            for tau in [t for t in CFG.TAUS if not np.isclose(t, clean)]:
                w0 = _wide_on_fold(df, protocol, metric, clean)
                wt = _wide_on_fold(df, protocol, metric, tau)
                if method in w0.columns and method in wt.columns:
                    paired = pd.concat([w0[method].rename("clean"),
                                        wt[method].rename("noisy")], axis=1).dropna()
                    W, p, n, md = wilcoxon_paired(paired["noisy"].values,
                                                  paired["clean"].values)
                else:
                    W, p, n, md = (np.nan, np.nan, 0, np.nan)
                block.append(dict(metric=metric, method=method, tau=tau,
                                  W=W, p_raw=p, n=n, mean_delta=md))
            padj = holm([b["p_raw"] for b in block])
            for b, pa in zip(block, padj):
                b["p_holm"] = pa
                b["sig"] = sig_symbol(pa if CFG.SIG_USES_CORRECTED else b["p_raw"])
            recs.extend(block)
    return pd.DataFrame(recs)


def method_vs_method(df, protocol):
    """All pairwise method comparisons, paired Wilcoxon by fold, at each tau.
    Holm-corrected WITHIN tau across the pair family (your decision 2:
    appendix only)."""
    pairs = list(itertools.combinations(CFG.METHODS, 2))
    recs = []
    for metric in CFG.METRIC_DISPLAY:
        if metric not in df.columns:
            continue
        for tau in CFG.TAUS:
            w = _wide_on_fold(df, protocol, metric, tau)
            block = []
            for a, b in pairs:
                if a in w.columns and b in w.columns:
                    W, p, n, md = wilcoxon_paired(w[a].values, w[b].values)
                else:
                    W, p, n, md = (np.nan, np.nan, 0, np.nan)
                block.append(dict(metric=metric, tau=tau, method_a=a, method_b=b,
                                  W=W, p_raw=p, n=n, mean_delta=md))
            padj = holm([x["p_raw"] for x in block])
            for x, pa in zip(block, padj):
                x["p_holm"] = pa
                x["sig"] = sig_symbol(pa if CFG.SIG_USES_CORRECTED else x["p_raw"])
            recs.extend(block)
    return pd.DataFrame(recs)


# ============================================================================
# plotting
# ============================================================================
def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": CFG.FIG_DPI,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _yerr(sub_metric_method_tau):
    """Return asymmetric yerr [[lower],[upper]] from mean/lo/hi rows, clipped
    to be non-negative."""
    means = sub_metric_method_tau["mean"].values
    lo = sub_metric_method_tau["lo"].values
    hi = sub_metric_method_tau["hi"].values
    lower = np.clip(means - lo, 0, None)
    upper = np.clip(hi - means, 0, None)
    return np.vstack([lower, upper])


def _grouped_bar_panel(ax, summary, mvb_stats, metric, protocol, show_sig=True):
    taus = list(CFG.TAUS)
    methods = list(CFG.METHODS)
    n_m = len(methods)
    x = np.arange(len(taus))
    width = 0.8 / n_m

    metric_sum = summary[summary["metric"] == metric]
    top_of_group = np.zeros(len(taus))  # for placing sig markers / ylim

    for j, method in enumerate(methods):
        rows = (metric_sum[metric_sum["method"] == method]
                .set_index("tau").reindex(taus).reset_index())
        means = rows["mean"].values
        yerr = _yerr(rows)
        offs = (j - (n_m - 1) / 2) * width
        ax.bar(x + offs, means, width=width, yerr=yerr,
               color=CFG.PALETTE.get(method, None),
               edgecolor="white", linewidth=0.6, capsize=2.5,
               error_kw=dict(elinewidth=0.9, alpha=0.85),
               label=CFG.METHOD_LABELS.get(method, method), zorder=3)
        tops = np.nan_to_num(rows["hi"].values, nan=0.0)
        top_of_group = np.maximum(top_of_group, tops)

        # significance markers above non-baseline bars
        if show_sig and method != CFG.BASELINE and mvb_stats is not None:
            st = mvb_stats[(mvb_stats["metric"] == metric)
                           & (mvb_stats["method"] == method)].set_index("tau")
            for xi, tau in zip(x, taus):
                if tau not in st.index:
                    continue
                pcol = "p_holm" if CFG.SIG_USES_CORRECTED else "p_raw"
                sym = sig_symbol(st.loc[tau, pcol], ns=CFG.SHOW_NS_IN_FIG)
                if not sym:
                    continue
                bar_top = np.nan_to_num(rows.loc[rows["tau"] == tau, "hi"].values, nan=0.0)
                bar_top = bar_top[0] if len(bar_top) else 0.0
                style = dict(ha="center", va="bottom", fontsize=8, zorder=5)
                if sym == CFG.NS_SYMBOL:
                    ax.text(xi + offs, bar_top + 0.012, sym, color="0.45",
                            fontsize=7, **{k: v for k, v in style.items()
                                           if k not in ("fontsize",)})
                else:
                    ax.text(xi + offs, bar_top + 0.012, sym, color="0.15", **style)

    disp, ylab, ymin, ymax = CFG.METRIC_DISPLAY[metric]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1f}" for t in taus])
    ax.set_xlabel(r"Noise rate $\tau$")
    ax.set_ylabel(ylab)
    ax.set_title(disp)
    headroom = 0.06 * (ymax - ymin)
    ax.set_ylim(ymin, min(ymax, float(np.nanmax(top_of_group)) + headroom + 0.04)
                if np.isfinite(np.nanmax(top_of_group)) else ymax)


def fig_money(summary, mvb_stats, protocol, metrics=None, fname=None):
    """Twin-panel grouped bars: BA (left) + Macro F1 (right)."""
    metrics = metrics or list(CFG.BODY_METRICS)
    _apply_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.6))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        _grouped_bar_panel(ax, summary, mvb_stats, metric, protocol)

    handles = [Patch(facecolor=CFG.PALETTE.get(m), edgecolor="white",
                     label=CFG.METHOD_LABELS.get(m, m)) for m in CFG.METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=len(CFG.METHODS),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    star_note = ("Stars: method vs. baseline, paired Wilcoxon, "
                 "Holm-corrected across $\\tau$ "
                 "($^{*}p<.05,\\ ^{**}p<.01,\\ ^{***}p<.001$).")
    fig.suptitle(f"Method comparison under label noise - protocol {protocol}",
                 y=1.0, fontsize=12.5)
    fig.text(0.5, -0.10, star_note, ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    fname = fname or f"fig_results3_money_{protocol}"
    _savefig(fig, fname)
    plt.close(fig)


def fig_auc(summary, mvb_stats, protocol, fname=None):
    """Optional appendix figure: single-panel Macro AUC grouped bars."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    _grouped_bar_panel(ax, summary, mvb_stats, "MacroAUC", protocol)
    handles = [Patch(facecolor=CFG.PALETTE.get(m), edgecolor="white",
                     label=CFG.METHOD_LABELS.get(m, m)) for m in CFG.METHODS]
    ax.legend(handles=handles, loc="lower left", ncol=2, frameon=False)
    ax.set_title(f"Macro AUC under label noise - protocol {protocol}")
    fig.tight_layout()
    fname = fname or f"fig_app_auc_{protocol}"
    _savefig(fig, fname)
    plt.close(fig)


def _savefig(fig, stem):
    CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)
    if CFG.SAVE_PDF:
        fig.savefig(CFG.FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    if CFG.SAVE_PNG:
        fig.savefig(CFG.FIG_DIR / f"{stem}.png", bbox_inches="tight")
    print(f"[fig] wrote {CFG.FIG_DIR / stem}.(pdf|png)")


# ============================================================================
# LaTeX tables
# ============================================================================
REQUIRED_PACKAGES = r"""% Required in your preamble for the generated tables:
%   \usepackage{booktabs}
%   \usepackage{makecell}
%   \usepackage{multirow}
%   \usepackage{graphicx}   % for \resizebox
% Reconcile fonts / caption style with your Results.2 tables.
"""


def _write_tex(stem, body):
    CFG.TAB_DIR.mkdir(parents=True, exist_ok=True)
    fp = CFG.TAB_DIR / f"{stem}.tex"
    with open(fp, "w") as fh:
        fh.write(REQUIRED_PACKAGES + "\n" + body + "\n")
    print(f"[tab] wrote {fp}")


def _cell_mean_ci(mean, lo, hi, sig, is_best):
    """Two-line cell: mean (with sig superscript) on top, CI below."""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return r"\makecell{--}"
    m = fmt_metric(mean)
    sup = f"^{{{sig}}}" if sig and sig != CFG.NS_SYMBOL else ""
    inner = rf"\mathbf{{{m}}}" if is_best else m
    line1 = rf"${inner}{sup}$"
    line2 = rf"{{\scriptsize $({fmt_metric(lo)},\,{fmt_metric(hi)})$}}"
    return rf"\makecell{{{line1}\\{line2}}}"


def tex_body_combined(summary, mvb_stats, protocol,
                      metrics=None, stem=None, caption=None, label=None):
    """One table: tau rows; metric-grouped method columns; mean+CI cells;
    method-vs-baseline sig stars; best method per (tau, metric) bold."""
    metrics = metrics or list(CFG.BODY_METRICS)
    methods = list(CFG.METHODS)
    stem = stem or f"tab_results3_body_{protocol}"
    label = label or f"tab:results3_body_{protocol}"
    caption = caption or (
        f"Method comparison under label noise (protocol {protocol}). "
        f"Cells give the mean over {CFG.N_FOLDS} folds with the 95\\% bootstrap "
        f"confidence interval below. Stars mark a significant method-vs-baseline "
        f"difference (paired Wilcoxon by fold, Holm-corrected across $\\tau$: "
        f"$^{{*}}p<.05$, $^{{**}}p<.01$, $^{{***}}p<.001$; no star = n.s.). "
        f"The best method per $\\tau$ and metric is in bold.")

    ncols = 1 + len(metrics) * len(methods)
    colspec = "l" + "".join(["*{%d}{c}" % len(methods) for _ in metrics])

    # header
    top = [r"\multirow{2}{*}{$\tau$}"]
    for metric in metrics:
        disp = CFG.METRIC_DISPLAY[metric][0]
        top.append(r"\multicolumn{%d}{c}{%s}" % (len(methods), disp))
    header1 = " & ".join(top) + r" \\"
    cmids = []
    start = 2
    for _ in metrics:
        cmids.append(r"\cmidrule(lr){%d-%d}" % (start, start + len(methods) - 1))
        start += len(methods)
    header_rule = "".join(cmids)
    header2 = " & ".join([""] + [CFG.METHOD_LABELS.get(m, m)
                                  for _ in metrics for m in methods]) + r" \\"

    # body rows
    body_rows = []
    for tau in CFG.TAUS:
        cells = [f"{tau:.1f}"]
        for metric in metrics:
            ms = summary[(summary["metric"] == metric) & (np.isclose(summary["tau"], tau))]
            best_method, best_val = None, -np.inf
            for method in methods:
                r = ms[ms["method"] == method]
                v = r["mean"].values[0] if len(r) else np.nan
                if not np.isnan(v) and v > best_val:
                    best_val, best_method = v, method
            for method in methods:
                r = ms[ms["method"] == method]
                mean = r["mean"].values[0] if len(r) else np.nan
                lo = r["lo"].values[0] if len(r) else np.nan
                hi = r["hi"].values[0] if len(r) else np.nan
                if method == CFG.BASELINE:
                    sig = ""
                else:
                    s = mvb_stats[(mvb_stats["metric"] == metric)
                                  & (mvb_stats["method"] == method)
                                  & (np.isclose(mvb_stats["tau"], tau))]
                    sig = s["sig"].values[0] if len(s) else ""
                cells.append(_cell_mean_ci(mean, lo, hi, sig, method == best_method))
        body_rows.append(" & ".join(cells) + r" \\")

    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(rf"\caption{{{caption}}}")
    tex.append(rf"\label{{{label}}}")
    tex.append(r"\resizebox{\textwidth}{!}{%")
    tex.append(rf"\begin{{tabular}}{{{colspec}}}")
    tex.append(r"\toprule")
    tex.append(header1)
    tex.append(header_rule)
    tex.append(header2)
    tex.append(r"\midrule")
    tex.extend(body_rows)
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}}")
    tex.append(r"\end{table}")
    _write_tex(stem, "\n".join(tex))

    if CFG.ALSO_EMIT_PER_METRIC_BODY_TABLES:
        for metric in metrics:
            tex_body_combined(summary, mvb_stats, protocol, metrics=[metric],
                              stem=f"{stem}_{metric}",
                              label=f"{label}_{metric.lower()}",
                              caption=caption.replace("metric.", f"metric ({CFG.METRIC_DISPLAY[metric][0]})."))


def tex_auc_table(summary, mvb_stats, protocol):
    """Macro AUC body-style table (appendix point 2)."""
    tex_body_combined(
        summary, mvb_stats, protocol, metrics=["MacroAUC"],
        stem=f"tab_app_auc_{protocol}",
        label=f"tab:app_auc_{protocol}",
        caption=(f"Macro AUC under label noise (protocol {protocol}); supporting "
                 f"metric. Mean over {CFG.N_FOLDS} folds with 95\\% bootstrap CI "
                 f"below; stars mark method-vs-baseline significance "
                 f"(paired Wilcoxon, Holm-corrected across $\\tau$). "
                 f"Ordering matches the body metrics."))


def tex_mvb_full_stats(mvb_stats, protocol, metrics=None):
    """Full method-vs-baseline Wilcoxon table (appendix point 3):
    per (metric, method, tau): mean delta, W, raw p, Holm p, sig."""
    metrics = metrics or list(CFG.METRIC_DISPLAY)
    metrics = [m for m in metrics if m in mvb_stats["metric"].unique()]
    stem = f"tab_app_mvb_wilcoxon_{protocol}"
    rows = []
    for metric in metrics:
        disp = CFG.METRIC_DISPLAY[metric][0]
        rows.append(r"\addlinespace")
        rows.append(r"\multicolumn{7}{l}{\textit{%s}} \\" % disp)
        for method in [m for m in CFG.METHODS if m != CFG.BASELINE]:
            st = (mvb_stats[(mvb_stats["metric"] == metric) & (mvb_stats["method"] == method)]
                  .sort_values("tau"))
            first = True
            for _, r in st.iterrows():
                mlabel = CFG.METHOD_LABELS.get(method, method) if first else ""
                first = False
                rows.append(" & ".join([
                    mlabel, f"{r['tau']:.1f}", fmt_signed(r["mean_delta"]),
                    fmt_W(r["W"]), fmt_p(r["p_raw"]), fmt_p(r["p_holm"]),
                    r["sig"] if r["sig"] != CFG.NS_SYMBOL else r"n.s.",
                ]) + r" \\")
    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(rf"\caption{{Method-vs-baseline paired Wilcoxon signed-rank tests "
               rf"(protocol {protocol}), by fold at each $\tau$. "
               rf"$\Delta$ is mean(method $-$ baseline) over folds; $W$ the test "
               rf"statistic; $p_{{\mathrm{{raw}}}}$ the uncorrected $p$; "
               rf"$p_{{\mathrm{{Holm}}}}$ Holm-corrected across $\tau$ within "
               rf"each method. Alternative: {CFG.WILCOXON_ALT}.}}")
    tex.append(rf"\label{{tab:app_mvb_wilcoxon_{protocol}}}")
    tex.append(r"\begin{tabular}{llrrrrl}")
    tex.append(r"\toprule")
    tex.append(r"Method & $\tau$ & $\Delta$ & $W$ & $p_{\mathrm{raw}}$ & "
               r"$p_{\mathrm{Holm}}$ & sig. \\")
    tex.append(r"\midrule")
    tex.extend(rows)
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    _write_tex(stem, "\n".join(tex))


def tex_noise_vs_clean(nvc_stats, protocol, metrics=None):
    """Noise-sensitivity (tau vs clean) for robust methods (optional pt 4)."""
    metrics = metrics or list(CFG.BODY_METRICS)
    metrics = [m for m in metrics if m in nvc_stats["metric"].unique()]
    stem = f"tab_app_noise_vs_clean_{protocol}"
    rows = []
    for metric in metrics:
        disp = CFG.METRIC_DISPLAY[metric][0]
        rows.append(r"\addlinespace")
        rows.append(r"\multicolumn{7}{l}{\textit{%s}} \\" % disp)
        for method in [m for m in CFG.METHODS if m != CFG.BASELINE]:
            st = (nvc_stats[(nvc_stats["metric"] == metric) & (nvc_stats["method"] == method)]
                  .sort_values("tau"))
            first = True
            for _, r in st.iterrows():
                mlabel = CFG.METHOD_LABELS.get(method, method) if first else ""
                first = False
                rows.append(" & ".join([
                    mlabel, f"{r['tau']:.1f}", fmt_signed(r["mean_delta"]),
                    fmt_W(r["W"]), fmt_p(r["p_raw"]), fmt_p(r["p_holm"]),
                    r["sig"] if r["sig"] != CFG.NS_SYMBOL else r"n.s.",
                ]) + r" \\")
    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(rf"\caption{{Noise-sensitivity of the robust methods "
               rf"(protocol {protocol}): each method at $\tau>0$ vs. the same "
               rf"method at $\tau=0$, paired Wilcoxon by fold. $\Delta$ is "
               rf"mean(noisy $-$ clean); $p_{{\mathrm{{Holm}}}}$ corrected across "
               rf"$\tau$ within each method. Supports the body claim that the "
               rf"robust methods degrade more gradually than the baseline.}}")
    tex.append(rf"\label{{tab:app_noise_vs_clean_{protocol}}}")
    tex.append(r"\begin{tabular}{llrrrrl}")
    tex.append(r"\toprule")
    tex.append(r"Method & $\tau$ & $\Delta$ & $W$ & $p_{\mathrm{raw}}$ & "
               r"$p_{\mathrm{Holm}}$ & sig. \\")
    tex.append(r"\midrule")
    tex.extend(rows)
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    _write_tex(stem, "\n".join(tex))


def tex_method_vs_method(mvm_stats, protocol, metrics=None):
    """Pairwise method-vs-method table, per tau (your decision 2: appendix)."""
    metrics = metrics or list(CFG.BODY_METRICS)
    metrics = [m for m in metrics if m in mvm_stats["metric"].unique()]
    stem = f"tab_app_method_vs_method_{protocol}"
    rows = []
    for metric in metrics:
        disp = CFG.METRIC_DISPLAY[metric][0]
        rows.append(r"\addlinespace")
        rows.append(r"\multicolumn{7}{l}{\textit{%s}} \\" % disp)
        sub = mvm_stats[mvm_stats["metric"] == metric]
        for tau in CFG.TAUS:
            st = sub[np.isclose(sub["tau"], tau)]
            first = True
            for _, r in st.iterrows():
                taul = f"{tau:.1f}" if first else ""
                first = False
                pair = f"{CFG.METHOD_LABELS.get(r['method_a'], r['method_a'])} vs. " \
                       f"{CFG.METHOD_LABELS.get(r['method_b'], r['method_b'])}"
                rows.append(" & ".join([
                    taul, pair, fmt_signed(r["mean_delta"]),
                    fmt_W(r["W"]), fmt_p(r["p_raw"]), fmt_p(r["p_holm"]),
                    r["sig"] if r["sig"] != CFG.NS_SYMBOL else r"n.s.",
                ]) + r" \\")
    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(rf"\caption{{Pairwise method-vs-method paired Wilcoxon tests "
               rf"(protocol {protocol}), by fold at each $\tau$. $\Delta$ is "
               rf"mean(first $-$ second) over folds; $p_{{\mathrm{{Holm}}}}$ "
               rf"corrected across the six pairs within each $\tau$. Reported in "
               rf"the appendix only; the body marks method-vs-baseline.}}")
    tex.append(rf"\label{{tab:app_method_vs_method_{protocol}}}")
    tex.append(r"\begin{tabular}{llrrrrl}")
    tex.append(r"\toprule")
    tex.append(r"$\tau$ & Pair & $\Delta$ & $W$ & $p_{\mathrm{raw}}$ & "
               r"$p_{\mathrm{Holm}}$ & sig. \\")
    tex.append(r"\midrule")
    tex.extend(rows)
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    _write_tex(stem, "\n".join(tex))


# ============================================================================
# prose helpers (feed the body commentary)
# ============================================================================
def print_prose_helpers(summary, mvb_stats, nvc_stats, protocol):
    print("\n" + "=" * 74)
    print(f"PROSE HELPER  -  facts for the Results.3 commentary (protocol {protocol})")
    print("=" * 74)
    pcol = "p_holm" if CFG.SIG_USES_CORRECTED else "p_raw"

    for metric in CFG.BODY_METRICS:
        disp = CFG.METRIC_DISPLAY[metric][0]
        print(f"\n[{disp}]")
        for method in [m for m in CFG.METHODS if m != CFG.BASELINE]:
            st = (mvb_stats[(mvb_stats["metric"] == metric) & (mvb_stats["method"] == method)]
                  .sort_values("tau"))
            sig_taus = st[st[pcol] < CFG.HOLM_ALPHA]["tau"].tolist()
            first_sig = f"{min(sig_taus):.1f}" if sig_taus else "never"
            # does the advantage widen with tau? slope of mean_delta on tau
            d = st.dropna(subset=["mean_delta"])
            if len(d) >= 2:
                slope = np.polyfit(d["tau"].values, d["mean_delta"].values, 1)[0]
                trend = "widens" if slope > 1e-4 else ("narrows" if slope < -1e-4 else "flat")
            else:
                trend = "n/a"
            d_lo = st[np.isclose(st["tau"], CFG.TAUS[1])]["mean_delta"]
            d_hi = st[np.isclose(st["tau"], CFG.TAUS[-1])]["mean_delta"]
            d_lo = d_lo.values[0] if len(d_lo) else np.nan
            d_hi = d_hi.values[0] if len(d_hi) else np.nan
            print(f"   {method:9s}: first significant at tau={first_sig}; "
                  f"gap {trend} (delta {fmt_signed(d_lo)} -> {fmt_signed(d_hi)} "
                  f"from tau={CFG.TAUS[1]:.1f} to {CFG.TAUS[-1]:.1f}).")

    # BA-vs-MacroF1 divergence at the top tau
    top = CFG.TAUS[-1]
    print(f"\n[BA vs Macro F1 divergence at tau={top:.1f}]")
    for metric in ("BA", "MacroF1"):
        ms = summary[(summary["metric"] == metric) & (np.isclose(summary["tau"], top))]
        if len(ms):
            best = ms.loc[ms["mean"].idxmax()]
            print(f"   best on {CFG.METRIC_DISPLAY[metric][0]:16s}: "
                  f"{CFG.METHOD_LABELS.get(best['method'], best['method'])} "
                  f"({fmt_metric(best['mean'])})")
    print("   -> if these differ, a method buys aggregate accuracy without "
          "fixing the minority classes; flag it (clinically relevant).")

    # noise-sensitivity one-liner
    print("\n[Noise-sensitivity summary (robust methods, tau vs clean)]")
    for metric in CFG.BODY_METRICS:
        for method in [m for m in CFG.METHODS if m != CFG.BASELINE]:
            st = nvc_stats[(nvc_stats["metric"] == metric) & (nvc_stats["method"] == method)]
            sig_taus = st[st[pcol] < CFG.HOLM_ALPHA]["tau"].tolist()
            first = f"{min(sig_taus):.1f}" if sig_taus else "never"
            print(f"   {CFG.METRIC_DISPLAY[metric][0]:12s} {method:9s}: "
                  f"degradation becomes significant at tau={first}")
    print("=" * 74 + "\n")


# ============================================================================
# main
# ============================================================================
def main():
    P = CFG.PRIMARY_PROTOCOL
    print(f"Loading data (SOURCE={CFG.SOURCE}) ...")
    df = load_long_df()
    print(f"[load] {len(df)} rows; protocols={sorted(df['protocol'].unique())}; "
          f"methods={sorted(df['method'].unique())}; "
          f"taus={sorted(df['tau'].unique())}")

    if P not in df["protocol"].unique():
        print(f"\n!! primary protocol '{P}' not found in the data. "
              f"Available: {sorted(df['protocol'].unique())}. "
              f"Set CONFIG.PRIMARY_PROTOCOL.\n")
        sys.exit(1)

    completeness_report(df, P)

    print("Computing summaries and statistics ...")
    summary = summarize(df, P)
    mvb = method_vs_baseline(df, P)
    nvc = noise_vs_clean(df, P)
    mvm = method_vs_method(df, P)

    print("Building figures ...")
    fig_money(summary, mvb, P)            # body money figure
    fig_auc(summary, mvb, P)              # optional appendix AUC figure

    print("Building LaTeX tables ...")
    tex_body_combined(summary, mvb, P)    # body table
    tex_auc_table(summary, mvb, P)        # appendix: AUC (pt 2)
    tex_mvb_full_stats(mvb, P)            # appendix: full Wilcoxon (pt 3)
    tex_noise_vs_clean(nvc, P)            # appendix: noise-sensitivity (pt 4)
    tex_method_vs_method(mvm, P)          # appendix: pairwise (decision 2)

    print_prose_helpers(summary, mvb, nvc, P)

    # tidy stat frames are handy to keep for the appendix-from-CSV workflow
    CFG.TAB_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(CFG.TAB_DIR / f"_summary_{P}.csv", index=False)
    mvb.to_csv(CFG.TAB_DIR / f"_mvb_{P}.csv", index=False)
    nvc.to_csv(CFG.TAB_DIR / f"_nvc_{P}.csv", index=False)
    mvm.to_csv(CFG.TAB_DIR / f"_mvm_{P}.csv", index=False)
    print(f"[csv] wrote tidy stat frames to {CFG.TAB_DIR}/_*.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()