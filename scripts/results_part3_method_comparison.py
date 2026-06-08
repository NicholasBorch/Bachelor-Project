"""
Results Part 3 - method comparison under label noise (RQ2) and appendix tables.

Reads results/main_experiment/{protocol}/training/{method}/tau_NN/fold_NN/
test_metrics.json and writes into results/method_comparison/: grouped-bar
figures (combined and per-metric), a body table (mean + bootstrap CI with
signed significance), and appendix stats tables (method-vs-baseline,
noise-vs-clean, method-vs-method).
"""

from __future__ import annotations

import json
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

import scripts.thesis_paired_stats as TPS


# config
@dataclass
class Config:
    EXPERIMENT_ROOT: Path = Path("./results/main_experiment")
    TRAINING_SUBDIR: str = "training"          
    METRICS_FILENAME: str = "test_metrics.json"
    TAU_DIR_FMT: str = "tau_{tt:02d}"          
    FOLD_DIR_FMT: str = "fold_{ff:02d}"

    PROTOCOL_DIRS: dict = field(default_factory=lambda: {
        "AP": "pretrained_adam",
        # "A":  "scratch_adam",
        # "SP": "pretrained_sgd",
        # "S":  "scratch_sgd",
    })

    # logical method -> folder name on disk
    METHOD_DIRS: dict = field(default_factory=lambda: {
        "baseline": "baseline",
        "SCE":      "sce",
        "ELR":      "elr",
        "AsyCo":    "asyco_divmix",
    })

    METRIC_KEYS: dict = field(default_factory=lambda: {
        "BA":       ["balanced_accuracy", "bacc", "balanced_acc", "BA", "bal_acc"],
        "MacroF1":  ["macro_f1", "f1_macro", "macro_F1", "f1macro", "f1_macro_avg"],
        "MacroAUC": ["macro_auc", "auc_macro", "macro_AUC", "roc_auc_macro", "auroc_macro"],
    })
    # nesting prefixes to try ("" = top level)
    METRIC_NEST_KEYS: tuple = ("", "test", "metrics", "test_metrics")

    METHODS: tuple = ("baseline", "SCE", "ELR", "AsyCo")
    BASELINE: str = "baseline"
    METHOD_LABELS: dict = field(default_factory=lambda: {
        "baseline": "Baseline", "SCE": "SCE", "ELR": "ELR", "AsyCo": "AsyCo",
    })
    PRIMARY_PROTOCOL: str = "AP"
    TAUS: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    N_FOLDS: int = 10

    BODY_METRICS: tuple = ("BA", "MacroF1")
    # metrics shown side-by-side in the combined figure, per protocol
    COMBINED_FIG_METRICS: dict = field(default_factory=lambda: {
        "AP": ("BA", "MacroF1"),
    })
    COMBINED_FIG_METRICS_DEFAULT: tuple = ("BA", "MacroF1", "MacroAUC")
    METRIC_DISPLAY: dict = field(default_factory=lambda: {
        "BA":       ("Balanced accuracy", "Balanced accuracy", 0.0, 1.0),
        "MacroF1":  ("Macro F1",          "Macro F1",          0.0, 1.0),
        "MacroAUC": ("Macro AUC",         "Macro AUC",         0.5, 1.0),
    })

    N_BOOT: int = 10000
    CI: float = 0.95
    SEED: int = 10
    WILCOXON_ALT: str = "two-sided"
    HOLM_ALPHA: float = 0.05
    SIG_LEVELS: tuple = ((0.001, "***"), (0.01, "**"), (0.05, "*"))
    NS_SYMBOL: str = "n.s."
    SHOW_NS_IN_FIG: bool = False
    SIG_USES_CORRECTED: bool = True

    PALETTE: dict = field(default_factory=lambda: {
        "baseline": "#9ec9e2", "SCE": "#2a9d8f", "ELR": "#e07a3f", "AsyCo": "#7b5cb8",
    })

    RESULTS_ROOT: Path = Path("./results")
    ANALYSIS_DIR: str = "method_comparison"
    FIG_DPI: int = 300
    SAVE_PNG: bool = True
    SAVE_PDF: bool = True
    ALSO_EMIT_PER_METRIC_BODY_TABLES: bool = False


CFG = Config()


def _out_dir() -> Path:
    d = CFG.RESULTS_ROOT / CFG.ANALYSIS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# small utilities
def _seed_for(*parts) -> int:
    h = hashlib.sha256(("|".join(map(str, parts))).encode()).hexdigest()
    return (CFG.SEED + int(h[:8], 16)) % (2**32 - 1)


def sig_symbol(p, ns=True):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    for thr, sym in CFG.SIG_LEVELS:
        if p < thr:
            return sym
    return CFG.NS_SYMBOL if ns else ""


def fmt_metric(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fmt_signed(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if abs(x) < 0.5 * 10 ** (-nd):
        return f"{0.0:+.{nd}f}"
    return f"{x:+.{nd}f}"


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    return r"$<0.001$" if p < 0.001 else f"{p:.3f}"


def fmt_W(w):
    if w is None or (isinstance(w, float) and np.isnan(w)):
        return "--"
    return f"{w:.0f}" if abs(w - round(w)) < 1e-9 else f"{w:.1f}"


# data loading
def _read_json(fp: Path) -> dict:
    with open(fp, "r") as fh:
        return json.load(fh)


def _flatten(d: dict, prefix: str) -> dict:
    if prefix and isinstance(d.get(prefix), dict):
        return d[prefix]
    return d


def _extract_metric(d: dict, aliases) -> Optional[float]:
    for nest in CFG.METRIC_NEST_KEYS:
        scope = _flatten(d, nest)
        if not isinstance(scope, dict):
            continue
        for k in aliases:
            if k in scope and scope[k] is not None:
                try:
                    return float(scope[k])
                except (TypeError, ValueError):
                    pass
    return None


def _all_keys(d: dict):
    keys = list(d.keys())
    for nest in CFG.METRIC_NEST_KEYS:
        if nest and isinstance(d.get(nest), dict):
            keys += [f"{nest}.{k}" for k in d[nest].keys()]
    return sorted(set(keys))


def _protocol_root(protocol: str) -> Path:
    base = CFG.EXPERIMENT_ROOT / CFG.PROTOCOL_DIRS[protocol]
    return base / CFG.TRAINING_SUBDIR if CFG.TRAINING_SUBDIR else base


def _scan_method_dirs(protocol: str):
    root = _protocol_root(protocol)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def load_long_df() -> pd.DataFrame:
    rows, sample, sample_path, missing = [], None, None, set()
    inv_method = {v: k for k, v in CFG.METHOD_DIRS.items()}

    for protocol in CFG.PROTOCOL_DIRS:
        proot = _protocol_root(protocol)
        if not proot.exists():
            print(f"[warn] protocol '{protocol}' dir not found: {proot} (skipped).")
            continue
        present = _scan_method_dirs(protocol)
        print(f"[scan] protocol {protocol} ({proot}):")
        print(f"[scan]   method dirs on disk = {present}")
        unmapped = [m for m in present if m not in inv_method and m != "figures_and_tables"]
        if unmapped:
            print(f"[scan]   NOT in CONFIG.METHOD_DIRS, ignored: {unmapped}")

        for method, mdir in CFG.METHOD_DIRS.items():
            mroot = proot / mdir
            if not mroot.exists():
                print(f"[warn]   method '{method}' dir not found: {mroot} (skipped).")
                continue
            for tau in CFG.TAUS:
                tt = int(round(tau * 100))
                tdir = mroot / CFG.TAU_DIR_FMT.format(tt=tt)
                if not tdir.exists():
                    continue
                for fold in range(CFG.N_FOLDS):
                    fp = tdir / CFG.FOLD_DIR_FMT.format(ff=fold) / CFG.METRICS_FILENAME
                    if not fp.exists():
                        continue
                    d = _read_json(fp)
                    if sample is None:
                        sample, sample_path = d, fp
                    rec = {"protocol": protocol, "method": method,
                           "tau": float(tau), "fold": int(fold)}
                    for logical, aliases in CFG.METRIC_KEYS.items():
                        val = _extract_metric(d, aliases)
                        if val is None:
                            missing.add(logical)
                        rec[logical] = val
                    rows.append(rec)

    if not rows:
        raise FileNotFoundError(
            f"No '{CFG.METRICS_FILENAME}' found under {CFG.EXPERIMENT_ROOT} for the "
            f"configured protocols/methods. Check EXPERIMENT_ROOT, TRAINING_SUBDIR, "
            f"PROTOCOL_DIRS, METHOD_DIRS and the tau_/fold_ naming. The [scan] lines "
            f"above list the method dirs that exist on disk.")

    if sample is not None:
        print(f"\n[schema] first metrics file: {sample_path}")
        print(f"[schema] keys present: {_all_keys(sample)}")
        resolved = {lg: _extract_metric(sample, al) for lg, al in CFG.METRIC_KEYS.items()}
        print("[schema] resolved -> " + ", ".join(
            f"{lg}={'OK' if v is not None else 'MISSING'}" for lg, v in resolved.items()))

    df = pd.DataFrame(rows)
    for logical in ("BA", "MacroF1"):
        if logical in df.columns and df[logical].isna().all():
            print(f"\n[schema] '{logical}' not found in ANY file. Map its JSON key in "
                  f"CONFIG.METRIC_KEYS (see keys above) and re-run.\n")
            raise SystemExit(1)

    df["tau"] = df["tau"].astype(float)
    df = df.drop_duplicates(subset=["protocol", "method", "tau", "fold"])
    df = df.sort_values(["protocol", "method", "tau", "fold"]).reset_index(drop=True)
    return df


def completeness_report(df, protocol):
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
                miss = sorted(set(range(CFG.N_FOLDS)) - set(folds))
                print(f"   ! {method:9s} tau={tau:.1f}: {len(folds)}/{CFG.N_FOLDS} "
                      f"folds (missing {miss})")
    if not any_missing:
        print("   OK - every method x tau has the full fold set.")
    print()


# statistics
def bootstrap_ci(values, n_boot=None, ci=None, seed=0):
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
    return (float(v.mean()),
            float(np.percentile(boot, (1 - ci) / 2 * 100)),
            float(np.percentile(boot, (1 + ci) / 2 * 100)))


def _wilcoxon_compat(a, b, alternative):
    try:
        res = stats.wilcoxon(a, b, alternative=alternative,
                             zero_method="wilcox", correction=False)
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return 0.0, 1.0


def wilcoxon_paired(a, b, alternative=None):
    alternative = CFG.WILCOXON_ALT if alternative is None else alternative
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if a.size == 0:
        return (np.nan, np.nan, 0, np.nan)
    d = a - b
    md = float(np.mean(d))
    if np.allclose(d, 0.0):
        return (0.0, 1.0, a.size, md)
    W, p = _wilcoxon_compat(a, b, alternative)
    return (W, p, a.size, md)


def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if idx.size == 0:
        return out
    pv = p[idx]; order = np.argsort(pv); m = pv.size; running = 0.0
    adj = np.empty(m)
    for rank, oi in enumerate(order):
        running = max(running, (m - rank) * pv[oi])
        adj[oi] = min(running, 1.0)
    out[idx] = adj
    return out


def _wide_on_fold(df, protocol, metric, tau):
    sub = df[(df["protocol"] == protocol) & (np.isclose(df["tau"], tau))]
    return sub.pivot_table(index="fold", columns="method", values=metric)


def summarize(df, protocol):
    recs = []
    for metric in CFG.METRIC_DISPLAY:
        if metric not in df.columns:
            continue
        for method in CFG.METHODS:
            for tau in CFG.TAUS:
                cell = df[(df["protocol"] == protocol) & (df["method"] == method)
                          & (np.isclose(df["tau"], tau))][metric].values
                mean, lo, hi = bootstrap_ci(cell, seed=_seed_for(protocol, metric, method, tau))
                recs.append(dict(metric=metric, method=method, tau=tau, mean=mean,
                                 lo=lo, hi=hi, n=int(np.sum(~np.isnan(cell)))))
    return pd.DataFrame(recs)


def method_vs_baseline(df, protocol):
    recs = []
    for metric in CFG.METRIC_DISPLAY:
        if metric not in df.columns:
            continue
        for method in [m for m in CFG.METHODS if m != CFG.BASELINE]:
            block = []
            for tau in CFG.TAUS:
                w = _wide_on_fold(df, protocol, metric, tau)
                if CFG.BASELINE in w.columns and method in w.columns:
                    a = w[method].values; b = w[CFG.BASELINE].values
                    m = ~(np.isnan(a) | np.isnan(b)); d = a[m] - b[m]
                    res = TPS.paired_compare(d, n_boot=CFG.N_BOOT,
                                             boot_seed=_seed_for(protocol, metric, method, tau))
                    rec = dict(metric=metric, method=method, tau=tau, **res.as_dict())
                    rec["mean_delta"] = rec["delta"]
                    rec["p_raw"] = rec["p_wilcoxon"]
                else:
                    rec = dict(metric=metric, method=method, tau=tau,
                               W=np.nan, p_wilcoxon=np.nan, p_perm=np.nan,
                               p_raw=np.nan, mean_delta=np.nan, delta=np.nan,
                               delta_ci_lo=np.nan, delta_ci_hi=np.nan,
                               r_rb=np.nan, direction=0, n=0)
                block.append(rec)
            TPS.add_holm_and_flags(block)
            for b in block:
                b["p_holm"] = b["p_wilcoxon_holm"]
            recs.extend(block)
    return pd.DataFrame(recs)


def noise_vs_clean(df, protocol, methods=None):
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
                    d = paired["noisy"].values - paired["clean"].values
                    res = TPS.paired_compare(d, n_boot=CFG.N_BOOT,
                                             boot_seed=_seed_for(protocol, metric, method, tau, "nvc"))
                    rec = dict(metric=metric, method=method, tau=tau, **res.as_dict())
                    rec["mean_delta"] = rec["delta"]; rec["p_raw"] = rec["p_wilcoxon"]
                else:
                    rec = dict(metric=metric, method=method, tau=tau,
                               W=np.nan, p_wilcoxon=np.nan, p_perm=np.nan,
                               p_raw=np.nan, mean_delta=np.nan, delta=np.nan,
                               delta_ci_lo=np.nan, delta_ci_hi=np.nan,
                               r_rb=np.nan, direction=0, n=0)
                block.append(rec)
            TPS.add_holm_and_flags(block)
            for b in block:
                b["p_holm"] = b["p_wilcoxon_holm"]
            recs.extend(block)
    return pd.DataFrame(recs)


def method_vs_method(df, protocol):
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
                    va = w[a].values; vb = w[b].values
                    m = ~(np.isnan(va) | np.isnan(vb)); d = va[m] - vb[m]
                    res = TPS.paired_compare(d, n_boot=CFG.N_BOOT,
                                             boot_seed=_seed_for(protocol, metric, tau, a, b))
                    rec = dict(metric=metric, tau=tau, method_a=a, method_b=b, **res.as_dict())
                    rec["mean_delta"] = rec["delta"]; rec["p_raw"] = rec["p_wilcoxon"]
                else:
                    rec = dict(metric=metric, tau=tau, method_a=a, method_b=b,
                               W=np.nan, p_wilcoxon=np.nan, p_perm=np.nan,
                               p_raw=np.nan, mean_delta=np.nan, delta=np.nan,
                               delta_ci_lo=np.nan, delta_ci_hi=np.nan,
                               r_rb=np.nan, direction=0, n=0)
                block.append(rec)
            TPS.add_holm_and_flags(block)
            for x in block:
                x["p_holm"] = x["p_wilcoxon_holm"]
            recs.extend(block)
    return pd.DataFrame(recs)


# plotting
def _apply_style():
    plt.rcParams.update({
        # serif (Palatino) matplotlib style, no LaTeX
        "font.family":        "serif",
        "font.serif":         ["Palatino", "Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
        "mathtext.fontset":   "cm",      # serif math
        "axes.unicode_minus": False,
        "figure.dpi": 150, "savefig.dpi": CFG.FIG_DPI, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#cccccc", "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def _yerr(rows):
    means = rows["mean"].values; lo = rows["lo"].values; hi = rows["hi"].values
    return np.vstack([np.clip(means - lo, 0, None), np.clip(hi - means, 0, None)])


def _grouped_bar_panel(ax, summary, mvb_stats, metric, protocol, show_sig=True,
                       ylim=None):
    taus = list(CFG.TAUS); methods = list(CFG.METHODS); n_m = len(methods)
    x = np.arange(len(taus)); width = 0.8 / n_m
    metric_sum = summary[summary["metric"] == metric]
    top_of_group = np.zeros(len(taus))
    for j, method in enumerate(methods):
        rows = (metric_sum[metric_sum["method"] == method]
                .set_index("tau").reindex(taus).reset_index())
        offs = (j - (n_m - 1) / 2) * width
        ax.bar(x + offs, rows["mean"].values, width=width, yerr=_yerr(rows),
               color=CFG.PALETTE.get(method, None), edgecolor="white", linewidth=0.6,
               capsize=2.5, error_kw=dict(elinewidth=0.9, alpha=0.85),
               label=CFG.METHOD_LABELS.get(method, method), zorder=3)
        top_of_group = np.maximum(top_of_group, np.nan_to_num(rows["hi"].values, nan=0.0))
        if show_sig and method != CFG.BASELINE and mvb_stats is not None:
            st = mvb_stats[(mvb_stats["metric"] == metric)
                           & (mvb_stats["method"] == method)].set_index("tau")
            pcol = "p_holm" if CFG.SIG_USES_CORRECTED else "p_raw"
            for xi, tau in zip(x, taus):
                if tau not in st.index:
                    continue
                sym = sig_symbol(st.loc[tau, pcol], ns=CFG.SHOW_NS_IN_FIG)
                if not sym:
                    continue
                # prepend +/- for better/worse than baseline
                if sym != CFG.NS_SYMBOL:
                    direction = st.loc[tau, "direction"] if "direction" in st.columns else np.sign(st.loc[tau, "mean_delta"])
                    if direction > 0:
                        sym = "+" + sym
                    elif direction < 0:
                        sym = "-" + sym
                bt = np.nan_to_num(rows.loc[rows["tau"] == tau, "hi"].values, nan=0.0)
                bt = bt[0] if len(bt) else 0.0
                color, fs = ("0.45", 7) if sym == CFG.NS_SYMBOL else ("0.15", 8)
                ax.text(xi + offs, bt + 0.012, sym, color=color, fontsize=fs,
                        ha="center", va="bottom", zorder=5)
    disp, ylab, ymin, ymax = CFG.METRIC_DISPLAY[metric]
    ax.set_xticks(x); ax.set_xticklabels([f"{t:.1f}" for t in taus])
    ax.set_xlabel(r"Noise rate $\tau$"); ax.set_ylabel(ylab); ax.set_title(disp)
    # shared y-range keeps panels comparable; else per-metric range
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        headroom = 0.06 * (ymax - ymin)
        if np.isfinite(np.nanmax(top_of_group)):
            ax.set_ylim(ymin, min(ymax, float(np.nanmax(top_of_group)) + headroom + 0.04))
        else:
            ax.set_ylim(ymin, ymax)


def _savefig(fig, stem):
    out = _out_dir()
    if CFG.SAVE_PDF:
        fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    if CFG.SAVE_PNG:
        fig.savefig(out / f"{stem}.png", bbox_inches="tight")
    print(f"[fig] wrote {out / stem}.(pdf|png)")


def fig_money(summary, mvb_stats, protocol, metrics=None, fname=None):
    # which metrics sit side-by-side depends on the protocol
    if metrics is None:
        metrics = list(CFG.COMBINED_FIG_METRICS.get(
            protocol, CFG.COMBINED_FIG_METRICS_DEFAULT))
    _apply_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4.6),
                             sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    # shared 0..1 y-axis so all panels are comparable
    shared_ylim = (0.0, 1.0)
    for ax, metric in zip(axes, metrics):
        _grouped_bar_panel(ax, summary, mvb_stats, metric, protocol,
                           ylim=shared_ylim)
    # leftmost panel keeps a generic label; clear the rest
    axes[0].set_ylabel("Score")
    for ax in axes[1:]:
        ax.set_ylabel("")
    handles = [Patch(facecolor=CFG.PALETTE.get(m), edgecolor="white",
                     label=CFG.METHOD_LABELS.get(m, m)) for m in CFG.METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=len(CFG.METHODS),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Method comparison under label noise - protocol {protocol}",
                 y=1.0, fontsize=12.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    _savefig(fig, fname or f"fig_results3_money_{protocol}")
    plt.close(fig)


def fig_single_metric(summary, mvb_stats, protocol, metric, fname=None):
    """One standalone grouped-bar plot for a single metric."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    _grouped_bar_panel(ax, summary, mvb_stats, metric, protocol, ylim=(0.0, 1.0))
    ax.set_ylabel("Score")
    handles = [Patch(facecolor=CFG.PALETTE.get(m), edgecolor="white",
                     label=CFG.METHOD_LABELS.get(m, m)) for m in CFG.METHODS]
    ax.legend(handles=handles, loc="lower left", ncol=2, frameon=False)
    disp = CFG.METRIC_DISPLAY[metric][0]
    ax.set_title(f"{disp} under label noise - protocol {protocol}")
    fig.tight_layout()
    _savefig(fig, fname or f"fig_results3_{metric}_{protocol}")
    plt.close(fig)


def fig_all_single_metrics(summary, mvb_stats, protocol):
    """Individual plots for all three metrics, for the given protocol."""
    for metric in CFG.METRIC_DISPLAY:
        if metric in summary["metric"].unique():
            fig_single_metric(summary, mvb_stats, protocol, metric)


# LaTeX tables
REQUIRED_PACKAGES = r"""% Preamble: \usepackage{booktabs,makecell,multirow,graphicx,longtable}
"""


def _write_tex(stem, body):
    fp = _out_dir() / f"{stem}.tex"
    with open(fp, "w") as fh:
        fh.write(REQUIRED_PACKAGES + "\n" + body + "\n")
    print(f"[tab] wrote {fp}")


def _cell_mean_ci(mean, lo, hi, sig, is_best):
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return r"\makecell{--}"
    m = fmt_metric(mean)
    sup = f"^{{{sig}}}" if sig and sig != CFG.NS_SYMBOL else ""
    inner = rf"\mathbf{{{m}}}" if is_best else m
    return (rf"\makecell{{${inner}{sup}$\\"
            rf"{{\scriptsize $({fmt_metric(lo)},\,{fmt_metric(hi)})$}}}}")


def tex_body_combined(summary, mvb_stats, protocol,
                      metrics=None, stem=None, caption=None, label=None):
    metrics = metrics or ["BA", "MacroF1", "MacroAUC"]
    methods = list(CFG.METHODS)
    stem = stem or f"tab_results3_body_{protocol}"
    label = label or f"tab:results3_body_{protocol}"
    caption = caption or (
        f"Method comparison under label noise (protocol {protocol}). Cells give "
        f"the mean over {CFG.N_FOLDS} folds with the 95\\% bootstrap confidence "
        f"interval below. Stars mark a significant method-vs-baseline difference "
        f"(paired Wilcoxon by fold, Holm-corrected across $\\tau$: $^{{*}}p<.05$, "
        f"$^{{**}}p<.01$, $^{{***}}p<.001$; no star = n.s.), with a leading sign "
        f"giving the direction ($+$ better, $-$ worse than baseline). The best "
        f"method per $\\tau$ and metric is in bold.")
    colspec = "l" + "".join(["*{%d}{c}" % len(methods) for _ in metrics])
    top = [r"\multirow{2}{*}{$\tau$}"]
    for metric in metrics:
        top.append(r"\multicolumn{%d}{c}{%s}" % (len(methods), CFG.METRIC_DISPLAY[metric][0]))
    header1 = " & ".join(top) + r" \\"
    cmids, start = [], 2
    for _ in metrics:
        cmids.append(r"\cmidrule(lr){%d-%d}" % (start, start + len(methods) - 1))
        start += len(methods)
    header2 = " & ".join([""] + [CFG.METHOD_LABELS.get(m, m)
                                  for _ in metrics for m in methods]) + r" \\"
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
    tex = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}",
           rf"\label{{{label}}}", r"\resizebox{\textwidth}{!}{%",
           rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header1,
           "".join(cmids), header2, r"\midrule", *body_rows, r"\bottomrule",
           r"\end{tabular}}", r"\end{table}"]
    _write_tex(stem, "\n".join(tex))
    if CFG.ALSO_EMIT_PER_METRIC_BODY_TABLES:
        for metric in metrics:
            tex_body_combined(summary, mvb_stats, protocol, metrics=[metric],
                              stem=f"{stem}_{metric}", label=f"{label}_{metric.lower()}")


def tex_auc_table(summary, mvb_stats, protocol):
    tex_body_combined(
        summary, mvb_stats, protocol, metrics=["MacroAUC"],
        stem=f"tab_app_auc_{protocol}", label=f"tab:app_auc_{protocol}",
        caption=(f"Macro AUC under label noise (protocol {protocol}); supporting "
                 f"metric. Mean over {CFG.N_FOLDS} folds with 95\\% bootstrap CI "
                 f"below; stars mark method-vs-baseline significance (paired "
                 f"Wilcoxon, Holm-corrected across $\\tau$)."))


def _fmt_ci(lo, hi):
    if lo is None or (isinstance(lo, float) and np.isnan(lo)):
        return "--"
    return f"[{lo:+.3f},\\,{hi:+.3f}]"


def _stats_table(stats_df, protocol, stem, caption, label, kind):
    metrics = [m for m in CFG.METRIC_DISPLAY if m in stats_df["metric"].unique()]
    rows = []
    ncol = 9
    for metric in metrics:
        rows.append(r"\addlinespace")
        rows.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (ncol, CFG.METRIC_DISPLAY[metric][0]))
        sub = stats_df[stats_df["metric"] == metric]
        if kind == "mvm":
            for tau in CFG.TAUS:
                st = sub[np.isclose(sub["tau"], tau)]
                first = True
                for _, r in st.iterrows():
                    taul = f"{tau:.1f}" if first else ""; first = False
                    pair = (f"{CFG.METHOD_LABELS.get(r['method_a'], r['method_a'])} vs. "
                            f"{CFG.METHOD_LABELS.get(r['method_b'], r['method_b'])}")
                    rows.append(" & ".join([taul, pair, fmt_signed(r["mean_delta"]),
                        f"${_fmt_ci(r['delta_ci_lo'], r['delta_ci_hi'])}$",
                        f"{r['r_rb']:+.2f}" if not np.isnan(r['r_rb']) else "--",
                        fmt_W(r["W"]), fmt_p(r["p_raw"]), fmt_p(r["p_holm"]),
                        (r["sig"] if r["sig"] != TPS.NS_SYMBOL else "n.s.") + ((r"\,!") if r.get("flag") else "")]) + r" \\")
        else:
            for method in [m for m in CFG.METHODS if m != CFG.BASELINE]:
                st = sub[sub["method"] == method].sort_values("tau")
                first = True
                for _, r in st.iterrows():
                    ml = CFG.METHOD_LABELS.get(method, method) if first else ""; first = False
                    rows.append(" & ".join([ml, f"{r['tau']:.1f}", fmt_signed(r["mean_delta"]),
                        f"${_fmt_ci(r['delta_ci_lo'], r['delta_ci_hi'])}$",
                        f"{r['r_rb']:+.2f}" if not np.isnan(r['r_rb']) else "--",
                        fmt_W(r["W"]), fmt_p(r["p_raw"]), fmt_p(r["p_holm"]),
                        (r["sig"] if r["sig"] != TPS.NS_SYMBOL else "n.s.") + ((r"\,!") if r.get("flag") else "")]) + r" \\")
    if kind == "mvm":
        head = (r"$\tau$ & Pair & $\Delta$ & 95\% CI & $r$ & $W$ & "
                r"$p_{\mathrm{raw}}$ & $p_{\mathrm{Holm}}$ & sig. \\")
    else:
        head = (r"Method & $\tau$ & $\Delta$ & 95\% CI & $r$ & $W$ & "
                r"$p_{\mathrm{raw}}$ & $p_{\mathrm{Holm}}$ & sig. \\")
    colspec = "llrrrrrrl"
    ncols = len(colspec)
    # longtable so tall appendix tables break across pages
    tex = [
        r"\begin{small}",
        r"\setlength{\LTcapwidth}{\textwidth}",
        rf"\begin{{longtable}}{{{colspec}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule", head, r"\midrule", r"\endfirsthead",
        r"\multicolumn{%d}{l}{\itshape\small continued from previous page}\\" % ncols,
        r"\toprule", head, r"\midrule", r"\endhead",
        r"\midrule",
        r"\multicolumn{%d}{r}{\itshape\small continued on next page}\\" % ncols,
        r"\endfoot",
        r"\bottomrule", r"\endlastfoot",
        *rows,
        r"\end{longtable}",
        r"\end{small}",
    ]
    _write_tex(stem, "\n".join(tex))

def tex_mvb_full_stats(mvb, protocol):
    _stats_table(mvb, protocol, f"tab_app_mvb_wilcoxon_{protocol}",
                 (f"Method-vs-baseline paired Wilcoxon signed-rank tests (protocol "
                  f"{protocol}), by fold at each $\\tau$. $\\Delta$ is mean(method $-$ "
                  f"baseline); $W$ the statistic; $p_{{\\mathrm{{Holm}}}}$ Holm-"
                  f"corrected across $\\tau$ within each method. Alternative: "
                  f"{CFG.WILCOXON_ALT}."),
                 f"tab:app_mvb_wilcoxon_{protocol}", kind="mvb")


def tex_noise_vs_clean(nvc, protocol):
    _stats_table(nvc, protocol, f"tab_app_noise_vs_clean_{protocol}",
                 (f"Noise-sensitivity of the robust methods (protocol {protocol}): "
                  f"each method at $\\tau>0$ vs. the same method at $\\tau=0$, paired "
                  f"Wilcoxon by fold. $\\Delta$ is mean(noisy $-$ clean); "
                  f"$p_{{\\mathrm{{Holm}}}}$ corrected across $\\tau$ within method."),
                 f"tab:app_noise_vs_clean_{protocol}", kind="nvc")


def tex_method_vs_method(mvm, protocol):
    _stats_table(mvm, protocol, f"tab_app_method_vs_method_{protocol}",
                 (f"Pairwise method-vs-method paired Wilcoxon tests (protocol "
                  f"{protocol}), by fold at each $\\tau$. $\\Delta$ is mean(first $-$ "
                  f"second); $p_{{\\mathrm{{Holm}}}}$ corrected across the six pairs "
                  f"within each $\\tau$. Appendix only; the body marks "
                  f"method-vs-baseline."),
                 f"tab:app_method_vs_method_{protocol}", kind="mvm")


# prose helpers
def print_prose_helpers(summary, mvb_stats, nvc_stats, protocol):
    print("\n" + "=" * 74)
    print(f"PROSE HELPER  -  facts for the Results.3 commentary (protocol {protocol})")
    print("=" * 74)
    pcol = "p_holm" if CFG.SIG_USES_CORRECTED else "p_raw"
    for metric in CFG.BODY_METRICS:
        print(f"\n[{CFG.METRIC_DISPLAY[metric][0]}]")
        for method in [m for m in CFG.METHODS if m != CFG.BASELINE]:
            st = (mvb_stats[(mvb_stats["metric"] == metric) & (mvb_stats["method"] == method)]
                  .sort_values("tau"))
            sig_taus = st[st[pcol] < CFG.HOLM_ALPHA]["tau"].tolist()
            first_sig = f"{min(sig_taus):.1f}" if sig_taus else "never"
            d = st.dropna(subset=["mean_delta"])
            if len(d) >= 2:
                slope = np.polyfit(d["tau"].values, d["mean_delta"].values, 1)[0]
                trend = "widens" if slope > 1e-4 else ("narrows" if slope < -1e-4 else "flat")
            else:
                trend = "n/a"
            dl = st[np.isclose(st["tau"], CFG.TAUS[1])]["mean_delta"]
            dh = st[np.isclose(st["tau"], CFG.TAUS[-1])]["mean_delta"]
            dl = dl.values[0] if len(dl) else np.nan
            dh = dh.values[0] if len(dh) else np.nan
            print(f"   {method:9s}: first significant at tau={first_sig}; gap {trend} "
                  f"(delta {fmt_signed(dl)} -> {fmt_signed(dh)} "
                  f"from tau={CFG.TAUS[1]:.1f} to {CFG.TAUS[-1]:.1f}).")
    top = CFG.TAUS[-1]
    print(f"\n[BA vs Macro F1 divergence at tau={top:.1f}]")
    for metric in ("BA", "MacroF1"):
        ms = summary[(summary["metric"] == metric) & (np.isclose(summary["tau"], top))]
        if len(ms):
            best = ms.loc[ms["mean"].idxmax()]
            print(f"   best on {CFG.METRIC_DISPLAY[metric][0]:16s}: "
                  f"{CFG.METHOD_LABELS.get(best['method'], best['method'])} "
                  f"({fmt_metric(best['mean'])})")
    print("\n[Noise-sensitivity summary (robust methods, tau vs clean)]")
    for metric in CFG.BODY_METRICS:
        for method in [m for m in CFG.METHODS if m != CFG.BASELINE]:
            st = nvc_stats[(nvc_stats["metric"] == metric) & (nvc_stats["method"] == method)]
            sig_taus = st[st[pcol] < CFG.HOLM_ALPHA]["tau"].tolist()
            first = f"{min(sig_taus):.1f}" if sig_taus else "never"
            print(f"   {CFG.METRIC_DISPLAY[metric][0]:12s} {method:9s}: "
                  f"degradation significant at tau={first}")
    print("=" * 74 + "\n")


# main
def main():
    P = CFG.PRIMARY_PROTOCOL
    print(f"Loading data from {CFG.EXPERIMENT_ROOT} ...")
    df = load_long_df()
    print(f"\n[load] {len(df)} rows; protocols={sorted(df['protocol'].unique())}; "
          f"methods={sorted(df['method'].unique())}; taus={sorted(df['tau'].unique())}")

    if P not in df["protocol"].unique():
        print(f"\n!! primary protocol '{P}' not found. Available: "
              f"{sorted(df['protocol'].unique())}. Set CONFIG.PRIMARY_PROTOCOL.\n")
        sys.exit(1)

    completeness_report(df, P)

    print("Computing summaries and statistics ...")
    summary = summarize(df, P)
    mvb = method_vs_baseline(df, P)
    nvc = noise_vs_clean(df, P)
    mvm = method_vs_method(df, P)

    print("Building figures ...")
    fig_money(summary, mvb, P)
    fig_all_single_metrics(summary, mvb, P)

    print("Building LaTeX tables ...")
    tex_body_combined(summary, mvb, P)
    tex_mvb_full_stats(mvb, P)
    tex_noise_vs_clean(nvc, P)
    tex_method_vs_method(mvm, P)

    print_prose_helpers(summary, mvb, nvc, P)

    out = _out_dir()
    summary.to_csv(out / f"_summary_{P}.csv", index=False)
    mvb.to_csv(out / f"_mvb_{P}.csv", index=False)
    nvc.to_csv(out / f"_nvc_{P}.csv", index=False)
    mvm.to_csv(out / f"_mvm_{P}.csv", index=False)
    print(f"[csv] wrote tidy stat frames to {out}/_*.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()