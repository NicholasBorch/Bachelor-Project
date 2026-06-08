"""
Results.3 - best-vs-next-best method comparison.

For each protocol and metric, at every noise rate tau the four methods are ranked
by fold-mean score and the rank-1 method is compared against rank-2 on the ten
per-fold differences, using the shared paired machinery (thesis_paired_stats).
Holm correction is applied across the six tau within each (protocol, metric).

Writes into results/method_comparison/best_vs_next/:
  tab_best_vs_next_<P>.tex   LaTeX body table (rows = tau), label tab:method-between for AP.
  _best_vs_next_<P>.csv      tidy values behind the table.
"""

from __future__ import annotations

import json
import sys
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Shared statistics module (identical machinery to every other Results script)
import thesis_paired_stats as TPS


# Config
@dataclass
class Config:
    EXPERIMENT_ROOT: Path = Path("./results/main_experiment")
    TRAINING_SUBDIR: str = "training"          
    METRICS_FILENAME: str = "test_metrics.json"
    TAU_DIR_FMT: str = "tau_{tt:02d}"          
    FOLD_DIR_FMT: str = "fold_{ff:02d}"

    # logical protocol -> on-disk folder name.
    PROTOCOL_DIRS: dict = field(default_factory=lambda: {
        "AP": "pretrained_adam",
        # "A":  "scratch_adam",
        # "SP": "pretrained_sgd",
        # "S":  "scratch_sgd",
    })
    PROTOCOLS_TO_RUN: tuple = ("AP",)
    PRIMARY_PROTOCOL: str = "AP"   # gets the tab:method-between label

    # logical method -> on-disk folder name.
    METHOD_DIRS: dict = field(default_factory=lambda: {
        "baseline": "baseline",
        "SCE":      "sce",
        "ELR":      "elr",
        "AsyCo":    "asyco_divmix",
    })
    METHODS: tuple = ("baseline", "SCE", "ELR", "AsyCo")
    BASELINE: str = "baseline"
    METHOD_LABELS: dict = field(default_factory=lambda: {
        "baseline": "Baseline", "SCE": "SCE", "ELR": "ELR", "AsyCo": "AsyCo",
    })

    METRIC_KEYS: dict = field(default_factory=lambda: {
        "BA":       ["balanced_accuracy", "bacc", "balanced_acc", "BA", "bal_acc"],
        "MacroF1":  ["macro_f1", "f1_macro", "macro_F1", "f1macro", "f1_macro_avg"],
        "MacroAUC": ["macro_auc", "auc_macro", "macro_AUC", "roc_auc_macro", "auroc_macro"],
    })
    METRIC_NEST_KEYS: tuple = ("", "test", "metrics", "test_metrics")
    # metrics shown in the table, in order
    TABLE_METRICS: tuple = ("BA", "MacroF1", "MacroAUC")
    METRIC_DISPLAY: dict = field(default_factory=lambda: {
        "BA": "Balanced accuracy", "MacroF1": "Macro F1", "MacroAUC": "Macro AUC",
    })

    TAUS: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)   # full family incl. clean (choice 2a)
    N_FOLDS: int = 10
    MIN_PAIRED_FOLDS_FOR_TEST: int = 3

    N_BOOT: int = 10000
    SEED: int = 10
    HOLM_ALPHA: float = 0.05

    RESULTS_ROOT: Path = Path("./results")
    ANALYSIS_DIR: str = "method_comparison"
    SUBDIR: str = "best_vs_next"


CFG = Config()


def _out() -> Path:
    d = CFG.RESULTS_ROOT / CFG.ANALYSIS_DIR / CFG.SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# Small utilities
def _seed_for(*parts) -> int:
    h = hashlib.sha256(("|".join(map(str, parts))).encode()).hexdigest()
    return (CFG.SEED + int(h[:8], 16)) % (2 ** 32 - 1)


def _tau_dir(tau: float) -> str:
    return CFG.TAU_DIR_FMT.format(tt=int(round(float(tau) * 100)))


def _fold_dir(fold: int) -> str:
    return CFG.FOLD_DIR_FMT.format(ff=int(fold))


def _read_json(fp: Path) -> dict:
    try:
        with open(fp) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _extract_metric(d: dict, aliases) -> Optional[float]:
    """Find a metric value by trying alias keys at the top level and common nests."""
    flat = _flatten(d)
    # direct / nested-prefixed alias hits
    for nest in CFG.METRIC_NEST_KEYS:
        for alias in aliases:
            key = f"{nest}.{alias}" if nest else alias
            if key in d and not isinstance(d[key], dict):
                try:
                    return float(d[key])
                except (TypeError, ValueError):
                    pass
            if key in flat:
                try:
                    return float(flat[key])
                except (TypeError, ValueError):
                    pass
    # last resort: any flattened key whose final segment matches an alias
    for alias in aliases:
        for fk, fv in flat.items():
            if fk.split(".")[-1] == alias:
                try:
                    return float(fv)
                except (TypeError, ValueError):
                    pass
    return None


def _protocol_root(protocol: str) -> Path:
    folder = CFG.PROTOCOL_DIRS[protocol]
    root = CFG.EXPERIMENT_ROOT / folder
    return root / CFG.TRAINING_SUBDIR if CFG.TRAINING_SUBDIR else root


def _metrics_path(protocol: str, method: str, tau: float, fold: int) -> Path:
    mdir = CFG.METHOD_DIRS[method]
    return _protocol_root(protocol) / mdir / _tau_dir(tau) / _fold_dir(fold) / CFG.METRICS_FILENAME


# Load -> long DataFrame (protocol, method, tau, fold, <metric columns>)
def load_long_df(protocols) -> pd.DataFrame:
    rows = []
    for protocol in protocols:
        if protocol not in CFG.PROTOCOL_DIRS:
            print(f"[skip] {protocol}: not in PROTOCOL_DIRS")
            continue
        for method in CFG.METHODS:
            for tau in CFG.TAUS:
                for fold in range(CFG.N_FOLDS):
                    fp = _metrics_path(protocol, method, tau, fold)
                    if not fp.exists():
                        continue
                    raw = _read_json(fp)
                    rec = dict(protocol=protocol, method=method,
                               tau=float(tau), fold=int(fold))
                    for mkey, aliases in CFG.METRIC_KEYS.items():
                        rec[mkey] = _extract_metric(raw, aliases)
                    rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            "No test_metrics.json found. Check EXPERIMENT_ROOT, PROTOCOL_DIRS, "
            "METHOD_DIRS, and the tau/fold directory naming.")
    df = df.drop_duplicates(subset=["protocol", "method", "tau", "fold"])
    return df.sort_values(["protocol", "method", "tau", "fold"]).reset_index(drop=True)


def _wide_fold(df: pd.DataFrame, protocol: str, metric: str, tau: float) -> pd.DataFrame:
    """Rows = fold, columns = method, values = the metric. Index is fold id."""
    sub = df[(df.protocol == protocol) & np.isclose(df.tau, tau)]
    return sub.pivot_table(index="fold", columns="method", values=metric)


# Ranking + best-vs-next (the family)
def _fold_means(df: pd.DataFrame, protocol: str, metric: str, tau: float) -> pd.Series:
    """Mean over folds per method (the TRUE fold means used for ranking)."""
    w = _wide_fold(df, protocol, metric, tau)
    return w.mean(axis=0, skipna=True)


def best_vs_next(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """Build the six-tau block per metric, test rank-1 vs rank-2, Holm across tau."""
    recs = []
    for metric in CFG.TABLE_METRICS:
        block = []
        for tau in CFG.TAUS:
            means = _fold_means(df, protocol, metric, tau).dropna()
            means = means[[m for m in CFG.METHODS if m in means.index]]
            ranked = means.sort_values(ascending=False)
            if len(ranked) < 2:
                block.append(dict(protocol=protocol, metric=metric, tau=float(tau),
                                  best_method="--", next_method="--",
                                  W=np.nan, p_wilcoxon=np.nan, p_perm=np.nan,
                                  delta=np.nan, delta_ci_lo=np.nan, delta_ci_hi=np.nan,
                                  r_rb=np.nan, direction=0, n=0,
                                  mean_delta=np.nan, p_raw=np.nan, exploratory=True))
                continue
            best_method = str(ranked.index[0])
            next_method = str(ranked.index[1])
            wide = _wide_fold(df, protocol, metric, tau)
            if best_method in wide.columns and next_method in wide.columns:
                pair = wide[[best_method, next_method]].dropna()
            else:
                pair = pd.DataFrame()
            if len(pair) >= CFG.MIN_PAIRED_FOLDS_FOR_TEST:
                d = pair[best_method].values - pair[next_method].values  # best - next
                res = TPS.paired_compare(
                    d, n_boot=CFG.N_BOOT,
                    boot_seed=_seed_for("best-next", protocol, metric, tau,
                                        best_method, next_method))
                rec = dict(protocol=protocol, metric=metric, tau=float(tau),
                           best_method=best_method, next_method=next_method,
                           exploratory=True, **res.as_dict())
                rec["mean_delta"] = rec["delta"]
                rec["p_raw"] = rec["p_wilcoxon"]
            else:
                rec = dict(protocol=protocol, metric=metric, tau=float(tau),
                           best_method=best_method, next_method=next_method,
                           W=np.nan, p_wilcoxon=np.nan, p_perm=np.nan,
                           delta=np.nan, delta_ci_lo=np.nan, delta_ci_hi=np.nan,
                           r_rb=np.nan, direction=0, n=0,
                           mean_delta=np.nan, p_raw=np.nan, exploratory=True)
            block.append(rec)
        # Holm ACROSS the six tau within this metric (the m=6 family).
        TPS.add_holm_and_flags(block, alpha=CFG.HOLM_ALPHA)
        for b in block:
            b["p_holm"] = b.get("p_wilcoxon_holm", np.nan)
        recs.extend(block)
    return pd.DataFrame(recs)


# LaTeX table
def _fmt_signed(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if abs(x) < 0.5 * 10 ** (-nd):
        return f"{0.0:+.{nd}f}"
    return f"{x:+.{nd}f}"


def _fmt_ci(lo, hi, nd=3):
    if lo is None or (isinstance(lo, float) and np.isnan(lo)):
        return "--"
    return f"[{lo:+.{nd}f},\\,{hi:+.{nd}f}]"


def _fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    return r"$<0.001$" if p < 0.001 else f"{p:.4f}"


def _fmt_r(r):
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "--"
    return f"{r:+.2f}"


def _sig_cell(rec):
    sig = rec.get("sig", TPS.NS_SYMBOL)
    sig = sig if sig != TPS.NS_SYMBOL else "n.s."
    if rec.get("flag"):
        sig += r"\,!"
    return sig


def emit_table(stats_df: pd.DataFrame, protocol: str):
    label = "tab:method-between" if protocol == CFG.PRIMARY_PROTOCOL \
        else f"tab:method-between-{protocol}"
    rows = []
    for metric in CFG.TABLE_METRICS:
        sub = stats_df[stats_df.metric == metric].sort_values("tau")
        rows.append(r"\addlinespace")
        rows.append(r"\multicolumn{7}{l}{\textit{%s}} \\" % CFG.METRIC_DISPLAY[metric])
        for _, r in sub.iterrows():
            comp = (f"{CFG.METHOD_LABELS.get(r['best_method'], r['best_method'])} "
                    f"vs. {CFG.METHOD_LABELS.get(r['next_method'], r['next_method'])}")
            rows.append(" & ".join([
                f"{r['tau']:.1f}", comp,
                _fmt_signed(r["mean_delta"]),
                f"${_fmt_ci(r['delta_ci_lo'], r['delta_ci_hi'])}$",
                _fmt_r(r["r_rb"]),
                _fmt_p(r["p_holm"]),
                _sig_cell(r),
            ]) + r" \\")

    head = (r"$\tau$ & Comparison & $\Delta$ & 95\% CI & $r$ & "
            r"$p_{\mathrm{Holm}}$ & sig. \\")
    colspec = "llrrrrl"
    caption = (
        f"Best-vs-next-best method comparison under the {protocol} protocol. For "
        f"each metric and noise rate, the highest-mean method is compared against "
        f"the second-highest on the per-fold differences ($\\Delta = \\text{{best}} "
        f"- \\text{{next}}$), with its $95\\%$ bootstrap confidence interval, the "
        f"matched-pairs rank-biserial correlation $r$, and the Holm-corrected "
        f"$p$-value across the six noise rates within each metric. The tested pair "
        f"is rank-selected from the observed means, so this comparison is "
        f"exploratory and read descriptively; the pre-specified pairwise family "
        f"(Appendix~\\ref{{app:method-pairwise}}) is its confirmatory counterpart. "
        f"Rows where the runner-up is the baseline are retained, as in a "
        f"best-vs-next table this reports the true runner-up rather than "
        f"duplicating the method-vs-baseline comparison."
    )
    tex = [
        r"\begin{table}[h!]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", head, r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    fp = _out() / f"tab_best_vs_next_{protocol}.tex"
    fp.write_text("\n".join(tex) + "\n")
    print(f"[tab] wrote {fp}")


# Main
def main():
    print(f"Loading from {CFG.EXPERIMENT_ROOT} ...")
    df = load_long_df(CFG.PROTOCOLS_TO_RUN)
    print(f"[load] {len(df)} rows; protocols={sorted(df.protocol.unique())}; "
          f"methods={sorted(df.method.unique())}; taus={sorted(df.tau.unique())}")

    floor6 = 6 * TPS.p_floor(CFG.N_FOLDS)
    print(f"[floor] n={CFG.N_FOLDS}: raw two-sided floor = {TPS.p_floor(CFG.N_FOLDS):.5f}; "
          f"Holm m=6 floor = {floor6:.5f}")

    for protocol in CFG.PROTOCOLS_TO_RUN:
        if protocol not in df.protocol.unique():
            print(f"[skip] {protocol}: no data loaded")
            continue
        stats_df = best_vs_next(df, protocol)
        # report the effective family size per metric (non-NaN p-values)
        for metric in CFG.TABLE_METRICS:
            m_eff = int(stats_df[(stats_df.metric == metric)]["p_wilcoxon"].notna().sum())
            print(f"   [{protocol}] {metric}: effective Holm m = {m_eff}")
        stats_df.to_csv(_out() / f"_best_vs_next_{protocol}.csv", index=False)
        print(f"[csv] wrote {_out() / f'_best_vs_next_{protocol}.csv'}")
        emit_table(stats_df, protocol)

    print("Done.")


if __name__ == "__main__":
    main()