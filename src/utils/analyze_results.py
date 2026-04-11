# src/utils/analyze_results.py
#
# Loads all test_metrics.json files from all methods and noise types and
# produces comparison visualisations and summary statistics.
#
# Usage:
#   python -m src.utils.analyze_results --noise_type normalized_idn
#   python -m src.utils.analyze_results --noise_type feature_driven_idn
#   python -m src.utils.analyze_results --noise_type balanced_normalized_idn
#   python -m src.utils.analyze_results --noise_type balanced_feature_driven_idn
#   python -m src.utils.analyze_results --noise_type all

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.common.io import project_root

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

METHODS = ["baseline", "sce", "elr", "asyco"]
METHOD_LABELS = {
    "baseline": "Baseline (CE)",
    "sce":      "SCE",
    "elr":      "ELR",
    "asyco":    "AsyCo",
}
METHOD_COLORS = {
    "baseline": "#4C72B0",
    "sce":      "#DD8452",
    "elr":      "#55A868",
    "asyco":    "#C44E52",
}

NOISE_TYPE_LABELS = {
    "normalized_idn":     "Normalised IDN",
    "feature_driven_idn": "Feature-Driven IDN",
    "balanced_normalized_idn": "Balanced Normalised IDN",
    "balanced_feature_driven_idn": "Balanced Feature-Driven IDN",
}

TAU_DIRS   = ["clean", "tau05", "tau10", "tau15", "tau20", "tau25", "tau30"]
TAU_LABELS = ["clean", "τ=0.05", "τ=0.10", "τ=0.15", "τ=0.20", "τ=0.25", "τ=0.30"]
FOLD_DIRS  = [f"fold_{i:02d}" for i in range(10)]
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

SCALAR_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "kappa",
    "auc_macro_ovr",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_data(results_root: Path, method: str, noise_type: str) -> pd.DataFrame:
    method_dir = results_root / method / noise_type
    if not method_dir.exists():
        return pd.DataFrame()

    records = []
    for tau in TAU_DIRS:
        tau_path = method_dir / tau
        if not tau_path.exists():
            continue
        for fold in FOLD_DIRS:
            fold_path = tau_path / fold
            tm_path   = fold_path / "test_metrics.json"
            if not tm_path.exists():
                continue
            tm  = load_json(tm_path)
            row = {"method": method, "tau": tau, "fold": fold}
            for m in SCALAR_METRICS:
                row[m] = tm.get(m, np.nan)
            for cls in CLASS_NAMES:
                row[f"f1_{cls}"] = tm.get("per_class_f1", {}).get(cls, np.nan)
            row["confusion_matrix"] = tm.get("confusion_matrix", None)
            records.append(row)

    return pd.DataFrame(records)


def load_all(results_root: Path, noise_type: str) -> pd.DataFrame:
    frames = []
    for method in METHODS:
        df = collect_data(results_root, method, noise_type)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame, noise_label: str) -> None:
    print(f"\n{'='*100}")
    print(f"  PERFORMANCE SUMMARY — {noise_label}  (mean ± std across 10 folds)")
    print(f"{'='*100}")

    metric_labels = {
        "balanced_accuracy": "Bal. Acc",
        "macro_f1":          "Macro F1",
        "kappa":             "Kappa",
        "auc_macro_ovr":     "AUC OvR",
    }
    col_w = 18
    header = f"{'Method':<12} {'Tau':<10}" + "".join(f"{v:>{col_w}}" for v in metric_labels.values())
    print(header)
    print(f"{'-'*100}")

    for method in METHODS:
        if method not in df["method"].values:
            continue
        for tau, label in zip(TAU_DIRS, TAU_LABELS):
            sub = df[(df["method"] == method) & (df["tau"] == tau)]
            if sub.empty:
                continue
            row_str = f"{METHOD_LABELS[method]:<12} {label:<10}"
            for m in metric_labels:
                vals = sub[m].dropna()
                if len(vals):
                    row_str += f"  {vals.mean():.3f}±{vals.std():.3f}"
                else:
                    row_str += f"  {'N/A':>{col_w-2}}"
            print(row_str)
        print(f"{'-'*100}")
    print(f"{'='*100}\n")


# ── Plot 1: Key metrics vs tau — one line per method ──────────────────────────

def plot_metrics_vs_tau(df: pd.DataFrame, noise_label: str, plot_dir: Path) -> None:
    metrics = ["balanced_accuracy", "macro_f1", "kappa", "auc_macro_ovr"]
    labels  = ["Balanced Accuracy", "Macro F1", "Cohen's Kappa", "AUC (macro OvR)"]

    taus_present = [t for t in TAU_DIRS if t in df["tau"].values]
    x            = np.arange(len(taus_present))
    x_labels     = [TAU_LABELS[TAU_DIRS.index(t)] for t in taus_present]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        f"Key Metrics vs. Noise Rate — {noise_label}\n(mean ± std across 10 folds)",
        fontsize=14,
    )

    for ax, metric, label in zip(axes.flat, metrics, labels):
        for method in METHODS:
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            means, stds = [], []
            for tau in taus_present:
                vals = sub[sub["tau"] == tau][metric].dropna()
                means.append(vals.mean() if len(vals) else np.nan)
                stds.append(vals.std()  if len(vals) else np.nan)
            means = np.array(means)
            stds  = np.array(stds)
            ax.plot(x, means, marker="o", linewidth=2,
                    color=METHOD_COLORS[method], label=METHOD_LABELS[method])
            ax.fill_between(x, means - stds, means + stds,
                            alpha=0.12, color=METHOD_COLORS[method])

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Score")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)

    fig.savefig(plot_dir / "metrics_vs_tau.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: metrics_vs_tau.png")


# ── Plot 2: Per-class F1 heatmap per method at one tau ────────────────────────

def plot_perclass_f1_heatmaps(df: pd.DataFrame, noise_label: str,
                               plot_dir: Path) -> None:
    for tau, tau_label in zip(TAU_DIRS, TAU_LABELS):
        if tau == "clean":
            continue
        methods_present = [m for m in METHODS if m in df["method"].values
                           and not df[(df["method"] == m) & (df["tau"] == tau)].empty]
        if not methods_present:
            continue

        n_methods = len(methods_present)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5),
                                  constrained_layout=True)
        if n_methods == 1:
            axes = [axes]
        fig.suptitle(
            f"Per-Class F1 — {noise_label} | {tau_label}\n(mean across 10 folds)",
            fontsize=13,
        )

        for ax, method in zip(axes, methods_present):
            sub    = df[(df["method"] == method) & (df["tau"] == tau)]
            matrix = np.array([sub[f"f1_{cls}"].mean() for cls in CLASS_NAMES])
            im     = ax.imshow(matrix.reshape(-1, 1), aspect="auto",
                               cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_yticks(range(len(CLASS_NAMES)))
            ax.set_yticklabels([c.upper() for c in CLASS_NAMES], fontsize=10)
            ax.set_xticks([])
            ax.set_title(METHOD_LABELS[method], fontweight="bold")
            for i, cls in enumerate(CLASS_NAMES):
                val = matrix[i]
                if not np.isnan(val):
                    ax.text(0, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=10,
                            color="black" if 0.3 < val < 0.75 else "white")
            fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04)

        tag = f"tau{int(float(tau.replace('tau','')) ):02d}" if tau != "clean" else "clean"
        fig.savefig(plot_dir / f"perclass_f1_{tag}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: perclass_f1_{tag}.png")


# ── Plot 3: Clean vs noisy — all methods, paired Wilcoxon ────────────────────

def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000,
                   ci: float = 0.95, seed: int = 0):
    rng   = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo    = np.percentile(boots, 100 * (1 - ci) / 2)
    hi    = np.percentile(boots, 100 * (1 + ci) / 2)
    return values.mean(), lo, hi


def _significance_stars(p: float) -> str:
    if p < 0.001:  return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return "ns"


def plot_clean_vs_noisy(df: pd.DataFrame, noise_label: str,
                         plot_dir: Path) -> None:
    metrics  = ["balanced_accuracy", "macro_f1", "kappa", "auc_macro_ovr"]
    m_labels = ["Balanced Accuracy", "Macro F1", "Cohen's Kappa", "AUC (macro OvR)"]

    noisy_taus   = [t for t in TAU_DIRS if t != "clean" and t in df["tau"].values]
    noisy_labels = [TAU_LABELS[TAU_DIRS.index(t)] for t in noisy_taus]
    all_folds    = sorted(df["fold"].unique())

    methods_present = [m for m in METHODS if m in df["method"].values
                       and "clean" in df[df["method"] == m]["tau"].values]
    if not methods_present:
        print("  [skip] no clean baseline found")
        return

    print(f"\n{'='*90}")
    print(f"  WILCOXON SIGNED-RANK TEST — {noise_label}")
    print(f"  (paired by fold, clean vs each tau, per method)")
    print(f"{'='*90}")

    for method in methods_present:
        sub = df[df["method"] == method]

        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        fig.suptitle(
            f"{METHOD_LABELS[method]} — {noise_label}\n"
            "Clean vs. Noisy | 95% Bootstrap CI "
            "(Wilcoxon signed-rank, paired by fold)",
            fontsize=13,
        )

        for ax, metric, m_label in zip(axes.flat, metrics, m_labels):
            clean_vals = np.array([
                sub[(sub["tau"] == "clean") & (sub["fold"] == f)][metric].values[0]
                for f in all_folds
                if len(sub[(sub["tau"] == "clean") & (sub["fold"] == f)][metric].values)
            ])
            c_mean, c_lo, c_hi = _bootstrap_ci(clean_vals)

            x         = np.arange(len(noisy_taus) + 1)
            bar_width = 0.55
            noise_colours = plt.cm.YlOrRd(np.linspace(0.35, 0.85, len(noisy_taus)))

            ax.bar(0, c_mean, width=bar_width,
                   color=METHOD_COLORS[method], alpha=0.88, zorder=3)
            ax.errorbar(0, c_mean,
                        yerr=[[c_mean - c_lo], [c_hi - c_mean]],
                        fmt="none", color="black", capsize=5,
                        linewidth=1.8, zorder=5)
            ax.axhline(c_mean, color=METHOD_COLORS[method], linewidth=1.2,
                       linestyle="--", alpha=0.5, zorder=2)

            print(f"\n  {METHOD_LABELS[method]} | {m_label}")
            print(f"  {'Tau':<10}  {'clean':>7}  {'noisy':>7}  "
                  f"{'Δ':>7}  {'p-value':>10}  sig")
            print("  " + "─" * 55)

            for j, (tau, label, colour) in enumerate(
                    zip(noisy_taus, noisy_labels, noise_colours), start=1):
                noisy_vals = np.array([
                    sub[(sub["tau"] == tau) & (sub["fold"] == f)][metric].values[0]
                    for f in all_folds
                    if len(sub[(sub["tau"] == tau) & (sub["fold"] == f)][metric].values)
                ])
                n_mean, n_lo, n_hi = _bootstrap_ci(noisy_vals)

                diffs = clean_vals - noisy_vals
                if np.all(diffs == 0):
                    p_val = 1.0
                else:
                    try:
                        _, p_val = wilcoxon(clean_vals, noisy_vals,
                                            alternative="two-sided")
                    except ValueError:
                        p_val = 1.0

                stars = _significance_stars(p_val)
                delta = n_mean - c_mean

                print(f"  {label:<10}  {c_mean:>7.4f}  {n_mean:>7.4f}  "
                      f"{delta:>+7.4f}  {p_val:>10.4f}  {stars}")

                ax.bar(j, n_mean, width=bar_width, color=colour,
                       alpha=0.88, zorder=3)
                ax.errorbar(j, n_mean,
                            yerr=[[n_mean - n_lo], [n_hi - n_mean]],
                            fmt="none", color="black", capsize=5,
                            linewidth=1.8, zorder=5)

                if abs(delta) > 0.003:
                    ax.text(j, max(n_mean, 0) * 0.5, f"Δ{delta:+.3f}",
                            ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold", zorder=6)

                star_y = max(n_hi, n_mean) + 0.012
                ax.text(j, star_y, stars, ha="center", va="bottom",
                        fontsize=10,
                        color="#c0392b" if stars != "ns" else "#555555",
                        fontweight="bold", zorder=6)

            y_min = sub[sub["tau"].isin(["clean"] + noisy_taus)][metric].min()
            y_max = sub[sub["tau"].isin(["clean"] + noisy_taus)][metric].max()
            y_pad = (y_max - y_min) * 0.25
            ax.set_ylim(max(0, y_min - y_pad * 0.5), y_max + y_pad)
            ax.set_xticks(x)
            ax.set_xticklabels(["clean"] + noisy_labels, fontsize=9)
            ax.set_title(m_label, fontweight="bold")
            ax.set_ylabel("Score")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)

        print(f"{'='*90}")
        fig.savefig(plot_dir / f"clean_vs_noisy_{method}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: clean_vs_noisy_{method}.png")


# ── Plot 4: Method comparison at each tau — bar chart ────────────────────────

def plot_method_comparison(df: pd.DataFrame, noise_label: str,
                            plot_dir: Path) -> None:
    metrics  = ["balanced_accuracy", "macro_f1", "kappa", "auc_macro_ovr"]
    m_labels = ["Balanced Accuracy", "Macro F1", "Cohen's Kappa", "AUC (macro OvR)"]

    taus_present     = [t for t in TAU_DIRS if t in df["tau"].values]
    methods_present  = [m for m in METHODS if m in df["method"].values]
    n_methods        = len(methods_present)
    x                = np.arange(len(taus_present))
    width            = 0.8 / n_methods
    offsets          = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2,
                                    n_methods) * width

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        f"Method Comparison — {noise_label}\n(mean across 10 folds)",
        fontsize=14,
    )

    for ax, metric, label in zip(axes.flat, metrics, m_labels):
        for method, offset in zip(methods_present, offsets):
            sub   = df[df["method"] == method]
            means = []
            stds  = []
            for tau in taus_present:
                vals = sub[sub["tau"] == tau][metric].dropna()
                means.append(vals.mean() if len(vals) else np.nan)
                stds.append(vals.std()   if len(vals) else np.nan)
            means = np.array(means)
            stds  = np.array(stds)
            ax.bar(x + offset, means, width=width,
                   color=METHOD_COLORS[method], label=METHOD_LABELS[method],
                   alpha=0.85, zorder=3)
            ax.errorbar(x + offset, means,
                        yerr=stds, fmt="none", color="black",
                        capsize=3, linewidth=1.2, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([TAU_LABELS[TAU_DIRS.index(t)] for t in taus_present],
                           fontsize=9)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Score")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)

    fig.savefig(plot_dir / "method_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: method_comparison.png")


# ── Plot 5: Per-fold scatter — all methods at clean and highest tau ───────────

def plot_fold_scatter(df: pd.DataFrame, noise_label: str,
                      plot_dir: Path) -> None:
    metric = "balanced_accuracy"
    taus_present    = [t for t in TAU_DIRS if t in df["tau"].values]
    methods_present = [m for m in METHODS if m in df["method"].values]

    fig, axes = plt.subplots(1, len(taus_present),
                              figsize=(3 * len(taus_present), 5),
                              constrained_layout=True)
    if len(taus_present) == 1:
        axes = [axes]
    fig.suptitle(
        f"Balanced Accuracy per Fold — {noise_label}\n"
        "(dots = individual folds, bar = mean)",
        fontsize=13,
    )

    x       = np.arange(len(methods_present))
    spacing = 0.3

    for ax, tau in zip(axes, taus_present):
        for j, method in enumerate(methods_present):
            sub  = df[(df["method"] == method) & (df["tau"] == tau)]
            vals = sub[metric].dropna().values
            if len(vals) == 0:
                continue
            jitter = np.random.default_rng(42).uniform(
                -spacing / 2, spacing / 2, len(vals)
            )
            ax.scatter(x[j] + jitter, vals, alpha=0.6, s=35,
                       color=METHOD_COLORS[method], zorder=4)
            ax.plot([x[j] - spacing / 2, x[j] + spacing / 2],
                    [vals.mean(), vals.mean()],
                    color="black", linewidth=2.5, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods_present],
                           fontsize=8, rotation=15, ha="right")
        ax.set_title(TAU_LABELS[TAU_DIRS.index(tau)], fontweight="bold")
        ax.set_ylabel("Balanced Accuracy" if tau == taus_present[0] else "")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(plot_dir / "fold_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: fold_scatter.png")


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, plot_dir: Path) -> None:
    out = []
    for method in METHODS:
        for tau, label in zip(TAU_DIRS, TAU_LABELS):
            sub = df[(df["method"] == method) & (df["tau"] == tau)]
            if sub.empty:
                continue
            row = {"method": METHOD_LABELS[method], "noise_level": label}
            for m in SCALAR_METRICS:
                vals = sub[m].dropna()
                row[f"{m}_mean"] = round(vals.mean(), 4) if len(vals) else np.nan
                row[f"{m}_std"]  = round(vals.std(),  4) if len(vals) else np.nan
            for cls in CLASS_NAMES:
                vals = sub[f"f1_{cls}"].dropna()
                row[f"f1_{cls}_mean"] = round(vals.mean(), 4) if len(vals) else np.nan
                row[f"f1_{cls}_std"]  = round(vals.std(),  4) if len(vals) else np.nan
            out.append(row)

    csv_path = plot_dir / "aggregated_results.csv"
    pd.DataFrame(out).to_csv(csv_path, index=False)
    print(f"  Saved: aggregated_results.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noise_type",
        choices=["normalized_idn", "feature_driven_idn", "balanced_normalized_idn", "balanced_feature_driven_idn", "all"],
        required=True,
        help="Which noise type to analyse, or 'all' to run both",
    )
    args = parser.parse_args()

    results_root = project_root() / "results" / "HAM10000"

    noise_types = (
        ["normalized_idn", "feature_driven_idn", "balanced_normalized_idn", "balanced_feature_driven_idn"]
        if args.noise_type == "all"
        else [args.noise_type]
    )

    for noise_type in noise_types:
        noise_label = NOISE_TYPE_LABELS[noise_type]
        plot_dir    = results_root / "analysis" / noise_type
        plot_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Analysing: {noise_label}")
        print(f"  Output:    {plot_dir}")
        print(f"{'='*60}")

        df = load_all(results_root, noise_type)

        if df.empty:
            print(f"  No results found for {noise_type}. Check results directory.")
            continue

        methods_found = df["method"].unique().tolist()
        folds_found   = df["fold"].nunique()
        taus_found    = df["tau"].unique().tolist()
        print(f"\n  Methods found : {methods_found}")
        print(f"  Folds found   : {folds_found}")
        print(f"  Tau levels    : {taus_found}\n")

        print_summary_table(df, noise_label)
        plot_metrics_vs_tau(df, noise_label, plot_dir)
        plot_method_comparison(df, noise_label, plot_dir)
        plot_perclass_f1_heatmaps(df, noise_label, plot_dir)
        plot_clean_vs_noisy(df, noise_label, plot_dir)
        plot_fold_scatter(df, noise_label, plot_dir)
        save_csv(df, plot_dir)

        print(f"\n  All outputs written to: {plot_dir.resolve()}")


if __name__ == "__main__":
    main()