"""
analyze_noise_rates.py

Compares model performance across noise rates (tau levels) for the
feature_driven_idn baseline on HAM10000.

Run from:  src/  (or wherever your analysis scripts live)
Results:   ../../results/HAM10000/baseline/feature_driven_idn/
Outputs:   saves all figures to  results/HAM10000/baseline/feature_driven_idn/plots/
           also prints a summary table to the terminal.
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / ".." / ".." / "results" / "HAM10000" / "baseline" / "standardized_idn"
RESULTS_ROOT = RESULTS_ROOT.resolve()
PLOT_DIR     = RESULTS_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────────
TAU_DIRS   = ["clean", "tau05", "tau10", "tau15", "tau20", "tau25", "tau30"]
TAU_LABELS = ["clean", "τ=0.05", "τ=0.10", "τ=0.15", "τ=0.20", "τ=0.25", "τ=0.30"]
FOLD_DIRS  = [f"fold_{i:02d}" for i in range(5)]
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

SCALAR_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "kappa",
    "auc_macro_ovr",
]

PALETTE = plt.cm.tab10.colors  # one colour per tau level

# ── helpers ────────────────────────────────────────────────────────────────────

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def collect_data():
    """
    Returns
    -------
    records : list[dict]   – one entry per (tau, fold)
    training : dict        – tau → list of loss arrays (one per fold)
    """
    records  = []
    training = {}

    for tau in TAU_DIRS:
        tau_path = RESULTS_ROOT / tau
        if not tau_path.exists():
            print(f"  [skip] {tau_path} not found")
            continue

        fold_losses = []
        for fold in FOLD_DIRS:
            fold_path = tau_path / fold
            if not fold_path.exists():
                continue

            # ── test metrics ──────────────────────────────────────────────────
            tm_path = fold_path / "test_metrics.json"
            if tm_path.exists():
                tm = load_json(tm_path)
                row = {"tau": tau, "fold": fold}
                for m in SCALAR_METRICS:
                    row[m] = tm.get(m, np.nan)
                for cls in CLASS_NAMES:
                    row[f"f1_{cls}"] = tm.get("per_class_f1", {}).get(cls, np.nan)
                row["confusion_matrix"] = tm.get("confusion_matrix", None)
                records.append(row)

            # ── training log ──────────────────────────────────────────────────
            tl_path = fold_path / "training_log.json"
            if tl_path.exists():
                tl = load_json(tl_path)
                losses = [e["train_loss"] for e in tl if not np.isnan(e["train_loss"])]
                fold_losses.append(losses)

        if fold_losses:
            training[tau] = fold_losses

    return records, training


# ── figure 1 : summary table ───────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame):
    print("\n" + "═" * 90)
    print("  PERFORMANCE SUMMARY  (mean ± std across folds)")
    print("═" * 90)
    metric_labels = {
        "accuracy":       "Accuracy",
        "balanced_accuracy": "Bal. Acc",
        "macro_f1":       "Macro F1",
        "weighted_f1":    "Wtd F1",
        "kappa":          "Kappa",
        "auc_macro_ovr":  "AUC OvR",
    }
    header = f"{'Noise level':<12}" + "".join(f"{v:>16}" for v in metric_labels.values())
    print(header)
    print("─" * 90)
    for tau, label in zip(TAU_DIRS, TAU_LABELS):
        sub = df[df["tau"] == tau]
        if sub.empty:
            continue
        row_str = f"{label:<12}"
        for m in metric_labels:
            vals = sub[m].dropna()
            if len(vals):
                row_str += f"  {vals.mean():.3f}±{vals.std():.3f}"
            else:
                row_str += f"  {'N/A':>12}"
        print(row_str)
    print("═" * 90 + "\n")


# ── figure 2 : key metrics vs noise rate ──────────────────────────────────────

def plot_metrics_vs_tau(df: pd.DataFrame):
    metrics = ["balanced_accuracy", "macro_f1", "kappa", "auc_macro_ovr"]
    labels  = ["Balanced Accuracy", "Macro F1", "Cohen's Kappa", "AUC (macro OvR)"]

    taus_present = [t for t in TAU_DIRS if t in df["tau"].values]
    x = np.arange(len(taus_present))
    x_labels = [TAU_LABELS[TAU_DIRS.index(t)] for t in taus_present]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Key Metrics vs. Noise Rate\n(mean ± std across 5 folds)", fontsize=14)

    for ax, metric, label in zip(axes.flat, metrics, labels):
        means, stds = [], []
        for tau in taus_present:
            vals = df[df["tau"] == tau][metric].dropna()
            means.append(vals.mean())
            stds.append(vals.std())

        means, stds = np.array(means), np.array(stds)
        ax.plot(x, means, marker="o", linewidth=2, color="#2563eb")
        ax.fill_between(x, means - stds, means + stds, alpha=0.2, color="#2563eb")
        ax.scatter(x, means, zorder=5, color="#2563eb", s=60)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Score")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(PLOT_DIR / "metrics_vs_tau.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR / 'metrics_vs_tau.png'}")


# ── figure 3 : per-class F1 heatmap ───────────────────────────────────────────

def plot_perclass_f1_heatmap(df: pd.DataFrame):
    taus_present = [t for t in TAU_DIRS if t in df["tau"].values]
    x_labels = [TAU_LABELS[TAU_DIRS.index(t)] for t in taus_present]

    matrix = np.zeros((len(CLASS_NAMES), len(taus_present)))
    for j, tau in enumerate(taus_present):
        sub = df[df["tau"] == tau]
        for i, cls in enumerate(CLASS_NAMES):
            vals = sub[f"f1_{cls}"].dropna()
            matrix[i, j] = vals.mean() if len(vals) else np.nan

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(taus_present)))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels([c.upper() for c in CLASS_NAMES], fontsize=10)
    ax.set_title("Per-Class F1 Score across Noise Rates\n(mean across 5 folds)", fontsize=13)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(taus_present)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="black" if 0.3 < val < 0.75 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("F1 Score")

    fig.savefig(PLOT_DIR / "perclass_f1_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR / 'perclass_f1_heatmap.png'}")


# ── figure 4 : training loss curves ───────────────────────────────────────────

def plot_training_curves(training: dict):
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

    for i, tau in enumerate(TAU_DIRS):
        if tau not in training:
            continue
        folds = training[tau]
        max_len = max(len(f) for f in folds)
        # pad shorter folds with NaN so we can stack
        arr = np.full((len(folds), max_len), np.nan)
        for k, f in enumerate(folds):
            arr[k, : len(f)] = f

        mean_loss = np.nanmean(arr, axis=0)
        std_loss  = np.nanstd(arr, axis=0)
        epochs    = np.arange(1, max_len + 1)
        label     = TAU_LABELS[TAU_DIRS.index(tau)]
        color     = PALETTE[i % len(PALETTE)]

        ax.plot(epochs, mean_loss, label=label, color=color, linewidth=2)
        ax.fill_between(epochs,
                        mean_loss - std_loss,
                        mean_loss + std_loss,
                        alpha=0.15, color=color)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Training Loss", fontsize=11)
    ax.set_title("Training Loss Curves across Noise Rates\n(mean ± std across 5 folds)", fontsize=13)
    ax.legend(title="Noise level", fontsize=9, title_fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(PLOT_DIR / "training_loss_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR / 'training_loss_curves.png'}")


# ── figure 5 : confusion matrices ─────────────────────────────────────────────

def plot_confusion_matrices(df: pd.DataFrame):
    taus_present = [t for t in TAU_DIRS if t in df["tau"].values]
    n = len(taus_present)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 5, rows * 4.5),
                             constrained_layout=True)
    axes = np.array(axes).flatten()
    fig.suptitle("Confusion Matrices (summed across 5 folds)", fontsize=14)

    for idx, tau in enumerate(taus_present):
        ax = axes[idx]
        sub = df[df["tau"] == tau]
        cms = [np.array(r) for r in sub["confusion_matrix"].dropna() if r is not None]
        if not cms:
            ax.axis("off")
            continue

        cm_sum = np.sum(cms, axis=0).astype(float)
        # normalise row-wise (true-class recall)
        row_sums = cm_sum.sum(axis=1, keepdims=True)
        cm_norm  = np.divide(cm_sum, row_sums, where=row_sums != 0)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels([c.upper() for c in CLASS_NAMES], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([c.upper() for c in CLASS_NAMES], fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(TAU_LABELS[TAU_DIRS.index(tau)], fontweight="bold")

        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val = cm_norm[i, j]
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if val > 0.55 else "black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(taus_present), len(axes)):
        axes[idx].axis("off")

    fig.savefig(PLOT_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR / 'confusion_matrices.png'}")


# ── figure 6 : per-fold scatter strip (balanced acc) ──────────────────────────

def plot_fold_scatter(df: pd.DataFrame):
    """Shows every individual fold result alongside the mean, for balanced acc."""
    metrics = ["balanced_accuracy", "macro_f1", "kappa", "auc_macro_ovr"]
    labels  = ["Balanced Accuracy", "Macro F1", "Cohen's Kappa", "AUC (macro OvR)"]

    taus_present = [t for t in TAU_DIRS if t in df["tau"].values]
    x_labels = [TAU_LABELS[TAU_DIRS.index(t)] for t in taus_present]
    x = np.arange(len(taus_present))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Per-Fold Results with Mean (individual folds shown as dots)", fontsize=13)

    for ax, metric, label in zip(axes.flat, metrics, labels):
        for j, tau in enumerate(taus_present):
            vals = df[df["tau"] == tau][metric].dropna().values
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(x[j] + jitter, vals, alpha=0.7, s=40,
                       color=PALETTE[j % len(PALETTE)], zorder=4)
            ax.plot([x[j] - 0.25, x[j] + 0.25], [vals.mean(), vals.mean()],
                    color="black", linewidth=2.5, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Score")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(PLOT_DIR / "fold_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR / 'fold_scatter.png'}")


# ── figure 7 : CSV export ──────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame):
    out = []
    for tau, label in zip(TAU_DIRS, TAU_LABELS):
        sub = df[df["tau"] == tau]
        if sub.empty:
            continue
        row = {"noise_level": label}
        for m in SCALAR_METRICS:
            vals = sub[m].dropna()
            row[f"{m}_mean"] = round(vals.mean(), 4) if len(vals) else np.nan
            row[f"{m}_std"]  = round(vals.std(),  4) if len(vals) else np.nan
        for cls in CLASS_NAMES:
            vals = sub[f"f1_{cls}"].dropna()
            row[f"f1_{cls}_mean"] = round(vals.mean(), 4) if len(vals) else np.nan
            row[f"f1_{cls}_std"]  = round(vals.std(),  4) if len(vals) else np.nan
        out.append(row)

    csv_path = PLOT_DIR / "aggregated_results.csv"
    pd.DataFrame(out).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


# ── figure 8 : clean vs noisy comparison with CIs & significance ──────────────

def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 0):
    """Return (mean, lower, upper) via percentile bootstrap."""
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo = np.percentile(boots, 100 * (1 - ci) / 2)
    hi = np.percentile(boots, 100 * (1 + ci) / 2)
    return values.mean(), lo, hi


def _significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def plot_clean_vs_noisy(df: pd.DataFrame):
    """
    For each of the four key metrics, draw a bar chart where:
      • 'clean' bar is shown in a neutral colour with its 95 % bootstrap CI
      • each noisy tau bar is coloured by severity and carries its own CI
      • a paired Wilcoxon signed-rank test (clean vs each tau, matched by fold)
        is printed and significance stars are drawn above each bar
      • the absolute Δ drop from clean is annotated inside each noisy bar
    The same-fold pairing is essential: fold_00 clean vs fold_00 tau10, etc.
    """
    metrics = ["balanced_accuracy", "macro_f1", "kappa", "auc_macro_ovr"]
    m_labels = ["Balanced Accuracy", "Macro F1", "Cohen's Kappa", "AUC (macro OvR)"]

    noisy_taus   = [t for t in TAU_DIRS if t != "clean" and t in df["tau"].values]
    noisy_labels = [TAU_LABELS[TAU_DIRS.index(t)] for t in noisy_taus]

    if "clean" not in df["tau"].values:
        print("  [skip] clean data not found – skipping clean_vs_noisy plot")
        return

    # colour ramp: light → dark orange/red for increasing noise
    noise_colours = plt.cm.YlOrRd(np.linspace(0.35, 0.85, len(noisy_taus)))
    clean_colour  = "#4C9BE8"   # blue for clean

    # ── make sure folds are aligned for pairing ──────────────────────────────
    all_folds = sorted(df["fold"].unique())

    print("\n" + "═" * 80)
    print("  WILCOXON SIGNED-RANK TEST  (paired by fold, clean vs each tau)")
    print("  H₀: no difference in metric between clean and noisy training")
    print("═" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "Clean vs. Noisy Training — 95 % Bootstrap CI & Significance\n"
        "(Wilcoxon signed-rank, paired by fold;  * p<0.05  ** p<0.01  *** p<0.001)",
        fontsize=13,
    )

    for ax, metric, m_label in zip(axes.flat, metrics, m_labels):

        # ── clean baseline ────────────────────────────────────────────────────
        clean_vals = np.array([
            df[(df["tau"] == "clean") & (df["fold"] == f)][metric].values[0]
            for f in all_folds
            if len(df[(df["tau"] == "clean") & (df["fold"] == f)][metric].values)
        ])
        c_mean, c_lo, c_hi = _bootstrap_ci(clean_vals)

        # all tau positions: clean first, then noisy
        all_taus   = ["clean"] + noisy_taus
        all_labels = ["clean"] + noisy_labels
        colours    = [clean_colour] + list(noise_colours)
        x          = np.arange(len(all_taus))
        bar_width  = 0.55

        # draw clean bar
        ax.bar(0, c_mean, width=bar_width, color=clean_colour,
               alpha=0.88, zorder=3, label="clean")
        ax.errorbar(0, c_mean,
                    yerr=[[c_mean - c_lo], [c_hi - c_mean]],
                    fmt="none", color="black", capsize=5, linewidth=1.8, zorder=5)

        # horizontal reference line at clean mean
        ax.axhline(c_mean, color=clean_colour, linewidth=1.2,
                   linestyle="--", alpha=0.55, zorder=2)

        # ── noisy bars ────────────────────────────────────────────────────────
        print(f"\n  Metric: {m_label}")
        print(f"  {'Tau':<10}  {'clean':>7}  {'noisy':>7}  {'Δ':>7}  {'p-value':>10}  sig")
        print("  " + "─" * 55)

        for j, (tau, label, colour) in enumerate(
                zip(noisy_taus, noisy_labels, noise_colours), start=1):

            noisy_vals = np.array([
                df[(df["tau"] == tau) & (df["fold"] == f)][metric].values[0]
                for f in all_folds
                if len(df[(df["tau"] == tau) & (df["fold"] == f)][metric].values)
            ])

            n_mean, n_lo, n_hi = _bootstrap_ci(noisy_vals)

            # paired Wilcoxon (requires at least one non-zero difference)
            diffs = clean_vals - noisy_vals
            if np.all(diffs == 0):
                p_val = 1.0
            else:
                try:
                    _, p_val = wilcoxon(clean_vals, noisy_vals, alternative="two-sided")
                except ValueError:
                    p_val = 1.0

            stars = _significance_stars(p_val)
            delta = n_mean - c_mean   # negative = degraded

            print(f"  {label:<10}  {c_mean:>7.4f}  {n_mean:>7.4f}  "
                  f"{delta:>+7.4f}  {p_val:>10.4f}  {stars}")

            # bar
            ax.bar(j, n_mean, width=bar_width, color=colour,
                   alpha=0.88, zorder=3)
            ax.errorbar(j, n_mean,
                        yerr=[[n_mean - n_lo], [n_hi - n_mean]],
                        fmt="none", color="black", capsize=5, linewidth=1.8, zorder=5)

            # Δ annotation inside bar (only if bar is tall enough)
            bar_top = max(n_mean, 0)
            if abs(delta) > 0.003:
                ax.text(j, bar_top * 0.5, f"Δ{delta:+.3f}",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold", zorder=6)

            # significance stars above bar
            star_y = max(n_hi, n_mean) + 0.012
            ax.text(j, star_y, stars,
                    ha="center", va="bottom", fontsize=10,
                    color="#c0392b" if stars != "ns" else "#555555",
                    fontweight="bold", zorder=6)

        # ── axes formatting ───────────────────────────────────────────────────
        y_min = df[df["tau"].isin(all_taus)][metric].min()
        y_max = df[df["tau"].isin(all_taus)][metric].max()
        y_pad = (y_max - y_min) * 0.25
        ax.set_ylim(max(0, y_min - y_pad * 0.5), y_max + y_pad)

        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, fontsize=9)
        ax.set_title(m_label, fontweight="bold")
        ax.set_ylabel("Score")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    print("═" * 80 + "\n")

    fig.savefig(PLOT_DIR / "clean_vs_noisy_significance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {PLOT_DIR / 'clean_vs_noisy_significance.png'}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\nResults root : {RESULTS_ROOT}")
    print(f"Plots output : {PLOT_DIR}\n")

    print("Loading data …")
    records, training = collect_data()

    if not records:
        print("ERROR: No data found. Check RESULTS_ROOT path.")
        return

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df)} fold-records across "
          f"{df['tau'].nunique()} noise level(s): {df['tau'].unique().tolist()}\n")

    print("Generating outputs …")
    print_summary_table(df)
    plot_metrics_vs_tau(df)
    plot_perclass_f1_heatmap(df)
    plot_training_curves(training)
    plot_confusion_matrices(df)
    plot_fold_scatter(df)
    plot_clean_vs_noisy(df)
    save_csv(df)

    print(f"\nAll done! Plots saved to: {PLOT_DIR}\n")


if __name__ == "__main__":
    main()