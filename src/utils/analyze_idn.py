# src/utils/analyze_idn.py
#
# Loads all noise_report.json files from a CV folder and produces
# visualisations and summary statistics for one noise method.
#
# Usage:
#   python -m src.utils.analyze_idn --method standard
#   python -m src.utils.analyze_idn --method normalized
#   python -m src.utils.analyze_idn --method feature_driven
#   python -m src.utils.analyze_idn --method feature_driven_v2
#   python -m src.utils.analyze_idn --method balanced_normalized
#   python -m src.utils.analyze_idn --method balanced_feature_driven
#   python -m src.utils.analyze_idn --method balanced_standard

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.common.io import project_root

CLASS_ORDER = ["nv", "bkl", "mel", "bcc", "akiec", "vasc", "df"]
CLEAN_COLOR = "#4C72B0"
NOISY_COLOR = "#DD8452"

METHOD_ROOTS = {
    "standard":       "cv_standard",
    "normalized":     "cv_normalized",
    "feature_driven": "cv_feature_driven",
    "feature_driven_v2": "cv_feature_driven_v2",
    "balanced_normalized": "cv_balanced_normalized",
    "balanced_feature_driven": "cv_balanced_feature_driven",
    "balanced_standard": "cv_balanced_standard",
}
METHOD_LABELS = {
    "standard":       "Standard IDN",
    "normalized":     "Normalised IDN",
    "feature_driven": "Feature-Driven IDN",
    "feature_driven_v2": "Feature-Driven IDN v2",
    "balanced_normalized": "Balanced Normalised IDN",
    "balanced_feature_driven": "Balanced Feature-Driven IDN",
    "balanced_standard": "Balanced Standard IDN",
}


def _fold_from_path(report_path: Path) -> int:
    # Extract fold index from directory name, e.g. "fold_03" → 3
    return int(report_path.parent.name.split("_")[1])


def load_reports(cv_root: Path) -> pd.DataFrame:
    records = []
    for report_path in sorted(cv_root.glob("*/fold_*/noise_report.json")):
        with open(report_path) as f:
            r = json.load(f)
        fold    = _fold_from_path(report_path)
        tau     = r["tau"]
        n_train = r["n_train"]
        for cls in CLASS_ORDER:
            clean_count = r["class_counts_clean"].get(cls, 0)
            noisy_count = r["class_counts_noisy"].get(cls, 0)
            records.append({
                "tau": tau, "fold": fold, "class": cls,
                "clean_count": clean_count, "noisy_count": noisy_count,
                "ratio": noisy_count / max(clean_count, 1),
                "n_train": n_train,
            })
    return pd.DataFrame(records)


def compute_concentration(cv_root: Path) -> pd.DataFrame:
    records = []
    for report_path in sorted(cv_root.glob("*/fold_*/noise_report.json")):
        with open(report_path) as f:
            r = json.load(f)
        fold      = _fold_from_path(report_path)
        tau       = r["tau"]
        confusion = r["flip_confusion"]
        for src_class, targets in confusion.items():
            total_flips = sum(targets.values())
            if total_flips == 0:
                continue
            max_fraction = max(targets.values()) / total_flips
            top_target   = max(targets, key=targets.get)
            records.append({
                "tau": tau, "fold": fold,
                "src_class": src_class, "top_target": top_target,
                "max_fraction": max_fraction, "total_flips": total_flips,
            })
    return pd.DataFrame(records)


def compute_tvd(cv_root: Path) -> pd.DataFrame:
    records = []
    for report_path in sorted(cv_root.glob("*/fold_*/noise_report.json")):
        with open(report_path) as f:
            r = json.load(f)
        fold  = _fold_from_path(report_path)
        tau   = r["tau"]
        clean = np.array([r["class_counts_clean"].get(c, 0) for c in CLASS_ORDER], dtype=float)
        noisy = np.array([r["class_counts_noisy"].get(c, 0) for c in CLASS_ORDER], dtype=float)
        clean_freq = clean / max(clean.sum(), 1)
        noisy_freq = noisy / max(noisy.sum(), 1)
        tvd = 0.5 * np.sum(np.abs(clean_freq - noisy_freq))
        records.append({"tau": tau, "fold": fold, "tvd": tvd})
    return pd.DataFrame(records)


def compute_avg_confusion(cv_root: Path, tau: float) -> pd.DataFrame:
    matrix = pd.DataFrame(0.0, index=CLASS_ORDER, columns=CLASS_ORDER)
    count  = 0
    for report_path in sorted(cv_root.glob("*/fold_*/noise_report.json")):
        with open(report_path) as f:
            r = json.load(f)
        if abs(r["tau"] - tau) > 1e-6:
            continue
        for src in CLASS_ORDER:
            targets = r["flip_confusion"].get(src, {})
            total   = sum(targets.values())
            if total == 0:
                continue
            for tgt in CLASS_ORDER:
                matrix.loc[src, tgt] += targets.get(tgt, 0) / total
        count += 1
    if count > 0:
        matrix /= count
    return matrix


def plot_class_distribution_shift(df: pd.DataFrame, tau: float,
                                   method_label: str, save_path: Path) -> None:
    subset = df[np.isclose(df["tau"], tau)].groupby("class")[["clean_count", "noisy_count"]].mean()
    subset = subset.reindex(CLASS_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5))
    x, w = np.arange(len(CLASS_ORDER)), 0.35
    ax.bar(x - w / 2, subset["clean_count"], w, label="Clean", color=CLEAN_COLOR, alpha=0.85)
    ax.bar(x + w / 2, subset["noisy_count"], w, label=f"Noisy (τ={tau})", color=NOISY_COLOR, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER, fontsize=11)
    ax.set_ylabel("Sample count (avg. over folds)", fontsize=11)
    ax.set_title(f"Class distribution shift — {method_label} (τ={tau})", fontsize=13)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_confusion_heatmap(matrix: pd.DataFrame, tau: float,
                            method_label: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Fraction of flips"})
    ax.set_xlabel("Flip target class", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)
    ax.set_title(f"Average flip confusion — {method_label} (τ={tau})", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_tvd_over_tau(tvd_df: pd.DataFrame, method_label: str, save_path: Path) -> None:
    summary = tvd_df.groupby("tau")["tvd"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(summary["tau"], summary["mean"], marker="o", color=NOISY_COLOR, linewidth=2)
    ax.fill_between(summary["tau"],
                    summary["mean"] - summary["std"],
                    summary["mean"] + summary["std"],
                    alpha=0.2, color=NOISY_COLOR)
    ax.set_xlabel("Noise rate τ", fontsize=11)
    ax.set_ylabel("Total Variation Distance", fontsize=11)
    ax.set_title(f"Distributional distortion vs noise rate — {method_label}", fontsize=13)
    ax.set_ylim(0, None)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_concentration_over_tau(conc_df: pd.DataFrame, method_label: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    taus         = sorted(conc_df["tau"].unique())
    data_per_tau = [conc_df[np.isclose(conc_df["tau"], t)]["max_fraction"].values for t in taus]
    bp = ax.boxplot(data_per_tau, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor(NOISY_COLOR)
        patch.set_alpha(0.7)
    ax.axhline(1 / (len(CLASS_ORDER) - 1), color=CLEAN_COLOR, linestyle="--",
               linewidth=1.5, label="Uniform baseline (1/6)")
    ax.set_xticklabels([f"{t:.2f}" for t in taus], fontsize=10)
    ax.set_xlabel("Noise rate τ", fontsize=11)
    ax.set_ylabel("Max flip concentration", fontsize=11)
    ax.set_title(f"Flip target concentration — {method_label}", fontsize=13)
    ax.legend(fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def print_summary_table(df: pd.DataFrame, tvd_df: pd.DataFrame,
                         conc_df: pd.DataFrame, method_label: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {method_label} — summary per tau")
    print(f"{'tau':>6} | {'TVD mean':>10} {'TVD std':>8} | "
          f"{'Max conc. mean':>14} {'Max conc. std':>13}")
    print(f"{'-' * 70}")
    for tau in sorted(df["tau"].unique()):
        tvd_rows  = tvd_df[np.isclose(tvd_df["tau"], tau)]["tvd"]
        conc_rows = conc_df[np.isclose(conc_df["tau"], tau)]["max_fraction"]
        print(f"{tau:>6.2f} | {tvd_rows.mean():>10.4f} {tvd_rows.std():>8.4f} | "
              f"{conc_rows.mean():>14.4f} {conc_rows.std():>13.4f}")
    print(f"{'=' * 70}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["standard", "normalized", "feature_driven", "feature_driven_v2", "balanced_normalized", "balanced_feature_driven", "balanced_standard"],
        required=True,
        help="Which noise method to analyse",
    )
    args = parser.parse_args()

    cv_root      = project_root() / "data" / "processed" / "HAM10000" / METHOD_ROOTS[args.method]
    method_label = METHOD_LABELS[args.method]
    out_dir      = cv_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analysing: {method_label}")
    print(f"CV root:   {cv_root}")

    df      = load_reports(cv_root)
    tvd_df  = compute_tvd(cv_root)
    conc_df = compute_concentration(cv_root)

    if df.empty:
        print(f"No noise reports found under {cv_root}.")
        return

    taus = sorted(df["tau"].unique())
    print(f"Found tau values: {taus}")
    print(f"Found folds:      {sorted(df['fold'].unique())}")

    print_summary_table(df, tvd_df, conc_df, method_label)
    plot_tvd_over_tau(tvd_df, method_label, out_dir / "tvd_over_tau.png")
    plot_concentration_over_tau(conc_df, method_label, out_dir / "concentration_over_tau.png")

    for tau in taus:
        if np.isclose(tau, 0.0):
            continue
        tag = f"tau{int(tau * 100):02d}"
        plot_class_distribution_shift(df, tau, method_label,
                                       out_dir / f"distribution_shift_{tag}.png")
        confusion_matrix = compute_avg_confusion(cv_root, tau)
        plot_confusion_heatmap(confusion_matrix, tau, method_label,
                                out_dir / f"confusion_heatmap_{tag}.png")

    print(f"\nAll outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()