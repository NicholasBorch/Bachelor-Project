# src/utils/analyze_idn_v2.py
#
# Side-by-side comparison of feature-driven IDN v1 (softmax sampling) vs
# v2 (argmax) on the full imbalanced dataset.
#
# Follows the same structure as analyze_idn.py — loads noise_report.json
# files, computes TVD, concentration, confusion matrices, and distribution
# shifts — but produces comparative plots for both variants.
#
# Usage:
#   python -m src.utils.analyze_idn_v2

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

V1_COLOR = "#4C72B0"
V2_COLOR = "#C44E52"

V1_LABEL = "v1 (softmax sampling)"
V2_LABEL = "v2 (argmax)"


# ── Data loading (mirrors analyze_idn.py) ─────────────────────────────────────

def _fold_from_path(report_path: Path) -> int:
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


def compute_flip_entropy(cv_root: Path, tau: float) -> dict:
    """Per-class entropy of flip target distribution, averaged across folds."""
    per_fold_entropy = {cls: [] for cls in CLASS_ORDER}
    for report_path in sorted(cv_root.glob("*/fold_*/noise_report.json")):
        with open(report_path) as f:
            r = json.load(f)
        if abs(r["tau"] - tau) > 1e-6:
            continue
        confusion = r["flip_confusion"]
        for cls in CLASS_ORDER:
            targets = confusion.get(cls, {})
            total = sum(targets.values())
            if total == 0:
                per_fold_entropy[cls].append(0.0)
                continue
            probs = np.array([targets.get(tgt, 0) / total for tgt in CLASS_ORDER])
            probs = probs[probs > 0]
            per_fold_entropy[cls].append(float(-np.sum(probs * np.log2(probs))))
    return {cls: float(np.mean(vals)) if vals else 0.0
            for cls, vals in per_fold_entropy.items()}


# ── Comparative plots ─────────────────────────────────────────────────────────

def plot_confusion_comparison(mat_v1: pd.DataFrame, mat_v2: pd.DataFrame,
                               tau: float, save_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    fig.suptitle(
        f"Average Flip Confusion — Feature-Driven IDN (τ={tau:.2f})\n"
        f"(mean across 10 folds, full imbalanced dataset)",
        fontsize=13,
    )

    for ax, mat, title in [(ax1, mat_v1, V1_LABEL), (ax2, mat_v2, V2_LABEL)]:
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlOrRd",
                    vmin=0, vmax=1, linewidths=0.5, ax=ax,
                    cbar_kws={"label": "Fraction of flips"})
        ax.set_xlabel("Flip target class", fontsize=10)
        ax.set_ylabel("True class", fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=11)

    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_concentration_comparison(conc_v1: pd.DataFrame, conc_v2: pd.DataFrame,
                                   save_path: Path) -> None:
    taus = sorted(set(conc_v1["tau"].unique()) | set(conc_v2["tau"].unique()))
    taus = [t for t in taus if not np.isclose(t, 0.0)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True,
                                     sharey=True)
    fig.suptitle(
        "Flip Target Concentration — Feature-Driven IDN\n"
        "(full imbalanced dataset)",
        fontsize=13,
    )

    for ax, conc_df, title, color in [
        (ax1, conc_v1, V1_LABEL, V1_COLOR),
        (ax2, conc_v2, V2_LABEL, V2_COLOR),
    ]:
        data = [conc_df[np.isclose(conc_df["tau"], t)]["max_fraction"].values for t in taus]
        bp = ax.boxplot(data, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2})
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(1 / (len(CLASS_ORDER) - 1), color="#888888", linestyle="--",
                   linewidth=1.5, label="Uniform baseline (1/6)")
        ax.set_xticklabels([f"{t:.2f}" for t in taus], fontsize=10)
        ax.set_xlabel("Noise rate τ", fontsize=11)
        ax.set_ylabel("Max flip concentration", fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.legend(fontsize=9)
        sns.despine(ax=ax)

    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_tvd_comparison(tvd_v1: pd.DataFrame, tvd_v2: pd.DataFrame,
                         save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        "Distributional Distortion vs Noise Rate — Feature-Driven IDN\n"
        "(full imbalanced dataset)",
        fontsize=13,
    )

    for tvd_df, label, color in [
        (tvd_v1, V1_LABEL, V1_COLOR),
        (tvd_v2, V2_LABEL, V2_COLOR),
    ]:
        summary = tvd_df.groupby("tau")["tvd"].agg(["mean", "std"]).reset_index()
        ax.plot(summary["tau"], summary["mean"], marker="o", color=color,
                linewidth=2, label=label)
        ax.fill_between(summary["tau"],
                        summary["mean"] - summary["std"],
                        summary["mean"] + summary["std"],
                        alpha=0.15, color=color)

    ax.set_xlabel("Noise rate τ", fontsize=11)
    ax.set_ylabel("Total Variation Distance", fontsize=11)
    ax.set_ylim(0, None)
    ax.legend(fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_entropy_comparison(v1_root: Path, v2_root: Path, taus: list,
                             save_path: Path) -> None:
    n_taus = len(taus)
    fig, axes = plt.subplots(1, n_taus, figsize=(3.5 * n_taus, 5),
                              constrained_layout=True, sharey=True)
    if n_taus == 1:
        axes = [axes]
    fig.suptitle(
        "Flip Target Entropy by True Class — Feature-Driven IDN\n"
        "(mean across 10 folds, full imbalanced dataset)",
        fontsize=13,
    )

    x     = np.arange(len(CLASS_ORDER))
    width = 0.35

    for ax, tau in zip(axes, taus):
        ent_v1 = compute_flip_entropy(v1_root, tau)
        ent_v2 = compute_flip_entropy(v2_root, tau)

        vals_v1 = [ent_v1[cls] for cls in CLASS_ORDER]
        vals_v2 = [ent_v2[cls] for cls in CLASS_ORDER]

        ax.bar(x - width / 2, vals_v1, width, label=V1_LABEL,
               color=V1_COLOR, alpha=0.85)
        ax.bar(x + width / 2, vals_v2, width, label=V2_LABEL,
               color=V2_COLOR, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in CLASS_ORDER],
                           fontsize=8, rotation=45, ha="right")
        ax.set_title(f"τ={tau:.2f}", fontweight="bold", fontsize=10)
        ax.set_ylabel("Entropy (bits)" if ax == axes[0] else "")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        sns.despine(ax=ax)

    axes[-1].legend(fontsize=9, loc="upper right")

    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary_table(df_v1: pd.DataFrame, df_v2: pd.DataFrame,
                         tvd_v1: pd.DataFrame, tvd_v2: pd.DataFrame,
                         conc_v1: pd.DataFrame, conc_v2: pd.DataFrame) -> None:
    taus = sorted(set(df_v1["tau"].unique()) | set(df_v2["tau"].unique()))
    taus = [t for t in taus if not np.isclose(t, 0.0)]

    print(f"\n{'=' * 90}")
    print(f"  Feature-Driven IDN — v1 vs v2 summary per tau")
    print(f"{'=' * 90}")
    print(f"{'tau':>6} | {'TVD v1':>8} {'TVD v2':>8} | "
          f"{'Conc v1':>8} {'Conc v2':>8} | "
          f"{'Flipped v1':>10} {'Flipped v2':>10}")
    print(f"{'-' * 90}")

    for tau in taus:
        tvd_1 = tvd_v1[np.isclose(tvd_v1["tau"], tau)]["tvd"]
        tvd_2 = tvd_v2[np.isclose(tvd_v2["tau"], tau)]["tvd"]
        con_1 = conc_v1[np.isclose(conc_v1["tau"], tau)]["max_fraction"]
        con_2 = conc_v2[np.isclose(conc_v2["tau"], tau)]["max_fraction"]

        sub_v1 = df_v1[np.isclose(df_v1["tau"], tau)]
        sub_v2 = df_v2[np.isclose(df_v2["tau"], tau)]
        flip_v1 = sub_v1.groupby("fold").apply(
            lambda g: (g["noisy_count"].sum() - g["clean_count"].sum())
        ).abs().mean() if not sub_v1.empty else 0
        flip_v2 = sub_v2.groupby("fold").apply(
            lambda g: (g["noisy_count"].sum() - g["clean_count"].sum())
        ).abs().mean() if not sub_v2.empty else 0

        print(f"{tau:>6.2f} | {tvd_1.mean():>8.4f} {tvd_2.mean():>8.4f} | "
              f"{con_1.mean():>8.4f} {con_2.mean():>8.4f} | "
              f"{flip_v1:>10.0f} {flip_v2:>10.0f}")

    print(f"{'=' * 90}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    v1_root = project_root() / "data" / "processed" / "HAM10000" / "cv_feature_driven"
    v2_root = project_root() / "data" / "processed" / "HAM10000" / "cv_feature_driven_v2"
    out_dir = project_root() / "results" / "HAM10000" / "analysis" / "feature_driven_v2_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not v1_root.exists():
        raise FileNotFoundError(f"v1 directory not found: {v1_root}")
    if not v2_root.exists():
        raise FileNotFoundError(f"v2 directory not found: {v2_root}")

    print(f"{'=' * 60}")
    print(f"  Feature-Driven IDN: v1 vs v2 comparison")
    print(f"  v1: {v1_root}")
    print(f"  v2: {v2_root}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")

    # Load data
    df_v1   = load_reports(v1_root)
    df_v2   = load_reports(v2_root)
    tvd_v1  = compute_tvd(v1_root)
    tvd_v2  = compute_tvd(v2_root)
    conc_v1 = compute_concentration(v1_root)
    conc_v2 = compute_concentration(v2_root)

    if df_v1.empty or df_v2.empty:
        print("ERROR: No noise reports found in one or both directories.")
        return

    taus = sorted(df_v1["tau"].unique())
    taus_noisy = [t for t in taus if not np.isclose(t, 0.0)]
    print(f"  Found tau values: {taus}")

    # Summary table
    print_summary_table(df_v1, df_v2, tvd_v1, tvd_v2, conc_v1, conc_v2)

    # TVD comparison
    plot_tvd_comparison(tvd_v1, tvd_v2, out_dir / "tvd_comparison.png")

    # Concentration comparison
    plot_concentration_comparison(conc_v1, conc_v2,
                                  out_dir / "concentration_comparison.png")

    # Entropy comparison
    plot_entropy_comparison(v1_root, v2_root, taus_noisy,
                            out_dir / "flip_entropy_comparison.png")

    # Per-tau confusion heatmaps
    for tau in taus_noisy:
        tag    = f"tau{int(tau * 100):02d}"
        mat_v1 = compute_avg_confusion(v1_root, tau)
        mat_v2 = compute_avg_confusion(v2_root, tau)
        plot_confusion_comparison(mat_v1, mat_v2, tau,
                                  out_dir / f"confusion_comparison_{tag}.png")

    print(f"\nAll outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
