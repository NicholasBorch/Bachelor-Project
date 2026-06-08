"""
Stage 4 figures.

Uses matplotlib Agg so it runs headless. Every function takes a DataFrame in the
load_all_results schema and writes one PNG to output_path without modifying the
frame. Colors follow a single fixed per-method palette across all figures.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd 

from src.data.ham10000 import CLASS_NAMES

logger = logging.getLogger(__name__)

METHOD_ORDER: list[str] = ["baseline", "sce", "elr", "asyco", "asyco_divmix"]
METHOD_COLORS: dict[str, str] = {
    "baseline": "#555555",
    "sce": "#1f77b4",
    "elr": "#d62728",
    "asyco": "#2ca02c",
    "asyco_divmix": "#9467bd",
}
METHOD_LABELS: dict[str, str] = {
    "baseline": "Baseline (CE)",
    "sce": "SCE",
    "elr": "ELR",
    "asyco": "AsyCo",
    "asyco_divmix": "AsyCo+MixMatch",
}

# Metrics shown in the 4-panel "metrics vs τ" summary plot.
# BA (primary), Macro F1 (co-primary), Macro AUC (supportive), NTA
# (noise-label interaction diagnostic)
METRICS_FOR_TAU_PLOT: list[tuple[str, str]] = [
    ("balanced_accuracy", "Balanced accuracy"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
    ("macro_auc", "Macro AUC (OvR)"),
]


# helpers


def _filter(df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
    """Return rows where all key=value kwargs match."""
    mask = pd.Series(True, index=df.index)
    for k, v in kwargs.items():
        mask &= df[k] == v
    return df[mask]


def _methods_present(df: pd.DataFrame) -> list[str]:
    """Return METHOD_ORDER intersected with methods actually in ``df``."""
    present = set(df["method"].unique())
    return [m for m in METHOD_ORDER if m in present]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, output_path: Path) -> None:
    _ensure_parent(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", output_path)


# per-condition plots


def plot_metrics_vs_tau(
    df: pd.DataFrame,
    dataset: str,
    init: str,
    optim: str,
    output_path: Path,
) -> None:
    """Line plot of four metrics vs τ, one line per method, error bars from folds."""
    sub = _filter(df, dataset=dataset, init=init, optim=optim)
    if sub.empty:
        logger.warning(
            "No data for metrics_vs_tau %s/%s/%s; skipping.", dataset, init, optim
        )
        return

    methods = _methods_present(sub)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes = axes.flatten()

    for ax, (metric, pretty) in zip(axes, METRICS_FOR_TAU_PLOT):
        for method in methods:
            msub = sub[sub["method"] == method]
            if msub.empty or metric not in msub.columns:
                continue
            grouped = msub.groupby("tau")[metric].agg(["mean", "std"]).reset_index()
            if grouped.empty:
                continue
            ax.errorbar(
                grouped["tau"],
                grouped["mean"],
                yerr=grouped["std"],
                marker="o",
                capsize=3,
                linewidth=1.8,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        ax.set_title(pretty)
        ax.set_xlabel(r"Noise rate $\tau$")
        ax.set_ylabel(pretty)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(
        f"Metrics vs $\\tau$ — {dataset}, init={init}, optim={optim}",
        fontsize=13,
    )
    _save(fig, output_path)


def plot_noise_label_interaction(
    df: pd.DataFrame,
    dataset: str,
    init: str,
    optim: str,
    output_path: Path,
) -> None:
    """Two-panel NTA (top) and LNMR (bottom) vs tau per method; tau=0 is skipped (undefined)."""
    sub = _filter(df, dataset=dataset, init=init, optim=optim)
    if sub.empty or "nta" not in sub.columns or "lnmr" not in sub.columns:
        logger.warning(
            "No NTA/LNMR data for %s/%s/%s; skipping.", dataset, init, optim
        )
        return

    # Drop τ=0 (NTA/LNMR are NaN there by construction)
    sub = sub[sub["tau"] > 0].copy()
    if sub.empty:
        logger.warning(
            "No τ>0 rows for %s/%s/%s; NTA/LNMR plot skipped.", dataset, init, optim
        )
        return

    methods = _methods_present(sub)
    fig, (ax_nta, ax_lnmr) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for ax, metric, pretty in [
        (ax_nta, "nta", "NTA (noise transition accuracy)"),
        (ax_lnmr, "lnmr", "LNMR (label noise memorization rate)"),
    ]:
        for method in methods:
            msub = sub[sub["method"] == method]
            grouped = msub.groupby("tau")[metric].agg(["mean", "std"]).reset_index()
            grouped = grouped.dropna(subset=["mean"])
            if grouped.empty:
                continue
            ax.errorbar(
                grouped["tau"],
                grouped["mean"],
                yerr=grouped["std"],
                marker="o",
                capsize=3,
                linewidth=1.8,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        ax.set_ylabel(pretty)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=":", alpha=0.5)

    ax_lnmr.set_xlabel(r"Noise rate $\tau$")
    ax_nta.legend(loc="best", fontsize=9)
    fig.suptitle(
        f"Noise–label interaction — {dataset}, init={init}, optim={optim}",
        fontsize=13,
    )
    _save(fig, output_path)


def plot_method_comparison_bars(
    df: pd.DataFrame,
    dataset: str,
    init: str,
    optim: str,
    tau: float,
    output_path: Path,
    metric: str = "balanced_accuracy",
) -> None:
    """Grouped bar chart at a single τ: one bar per method with SEM error bars."""
    sub = _filter(df, dataset=dataset, init=init, optim=optim, tau=tau)
    if sub.empty:
        logger.warning(
            "No data for method_comparison_bars %s/%s/%s/tau=%.2f; skipping.",
            dataset, init, optim, tau,
        )
        return

    methods = _methods_present(sub)
    means, sems, labels, colors = [], [], [], []
    for method in methods:
        vals = sub[sub["method"] == method][metric].dropna().to_numpy()
        if vals.size == 0:
            continue
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0)
        labels.append(METHOD_LABELS[method])
        colors.append(METHOD_COLORS[method])

    if not means:
        logger.warning("No values for %s in bar plot; skipping.", metric)
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = np.arange(len(means))
    ax.bar(xs, means, yerr=sems, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(
        f"{metric.replace('_', ' ').title()} — {dataset}, {init}/{optim}, $\\tau={tau:.1f}$"
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    _save(fig, output_path)


def plot_per_class_f1_heatmap(
    df: pd.DataFrame,
    dataset: str,
    init: str,
    optim: str,
    tau: float,
    output_path: Path,
) -> None:
    """Heatmap of per-class F1 (7 classes × 4 methods) averaged across folds."""
    sub = _filter(df, dataset=dataset, init=init, optim=optim, tau=tau)
    if sub.empty:
        logger.warning(
            "No data for per_class_f1_heatmap %s/%s/%s/tau=%.2f; skipping.",
            dataset, init, optim, tau,
        )
        return

    methods = _methods_present(sub)
    matrix = np.full((len(CLASS_NAMES), len(methods)), np.nan)
    for j, method in enumerate(methods):
        msub = sub[sub["method"] == method]
        for i, cls in enumerate(CLASS_NAMES):
            col = f"per_class_f1_{cls}"
            if col not in msub.columns:
                continue
            vals = msub[col].dropna().to_numpy()
            if vals.size:
                matrix[i, j] = vals.mean()

    fig, ax = plt.subplots(figsize=(1.4 * len(methods) + 2, 0.7 * len(CLASS_NAMES) + 1.5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=0)
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                txt = "—"
                color = "white"
            else:
                txt = f"{val:.2f}"
                color = "white" if val < 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, label="F1")
    ax.set_title(
        f"Per-class F1 — {dataset}, {init}/{optim}, $\\tau={tau:.1f}$"
    )
    _save(fig, output_path)


# cross-condition plots


def plot_init_optim_ablation(
    df: pd.DataFrame,
    dataset: str,
    output_path: Path,
    metric: str = "balanced_accuracy",
) -> None:
    """2×2 grid of BA-vs-τ curves over the four (init, optim) combinations."""
    sub = df[df["dataset"] == dataset]
    if sub.empty:
        logger.warning("No data for init_optim_ablation dataset=%s; skipping.", dataset)
        return

    inits = ["pretrained", "scratch"]
    optims = ["sgd", "adam"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

    for i, init in enumerate(inits):
        for j, optim in enumerate(optims):
            ax = axes[i, j]
            cell = _filter(sub, init=init, optim=optim)
            if cell.empty:
                ax.set_title(f"{init} / {optim} (no data)")
                continue
            for method in _methods_present(cell):
                msub = cell[cell["method"] == method]
                grouped = msub.groupby("tau")[metric].agg(["mean", "std"]).reset_index()
                if grouped.empty:
                    continue
                ax.errorbar(
                    grouped["tau"],
                    grouped["mean"],
                    yerr=grouped["std"],
                    marker="o",
                    capsize=3,
                    linewidth=1.5,
                    label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method],
                )
            ax.set_title(f"{init} / {optim}")
            ax.grid(True, linestyle=":", alpha=0.5)
            if i == 1:
                ax.set_xlabel(r"Noise rate $\tau$")
            if j == 0:
                ax.set_ylabel(metric.replace("_", " ").title())

    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc="upper center", ncol=len(handles), fontsize=10)
    fig.suptitle(
        f"Init/optim ablation on {dataset} ({metric.replace('_', ' ')})",
        fontsize=13,
        y=1.02,
    )
    _save(fig, output_path)


def plot_dataset_comparison(
    df: pd.DataFrame,
    init: str,
    optim: str,
    output_path: Path,
    metric: str = "balanced_accuracy",
) -> None:
    """Balanced vs imbalanced side-by-side at fixed (init, optim)."""
    sub = _filter(df, init=init, optim=optim)
    if sub.empty:
        logger.warning(
            "No data for dataset_comparison %s/%s; skipping.", init, optim
        )
        return

    datasets = ["balanced", "imbalanced"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, dataset in zip(axes, datasets):
        cell = sub[sub["dataset"] == dataset]
        if cell.empty:
            ax.set_title(f"{dataset} (no data)")
            continue
        for method in _methods_present(cell):
            msub = cell[cell["method"] == method]
            grouped = msub.groupby("tau")[metric].agg(["mean", "std"]).reset_index()
            if grouped.empty:
                continue
            ax.errorbar(
                grouped["tau"],
                grouped["mean"],
                yerr=grouped["std"],
                marker="o",
                capsize=3,
                linewidth=1.8,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        ax.set_title(dataset)
        ax.set_xlabel(r"Noise rate $\tau$")
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].set_ylabel(metric.replace("_", " ").title())
    handles, labels_ = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(
        f"Dataset comparison — init={init}, optim={optim}",
        fontsize=13,
    )
    _save(fig, output_path)