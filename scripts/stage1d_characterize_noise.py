"""Stage 1d: noise characterization plots and metrics.

For a given (dataset, noise_type): aggregate all noisy train CSVs across folds
at each tau, compute the confusion matrix, concentration, TVD, and distribution
shift. Save plots to results/noise_characterization/{dataset}/{noise_type}/
and numerical data to CSV.

Run: python -m scripts.stage1d_characterize_noise --dataset imbalanced --noise-type standard

Report figures
--------------
concentration_vs_tau_report.pdf
    Boxplot of max flip concentration per source class, one box per tau.
    Each box pools all (fold x class) values. Boxes are coloured along the
    YlOrBr ramp so lighter yellows map to low noise and dark browns to high.

tvd_vs_tau_report.pdf
    Mean TVD(clean class distribution, noisy class distribution) across folds
    with +/-1 SD shaded band, anchored at (0, 0).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.ham10000 import CLASS_NAMES
from src.noise.characterize import (
    class_distribution,
    concentration,
    confusion_matrix_from_labels,
    total_variation_distance,
)
from src.utils.io import ensure_dir, load_config, project_root
from src.utils.manifest import write_manifest


# ---------------------------------------------------------------------------
# Colour palettes — muted orange and pastel orange variants
# ---------------------------------------------------------------------------

# Orange is index 1 in every seaborn qualitative palette
_MUTED_ORANGE  = sns.color_palette("muted")[1]
_PASTEL_ORANGE = sns.color_palette("pastel")[1]


def _darken(rgb, factor: float = 0.55):
    """Return a darkened RGBA tuple from an RGB(A) input."""
    return (rgb[0] * factor, rgb[1] * factor, rgb[2] * factor, 1.0)


# Per-palette spec used by both plot functions
_PALETTES = {
    "muted": {
        "box":        _MUTED_ORANGE,
        "edge":       _darken(_MUTED_ORANGE, 0.58),
        "median":     _darken(_MUTED_ORANGE, 0.30),
        "line":       _darken(_MUTED_ORANGE, 0.70),
        "fill":       _MUTED_ORANGE,
        "fill_alpha": 0.25,
    },
    "pastel": {
        "box":        _PASTEL_ORANGE,
        "edge":       _darken(_PASTEL_ORANGE, 0.62),
        "median":     _darken(_PASTEL_ORANGE, 0.38),
        "line":       _darken(_PASTEL_ORANGE, 0.72),
        "fill":       _PASTEL_ORANGE,
        "fill_alpha": 0.30,
    },
}

_BASE_C = "#4472C4"   # blue dashed baseline — contrast against warm boxes


# ---------------------------------------------------------------------------
# Shared axis style
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    # LaTeX-style serif font — Computer Modern if installed, graceful fallback
    "font.family":        "serif",
    "font.serif":         ["Computer Modern Roman", "CMU Serif",
                           "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset":   "cm",
    # Sizes — match typical LaTeX figure body (~9 pt, small captions)
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    # Spines / grid
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.grid":          True,
    "grid.color":         "#E4E4E4",
    "grid.linewidth":     0.6,
    "axes.axisbelow":     True,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "axes.labelcolor":    "#333333",
    "xtick.color":        "#555555",
    "ytick.color":        "#555555",
    "text.color":         "#333333",
})


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors="#555555", labelsize=11)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _tau_dirname(tau: float) -> str:
    return f"tau_{int(round(tau * 100)):02d}"


def _aggregate_across_folds(
    cv_root: Path, dataset: str, noise_type: str, tau: float, n_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    clean_all, noisy_all = [], []
    for fold in range(n_folds):
        path = (cv_root / dataset / noise_type / _tau_dirname(tau)
                / f"fold_{fold:02d}" / "train_noisy.csv")
        if not path.exists():
            raise FileNotFoundError(f"Missing noisy fold file: {path}")
        df = pd.read_csv(path)
        clean_all.extend(df["dx_clean"].tolist())
        noisy_all.extend(df["dx"].tolist())
    return np.array(clean_all), np.array(noisy_all)


def _load_fold_labels(
    cv_root: Path, dataset: str, noise_type: str, tau: float, fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = (cv_root / dataset / noise_type / _tau_dirname(tau)
            / f"fold_{fold:02d}" / "train_noisy.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing noisy fold file: {path}")
    df = pd.read_csv(path)
    return np.array(df["dx_clean"].tolist()), np.array(df["dx"].tolist())


def _load_fold_matrix(
    cv_root: Path, dataset: str, noise_type: str, tau: float, fold: int,
) -> np.ndarray:
    clean, noisy = _load_fold_labels(cv_root, dataset, noise_type, tau, fold)
    return confusion_matrix_from_labels(clean, noisy, normalize="row")


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def _max_flip_concentration_per_class(M: np.ndarray) -> np.ndarray:
    """Max flip concentration for every source class.

    For class c: zero the diagonal, then
        conc_c = max(off-diag row) / sum(off-diag row)
    i.e. the fraction of flip mass assigned to the single most-probable
    wrong class.  Returns shape (n_classes,).
    """
    n = M.shape[0]
    out = np.empty(n)
    for c in range(n):
        row = M[c].copy()
        row[c] = 0.0
        total = row.sum()
        out[c] = row.max() / total if total > 1e-9 else 1.0
    return out


# ---------------------------------------------------------------------------
# Original (unchanged) plot helpers
# ---------------------------------------------------------------------------

def _plot_confusion(M, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "P(noisy | clean)"})
    ax.set_xlabel("Noisy label"); ax.set_ylabel("Clean label"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _plot_concentration(concentrations, out_path, noise_type):
    taus = sorted(concentrations); vals = [concentrations[t] for t in taus]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, vals, marker="o")
    ax.set_xlabel("τ"); ax.set_ylabel("Mean concentration (off-diag)")
    ax.set_title(f"Off-diagonal concentration — {noise_type}")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _plot_tvd(tvds, out_path, noise_type):
    taus = sorted(tvds); vals = [tvds[t] for t in taus]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, vals, marker="s", color="C1")
    ax.set_xlabel("τ"); ax.set_ylabel("TVD(clean, noisy)")
    ax.set_title(f"Class-distribution shift — {noise_type}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _plot_distribution_shift(clean_dist, noisy_dist, tau, out_path, noise_type):
    x = np.arange(len(CLASS_NAMES)); w = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, clean_dist, w, label="clean")
    ax.bar(x + w/2, noisy_dist, w, label="noisy")
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Frequency"); ax.set_title(f"Distribution shift — {noise_type} τ={tau}")
    ax.legend(); fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Report-quality diagnostic plots
# ---------------------------------------------------------------------------

def _plot_concentration_report(
    per_fold_class_conc: dict[float, np.ndarray],
    out_path: Path,
    noise_type: str,
    n_classes: int,
    palette: str = "muted",
) -> None:
    """Boxplot of max flip concentration per source class.

    per_fold_class_conc: tau -> array (n_folds, n_classes).
    Each box pools all fold x class values.
    palette: 'muted' or 'pastel'.
    """
    pal    = _PALETTES[palette]
    taus   = [t for t in sorted(per_fold_class_conc.keys()) if t > 0]
    n_tau  = len(taus)
    data   = [per_fold_class_conc[t].ravel() for t in taus]
    labels = [f"{t:.2f}" for t in taus]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    bp = ax.boxplot(
        data,
        positions=range(n_tau),
        widths=0.52,
        patch_artist=True,
        medianprops=dict(color=pal["median"], linewidth=2.2),
        whiskerprops=dict(color="#888888", linewidth=0.9, linestyle="-"),
        capprops=dict(color="#888888", linewidth=1.1),
        flierprops=dict(marker="o", markersize=2.5, alpha=0.40,
                        markeredgewidth=0.0),
        showfliers=True,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(pal["box"])
        patch.set_edgecolor(pal["edge"])
        patch.set_linewidth(0.9)

    for flier in bp["fliers"]:
        flier.set_markerfacecolor(pal["box"])
        flier.set_markeredgecolor("none")

    # Uniform baseline
    uniform = 1.0 / n_classes
    ax.axhline(uniform, color=_BASE_C, linestyle="--", linewidth=1.3,
               label=f"Uniform baseline (1/{n_classes})", zorder=2)

    ax.set_xticks(range(n_tau))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Noise rate τ", labelpad=6)
    ax.set_ylabel("Max flip concentration", labelpad=6)
    ax.set_title(
        f"Flip target concentration per source class — "
        f"{noise_type.replace('_', ' ')} IDN", pad=10,
    )
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="lower right")

    _style_ax(ax)
    fig.tight_layout(pad=1.6)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage1d] saved concentration boxplot → {out_path}")


def _plot_tvd_report(
    per_fold_tvd: dict[float, np.ndarray],
    out_path: Path,
    noise_type: str,
    palette: str = "muted",
) -> None:
    """TVD(clean dist, noisy dist) mean +/- 1 SD across folds, anchored at (0,0).
    palette: 'muted' or 'pastel'.
    """
    pal   = _PALETTES[palette]
    taus  = [0.0] + sorted(per_fold_tvd.keys())
    means = np.array([0.0] + [per_fold_tvd[t].mean()      for t in taus[1:]])
    stds  = np.array([0.0] + [per_fold_tvd[t].std(ddof=1) for t in taus[1:]])

    lc = pal["line"]
    fc = (*pal["fill"][:3], pal["fill_alpha"])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    ax.fill_between(
        taus,
        np.maximum(means - stds, 0),
        means + stds,
        color=fc,
        linewidth=0,
    )
    ax.plot(
        taus, means,
        color=lc, linewidth=2.2,
        marker="o", markersize=5.5,
        markerfacecolor=lc, markeredgecolor="white", markeredgewidth=0.9,
        zorder=3,
    )

    ax.set_xlabel("Noise rate τ", labelpad=6)
    ax.set_ylabel("Total Variation Distance", labelpad=6)
    ax.set_title(
        f"Distributional distortion vs noise rate — "
        f"{noise_type.replace('_', ' ')} IDN", pad=10,
    )
    ax.set_xlim(left=-0.005)
    ax.set_ylim(bottom=0)

    _style_ax(ax)
    fig.tight_layout(pad=1.6)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage1d] saved TVD report → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    cfg       = load_config("base.yaml", f"data/{args.dataset}.yaml")
    root      = project_root()
    cv_root   = root / cfg["paths"]["cv_folds"]
    out_dir   = ensure_dir(
        root / cfg["paths"]["results"]
        / "noise_characterization" / args.dataset / args.noise_type
    )
    n_folds   = int(cfg["folds"])
    n_classes = len(CLASS_NAMES)

    concentrations: dict[float, float] = {}
    tvds:           dict[float, float] = {}
    rows_for_csv = []

    per_fold_class_conc: dict[float, np.ndarray] = {}
    per_fold_tvd:        dict[float, np.ndarray] = {}

    for tau in cfg["noise_rates"]:
        tau = float(tau)

        # ---- aggregated (original outputs) ----------------------------------
        clean_labels, noisy_labels = _aggregate_across_folds(
            cv_root, args.dataset, args.noise_type, tau, n_folds,
        )
        M   = confusion_matrix_from_labels(clean_labels, noisy_labels, normalize="row")
        c   = concentration(M)
        cd  = class_distribution(clean_labels)
        nd  = class_distribution(noisy_labels)
        tvd = total_variation_distance(cd, nd)

        concentrations[tau] = c
        tvds[tau]           = tvd

        _plot_confusion(M, f"{args.noise_type} τ={tau:.2f}",
                        out_dir / f"confusion_tau{int(round(tau*100)):02d}.png")
        _plot_distribution_shift(cd, nd, tau,
                        out_dir / f"distshift_tau{int(round(tau*100)):02d}.png",
                        args.noise_type)
        np.savetxt(out_dir / f"confusion_tau{int(round(tau*100)):02d}.csv",
                   M, delimiter=",", fmt="%.6f",
                   header=",".join(CLASS_NAMES), comments="")

        rows_for_csv.append({
            "tau": tau, "concentration": c, "tvd": tvd,
            "empirical_rate": float((clean_labels != noisy_labels).mean()),
        })
        print(f"[stage1d] τ={tau:.2f}: concentration={c:.4f}, tvd={tvd:.4f}")

        # ---- per-fold (report figures) --------------------------------------
        fold_concs = np.empty((n_folds, n_classes))
        fold_tvds  = np.empty(n_folds)

        for fold in range(n_folds):
            Mf              = _load_fold_matrix(cv_root, args.dataset, args.noise_type, tau, fold)
            fold_concs[fold] = _max_flip_concentration_per_class(Mf)
            cl, nl          = _load_fold_labels(cv_root, args.dataset, args.noise_type, tau, fold)
            fold_tvds[fold] = total_variation_distance(
                class_distribution(cl), class_distribution(nl)
            )

        per_fold_class_conc[tau] = fold_concs
        per_fold_tvd[tau]        = fold_tvds
        print(
            f"          conc: median={np.median(fold_concs):.3f} "
            f"IQR=[{np.percentile(fold_concs,25):.3f},{np.percentile(fold_concs,75):.3f}] | "
            f"tvd: mean={fold_tvds.mean():.4f} std={fold_tvds.std(ddof=1):.4f}"
        )

    # ---- original summary plots ---------------------------------------------
    _plot_concentration(concentrations, out_dir / "concentration_vs_tau.png", args.noise_type)
    _plot_tvd(tvds, out_dir / "tvd_vs_tau.png", args.noise_type)

    # ---- report-quality plots -----------------------------------------------
    _plot_concentration_report(
        per_fold_class_conc,
        out_dir / "concentration_vs_tau_report.pdf",
        args.noise_type,
        n_classes=n_classes,
        palette="muted",
    )
    _plot_tvd_report(
        per_fold_tvd,
        out_dir / "tvd_vs_tau_report.pdf",
        args.noise_type,
        palette="muted",
    )

    # ---- summary CSV --------------------------------------------------------
    summary_df = pd.DataFrame(rows_for_csv).sort_values("tau")
    summary_df["conc_median"] = [float(np.median(per_fold_class_conc[t])) for t in summary_df["tau"]]
    summary_df["conc_q25"]    = [float(np.percentile(per_fold_class_conc[t], 25)) for t in summary_df["tau"]]
    summary_df["conc_q75"]    = [float(np.percentile(per_fold_class_conc[t], 75)) for t in summary_df["tau"]]
    summary_df["tvd_mean"]    = [per_fold_tvd[t].mean()      for t in summary_df["tau"]]
    summary_df["tvd_std"]     = [per_fold_tvd[t].std(ddof=1) for t in summary_df["tau"]]
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"[stage1d] wrote {out_dir / 'summary.csv'}")

    manifest_path = (
        root / cfg["paths"]["manifests"]
        / f"stage1d_{args.dataset}_{args.noise_type}.json"
    )
    write_manifest(manifest_path, stage="stage1d",
                   params={"dataset": args.dataset, "noise_type": args.noise_type},
                   outputs=[str(out_dir.relative_to(root))])
    print("[stage1d] DONE")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1d: noise characterization")
    p.add_argument("--dataset",    required=True, choices=["balanced", "imbalanced"])
    p.add_argument("--noise-type", required=True,
                   choices=["standard", "normalized", "feature_driven"])
    sys.exit(main(p.parse_args()))