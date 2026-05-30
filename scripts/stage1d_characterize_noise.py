"""Stage 4 (RQ1): does Feature-Driven IDN reproduce human confusion better
than random-projection (Normalized) IDN?

This is the RESULTS-side analysis for Research Question 1. It builds on the
characterization in stage1e/stage1f but adds the two things the thesis Results
chapter needs and the earlier stages did not produce:

    1. PER-FOLD off-diagonal MAE between each synthetic noise model and the
       human reference matrix (10 values per noise-type x tau). The earlier
       stages pooled labels across folds and produced a single MAE; the paired
       Wilcoxon test and the bootstrap confidence intervals both require the
       per-fold vector, so it is computed here from scratch.

    2. A PAIRED comparison between Feature-Driven and Normalized IDN at each
       tau (Wilcoxon signed-rank on the per-fold MAE differences), plus
       95% bootstrap CIs on each model's mean MAE. This is the correct test
       for RQ1: "is Feature-Driven's distance to the human pattern smaller,
       and is the gap real beyond fold variation?" -- NOT "is the human matrix
       inside a CI", which is a category error.

The comparison is done on FLIP-ONLY matrices: each confusion matrix has its
diagonal zeroed and every row renormalized to sum to one, so the comparison
is purely about WHERE errors are directed (the confusion structure), with the
overall flip rate removed. This matches the methodology described in
Section "Comparison with Human Annotation Patterns".

Reference matrix: data/external/tschandl_confusion_matrix.csv
    (all-readers majority-vote confusion matrix from Tschandl et al. 2019),
    same file used by stage1e/stage1f. Rows/cols MUST be in CLASS_NAMES order.

Run:
    python -m scripts.stage4_rq1_human_confusion
    python -m scripts.stage4_rq1_human_confusion --dataset imbalanced
    python -m scripts.stage4_rq1_human_confusion --representative-tau 0.2

Outputs (new folder):
    results/rq1_human_confusion/{dataset}/
        per_fold_mae.csv            # tidy: noise_type, tau, fold, mae
        mae_summary.csv             # noise_type, tau, mean, ci_lo, ci_hi, n_folds
        paired_tests.csv            # tau, mean_fd, mean_norm, delta, W, p, p_holm, sig
        mae_vs_tau.png              # line plot, one line per noise model, CI bands
        confusion_triplet_tau{NN}.png  # human | Normalized | Feature-Driven (flip-only)
        manifest .json (via write_manifest)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / HPC login nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.stats import _significance_code, _wilcoxon_safe, bootstrap_ci
from src.data.ham10000 import CLASS_NAMES, NUM_CLASSES
from src.noise.characterize import confusion_matrix_from_labels, off_diagonal_mae
from src.utils.io import ensure_dir, load_config, project_root
from src.utils.manifest import write_manifest

_DATASETS = ("balanced", "imbalanced")

# The two noise models compared against humans for RQ1. Feature-Driven is the
# proposed method; Normalized is the random-projection baseline (the strongest
# random-projection variant, used rather than Standard because Standard's
# concentration collapse makes it a trivially poor comparison).
_NOISE_TYPES = ("feature_driven", "normalized")
_NOISE_LABELS = {
    "feature_driven": "Feature-Driven IDN",
    "normalized": "Normalized IDN",
}
_NOISE_COLORS = {
    "feature_driven": "#b2182b",  # deep warm red (proposed method)
    "normalized": "#ef8a62",      # warm orange (baseline)
}

_HUMAN_MATRIX_FILE = "tschandl_confusion_matrix.csv"


# ──────────────────────────────────────────────────────────────────────────
# Matrix helpers
# ──────────────────────────────────────────────────────────────────────────
def _flip_only(M: np.ndarray) -> np.ndarray:
    """Zero the diagonal and row-renormalize so each row sums to 1.

    Produces a distribution over INCORRECT classes only. Rows whose
    off-diagonal mass is zero (a class that never flips) are left as zeros so
    they contribute nothing to the off-diagonal MAE rather than injecting a
    spurious uniform row.
    """
    M = M.astype(np.float64).copy()
    np.fill_diagonal(M, 0.0)
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    return M / row_sums


def _load_human_matrix(root: Path) -> np.ndarray:
    """Load and flip-only-normalize the human reference confusion matrix."""
    path = root / "data" / "external" / _HUMAN_MATRIX_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Human reference matrix not found at {path}. "
            "This file is committed to the repo (used by stage1e/1f) and "
            "must be present for the RQ1 analysis."
        )
    df = pd.read_csv(path)
    rows = df["true_class"].tolist()
    cols = [c for c in df.columns if c != "true_class"]
    if rows != CLASS_NAMES or cols != CLASS_NAMES:
        raise ValueError(
            f"Tschandl CSV row/column order must match CLASS_NAMES={CLASS_NAMES}. "
            f"Got rows={rows}, cols={cols}."
        )
    M = df[cols].values.astype(np.float64)
    # Row-normalize raw counts/rates first (defensive: file may be counts),
    # then convert to flip-only form so it matches the synthetic matrices.
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    M = M / row_sums
    return _flip_only(M)


def _fold_csv_path(
    cv_root: Path, dataset: str, noise_type: str, tau: float, fold: int
) -> Path:
    return (
        cv_root / dataset / noise_type
        / f"tau_{int(round(tau * 100)):02d}"
        / f"fold_{fold:02d}" / "train_noisy.csv"
    )


def _per_fold_flip_matrix(
    cv_root: Path, dataset: str, noise_type: str, tau: float, fold: int
) -> np.ndarray:
    """Flip-only confusion matrix for ONE fold."""
    path = _fold_csv_path(cv_root, dataset, noise_type, tau, fold)
    if not path.exists():
        raise FileNotFoundError(f"Missing noise CSV: {path}")
    df = pd.read_csv(path)
    M = confusion_matrix_from_labels(
        df["dx_clean"].values, df["dx"].values, normalize="row",
    )
    return _flip_only(M)


def _pooled_flip_matrix(
    cv_root: Path, dataset: str, noise_type: str, tau: float, n_folds: int
) -> np.ndarray:
    """Flip-only confusion matrix pooled over all folds (for the heatmap)."""
    clean_all: list[str] = []
    noisy_all: list[str] = []
    for fold in range(n_folds):
        path = _fold_csv_path(cv_root, dataset, noise_type, tau, fold)
        if not path.exists():
            raise FileNotFoundError(f"Missing noise CSV: {path}")
        df = pd.read_csv(path)
        clean_all.extend(df["dx_clean"].tolist())
        noisy_all.extend(df["dx"].tolist())
    M = confusion_matrix_from_labels(
        np.array(clean_all), np.array(noisy_all), normalize="row",
    )
    return _flip_only(M)


# ──────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────
def _plot_mae_vs_tau(summary: pd.DataFrame, out_path: Path, dataset: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for noise_type in _NOISE_TYPES:
        sub = summary[summary["noise_type"] == noise_type].sort_values("tau")
        if sub.empty:
            continue
        taus = sub["tau"].values
        means = sub["mae_mean"].values
        lo = sub["ci_lo"].values
        hi = sub["ci_hi"].values
        color = _NOISE_COLORS[noise_type]
        ax.plot(taus, means, marker="o", markersize=5, linewidth=2.0,
                label=_NOISE_LABELS[noise_type], color=color)
        ax.fill_between(taus, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel(r"Noise rate $\tau$")
    ax.set_ylabel("Off-diagonal MAE vs. human reference\n(flip-only, lower = closer)")
    ax.set_title(f"Alignment with human confusion patterns — {dataset}", fontsize=12)
    # Clean look: drop the box, keep only a light horizontal grid, no tick marks.
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(length=0)
    ax.set_xticks(taus)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_confusion_triplet(
    human: np.ndarray,
    norm: np.ndarray,
    feat: np.ndarray,
    tau: float,
    out_path: Path,
    dataset: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    mats = [human, norm, feat]
    titles = [
        "Human (all readers)",
        f"Normalized IDN ($\\tau={tau:.2f}$)",
        f"Feature-Driven IDN ($\\tau={tau:.2f}$)",
    ]
    for i, (ax, M, title) in enumerate(zip(axes, mats, titles)):
        last = i == len(axes) - 1
        sns.heatmap(
            M, annot=True, fmt=".2f", cmap="OrRd",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            vmin=0, vmax=1, ax=ax, square=True,
            cbar=last,  # single shared colorbar on the rightmost panel
            cbar_kws={"shrink": 0.8, "label": "Flip probability"} if last else None,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 7, "color": "#333333"},
        )
        ax.set_xlabel("Flipped-to class", fontsize=9)
        # Only the leftmost panel keeps the y-label (shared class axis).
        ax.set_ylabel("True class" if i == 0 else "", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.tick_params(length=0)  # no protruding tick marks
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.suptitle(
        f"Flip-only confusion structure — {dataset} "
        f"(diagonal removed, rows renormalized)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Multiple-testing (Holm within the per-tau family)
# ──────────────────────────────────────────────────────────────────────────
def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, preserving input order.

    NaN p-values are passed through unchanged and excluded from the family
    size used for correction.
    """
    idx_valid = [i for i, p in enumerate(pvals) if not (p is None or np.isnan(p))]
    m = len(idx_valid)
    adj = [float("nan")] * len(pvals)
    if m == 0:
        return adj
    order = sorted(idx_valid, key=lambda i: pvals[i])
    running_max = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running_max = max(running_max, val)
        adj[i] = min(running_max, 1.0)
    return adj


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def _run_dataset(
    dataset: str,
    representative_tau: float,
    n_bootstrap: int,
    boot_seed: int,
) -> dict:
    root = project_root()
    cfg = load_config("base.yaml", f"data/{dataset}.yaml")
    cv_root = root / cfg["paths"]["cv_folds"]
    n_folds = int(cfg["folds"])
    # tau=0 is clean (no flips) -> excluded from the human comparison.
    taus = [float(t) for t in cfg["noise_rates"] if float(t) > 0.0]

    out_dir = ensure_dir(root / cfg["paths"]["results"] / "rq1_human_confusion" / dataset)
    human = _load_human_matrix(root)

    # ---- 1. Per-fold MAE ----------------------------------------------------
    per_fold_rows: list[dict] = []
    # mae_by[noise_type][tau] = np.array of per-fold MAE (len n_folds)
    mae_by: dict[str, dict[float, np.ndarray]] = {nt: {} for nt in _NOISE_TYPES}

    for noise_type in _NOISE_TYPES:
        for tau in taus:
            vals = np.empty(n_folds, dtype=np.float64)
            for fold in range(n_folds):
                M = _per_fold_flip_matrix(cv_root, dataset, noise_type, tau, fold)
                mae = off_diagonal_mae(M, human)
                vals[fold] = mae
                per_fold_rows.append({
                    "dataset": dataset, "noise_type": noise_type,
                    "tau": tau, "fold": fold, "mae": mae,
                })
            mae_by[noise_type][tau] = vals
            print(f"[rq1] {dataset:10s} {noise_type:16s} tau={tau:.2f}  "
                  f"mean MAE={vals.mean():.5f}  (n={n_folds})")

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv(out_dir / "per_fold_mae.csv", index=False)

    # ---- 2. Summary (mean + bootstrap CI) ----------------------------------
    summary_rows: list[dict] = []
    for noise_type in _NOISE_TYPES:
        for tau in taus:
            vals = mae_by[noise_type][tau]
            lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap,
                                  alpha=0.05, random_state=boot_seed)
            summary_rows.append({
                "dataset": dataset, "noise_type": noise_type, "tau": tau,
                "mae_mean": float(vals.mean()),
                "mae_std": float(vals.std(ddof=0)),
                "ci_lo": lo, "ci_hi": hi, "n_folds": int(len(vals)),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "mae_summary.csv", index=False)

    # ---- 3. Paired Wilcoxon: Feature-Driven vs. Normalized, per tau --------
    # diff = feature_driven - normalized; negative => FD closer to humans.
    paired_rows: list[dict] = []
    raw_p: list[float] = []
    for tau in taus:
        fd = mae_by["feature_driven"][tau]
        nm = mae_by["normalized"][tau]
        diffs = fd - nm
        stat, p = _wilcoxon_safe(diffs)
        raw_p.append(p)
        paired_rows.append({
            "dataset": dataset, "tau": tau,
            "mean_feature_driven": float(fd.mean()),
            "mean_normalized": float(nm.mean()),
            "delta_fd_minus_norm": float(diffs.mean()),
            "wilcoxon_W": stat, "p_value": p,
        })
    holm_p = _holm(raw_p)
    for r, hp in zip(paired_rows, holm_p):
        r["p_value_holm"] = hp
        r["significant_holm"] = bool(hp < 0.05) if not np.isnan(hp) else False
        r["sig_code"] = _significance_code(hp)
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(out_dir / "paired_tests.csv", index=False)

    # ---- 4. Plots ----------------------------------------------------------
    _plot_mae_vs_tau(summary_df, out_dir / "mae_vs_tau.png", dataset)

    rep_tau = representative_tau
    if rep_tau not in taus:
        rep_tau = min(taus, key=lambda t: abs(t - representative_tau))
        print(f"[rq1] representative tau {representative_tau} not available; "
              f"using closest = {rep_tau}")
    norm_pooled = _pooled_flip_matrix(cv_root, dataset, "normalized", rep_tau, n_folds)
    feat_pooled = _pooled_flip_matrix(cv_root, dataset, "feature_driven", rep_tau, n_folds)
    triplet_path = out_dir / f"confusion_triplet_tau{int(round(rep_tau * 100)):02d}.png"
    _plot_confusion_triplet(human, norm_pooled, feat_pooled, rep_tau, triplet_path, dataset)

    # ---- 5. Console summary ------------------------------------------------
    print(f"\n[rq1] === {dataset}: paired Feature-Driven vs. Normalized (Holm) ===")
    for r in paired_rows:
        better = "FD closer" if r["delta_fd_minus_norm"] < 0 else "Norm closer"
        print(f"  tau={r['tau']:.2f}  FD={r['mean_feature_driven']:.4f}  "
              f"Norm={r['mean_normalized']:.4f}  Δ={r['delta_fd_minus_norm']:+.4f}  "
              f"p_holm={r['p_value_holm']:.4g} {r['sig_code']:>3s}  ({better})")

    return {
        "out_dir": str(out_dir.relative_to(root)),
        "n_folds": n_folds,
        "taus": taus,
        "representative_tau": rep_tau,
        "outputs": [
            str((out_dir / f).relative_to(root))
            for f in ("per_fold_mae.csv", "mae_summary.csv", "paired_tests.csv",
                      "mae_vs_tau.png", triplet_path.name)
        ],
    }


def main(args: argparse.Namespace) -> int:
    root = project_root()
    datasets = (args.dataset,) if args.dataset else _DATASETS
    all_outputs: list[str] = []
    for dataset in datasets:
        info = _run_dataset(
            dataset=dataset,
            representative_tau=float(args.representative_tau),
            n_bootstrap=int(args.n_bootstrap),
            boot_seed=int(args.bootstrap_seed),
        )
        all_outputs.extend(info["outputs"])

    manifest_path = root / load_config("base.yaml")["paths"]["manifests"] / "stage4_rq1_human_confusion.json"
    write_manifest(
        manifest_path,
        stage="stage4_rq1_human_confusion",
        params={
            "datasets": list(datasets),
            "noise_types": list(_NOISE_TYPES),
            "metric": "off_diagonal_MAE_flip_only_vs_tschandl_all_readers",
            "test": "paired_wilcoxon_feature_driven_vs_normalized_per_tau_holm",
            "n_bootstrap": int(args.n_bootstrap),
            "representative_tau": float(args.representative_tau),
        },
        outputs=all_outputs,
    )
    print(f"\n[rq1] DONE. Manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="RQ1: human-confusion alignment of Feature-Driven vs. Normalized IDN."
    )
    p.add_argument("--dataset", choices=list(_DATASETS), default=None,
                   help="Run a single dataset; default runs both.")
    p.add_argument("--representative-tau", type=float, default=0.2,
                   help="tau for the side-by-side confusion-triplet heatmap (default 0.2).")
    p.add_argument("--n-bootstrap", type=int, default=2000,
                   help="Bootstrap resamples for the CI (default 2000, matches thesis).")
    p.add_argument("--bootstrap-seed", type=int, default=0,
                   help="Seed for the bootstrap RNG (default 0).")
    sys.exit(main(p.parse_args()))