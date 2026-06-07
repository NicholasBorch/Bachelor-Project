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
        per_fold_mae.csv            # tidy: noise_type, tau, fold, mae, cosine
        mae_summary.csv             # ...mae_mean, ci_lo/hi, cosine_mean, cosine_ci_lo/hi
        paired_tests.csv            # MAE: tau, mean_fd, mean_norm, delta, W, p, p_holm, sig
        paired_tests_cosine.csv     # same paired test on the per-row cosine metric
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
import scripts.thesis_paired_stats as TPS
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

# Match the thesis figure typography WITHOUT LaTeX: matplotlib's native text
# renderer in a Palatino serif stack, exactly as the other figure scripts do.
# This keeps the look consistent and avoids any dependency on a LaTeX install
# (important for the headless HPC run). mathtext.fontset="cm" makes the $\tau$
# symbols render in a matching serif rather than the default sans-serif.
plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Palatino", "Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})


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


def _per_row_cosine(M: np.ndarray, human: np.ndarray) -> float:
    """Mean per-row cosine similarity between two flip-only matrices.

    Each row of a flip-only matrix is a distribution over the classes a given
    true class is confused *into*. Cosine similarity on these rows measures
    whether two noise models direct their errors toward the SAME wrong classes
    -- the SHAPE of confusion -- independently of the overall magnitude that the
    off-diagonal MAE already captures. This is the head-on answer to RQ1's
    "does it confuse in the same direction".

    Only rows that are non-degenerate (non-zero) in BOTH matrices are scored: a
    class that never flips carries no directional information. Returns the
    unweighted mean over scored rows (every class an equal vote), or NaN if no
    row qualifies. Switch to a support-weighted mean here if rare-class human
    rows should count less.
    """
    sims: list[float] = []
    for r in range(M.shape[0]):
        a = M[r]
        b = human[r]
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            continue
        sims.append(float(np.dot(a, b) / (na * nb)))
    if not sims:
        return float("nan")
    return float(np.mean(sims))


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


def _true_class_support(
    cv_root: Path, dataset: str, noise_type: str, tau: float, n_folds: int
) -> dict[str, int]:
    """Count clean (true) labels per class across all folds.

    Used only to order the heatmap axes by class size. dx_clean is the original
    label and is identical across noise types and taus, so any available split
    yields the same support.
    """
    counts = {c: 0 for c in CLASS_NAMES}
    for fold in range(n_folds):
        path = _fold_csv_path(cv_root, dataset, noise_type, tau, fold)
        if not path.exists():
            raise FileNotFoundError(f"Missing noise CSV: {path}")
        vc = pd.read_csv(path)["dx_clean"].value_counts()
        for c, n in vc.items():
            if c in counts:
                counts[c] += int(n)
    return counts


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
    order: list[str] | None = None,
) -> None:
    """Side-by-side flip-only heatmaps: human | Normalized | Feature-Driven.

    ``order`` is the list of class names to display along both axes; when given
    (e.g. ranked by class size) the matrices and tick labels are permuted to
    match. Defaults to CLASS_NAMES order. Text uses the module-level serif
    (Palatino) style so the figure matches the thesis body typography.
    """
    if order is None:
        order = list(CLASS_NAMES)
    perm = [CLASS_NAMES.index(c) for c in order]
    human = human[np.ix_(perm, perm)]
    norm = norm[np.ix_(perm, perm)]
    feat = feat[np.ix_(perm, perm)]
    labels = list(order)

    # 3 equal panels + a slim 4th column reserved for the shared colorbar, so
    # the rightmost heatmap matches the other two in width.
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(
        1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.15,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])  # dedicated colorbar axis

    mats = [human, norm, feat]
    titles = [
        "Human (all readers)",
        rf"Normalized IDN ($\tau={tau:.2f}$)",
        rf"Feature-Driven IDN ($\tau={tau:.2f}$)",
    ]
    for i, (ax, M, title) in enumerate(zip(axes, mats, titles)):
        last = i == len(axes) - 1
        sns.heatmap(
            M, annot=True, fmt=".2f", cmap="OrRd",
            xticklabels=labels, yticklabels=labels,
            vmin=0, vmax=1, ax=ax, square=True,
            cbar=last,                       # draw the colorbar once...
            cbar_ax=cax if last else None,   # ...into the dedicated axis
            cbar_kws={"label": "Flip probability"} if last else None,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 9, "color": "#333333"},
        )
        ax.set_xlabel("Flipped-to class", fontsize=9)
        ax.set_ylabel("True class" if i == 0 else "", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.tick_params(length=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", fontsize=8,
        )
    fig.suptitle("Flip-only confusion structure", fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.94, top=0.90, bottom=0.12)
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

    # ---- 1. Per-fold MAE + structural cosine -------------------------------
    per_fold_rows: list[dict] = []
    # mae_by[noise_type][tau] / cos_by[noise_type][tau] = per-fold vectors.
    mae_by: dict[str, dict[float, np.ndarray]] = {nt: {} for nt in _NOISE_TYPES}
    cos_by: dict[str, dict[float, np.ndarray]] = {nt: {} for nt in _NOISE_TYPES}

    for noise_type in _NOISE_TYPES:
        for tau in taus:
            mae_vals = np.empty(n_folds, dtype=np.float64)
            cos_vals = np.empty(n_folds, dtype=np.float64)
            for fold in range(n_folds):
                M = _per_fold_flip_matrix(cv_root, dataset, noise_type, tau, fold)
                mae = off_diagonal_mae(M, human)
                cos = _per_row_cosine(M, human)
                mae_vals[fold] = mae
                cos_vals[fold] = cos
                per_fold_rows.append({
                    "dataset": dataset, "noise_type": noise_type,
                    "tau": tau, "fold": fold, "mae": mae, "cosine": cos,
                })
            mae_by[noise_type][tau] = mae_vals
            cos_by[noise_type][tau] = cos_vals
            print(f"[rq1] {dataset:10s} {noise_type:16s} tau={tau:.2f}  "
                  f"mean MAE={mae_vals.mean():.5f}  "
                  f"mean cos={np.nanmean(cos_vals):.4f}  (n={n_folds})")

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv(out_dir / "per_fold_mae.csv", index=False)

    # ---- 2. Summary (mean + bootstrap CI) ----------------------------------
    # MAE (magnitude, lower = closer) and cosine (shape, higher = closer) are
    # summarized side by side. ci_lo/ci_hi remain the MAE CI for back-compat;
    # the cosine CI is in dedicated cosine_* columns.
    summary_rows: list[dict] = []
    for noise_type in _NOISE_TYPES:
        for tau in taus:
            mae_vals = mae_by[noise_type][tau]
            cos_vals = cos_by[noise_type][tau]
            mae_lo, mae_hi = bootstrap_ci(mae_vals, n_bootstrap=n_bootstrap,
                                          alpha=0.05, random_state=boot_seed)
            cos_lo, cos_hi = bootstrap_ci(cos_vals, n_bootstrap=n_bootstrap,
                                          alpha=0.05, random_state=boot_seed)
            summary_rows.append({
                "dataset": dataset, "noise_type": noise_type, "tau": tau,
                "mae_mean": float(mae_vals.mean()),
                "mae_std": float(mae_vals.std(ddof=0)),
                "ci_lo": mae_lo, "ci_hi": mae_hi,
                "cosine_mean": float(np.nanmean(cos_vals)),
                "cosine_std": float(np.nanstd(cos_vals)),
                "cosine_ci_lo": cos_lo, "cosine_ci_hi": cos_hi,
                "n_folds": int(len(mae_vals)),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "mae_summary.csv", index=False)

    # ---- 3. Paired Wilcoxon: Feature-Driven vs. Normalized, per tau --------
    # diff = feature_driven - normalized; negative => FD closer to humans.
    # Paired comparison via the shared thesis statistics module so RQ1 uses the
    # same machinery as Results.2/3: exact Wilcoxon + exact sign-flip
    # permutation + bootstrap CI on the paired difference + rank-biserial r,
    # with directional Holm and a Wilcoxon-vs-permutation concordance flag.
    # diff = feature_driven - normalized; lower MAE = closer to humans, so a
    # SIGNIFICANT FD advantage has direction = -1 and prints "-*", "-**", ...
    paired_rows: list[dict] = []
    for tau in taus:
        fd = np.asarray(mae_by["feature_driven"][tau], dtype=float)
        nm = np.asarray(mae_by["normalized"][tau], dtype=float)
        d = fd - nm
        res = TPS.paired_compare(d, n_boot=n_bootstrap, boot_seed=boot_seed)
        paired_rows.append({
            "dataset": dataset, "tau": tau,
            "mean_feature_driven": float(fd.mean()),
            "mean_normalized": float(nm.mean()),
            "delta_fd_minus_norm": res.delta,
            "delta_ci_lo": res.delta_ci_lo, "delta_ci_hi": res.delta_ci_hi,
            "r_rb": res.r_rb,
            "wilcoxon_W": res.W,
            "p_value": res.p_wilcoxon, "p_perm": res.p_perm,
            "direction": res.direction,
        })
    TPS.add_holm_and_flags(paired_rows, pkey_w="p_value", pkey_perm="p_perm")
    for r in paired_rows:
        r["p_value_holm"] = r["p_wilcoxon_holm"]
        r["p_perm_holm"] = r["p_perm_holm"]
        r["significant_holm"] = (not np.isnan(r["p_wilcoxon_holm"])
                                 and r["p_wilcoxon_holm"] < 0.05)
        # directional code: "-*" means FD significantly CLOSER to humans
        r["sig_code"] = r["sig"]
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(out_dir / "paired_tests.csv", index=False)

    # ---- 3b. Paired structural test: same machinery on the cosine metric ---
    # diff = feature_driven - normalized; HIGHER cosine = closer to humans, so
    # here a SIGNIFICANT FD advantage has a POSITIVE delta (direction = +1 and
    # prints "+*", "+**", ...). MAE small for the wrong reasons cannot survive
    # this: it asks whether FD confuses in the same DIRECTION as humans.
    paired_cos_rows: list[dict] = []
    for tau in taus:
        fd = np.asarray(cos_by["feature_driven"][tau], dtype=float)
        nm = np.asarray(cos_by["normalized"][tau], dtype=float)
        d = fd - nm
        res = TPS.paired_compare(d, n_boot=n_bootstrap, boot_seed=boot_seed)
        paired_cos_rows.append({
            "dataset": dataset, "tau": tau,
            "mean_feature_driven": float(fd.mean()),
            "mean_normalized": float(nm.mean()),
            "delta_fd_minus_norm": res.delta,
            "delta_ci_lo": res.delta_ci_lo, "delta_ci_hi": res.delta_ci_hi,
            "r_rb": res.r_rb,
            "wilcoxon_W": res.W,
            "p_value": res.p_wilcoxon, "p_perm": res.p_perm,
            "direction": res.direction,
        })
    TPS.add_holm_and_flags(paired_cos_rows, pkey_w="p_value", pkey_perm="p_perm")
    for r in paired_cos_rows:
        r["p_value_holm"] = r["p_wilcoxon_holm"]
        r["p_perm_holm"] = r["p_perm_holm"]
        r["significant_holm"] = (not np.isnan(r["p_wilcoxon_holm"])
                                 and r["p_wilcoxon_holm"] < 0.05)
        # directional code: "+*" means FD significantly closer in SHAPE
        r["sig_code"] = r["sig"]
    paired_cos_df = pd.DataFrame(paired_cos_rows)
    paired_cos_df.to_csv(out_dir / "paired_tests_cosine.csv", index=False)

    # ---- 4. Plots ----------------------------------------------------------
    _plot_mae_vs_tau(summary_df, out_dir / "mae_vs_tau.png", dataset)

    rep_tau = representative_tau
    if rep_tau not in taus:
        rep_tau = min(taus, key=lambda t: abs(t - representative_tau))
        print(f"[rq1] representative tau {representative_tau} not available; "
              f"using closest = {rep_tau}")
    norm_pooled = _pooled_flip_matrix(cv_root, dataset, "normalized", rep_tau, n_folds)
    feat_pooled = _pooled_flip_matrix(cv_root, dataset, "feature_driven", rep_tau, n_folds)
    # Order the heatmap axes by class size (largest first) rather than
    # alphabetically. dx_clean support is identical across noise types, so any
    # split gives the ordering; ties (e.g. balanced data) keep CLASS_NAMES order.
    support = _true_class_support(cv_root, dataset, "feature_driven", rep_tau, n_folds)
    size_order = sorted(CLASS_NAMES, key=lambda c: support.get(c, 0), reverse=True)
    triplet_path = out_dir / f"confusion_triplet_tau{int(round(rep_tau * 100)):02d}.png"
    _plot_confusion_triplet(
        human, norm_pooled, feat_pooled, rep_tau, triplet_path, dataset,
        order=size_order,
    )

    # ---- 5. Console summary ------------------------------------------------
    print(f"\n[rq1] === {dataset}: paired Feature-Driven vs. Normalized (Holm) ===")
    print("  [MAE: magnitude, lower = closer to humans]")
    for r in paired_rows:
        better = "FD closer" if r["delta_fd_minus_norm"] < 0 else "Norm closer"
        print(f"  tau={r['tau']:.2f}  FD={r['mean_feature_driven']:.4f}  "
              f"Norm={r['mean_normalized']:.4f}  Δ={r['delta_fd_minus_norm']:+.4f}  "
              f"p_holm={r['p_value_holm']:.4g} {r['sig_code']:>3s}  ({better})")
    print("  [Cosine: shape, higher = closer to humans]")
    for r in paired_cos_rows:
        better = "FD closer" if r["delta_fd_minus_norm"] > 0 else "Norm closer"
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
                      "paired_tests_cosine.csv", "mae_vs_tau.png",
                      triplet_path.name)
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
            "metric": "off_diagonal_MAE_and_per_row_cosine_flip_only_vs_tschandl_all_readers",
            "test": "paired_wilcoxon+permutation+bootstrapCI_fd_vs_norm_per_tau_holm_on_mae_and_cosine",
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
    p.add_argument("--n-bootstrap", type=int, default=10000,
                   help="Bootstrap resamples for the CIs (default 10000).")
    p.add_argument("--bootstrap-seed", type=int, default=10,
                   help="Seed for the bootstrap RNG (default 10).")
    sys.exit(main(p.parse_args()))