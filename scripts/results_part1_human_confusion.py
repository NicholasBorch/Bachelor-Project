"""
Results.1 (RQ1): does Feature-Driven IDN reproduce human confusion better than random-projection (Normalized) IDN?

Per tau and fold, each synthetic flip-only confusion matrix (diagonal zeroed,
rows renormalized) is compared to the Tschandl all-readers human reference by
off-diagonal MAE (magnitude) and mean per-row cosine (shape) (Not used in the final thesis). Feature-Driven
and Normalized IDN are compared per tau with paired Wilcoxon + permutation +
bootstrap CIs and Holm correction, plus a side-by-side confusion triplet at a
representative tau.

Reference: data/external/tschandl_confusion_matrix.csv (rows/cols in
CLASS_NAMES order).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg") 
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

_DATASET = "imbalanced"

# Feature-Driven (proposed) vs Normalized (random-projection baseline).
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

# Serif (Palatino) figure style, no LaTeX.
plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Palatino", "Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})


def _flip_only(M: np.ndarray) -> np.ndarray:
    """Zero the diagonal and row-renormalize so each row sums to 1."""
    M = M.astype(np.float64).copy()
    np.fill_diagonal(M, 0.0)
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    return M / row_sums


def _per_row_cosine(M: np.ndarray, human: np.ndarray) -> float:
    """Mean per-row cosine similarity between two flip-only matrices."""
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
            "This file is committed to the repo and "
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
    # Row-normalize (file may be counts), then flip-only.
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
    """Count clean (true) labels per class across all folds (for axis ordering)."""
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
    # light horizontal grid, no box or ticks
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


def _plot_single_confusion(
    M: np.ndarray,
    title: str,
    out_path: Path,
    labels: list[str],
) -> None:
    """One flip-only confusion heatmap saved as a standalone figure."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        M, annot=True, fmt=".2f", cmap="OrRd",
        xticklabels=labels, yticklabels=labels,
        vmin=0, vmax=1, ax=ax, square=True,
        cbar=True, cbar_kws={"label": "Flip probability"},
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 9, "color": "#333333"},
    )
    ax.set_xlabel("Flipped-to class", fontsize=9)
    ax.set_ylabel("True class", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.tick_params(length=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
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
    """Side-by-side flip-only heatmaps: human | Normalized | Feature-Driven."""
    if order is None:
        order = list(CLASS_NAMES)
    perm = [CLASS_NAMES.index(c) for c in order]
    human = human[np.ix_(perm, perm)]
    norm = norm[np.ix_(perm, perm)]
    feat = feat[np.ix_(perm, perm)]
    labels = list(order)

    # 3 panels + a slim 4th column for the shared colorbar
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(
        1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.15,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

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
            cbar=last,
            cbar_ax=cax if last else None,
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

    # also save each panel standalone
    for M, title, suffix in zip(
        mats, titles, ("human", "normalized", "feature_driven"),
    ):
        single_path = out_path.with_name(f"{out_path.stem}_{suffix}.png")
        _plot_single_confusion(M, title, single_path, labels)


def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, preserving input order."""
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
    # tau=0 is clean; excluded
    taus = [float(t) for t in cfg["noise_rates"] if float(t) > 0.0]

    out_dir = ensure_dir(root / cfg["paths"]["results"] / "human_confusion" / dataset)
    human = _load_human_matrix(root)

    # per-fold MAE + cosine
    per_fold_rows: list[dict] = []
    # [noise_type][tau] -> per-fold vectors
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

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv(out_dir / "per_fold_mae.csv", index=False)

    # summary: per-tau mean + bootstrap CI
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

    # paired Wilcoxon: FD vs Normalized (diff = FD - norm, negative => FD closer)
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
        # "-*" = FD significantly closer
        r["sig_code"] = r["sig"]
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(out_dir / "paired_tests.csv", index=False)

    # same paired test on the cosine metric (higher cosine = closer)
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
        # "+*" = FD significantly closer in shape
        r["sig_code"] = r["sig"]
    paired_cos_df = pd.DataFrame(paired_cos_rows)
    paired_cos_df.to_csv(out_dir / "paired_tests_cosine.csv", index=False)

    # plots
    _plot_mae_vs_tau(summary_df, out_dir / "mae_vs_tau.png", dataset)

    rep_tau = representative_tau
    if rep_tau not in taus:
        rep_tau = min(taus, key=lambda t: abs(t - representative_tau))
        print(f"[rq1] representative tau {representative_tau} not available; "
              f"using closest = {rep_tau}")
    norm_pooled = _pooled_flip_matrix(cv_root, dataset, "normalized", rep_tau, n_folds)
    feat_pooled = _pooled_flip_matrix(cv_root, dataset, "feature_driven", rep_tau, n_folds)
    # order heatmap axes by class size (largest first)
    support = _true_class_support(cv_root, dataset, "feature_driven", rep_tau, n_folds)
    size_order = sorted(CLASS_NAMES, key=lambda c: support.get(c, 0), reverse=True)
    triplet_path = out_dir / f"confusion_triplet_tau{int(round(rep_tau * 100)):02d}.png"
    _plot_confusion_triplet(
        human, norm_pooled, feat_pooled, rep_tau, triplet_path, dataset,
        order=size_order,
    )

    triplet_singles = [
        triplet_path.with_name(f"{triplet_path.stem}_{s}.png").name
        for s in ("human", "normalized", "feature_driven")
    ]
    return {
        "out_dir": str(out_dir.relative_to(root)),
        "n_folds": n_folds,
        "taus": taus,
        "representative_tau": rep_tau,
        "outputs": [
            str((out_dir / f).relative_to(root))
            for f in ("per_fold_mae.csv", "mae_summary.csv", "paired_tests.csv",
                      "paired_tests_cosine.csv", "mae_vs_tau.png",
                      triplet_path.name, *triplet_singles)
        ],
    }


def main(args: argparse.Namespace) -> int:
    root = project_root()
    info = _run_dataset(
        dataset=_DATASET,
        representative_tau=float(args.representative_tau),
        n_bootstrap=int(args.n_bootstrap),
        boot_seed=int(args.bootstrap_seed),
    )
    all_outputs = info["outputs"]

    manifest_path = root / load_config("base.yaml")["paths"]["manifests"] / "stage4_rq1_human_confusion.json"
    write_manifest(
        manifest_path,
        stage="stage4_rq1_human_confusion",
        params={
            "dataset": _DATASET,
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
    p.add_argument("--representative-tau", type=float, default=0.2,
                   help="tau for the side-by-side confusion-triplet heatmap (default 0.2).")
    p.add_argument("--n-bootstrap", type=int, default=10000,
                   help="Bootstrap resamples for the CIs (default 10000).")
    p.add_argument("--bootstrap-seed", type=int, default=10,
                   help="Seed for the bootstrap RNG (default 10).")
    sys.exit(main(p.parse_args()))