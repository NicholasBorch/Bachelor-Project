"""Results.2: does label noise meaningfully degrade the baseline?

This is the "gate" analysis. Before any noise-robust method is compared, it
must be shown that the injected noise actually corrupts learning under the
primary protocol; otherwise every downstream comparison is uninterpretable.

Primary protocol: AP = ImageNet-pretrained backbone trained with Adam
(init=pretrained, optim=adam), the clinically practical setting motivating the
thesis. Only the BASELINE method is analyzed here; the noise-robust methods and
the other three protocols are handled in Results.3-5.

What it produces:
    1. A twin-panel figure (left: balanced accuracy, right: macro F1) of the
       baseline vs. tau, each panel with 95% bootstrap CI bands across the ten
       folds and significance markers above each tau > 0 (paired Wilcoxon vs.
       the clean tau = 0 condition, Holm-corrected across the five noise rates).
    2. A body table (BA and macro F1 per tau, mean +/- CI, Holm-corrected
       p-value vs. clean).
    3. An appendix table of the full per-fold scores (reproducibility).
    4. An appendix macro-AUC-vs-tau figure (supporting metric, same treatment).

The statistical test reuses src.analysis.stats.wilcoxon_vs_clean (the per-method
tau-vs-clean noise-sensitivity test defined for the thesis). That function
returns RAW p-values; Holm correction across the five tau is applied here so the
reported significance matches the multiple-comparison policy used elsewhere.

Run:
    python -m scripts.stage5_results2_baseline_degradation
    python -m scripts.stage5_results2_baseline_degradation --dataset imbalanced
    python -m scripts.stage5_results2_baseline_degradation --init pretrained --optim adam

Outputs (new folder):
    results/results2_baseline_degradation/{dataset}/{init}_{optim}/
        baseline_metrics_per_fold.csv     # tidy: tau, fold, balanced_accuracy, macro_f1, macro_auc
        baseline_summary.csv              # tau, metric, mean, ci_lo, ci_hi, p_holm, sig
        baseline_degradation.png          # twin panel BA | macro F1
        baseline_macro_auc.png            # appendix supporting figure
        manifest .json (via write_manifest)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / HPC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.stats import bootstrap_ci, wilcoxon_vs_clean
from src.utils.io import ensure_dir, load_config, project_root
from src.utils.manifest import write_manifest

_DATASETS = ("imbalanced",)  # thesis only uses the imbalanced split
_BASELINE = "baseline"

# Metrics shown in the body twin panel, and the supporting metric (appendix).
_PRIMARY_METRICS = ("balanced_accuracy", "macro_f1")
_SUPPORTING_METRIC = "macro_auc"
_METRIC_LABELS = {
    "balanced_accuracy": "Balanced Accuracy",
    "macro_f1": "Macro F1",
    "macro_auc": "Macro AUC",
}

# Shared 4-method color palette, reused across Results.2-5 so each method has
# the SAME color in every figure. Colorblind-safe, qualitatively distinct.
_METHOD_COLORS = {
    "baseline":     "#9fb6cd",  # greyish light blue (the reference)
    "sce":          "#1b9e77",  # teal-green
    "elr":          "#d95f02",  # orange
    "asyco_divmix": "#7570b3",  # violet
}
_METHOD_LABELS = {
    "baseline":     "Baseline",
    "sce":          "SCE",
    "elr":          "ELR",
    "asyco_divmix": "AsyCo",
}


def _tau_dirname(tau: float) -> str:
    return f"tau_{int(round(tau * 100)):02d}"


def _fold_dirname(fold: int) -> str:
    return f"fold_{fold:02d}"


def _metrics_path(
    root: Path, method: str, dataset: str, init: str, optim: str, tau: float, fold: int
) -> Path:
    """Mirror the main-experiment output tree.

    The HPC submission writes each protocol into its own top-level grouping
    directory, so the protocol appears twice in the path: once right after
    ``main_experiment/`` (the per-protocol job tree) and once in the
    ``{init}_{optim}`` position. Concretely::

        results/main_experiment/{init}_{optim}/training/
            {method}/{dataset}/{init}_{optim}/tau_NN/fold_NN/test_metrics.json
    """
    protocol = f"{init}_{optim}"
    return (
        root / "results" / "main_experiment" / protocol / "training"
        / method / dataset / protocol
        / _tau_dirname(tau) / _fold_dirname(fold) / "test_metrics.json"
    )


# ──────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────
def _protocol_root(root: Path, method: str, dataset: str, init: str, optim: str) -> Path:
    """Directory that contains the tau_NN/fold_NN tree for one (method, protocol).

        results/main_experiment/{init}_{optim}/training/{method}/{dataset}/{init}_{optim}/
    """
    protocol = f"{init}_{optim}"
    return (
        root / "results" / "main_experiment" / protocol / "training"
        / method / dataset / protocol
    )


def _load_baseline_long(
    root: Path, dataset: str, init: str, optim: str, taus: list[float], n_folds: int
) -> pd.DataFrame:
    """Collect baseline per-fold scalar metrics into a tidy long DataFrame.

    Columns: dataset, init, optim, method, tau, fold,
             balanced_accuracy, macro_f1, macro_auc

    Folds and tau directories are DISCOVERED by globbing the protocol tree
    rather than generated from ``range(n_folds)``. This makes the loader
    agnostic to fold numbering (the runs use 1-indexed ``fold_01..fold_10``,
    not 0-indexed) and tolerant of a partially-complete grid. ``taus`` is used
    only to keep the noise-rate ordering consistent with the config; any tau
    found on disk that is also in the config is loaded.
    """
    proto_dir = _protocol_root(root, _BASELINE, dataset, init, optim)
    if not proto_dir.exists():
        raise FileNotFoundError(
            f"Protocol directory not found: {proto_dir.relative_to(root)}. "
            "Has the main experiment been run for the baseline under this protocol?"
        )

    # Map config tau values to their on-disk directory names, then discover folds.
    rows: list[dict] = []
    found_paths = sorted(proto_dir.glob("tau_*/fold_*/test_metrics.json"))
    if not found_paths:
        raise FileNotFoundError(
            f"No test_metrics.json under {proto_dir.relative_to(root)} "
            "(searched tau_*/fold_*/). Check that runs have completed."
        )

    # Build a lookup from tau-dirname -> float tau, restricted to config taus
    # so a stray directory cannot inject an unexpected noise rate.
    taudir_to_tau = {_tau_dirname(t): float(t) for t in taus}

    for path in found_paths:
        tau_dirname = path.parent.parent.name  # tau_NN
        fold_dirname = path.parent.name        # fold_NN
        if tau_dirname not in taudir_to_tau:
            continue  # tau not in the configured grid; skip quietly
        try:
            fold_id = int(fold_dirname.split("_")[1])
        except (IndexError, ValueError):
            continue
        with open(path) as fh:
            m = json.load(fh)
        rows.append({
            "dataset": dataset, "init": init, "optim": optim,
            "method": _BASELINE, "tau": taudir_to_tau[tau_dirname], "fold": fold_id,
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "macro_f1": float(m["macro_f1"]),
            "macro_auc": float(m["macro_auc"]),
        })

    if not rows:
        raise FileNotFoundError(
            f"Found metric files under {proto_dir.relative_to(root)} but none "
            f"matched the configured noise rates {sorted(taudir_to_tau)}."
        )

    df = pd.DataFrame(rows)

    # Report any (tau, fold) gaps relative to what is present, so a partial
    # grid is visible without crashing.
    n_per_tau = df.groupby("tau")["fold"].nunique()
    folds_seen = sorted(df["fold"].unique())
    print(f"[results2] {dataset}/{init}_{optim}: loaded {len(df)} files; "
          f"folds present = {folds_seen}; per-tau fold counts:")
    for tau in sorted(df["tau"].unique()):
        print(f"    tau={tau:.2f}: {int(n_per_tau.get(tau, 0))} folds")
    return df


# ──────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────
def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, preserving input order.

    NaN p-values pass through unchanged and are excluded from the family size.
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


def _sig_code(p: float) -> str:
    if p is None or np.isnan(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _build_summary(
    long_df: pd.DataFrame,
    taus: list[float],
    n_bootstrap: int,
    boot_seed: int,
) -> pd.DataFrame:
    """Per-tau mean + bootstrap CI per metric, plus Holm-corrected vs.-clean p.

    The vs.-clean Wilcoxon (tau > 0 against tau = 0) is computed with the
    thesis helper, then Holm-corrected across the five noise rates per metric.
    """
    all_metrics = list(_PRIMARY_METRICS) + [_SUPPORTING_METRIC]
    summary_rows: list[dict] = []

    # Wilcoxon vs clean (raw p) -> Holm, per metric.
    holm_lookup: dict[tuple[str, float], tuple[float, str]] = {}
    for metric in all_metrics:
        vc = wilcoxon_vs_clean(long_df, metric=metric)  # raw p_value per tau
        vc = vc.sort_values("tau").reset_index(drop=True)
        raw_p = vc["p_value"].tolist()
        holm_p = _holm(raw_p)
        for (_, r), hp in zip(vc.iterrows(), holm_p):
            holm_lookup[(metric, float(r["tau"]))] = (hp, _sig_code(hp))

    for metric in all_metrics:
        for tau in taus:
            vals = long_df[np.isclose(long_df["tau"], tau)][metric].dropna().to_numpy()
            if vals.size == 0:
                continue
            lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap, alpha=0.05,
                                  random_state=boot_seed)
            is_clean = np.isclose(tau, 0.0)
            hp, sig = ("nan", "---") if is_clean else holm_lookup.get(
                (metric, float(tau)), (float("nan"), "n.s.")
            )
            summary_rows.append({
                "metric": metric, "tau": float(tau),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "ci_lo": lo, "ci_hi": hi, "n_folds": int(vals.size),
                "p_holm_vs_clean": hp,
                "sig_vs_clean": sig,
            })
    return pd.DataFrame(summary_rows)


# ──────────────────────────────────────────────────────────────────────────
# Plots (bar charts)
# ──────────────────────────────────────────────────────────────────────────
def _asymmetric_err(sub: pd.DataFrame) -> np.ndarray:
    """CI bounds -> matplotlib yerr (distances from the mean): shape (2, n)."""
    mean = sub["mean"].to_numpy()
    lo = sub["ci_lo"].to_numpy()
    hi = sub["ci_hi"].to_numpy()
    lower = np.clip(mean - lo, 0, None)
    upper = np.clip(hi - mean, 0, None)
    return np.vstack([lower, upper])


def _bar_panel(ax, sub: pd.DataFrame, color: str, ylabel: str) -> None:
    """One bar chart: baseline mean per tau, asymmetric CI error bars,
    significance codes above each tau > 0 bar."""
    sub = sub.sort_values("tau")
    taus = sub["tau"].to_numpy()
    means = sub["mean"].to_numpy()
    yerr = _asymmetric_err(sub)
    x = np.arange(len(taus))
    ax.bar(
        x, means, width=0.66, color=color, edgecolor="none",
        yerr=yerr, capsize=0,
        error_kw={"elinewidth": 1.0, "ecolor": "#5a5a5a", "alpha": 0.8},
    )
    # Significance codes above the error-bar top for tau > 0.
    for xi, (_, r) in zip(x, sub.iterrows()):
        if np.isclose(r["tau"], 0.0):
            continue
        code = r["sig_vs_clean"]
        if code in ("", "---"):
            continue
        ax.annotate(code, xy=(xi, r["ci_hi"]), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=10, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1f}" for t in taus])
    ax.set_xlabel(r"Noise rate $\tau$")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel, fontsize=12)
    # Clean look: drop the box, keep only a light horizontal grid.
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(length=0)  # no tick marks, just labels
    # Headroom so significance codes are not clipped.
    ymax = float((sub["ci_hi"]).max())
    ax.set_ylim(0, min(1.0, ymax * 1.12))


def _plot_twin_panel(
    summary: pd.DataFrame, taus: list[float], out_path: Path, protocol: str
) -> None:
    color = _METHOD_COLORS[_BASELINE]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for metric, ax in (("balanced_accuracy", axes[0]), ("macro_f1", axes[1])):
        sub = summary[summary["metric"] == metric]
        if sub.empty:
            continue
        _bar_panel(ax, sub, color, _METRIC_LABELS[metric])
    fig.suptitle(f"Baseline degradation under label noise — {protocol}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_supporting(
    summary: pd.DataFrame, taus: list[float], out_path: Path, protocol: str
) -> None:
    sub = summary[summary["metric"] == _SUPPORTING_METRIC]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    _bar_panel(ax, sub, _METHOD_COLORS[_BASELINE], _METRIC_LABELS[_SUPPORTING_METRIC])
    ax.set_title(f"Baseline macro AUC under label noise — {protocol}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def _run(
    dataset: str, init: str, optim: str, n_bootstrap: int, boot_seed: int
) -> dict:
    root = project_root()
    cfg = load_config("base.yaml", f"data/{dataset}.yaml")
    taus = [float(t) for t in cfg["noise_rates"]]   # includes 0.0 here
    n_folds = int(cfg["folds"])
    protocol = f"{init}_{optim}"

    out_dir = ensure_dir(
        root / cfg["paths"]["results"] / "results2_baseline_degradation"
        / dataset / protocol
    )

    long_df = _load_baseline_long(root, dataset, init, optim, taus, n_folds)
    long_df.to_csv(out_dir / "baseline_metrics_per_fold.csv", index=False)

    summary = _build_summary(long_df, taus, n_bootstrap, boot_seed)
    summary.insert(0, "dataset", dataset)
    summary.insert(1, "protocol", protocol)
    summary.to_csv(out_dir / "baseline_summary.csv", index=False)

    twin_path = out_dir / "baseline_degradation.png"
    _plot_twin_panel(summary, taus, twin_path, protocol)
    auc_path = out_dir / "baseline_macro_auc.png"
    _plot_supporting(summary, taus, auc_path, protocol)

    # Console summary: where does degradation become significant?
    print(f"\n[results2] === {dataset} / {protocol}: baseline vs. clean (Holm) ===")
    for metric in _PRIMARY_METRICS:
        sub = summary[summary["metric"] == metric].sort_values("tau")
        clean = sub[np.isclose(sub["tau"], 0.0)]
        clean_mean = float(clean["mean"].iloc[0]) if not clean.empty else float("nan")
        print(f"  {_METRIC_LABELS[metric]} (clean = {clean_mean:.4f}):")
        for _, r in sub.iterrows():
            if np.isclose(r["tau"], 0.0):
                continue
            drop = clean_mean - r["mean"]
            p = r["p_holm_vs_clean"]
            p_str = f"{p:.4g}" if not (isinstance(p, str) or np.isnan(p)) else "n/a"
            print(f"    tau={r['tau']:.2f}  mean={r['mean']:.4f}  "
                  f"drop={drop:+.4f}  p_holm={p_str:>8s} {r['sig_vs_clean']:>4s}")

    return {
        "out_dir": str(out_dir.relative_to(root)),
        "protocol": protocol,
        "outputs": [
            str((out_dir / f).relative_to(root))
            for f in ("baseline_metrics_per_fold.csv", "baseline_summary.csv",
                      twin_path.name, auc_path.name)
        ],
    }


def main(args: argparse.Namespace) -> int:
    root = project_root()
    datasets = (args.dataset,) if args.dataset else _DATASETS  # _DATASETS = ("imbalanced",)
    all_outputs: list[str] = []
    for dataset in datasets:
        info = _run(
            dataset=dataset, init=args.init, optim=args.optim,
            n_bootstrap=int(args.n_bootstrap), boot_seed=int(args.bootstrap_seed),
        )
        all_outputs.extend(info["outputs"])

    manifest_path = (
        root / load_config("base.yaml")["paths"]["manifests"]
        / "stage5_results2_baseline_degradation.json"
    )
    write_manifest(
        manifest_path,
        stage="stage5_results2_baseline_degradation",
        params={
            "datasets": list(datasets),
            "method": _BASELINE,
            "init": args.init, "optim": args.optim,
            "primary_metrics": list(_PRIMARY_METRICS),
            "supporting_metric": _SUPPORTING_METRIC,
            "test": "paired_wilcoxon_vs_clean_per_tau_holm",
            "n_bootstrap": int(args.n_bootstrap),
        },
        outputs=all_outputs,
    )
    print(f"\n[results2] DONE. Manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Results.2: baseline degradation under label noise (primary protocol)."
    )
    p.add_argument("--dataset", choices=list(_DATASETS), default=None,
                   help="Dataset to analyze; defaults to imbalanced (the only "
                        "split used in this thesis).")
    p.add_argument("--init", default="pretrained",
                   help="Initialization of the primary protocol (default: pretrained).")
    p.add_argument("--optim", default="adam",
                   help="Optimizer of the primary protocol (default: adam). "
                        "Together with --init this selects AP by default.")
    p.add_argument("--n-bootstrap", type=int, default=2000,
                   help="Bootstrap resamples for the CI (default 2000, matches thesis).")
    p.add_argument("--bootstrap-seed", type=int, default=0,
                   help="Seed for the bootstrap RNG (default 0).")
    sys.exit(main(p.parse_args()))