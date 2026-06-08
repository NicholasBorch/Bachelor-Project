"""
Results.2 (gate analysis): does label noise degrade the baseline under the
primary protocol (AP = pretrained + Adam)?

Only the baseline method is analysed. For each tau, computes per-fold balanced
accuracy / macro F1 / macro AUC, a paired Wilcoxon vs. the clean (tau=0)
condition (Holm-corrected across noise rates), and bootstrap CIs. Writes a tidy
per-fold CSV, a summary CSV, and a degradation line figure into
results/baseline_degradation/{dataset}/{init}_{optim}/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Palatino", "Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset":   "cm",      
    "axes.unicode_minus": False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

from src.analysis.stats import bootstrap_ci, wilcoxon_vs_clean
import scripts.thesis_paired_stats as TPS
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

# Shared 4-method palette, reused across Results.2-5; colorblind-safe.
_METHOD_COLORS = {
    "baseline":     "#9ec9e2",  # light blue, matched to Results.3 palette
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

# Display labels for protocol titles only; the {init}_{optim} string is unchanged.
_PROTOCOL_LABELS = {
    "pretrained_adam": "Protocol AP",
    "pretrained_sgd":  "Protocol SP",
    "scratch_adam":    "Protocol AS",
    "scratch_sgd":     "Protocol SS",
}


def _protocol_label(protocol: str) -> str:
    return _PROTOCOL_LABELS.get(protocol, protocol)


def _tau_dirname(tau: float) -> str:
    return f"tau_{int(round(tau * 100)):02d}"


def _fold_dirname(fold: int) -> str:
    return f"fold_{fold:02d}"


def _metrics_path(
    root: Path, method: str, dataset: str, init: str, optim: str, tau: float, fold: int
) -> Path:
    """Path to one job's test_metrics.json in the per-protocol main-experiment tree."""
    protocol = f"{init}_{optim}"
    # Tree: results/main_experiment/{protocol}/training/{method}/tau_NN/fold_NN/
    return (
        root / "results" / "main_experiment" / protocol / "training"
        / method
        / _tau_dirname(tau) / _fold_dirname(fold) / "test_metrics.json"
    )


# Data loading
def _protocol_root(root: Path, method: str, dataset: str, init: str, optim: str) -> Path:
    """Directory containing the tau_NN/fold_NN tree for one (method, protocol)."""
    protocol = f"{init}_{optim}"
    return (
        root / "results" / "main_experiment" / protocol / "training"
        / method
    )


def _load_baseline_long(
    root: Path, dataset: str, init: str, optim: str, taus: list[float], n_folds: int
) -> pd.DataFrame:
    """Collect baseline per-fold scalar metrics into a tidy long DataFrame."""
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

    # Map tau-dirname -> float tau, restricted to config taus.
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

    # Report per-tau fold counts so a partial grid is visible.
    n_per_tau = df.groupby("tau")["fold"].nunique()
    folds_seen = sorted(df["fold"].unique())
    print(f"[results2] {dataset}/{init}_{optim}: loaded {len(df)} files; "
          f"folds present = {folds_seen}; per-tau fold counts:")
    for tau in sorted(df["tau"].unique()):
        print(f"    tau={tau:.2f}: {int(n_per_tau.get(tau, 0))} folds")
    return df


# Statistics
def _holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, preserving input order (NaNs excluded)."""
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
    """Per-tau mean + bootstrap CI per metric, plus Holm-corrected vs-clean p."""
    all_metrics = list(_PRIMARY_METRICS) + [_SUPPORTING_METRIC]
    summary_rows: list[dict] = []

    # vs-clean test (diff = noisy - clean): paired Wilcoxon + permutation + bootstrap CI, directional Holm.
    holm_lookup: dict[tuple[str, float], tuple[float, str]] = {}
    extra_lookup: dict[tuple[str, float], dict] = {}
    taus_nz = [t for t in taus if not np.isclose(t, 0.0)]
    for metric in all_metrics:
        clean = (long_df[np.isclose(long_df["tau"], 0.0)]
                 .set_index("fold")[metric])
        block = []
        for tau in taus_nz:
            noisy = (long_df[np.isclose(long_df["tau"], tau)]
                     .set_index("fold")[metric])
            paired = pd.concat([clean.rename("clean"),
                                noisy.rename("noisy")], axis=1).dropna()
            d = paired["noisy"].to_numpy() - paired["clean"].to_numpy()
            res = TPS.paired_compare(d, n_boot=n_bootstrap, boot_seed=boot_seed)
            block.append(dict(metric=metric, tau=float(tau), **res.as_dict()))
        TPS.add_holm_and_flags(block)
        for rec in block:
            key = (metric, rec["tau"])
            holm_lookup[key] = (rec["p_wilcoxon_holm"], rec["sig"])
            extra_lookup[key] = rec

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
            ex = extra_lookup.get((metric, float(tau)), {})
            summary_rows.append({
                "metric": metric, "tau": float(tau),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "ci_lo": lo, "ci_hi": hi, "n_folds": int(vals.size),
                "p_holm_vs_clean": hp,
                "sig_vs_clean": sig,
                "delta_vs_clean": ex.get("delta", float("nan")),
                "delta_ci_lo": ex.get("delta_ci_lo", float("nan")),
                "delta_ci_hi": ex.get("delta_ci_hi", float("nan")),
                "r_rb": ex.get("r_rb", float("nan")),
                "p_perm_vs_clean": ex.get("p_perm", float("nan")),
                "p_perm_holm_vs_clean": ex.get("p_perm_holm", float("nan")),
                "concordant": ex.get("concordant", True),
            })
    return pd.DataFrame(summary_rows)


# Plot: all three metrics degrading under increasing noise
# Per-metric line colours. Colourblind-safe, qualitatively distinct
_METRIC_LINE_COLORS = {
    "balanced_accuracy": "#0072B2",  # blue
    "macro_f1":          "#D55E00",  # vermillion
    "macro_auc":         "#009E73",  # green
}
_METRIC_MARKERS = {
    "balanced_accuracy": "o",
    "macro_f1":          "s",
    "macro_auc":         "^",
}


def _metric_curve(ax, sub: pd.DataFrame, metric: str) -> None:
    """Plot one metric's mean-vs-tau line with a bootstrap CI band and significance codes."""
    sub = sub.sort_values("tau")
    taus = sub["tau"].to_numpy()
    means = sub["mean"].to_numpy()
    lo = sub["ci_lo"].to_numpy()
    hi = sub["ci_hi"].to_numpy()
    color = _METRIC_LINE_COLORS[metric]

    ax.fill_between(taus, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=2)
    ax.plot(
        taus, means, color=color, marker=_METRIC_MARKERS[metric],
        markersize=5.5, linewidth=1.8, label=_METRIC_LABELS[metric], zorder=3,
    )
    # Significance codes above the CI band for tau > 0.
    for _, r in sub.iterrows():
        if np.isclose(r["tau"], 0.0):
            continue
        code = r["sig_vs_clean"]
        if code in ("", "---"):
            continue
        ax.annotate(code, xy=(r["tau"], r["ci_hi"]), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=color)


def _plot_degradation(
    summary: pd.DataFrame, taus: list[float], out_path: Path, protocol: str
) -> None:
    """Single figure: all three metrics' mean (with bootstrap CI bands) vs tau."""
    metrics = list(_PRIMARY_METRICS) + [_SUPPORTING_METRIC]
    # Only the tau values actually present in the summary, in order.
    taus_present = sorted(float(t) for t in summary["tau"].unique())

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for metric in metrics:
        sub = summary[summary["metric"] == metric]
        if sub.empty:
            continue
        _metric_curve(ax, sub, metric)

    ax.set_xlabel(r"Noise rate $\tau$")
    ax.set_ylabel("Score")
    ax.set_title(f"Baseline degradation under label noise — {_protocol_label(protocol)}", fontsize=12)

    # X-axis ticks only at tau values we actually have data for (no in-between).
    ax.set_xticks(taus_present)
    ax.set_xticklabels([f"{t:.1f}" for t in taus_present])
    # Pin the left (y) axis at tau = 0 rather than leaving padding to its left.
    ax.set_xlim(0.0, max(taus_present) + 0.02)
    ax.spines["left"].set_position(("data", 0.0))

    # Y ticks at 0.1 intervals; axis bottom stays at 0.
    ax.set_yticks(np.arange(0.1, 1.0 + 1e-9, 0.1))
    ax.set_ylim(0.0, 1.0)

    # Clean look: drop top/right spines, keep bottom + left (x/y axes), light grid.
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#cccccc")
    ax.tick_params(length=0)

    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# Main
def _run(
    dataset: str, init: str, optim: str, n_bootstrap: int, boot_seed: int
) -> dict:
    root = project_root()
    cfg = load_config("base.yaml", f"data/{dataset}.yaml")
    taus = [float(t) for t in cfg["noise_rates"]]   # includes 0.0 here
    n_folds = int(cfg["folds"])
    protocol = f"{init}_{optim}"

    out_dir = ensure_dir(
        root / cfg["paths"]["results"] / "baseline_degradation"
        / dataset / protocol
    )

    long_df = _load_baseline_long(root, dataset, init, optim, taus, n_folds)
    long_df.to_csv(out_dir / "baseline_metrics_per_fold.csv", index=False)

    summary = _build_summary(long_df, taus, n_bootstrap, boot_seed)
    summary.insert(0, "dataset", dataset)
    summary.insert(1, "protocol", protocol)
    summary.to_csv(out_dir / "baseline_summary.csv", index=False)

    fig_path = out_dir / "baseline_degradation.png"
    _plot_degradation(summary, taus, fig_path, protocol)

    # console summary
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
                      fig_path.name)
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
            "test": "paired_wilcoxon+permutation+bootstrapCI_vs_clean_per_tau_holm",
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
    p.add_argument("--n-bootstrap", type=int, default=10000,
                   help="Bootstrap resamples for the CI (default 10000, matches thesis).")
    p.add_argument("--bootstrap-seed", type=int, default=10,
                   help="Seed for the bootstrap RNG (default 10).")
    sys.exit(main(p.parse_args()))