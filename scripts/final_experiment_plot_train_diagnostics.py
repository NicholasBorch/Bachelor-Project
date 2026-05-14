"""Plot per-epoch and per-class noise-label diagnostics from the main experiment.

Reads two sources:
  - training_log.jsonl per job: extracts per-epoch `train_diagnostics` blocks
    (scalar NTA / LNMR + per-class breakdowns, recorded every N epochs)
  - test_metrics.json per job: extracts end-of-training per-class breakdowns
    (these are the `per_class_*` keys merged at the end of run_training)

Outputs, under results/main_experiment/figures_and_tables/train_diagnostics/:

  per_epoch/
    nta_lnmr_curves_{method}_tau{XX}.png
        Per-epoch scalar NTA & LNMR for each fold of (method, tau).
    nta_lnmr_curves_combined_tau{XX}.png
        Same but with all 4 methods overlaid (mean across folds).

  per_class/
    per_class_nta_by_clean_{method}.png
        Heatmap: rows = classes, columns = tau values. Cells are mean
        per-class NTA across folds.
    per_class_lnmr_by_clean_{method}.png
        Same for LNMR.
    per_class_nta_by_noisy_{method}.png
        Conditioned on noisy label instead of clean (asks: "for samples
        mislabeled AS class c, how often does the model resist memorizing
        c?").
    per_class_lnmr_by_noisy_{method}.png

Usage:
    python -m scripts.final_experiment_plot_train_diagnostics
    python -m scripts.final_experiment_plot_train_diagnostics --force
    python -m scripts.final_experiment_plot_train_diagnostics \\
        --methods baseline sce elr asyco_divmix \\
        --taus 0.1 0.2 0.3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.utils.io import ensure_dir, project_root  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_METHODS = ["baseline", "sce", "elr", "asyco_divmix"]
DEFAULT_TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_FOLDS = list(range(10))
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def _tau_dirname(tau: float) -> str:
    return f"tau_{int(round(tau * 100)):02d}"


def _tau_tag(tau: float) -> str:
    return f"tau{int(round(tau * 100)):02d}"


def _fold_dirname(fold: int) -> str:
    return f"fold_{fold:02d}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _job_dir(
    base: Path, method: str, dataset: str, init: str, optim: str,
    tau: float, fold: int,
) -> Path:
    return (
        base / method / dataset / f"{init}_{optim}"
        / _tau_dirname(tau) / _fold_dirname(fold)
    )


# ============================================================================
# Per-epoch scalar NTA / LNMR curves
# ============================================================================

def _collect_per_epoch_diagnostics(
    base: Path, methods, dataset, init, optim, taus, folds,
) -> dict:
    """Return nested dict: {(method, tau): {fold: list of (epoch, nta, lnmr)}}."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for method in methods:
        for tau in taus:
            for fold in folds:
                log = _read_jsonl(
                    _job_dir(base, method, dataset, init, optim, tau, fold)
                    / "training_log.jsonl"
                )
                for record in log:
                    diag = record.get("train_diagnostics")
                    if diag is None:
                        continue
                    nta = diag.get("nta")
                    lnmr = diag.get("lnmr")
                    epoch = record.get("epoch")
                    if epoch is None:
                        continue
                    out[(method, tau)][fold].append(
                        (int(epoch), nta, lnmr)
                    )
    return out


def _plot_per_method_per_tau_curves(
    data: dict, out_dir: Path, methods, taus,
) -> int:
    """One figure per (method, tau): NTA & LNMR over epochs, one line per fold."""
    n_written = 0
    for method in methods:
        for tau in taus:
            entries = data.get((method, tau), {})
            if not entries:
                continue
            # Pull values
            all_epochs = sorted({
                e for fold_data in entries.values() for e, _, _ in fold_data
            })
            if not all_epochs:
                continue

            fig, (ax_nta, ax_lnmr) = plt.subplots(
                1, 2, figsize=(12, 4.5), sharex=True,
            )
            cmap = plt.get_cmap("tab10")
            for i, (fold, fold_data) in enumerate(sorted(entries.items())):
                ep = [e for e, _, _ in fold_data]
                ntas = [n if n is not None else float("nan")
                        for _, n, _ in fold_data]
                lnmrs = [l if l is not None else float("nan")
                         for _, _, l in fold_data]
                color = cmap(fold % 10)
                ax_nta.plot(ep, ntas, color=color, marker="o", markersize=4,
                            linewidth=1.2, label=f"fold {fold}")
                ax_lnmr.plot(ep, lnmrs, color=color, marker="o", markersize=4,
                             linewidth=1.2, label=f"fold {fold}")

            ax_nta.set_title(f"NTA — {method} (tau={tau:.1f})")
            ax_nta.set_xlabel("epoch")
            ax_nta.set_ylabel("noisy training accuracy")
            ax_nta.grid(True, alpha=0.3)
            ax_nta.set_ylim(-0.02, 1.02)

            ax_lnmr.set_title(f"LNMR — {method} (tau={tau:.1f})")
            ax_lnmr.set_xlabel("epoch")
            ax_lnmr.set_ylabel("label noise memorization rate")
            ax_lnmr.grid(True, alpha=0.3)
            ax_lnmr.set_ylim(-0.02, 1.02)

            ax_lnmr.legend(fontsize=7, loc="best", ncol=2)
            fig.tight_layout()
            out_path = (
                out_dir / f"nta_lnmr_curves_{method}_{_tau_tag(tau)}.png"
            )
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            n_written += 1
    return n_written


def _plot_combined_curves(
    data: dict, out_dir: Path, methods, taus,
) -> int:
    """One figure per tau: all methods overlaid, fold-mean."""
    n_written = 0
    method_colors = {
        "baseline":     "#777777",
        "sce":          "#1f77b4",
        "elr":          "#2ca02c",
        "asyco_divmix": "#d62728",
    }
    for tau in taus:
        fig, (ax_nta, ax_lnmr) = plt.subplots(
            1, 2, figsize=(12, 4.5), sharex=True,
        )
        any_data = False
        for method in methods:
            entries = data.get((method, tau), {})
            if not entries:
                continue
            # Collect all (epoch -> mean across folds)
            per_epoch_nta: dict[int, list[float]] = defaultdict(list)
            per_epoch_lnmr: dict[int, list[float]] = defaultdict(list)
            for fold_data in entries.values():
                for e, n, l in fold_data:
                    if n is not None:
                        per_epoch_nta[e].append(n)
                    if l is not None:
                        per_epoch_lnmr[e].append(l)
            if not per_epoch_nta:
                continue
            any_data = True
            epochs = sorted(per_epoch_nta.keys())
            nta_mean = [float(np.mean(per_epoch_nta[e])) for e in epochs]
            lnmr_mean = [float(np.mean(per_epoch_lnmr[e]))
                         if per_epoch_lnmr[e] else float("nan")
                         for e in epochs]
            nta_std = [float(np.std(per_epoch_nta[e]))
                       if len(per_epoch_nta[e]) > 1 else 0.0
                       for e in epochs]
            lnmr_std = [float(np.std(per_epoch_lnmr[e]))
                        if len(per_epoch_lnmr[e]) > 1 else 0.0
                        for e in epochs]

            color = method_colors.get(method, "black")
            ax_nta.plot(epochs, nta_mean, color=color, marker="o",
                        markersize=4, linewidth=1.6, label=method)
            ax_nta.fill_between(
                epochs,
                [m - s for m, s in zip(nta_mean, nta_std)],
                [m + s for m, s in zip(nta_mean, nta_std)],
                color=color, alpha=0.15,
            )
            ax_lnmr.plot(epochs, lnmr_mean, color=color, marker="o",
                         markersize=4, linewidth=1.6, label=method)
            ax_lnmr.fill_between(
                epochs,
                [m - s for m, s in zip(lnmr_mean, lnmr_std)],
                [m + s for m, s in zip(lnmr_mean, lnmr_std)],
                color=color, alpha=0.15,
            )

        if not any_data:
            plt.close(fig)
            continue

        ax_nta.set_title(f"NTA across methods — tau={tau:.1f}")
        ax_nta.set_xlabel("epoch")
        ax_nta.set_ylabel("noisy training accuracy")
        ax_nta.grid(True, alpha=0.3)
        ax_nta.set_ylim(-0.02, 1.02)
        ax_nta.legend(fontsize=9, loc="best")

        ax_lnmr.set_title(f"LNMR across methods — tau={tau:.1f}")
        ax_lnmr.set_xlabel("epoch")
        ax_lnmr.set_ylabel("label noise memorization rate")
        ax_lnmr.grid(True, alpha=0.3)
        ax_lnmr.set_ylim(-0.02, 1.02)
        ax_lnmr.legend(fontsize=9, loc="best")

        fig.tight_layout()
        out_path = out_dir / f"nta_lnmr_curves_combined_{_tau_tag(tau)}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        n_written += 1
    return n_written


# ============================================================================
# End-of-training per-class heatmaps
# ============================================================================

def _collect_per_class_eot(
    base: Path, methods, dataset, init, optim, taus, folds,
) -> dict:
    """Returns: {(method, conditioning, metric): np.ndarray of shape (num_taus, num_classes)}
    where each cell is the mean across folds (NaN if missing).

    conditioning in {"by_clean", "by_noisy"}; metric in {"nta", "lnmr"}.
    """
    out: dict = {}
    num_classes = len(CLASS_NAMES)
    for method in methods:
        for cond in ("by_clean", "by_noisy"):
            for metric in ("nta", "lnmr"):
                key = (method, cond, metric)
                # rows = taus, cols = classes
                matrix = np.full((len(taus), num_classes), np.nan,
                                 dtype=np.float64)
                json_key = f"per_class_{metric}_{cond}"
                for i, tau in enumerate(taus):
                    fold_vals = [[] for _ in range(num_classes)]
                    for fold in folds:
                        m = _read_json(
                            _job_dir(base, method, dataset, init, optim,
                                     tau, fold) / "test_metrics.json"
                        )
                        vals = m.get(json_key)
                        if not isinstance(vals, list):
                            continue
                        for c, v in enumerate(vals):
                            if c < num_classes and isinstance(v, (int, float)):
                                fold_vals[c].append(float(v))
                    for c in range(num_classes):
                        if fold_vals[c]:
                            matrix[i, c] = float(np.mean(fold_vals[c]))
                out[key] = matrix
    return out


def _plot_per_class_heatmaps(
    data: dict, out_dir: Path, methods, taus,
) -> int:
    """4 figures per method (nta_by_clean, lnmr_by_clean, nta_by_noisy, lnmr_by_noisy)."""
    n_written = 0
    for method in methods:
        for cond in ("by_clean", "by_noisy"):
            for metric in ("nta", "lnmr"):
                key = (method, cond, metric)
                matrix = data.get(key)
                if matrix is None or np.isnan(matrix).all():
                    continue
                fig, ax = plt.subplots(figsize=(8, 4.5))
                im = ax.imshow(matrix, aspect="auto",
                               cmap="RdYlGn" if metric == "nta" else "RdYlGn_r",
                               vmin=0.0, vmax=1.0)
                ax.set_xticks(range(len(CLASS_NAMES)))
                ax.set_xticklabels(CLASS_NAMES)
                ax.set_yticks(range(len(taus)))
                ax.set_yticklabels([f"{t:.1f}" for t in taus])
                ax.set_xlabel(
                    "class (true label)" if cond == "by_clean"
                    else "class (noisy label)"
                )
                ax.set_ylabel("tau")
                metric_long = (
                    "NTA (model -> clean)" if metric == "nta"
                    else "LNMR (model -> noisy)"
                )
                cond_long = (
                    "conditioned on TRUE class" if cond == "by_clean"
                    else "conditioned on NOISY class"
                )
                ax.set_title(f"{method} — {metric_long}, {cond_long}")
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label(metric.upper())
                # annotate cells
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        v = matrix[i, j]
                        if not np.isnan(v):
                            ax.text(j, i, f"{v:.2f}", ha="center",
                                    va="center", fontsize=8,
                                    color="black" if v > 0.5 else "white")
                fig.tight_layout()
                out_path = (
                    out_dir
                    / f"per_class_{metric}_{cond}_{method}.png"
                )
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                n_written += 1
    return n_written


# ============================================================================
# Main
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot per-epoch and per-class NTA/LNMR diagnostics",
    )
    p.add_argument("--dataset", default="imbalanced",
                   choices=["balanced", "imbalanced"])
    p.add_argument("--init", default="pretrained",
                   choices=["pretrained", "scratch"])
    p.add_argument("--optim", default="adam", choices=["sgd", "adam"])
    p.add_argument("--methods", nargs="*", default=DEFAULT_METHODS,
                   choices=DEFAULT_METHODS)
    p.add_argument("--taus", nargs="*", type=float, default=DEFAULT_TAUS)
    p.add_argument("--folds", nargs="*", type=int, default=DEFAULT_FOLDS)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--force", action="store_true",
                   help="Re-run even if output dir has content")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    root = project_root()
    base = root / "results" / "main_experiment" / "training"
    if not base.exists():
        logger.error("Results dir %s does not exist.", base)
        return 2

    output_dir = (
        args.output_dir
        or (root / "results" / "main_experiment" / "figures_and_tables"
            / "train_diagnostics")
    )
    output_dir = Path(output_dir)
    if (output_dir.exists() and any(output_dir.iterdir())
            and not args.force):
        logger.error(
            "Output dir %s already has content. Re-run with --force.",
            output_dir,
        )
        return 3
    ensure_dir(output_dir)
    per_epoch_dir = output_dir / "per_epoch"
    per_class_dir = output_dir / "per_class"
    ensure_dir(per_epoch_dir)
    ensure_dir(per_class_dir)

    logger.info("Collecting per-epoch diagnostics ...")
    per_epoch = _collect_per_epoch_diagnostics(
        base, args.methods, args.dataset, args.init, args.optim,
        args.taus, args.folds,
    )
    n_have = sum(len(v) for v in per_epoch.values())
    logger.info("  found per-epoch data for %d (method, tau, fold) combos",
                n_have)

    n_pm_pt = _plot_per_method_per_tau_curves(
        per_epoch, per_epoch_dir, args.methods, args.taus,
    )
    n_combined = _plot_combined_curves(
        per_epoch, per_epoch_dir, args.methods, args.taus,
    )
    logger.info(
        "  wrote %d per-(method, tau) and %d combined plots in %s",
        n_pm_pt, n_combined, per_epoch_dir,
    )

    logger.info("Collecting end-of-training per-class breakdowns ...")
    per_class = _collect_per_class_eot(
        base, args.methods, args.dataset, args.init, args.optim,
        args.taus, args.folds,
    )

    n_heatmaps = _plot_per_class_heatmaps(
        per_class, per_class_dir, args.methods, args.taus,
    )
    logger.info("  wrote %d per-class heatmaps in %s",
                n_heatmaps, per_class_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
