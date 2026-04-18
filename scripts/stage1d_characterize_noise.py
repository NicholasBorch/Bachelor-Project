"""Stage 1d: noise characterization plots and metrics.

For a given (dataset, noise_type): aggregate all noisy train CSVs across folds
at each tau, compute the confusion matrix, concentration, TVD, and distribution
shift. Save plots to results/noise_characterization/{dataset}/{noise_type}/
and numerical data to CSV.

Run: python -m scripts.stage1d_characterize_noise --dataset imbalanced --noise-type feature_driven
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _tau_dirname(tau: float) -> str:
    return f"tau_{int(round(tau * 100)):02d}"


def _aggregate_across_folds(
    cv_root: Path, dataset: str, noise_type: str, tau: float, n_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate clean and noisy labels across all folds for a given tau.
    Returns (clean_labels, noisy_labels) as arrays of class names.
    """
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


def _plot_confusion(M: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        M, annot=True, fmt=".2f", cmap="viridis",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        vmin=0, vmax=1, ax=ax, cbar_kws={"label": "P(noisy | clean)"},
    )
    ax.set_xlabel("Noisy label")
    ax.set_ylabel("Clean label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_concentration(concentrations: dict[float, float], out_path: Path, noise_type: str) -> None:
    taus = sorted(concentrations.keys())
    vals = [concentrations[t] for t in taus]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, vals, marker="o")
    ax.set_xlabel("τ (target noise rate)")
    ax.set_ylabel("Mean concentration (off-diag)")
    ax.set_title(f"Off-diagonal concentration — {noise_type}")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_tvd(tvds: dict[float, float], out_path: Path, noise_type: str) -> None:
    taus = sorted(tvds.keys())
    vals = [tvds[t] for t in taus]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, vals, marker="s", color="C1")
    ax.set_xlabel("τ")
    ax.set_ylabel("TVD(clean distribution, noisy distribution)")
    ax.set_title(f"Class-distribution shift — {noise_type}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_distribution_shift(
    clean_dist: np.ndarray, noisy_dist: np.ndarray,
    tau: float, out_path: Path, noise_type: str,
) -> None:
    x = np.arange(len(CLASS_NAMES))
    width = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, clean_dist, width, label="clean")
    ax.bar(x + width / 2, noisy_dist, width, label="noisy")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution shift — {noise_type} τ={tau}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(args: argparse.Namespace) -> int:
    cfg = load_config("base.yaml", f"data/{args.dataset}.yaml")
    root = project_root()
    cv_root = root / cfg["paths"]["cv_folds"]
    out_dir = ensure_dir(
        root / cfg["paths"]["results"]
        / "noise_characterization" / args.dataset / args.noise_type
    )

    concentrations: dict[float, float] = {}
    tvds: dict[float, float] = {}
    rows_for_csv = []
    confusion_matrices: dict[float, np.ndarray] = {}

    for tau in cfg["noise_rates"]:
        tau = float(tau)
        clean_labels, noisy_labels = _aggregate_across_folds(
            cv_root, args.dataset, args.noise_type, tau, int(cfg["folds"]),
        )
        M = confusion_matrix_from_labels(clean_labels, noisy_labels, normalize="row")
        confusion_matrices[tau] = M

        c = concentration(M)
        concentrations[tau] = c

        cd = class_distribution(clean_labels)
        nd = class_distribution(noisy_labels)
        tvd = total_variation_distance(cd, nd)
        tvds[tau] = tvd

        _plot_confusion(
            M, f"{args.noise_type} τ={tau:.2f}",
            out_dir / f"confusion_tau{int(round(tau * 100)):02d}.png",
        )
        _plot_distribution_shift(
            cd, nd, tau,
            out_dir / f"distshift_tau{int(round(tau * 100)):02d}.png",
            args.noise_type,
        )

        # numerical dump
        np.savetxt(
            out_dir / f"confusion_tau{int(round(tau * 100)):02d}.csv",
            M, delimiter=",", fmt="%.6f",
            header=",".join(CLASS_NAMES), comments="",
        )

        rows_for_csv.append({
            "tau": tau,
            "concentration": c,
            "tvd": tvd,
            "empirical_rate": float((clean_labels != noisy_labels).mean()),
        })
        print(f"[stage1d] τ={tau:.2f}: concentration={c:.4f}, tvd={tvd:.4f}")

    _plot_concentration(concentrations, out_dir / "concentration_vs_tau.png", args.noise_type)
    _plot_tvd(tvds, out_dir / "tvd_vs_tau.png", args.noise_type)

    summary_df = pd.DataFrame(rows_for_csv).sort_values("tau")
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"[stage1d] wrote {out_dir / 'summary.csv'}")

    manifest_path = (
        root / cfg["paths"]["manifests"]
        / f"stage1d_{args.dataset}_{args.noise_type}.json"
    )
    write_manifest(
        manifest_path, stage="stage1d",
        params={"dataset": args.dataset, "noise_type": args.noise_type},
        outputs=[str(out_dir.relative_to(root))],
    )
    print("[stage1d] DONE")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1d: noise characterization")
    p.add_argument("--dataset", required=True, choices=["balanced", "imbalanced"])
    p.add_argument("--noise-type", required=True,
                   choices=["standard", "normalized", "feature_driven"])
    sys.exit(main(p.parse_args()))
