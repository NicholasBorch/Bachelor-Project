"""Stage 1e: human annotator confusion comparison.

Compares the empirical confusion matrices of normalized IDN and feature-driven
IDN (both on the IMBALANCED dataset to match Tschandl's reader study priors)
against the Tschandl et al. 2019 all-readers majority-vote confusion matrix
from Supplementary Figure 6, top-left panel.

Metric: mean absolute error on off-diagonal entries. The diagonal is
(approximately) 1 - accuracy for Tschandl and 1 - tau for our matrices, so
including it would dominate the comparison with a quantity that is not
really about *confusion patterns* between classes.

Run: python -m scripts.stage1e_human_comparison

Outputs (results/human_comparison/):
    - mae_vs_tau.png
    - matrix_sidebyside_tau{NN}.png  (at best-aligned tau for each noise type)
    - mae_table.csv
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
from src.noise.characterize import confusion_matrix_from_labels, off_diagonal_mae
from src.utils.io import ensure_dir, load_config, project_root
from src.utils.manifest import write_manifest

_NOISE_TYPES = ("normalized", "feature_driven")
_DATASET = "imbalanced"


def _tau_dirname(tau: float) -> str:
    return f"tau_{int(round(tau * 100)):02d}"


def _load_tschandl(root: Path) -> np.ndarray:
    path = root / "data" / "external" / "tschandl_confusion_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Tschandl matrix not found at {path}. "
            f"This file is committed to the repo and should not be missing."
        )
    df = pd.read_csv(path)
    # Validate column order matches CLASS_NAMES
    rows = df["true_class"].tolist()
    cols = [c for c in df.columns if c != "true_class"]
    if rows != CLASS_NAMES or cols != CLASS_NAMES:
        raise ValueError(
            f"Tschandl CSV row/column order must match CLASS_NAMES={CLASS_NAMES}. "
            f"Got rows={rows}, cols={cols}."
        )
    M = df[cols].values.astype(np.float64)
    # Normalize rows in case values are rounded
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return M / row_sums


def _load_empirical(cv_root: Path, noise_type: str, tau: float, n_folds: int) -> np.ndarray:
    clean_all, noisy_all = [], []
    for fold in range(n_folds):
        path = (cv_root / _DATASET / noise_type / _tau_dirname(tau)
                / f"fold_{fold:02d}" / "train_noisy.csv")
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        df = pd.read_csv(path)
        clean_all.extend(df["dx_clean"].tolist())
        noisy_all.extend(df["dx"].tolist())
    return confusion_matrix_from_labels(np.array(clean_all), np.array(noisy_all), normalize="row")


def _plot_mae_vs_tau(mae_data: dict[str, dict[float, float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for noise_type, by_tau in mae_data.items():
        taus = sorted(by_tau.keys())
        vals = [by_tau[t] for t in taus]
        ax.plot(taus, vals, marker="o", label=noise_type)
    ax.set_xlabel("τ (target noise rate)")
    ax.set_ylabel("Off-diagonal MAE vs Tschandl (all readers)")
    ax.set_title("Alignment of IDN variants with human confusion patterns")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_sidebyside(
    tschandl: np.ndarray, ours: np.ndarray,
    noise_type: str, tau: float, out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, M, title in zip(axes, [tschandl, ours], ["Tschandl (humans, all)", f"{noise_type} τ={tau:.2f}"]):
        sns.heatmap(
            M, annot=True, fmt=".2f", cmap="viridis",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            vmin=0, vmax=1, ax=ax, cbar=True,
        )
        ax.set_xlabel("Predicted / noisy class")
        ax.set_ylabel("True class")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(args: argparse.Namespace) -> int:
    cfg = load_config("base.yaml", f"data/{_DATASET}.yaml")
    root = project_root()
    cv_root = root / cfg["paths"]["cv_folds"]
    out_dir = ensure_dir(root / cfg["paths"]["results"] / "human_comparison")

    tschandl = _load_tschandl(root)
    print(f"[stage1e] loaded Tschandl matrix, shape {tschandl.shape}")

    rows = []
    mae_by_type: dict[str, dict[float, float]] = {nt: {} for nt in _NOISE_TYPES}
    best_tau: dict[str, tuple[float, np.ndarray]] = {}

    for noise_type in _NOISE_TYPES:
        for tau in cfg["noise_rates"]:
            tau = float(tau)
            M = _load_empirical(cv_root, noise_type, tau, int(cfg["folds"]))
            mae = off_diagonal_mae(M, tschandl)
            mae_by_type[noise_type][tau] = mae
            rows.append({"noise_type": noise_type, "tau": tau, "mae_off_diagonal": mae})
            if (noise_type not in best_tau) or (mae < best_tau[noise_type][0]):
                best_tau[noise_type] = (mae, M, tau)
            print(f"[stage1e] {noise_type:16s} τ={tau:.2f}  off-diag MAE={mae:.5f}")

    _plot_mae_vs_tau(mae_by_type, out_dir / "mae_vs_tau.png")

    # Side-by-side at best-aligned tau for each noise type.
    for noise_type, (_mae, M, tau) in best_tau.items():
        out_path = out_dir / f"sidebyside_{noise_type}_tau{int(round(tau * 100)):02d}.png"
        _plot_sidebyside(tschandl, M, noise_type, tau, out_path)

    table = pd.DataFrame(rows).sort_values(["noise_type", "tau"])
    table.to_csv(out_dir / "mae_table.csv", index=False)
    print(f"[stage1e] wrote {out_dir / 'mae_table.csv'}")

    manifest_path = root / cfg["paths"]["manifests"] / "stage1e.json"
    write_manifest(
        manifest_path, stage="stage1e",
        params={},
        outputs=[str(out_dir.relative_to(root))],
        extra={"best_tau_per_noise_type": {k: v[2] for k, v in best_tau.items()}},
    )
    print("[stage1e] DONE")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1e: human annotator comparison")
    sys.exit(main(p.parse_args()))
