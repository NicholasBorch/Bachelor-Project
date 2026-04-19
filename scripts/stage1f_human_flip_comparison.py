"""Stage 1f: human flip-pattern comparison.

Compares empirical noisy-label confusion matrices against a human annotator
reference matrix loaded from CSV. Both matrices are transformed into flip-only
form by zeroing the diagonal and re-normalising each row, so the comparison
focuses only on where errors/flips go.

Runs for both balanced and imbalanced datasets.

Outputs:
    results/human_flip_comparison/{dataset}/
        - mae_vs_tau.png
        - mae_table.csv
        - sidebyside_{noise_type}_tau{NN}.png
        - sidebyside_best_{noise_type}_tau{NN}.png

Run:
    python -m scripts.stage1f_human_flip_comparison
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

_DATASETS = ("balanced", "imbalanced")
_NOISE_TYPES = ("normalized", "feature_driven")
_HUMAN_MATRIX_FILE = "tschandl_confusion_matrix.csv"


def _tau_dirname(tau: float) -> str:
    return f"tau_{int(round(tau * 100)):02d}"


def _flip_only_row_normalize(M: np.ndarray) -> np.ndarray:
    M = M.astype(np.float64).copy()
    np.fill_diagonal(M, 0.0)
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return M / row_sums


def _load_human_flip_matrix(root: Path) -> np.ndarray:
    path = root / "data" / "external" / _HUMAN_MATRIX_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Human confusion matrix not found at {path}."
        )

    df = pd.read_csv(path)
    rows = df["true_class"].tolist()
    cols = [c for c in df.columns if c != "true_class"]

    expected_raw_order = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    if rows != expected_raw_order or cols != expected_raw_order:
        raise ValueError(
            "Human CSV row/column order must be "
            f"{expected_raw_order}. Got rows={rows}, cols={cols}."
        )

    df = df.set_index("true_class")
    df = df.loc[CLASS_NAMES, CLASS_NAMES]

    M = df.values.astype(np.float64)
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    M = M / row_sums

    return _flip_only_row_normalize(M)


def _load_empirical_flip_matrix(
    cv_root: Path, dataset: str, noise_type: str, tau: float, n_folds: int
) -> np.ndarray:
    clean_all, noisy_all = [], []

    for fold in range(n_folds):
        path = (
            cv_root
            / dataset
            / noise_type
            / _tau_dirname(tau)
            / f"fold_{fold:02d}"
            / "train_noisy.csv"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)
        clean_all.extend(df["dx_clean"].tolist())
        noisy_all.extend(df["dx"].tolist())

    M = confusion_matrix_from_labels(
        np.array(clean_all),
        np.array(noisy_all),
        normalize="row",
    )
    return _flip_only_row_normalize(M)


def _plot_mae_vs_tau(
    mae_data: dict[str, dict[float, float]],
    out_path: Path,
    dataset: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for noise_type, by_tau in mae_data.items():
        taus = sorted(by_tau.keys())
        vals = [by_tau[t] for t in taus]
        ax.plot(taus, vals, marker="o", label=noise_type)

    ax.set_xlabel("τ (target noise rate)")
    ax.set_ylabel("Off-diagonal MAE vs human flip matrix")
    ax.set_title(f"Alignment with human flip patterns — {dataset}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_sidebyside(
    human_M: np.ndarray,
    ours_M: np.ndarray,
    noise_type: str,
    tau: float,
    out_path: Path,
    dataset: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    matrices = [
        pd.DataFrame(ours_M, index=CLASS_NAMES, columns=CLASS_NAMES),
        pd.DataFrame(human_M, index=CLASS_NAMES, columns=CLASS_NAMES),
    ]
    titles = [
        f"{dataset} — {noise_type} τ={tau:.2f}",
        "Human annotators (re-normalised flips)",
    ]

    for ax, matrix, title in zip(axes, matrices, titles):
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Fraction of flips"},
        )
        ax.set_xlabel("Flip target class")
        ax.set_ylabel("True class")
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(args: argparse.Namespace) -> int:
    root = project_root()
    human_flip = _load_human_flip_matrix(root)
    print(f"[stage1f] loaded human flip matrix, shape {human_flip.shape}")

    for dataset in _DATASETS:
        cfg = load_config("base.yaml", f"data/{dataset}.yaml")
        cv_root = root / cfg["paths"]["cv_folds"]
        out_dir = ensure_dir(root / cfg["paths"]["results"] / "human_flip_comparison" / dataset)

        rows = []
        mae_by_type: dict[str, dict[float, float]] = {nt: {} for nt in _NOISE_TYPES}
        best_tau: dict[str, tuple[float, np.ndarray, float]] = {}

        print(f"[stage1f] dataset={dataset}")

        for noise_type in _NOISE_TYPES:
            for tau in cfg["noise_rates"]:
                tau = float(tau)

                M = _load_empirical_flip_matrix(
                    cv_root=cv_root,
                    dataset=dataset,
                    noise_type=noise_type,
                    tau=tau,
                    n_folds=int(cfg["folds"]),
                )
                mae = off_diagonal_mae(M, human_flip)

                mae_by_type[noise_type][tau] = mae
                rows.append({
                    "dataset": dataset,
                    "noise_type": noise_type,
                    "tau": tau,
                    "mae_off_diagonal": mae,
                })

                if (noise_type not in best_tau) or (mae < best_tau[noise_type][0]):
                    best_tau[noise_type] = (mae, M, tau)

                out_path = out_dir / f"sidebyside_{noise_type}_tau{int(round(tau * 100)):02d}.png"
                _plot_sidebyside(human_flip, M, noise_type, tau, out_path, dataset)

                print(
                    f"[stage1f] {dataset:10s} {noise_type:16s} "
                    f"τ={tau:.2f}  MAE={mae:.5f}"
                )

        _plot_mae_vs_tau(mae_by_type, out_dir / "mae_vs_tau.png", dataset)

        for noise_type, (_mae, M, tau) in best_tau.items():
            out_path = out_dir / f"sidebyside_best_{noise_type}_tau{int(round(tau * 100)):02d}.png"
            _plot_sidebyside(human_flip, M, noise_type, tau, out_path, dataset)

        table = pd.DataFrame(rows).sort_values(["noise_type", "tau"])
        table.to_csv(out_dir / "mae_table.csv", index=False)
        print(f"[stage1f] wrote {out_dir / 'mae_table.csv'}")

        manifest_path = root / cfg["paths"]["manifests"] / f"stage1f_{dataset}.json"
        write_manifest(
            manifest_path,
            stage="stage1f",
            params={"dataset": dataset},
            outputs=[str(out_dir.relative_to(root))],
            extra={"best_tau_per_noise_type": {k: v[2] for k, v in best_tau.items()}},
        )

    print("[stage1f] DONE")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1f: human flip comparison")
    sys.exit(main(p.parse_args()))