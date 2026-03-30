# src/utils/aggregate_epoch_budget.py
#
# Aggregates per-fold epoch budget curves produced by find_epoch_budget.py
# and produces the averaged curve and plot.
#
# Run AFTER all find_epoch_budget fold jobs have completed:
#   python -m src.utils.aggregate_epoch_budget

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.common.io import project_root
from configs.classification_default import FOLDS


def plot_curves(avg_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs  = avg_df["epoch"].values

    ax.plot(epochs, avg_df["train_loss_mean"],
            label="Training Loss", color="#1f77b4", linewidth=2)
    ax.fill_between(
        epochs,
        avg_df["train_loss_mean"] - avg_df["train_loss_std"],
        avg_df["train_loss_mean"] + avg_df["train_loss_std"],
        alpha=0.2, color="#1f77b4",
    )
    ax.plot(epochs, avg_df["val_loss_mean"],
            label="Validation Loss", color="#d62728", linewidth=2)
    ax.fill_between(
        epochs,
        avg_df["val_loss_mean"] - avg_df["val_loss_std"],
        avg_df["val_loss_mean"] + avg_df["val_loss_std"],
        alpha=0.2, color="#d62728",
    )

    best_epoch = int(avg_df.loc[avg_df["val_loss_mean"].idxmin(), "epoch"])
    best_val   = avg_df["val_loss_mean"].min()
    ax.axvline(x=best_epoch, color="#2ca02c", linestyle="--",
               linewidth=1.5, alpha=0.8)
    ax.scatter([best_epoch], [best_val], color="#2ca02c", s=80, zorder=5)
    ax.annotate(
        f"Min val loss: epoch {best_epoch}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 3, best_val + 0.05),
        fontsize=10, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(
        "Epoch Selection — Clean Baseline (mean ± std across 10 folds)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def main() -> None:
    out_dir = project_root() / "results" / "HAM10000" / "epoch_selection"

    # Load all per-fold files
    fold_files = sorted(out_dir.glob("fold_*_curves.csv"))
    if not fold_files:
        raise FileNotFoundError(
            f"No fold curve files found in {out_dir}. "
            "Run find_epoch_budget.py for all folds first."
        )

    print(f"Found {len(fold_files)} fold files:")
    for f in fold_files:
        print(f"  {f.name}")

    curves_df = pd.concat([pd.read_csv(f) for f in fold_files], ignore_index=True)
    curves_df.to_csv(out_dir / "curves_per_fold.csv", index=False)

    avg_df = (
        curves_df
        .groupby("epoch")
        .agg(
            train_loss_mean=("train_loss", "mean"),
            train_loss_std=("train_loss", "std"),
            val_loss_mean=("val_loss",   "mean"),
            val_loss_std=("val_loss",   "std"),
        )
        .reset_index()
    )
    avg_df.to_csv(out_dir / "curves_averaged.csv", index=False)

    best_epoch = int(avg_df.loc[avg_df["val_loss_mean"].idxmin(), "epoch"])
    best_val   = avg_df["val_loss_mean"].min()

    print(f"\n{'='*60}")
    print(f"  Best epoch (min avg val loss): {best_epoch}")
    print(f"  Val loss at best epoch:        {best_val:.4f}")
    print(f"{'='*60}")

    plot_curves(avg_df, out_dir / "epoch_selection_curves.png")
    print(f"\nAll outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()