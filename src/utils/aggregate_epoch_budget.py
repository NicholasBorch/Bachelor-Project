# src/utils/aggregate_epoch_budget.py
#
# Aggregates per-fold epoch budget curves produced by find_epoch_budget.py.
# Always overwrites existing aggregated results and plots.
#
# Run locally after pulling all fold results from HPC:
#   python -m src.utils.aggregate_epoch_budget

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.common.io import project_root
from configs.classification_default import FOLDS


def plot_curves(avg_df: pd.DataFrame, out_dir: Path) -> None:
    """Three-panel plot: loss (left), balanced accuracy (centre), macro F1 (right)."""
    epochs = avg_df["epoch"].values

    best_loss_epoch = int(avg_df.loc[avg_df["val_loss_mean"].idxmin(),      "epoch"])
    best_acc_epoch  = int(avg_df.loc[avg_df["val_bal_acc_mean"].idxmax(),   "epoch"])
    best_f1_epoch   = int(avg_df.loc[avg_df["val_macro_f1_mean"].idxmax(),  "epoch"])

    fig, (ax_loss, ax_acc, ax_f1) = plt.subplots(
        1, 3, figsize=(20, 5), constrained_layout=True
    )
    fig.suptitle(
        "Epoch Budget Selection — Clean Baseline (mean ± std across 10 folds)",
        fontsize=14,
    )

    def _mark_best(ax, x_epoch, y_val, label):
        ax.axvline(x=x_epoch, color="#2ca02c", linestyle="--",
                   linewidth=1.5, alpha=0.8)
        ax.scatter([x_epoch], [y_val], color="#2ca02c", s=80, zorder=5)
        ax.annotate(
            label,
            xy=(x_epoch, y_val),
            xytext=(x_epoch + 4, y_val),
            fontsize=9, color="#2ca02c",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
        )

    # ── Loss ──────────────────────────────────────────────────────────────
    ax_loss.plot(epochs, avg_df["train_loss_mean"],
                 label="Training Loss", color="#1f77b4", linewidth=2)
    ax_loss.fill_between(
        epochs,
        avg_df["train_loss_mean"] - avg_df["train_loss_std"],
        avg_df["train_loss_mean"] + avg_df["train_loss_std"],
        alpha=0.2, color="#1f77b4",
    )
    ax_loss.plot(epochs, avg_df["val_loss_mean"],
                 label="Validation Loss", color="#d62728", linewidth=2)
    ax_loss.fill_between(
        epochs,
        avg_df["val_loss_mean"] - avg_df["val_loss_std"],
        avg_df["val_loss_mean"] + avg_df["val_loss_std"],
        alpha=0.2, color="#d62728",
    )
    _mark_best(ax_loss, best_loss_epoch, avg_df["val_loss_mean"].min(),
               f"Min val loss: epoch {best_loss_epoch}")
    ax_loss.set_xlabel("Epoch", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.set_title("Loss Curves", fontweight="bold")
    ax_loss.legend(fontsize=10)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.spines[["top", "right"]].set_visible(False)

    # ── Balanced Accuracy ─────────────────────────────────────────────────
    ax_acc.plot(epochs, avg_df["val_bal_acc_mean"],
                label="Val Balanced Accuracy", color="#9467bd", linewidth=2)
    ax_acc.fill_between(
        epochs,
        avg_df["val_bal_acc_mean"] - avg_df["val_bal_acc_std"],
        avg_df["val_bal_acc_mean"] + avg_df["val_bal_acc_std"],
        alpha=0.2, color="#9467bd",
    )
    _mark_best(ax_acc, best_acc_epoch, avg_df["val_bal_acc_mean"].max(),
               f"Max bal acc: epoch {best_acc_epoch}")
    ax_acc.set_xlabel("Epoch", fontsize=12)
    ax_acc.set_ylabel("Balanced Accuracy", fontsize=12)
    ax_acc.set_title("Validation Balanced Accuracy", fontweight="bold")
    ax_acc.legend(fontsize=10)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.spines[["top", "right"]].set_visible(False)

    # ── Macro F1 ──────────────────────────────────────────────────────────
    ax_f1.plot(epochs, avg_df["val_macro_f1_mean"],
               label="Val Macro F1", color="#e377c2", linewidth=2)
    ax_f1.fill_between(
        epochs,
        avg_df["val_macro_f1_mean"] - avg_df["val_macro_f1_std"],
        avg_df["val_macro_f1_mean"] + avg_df["val_macro_f1_std"],
        alpha=0.2, color="#e377c2",
    )
    _mark_best(ax_f1, best_f1_epoch, avg_df["val_macro_f1_mean"].max(),
               f"Max macro F1: epoch {best_f1_epoch}")
    ax_f1.set_xlabel("Epoch", fontsize=12)
    ax_f1.set_ylabel("Macro F1", fontsize=12)
    ax_f1.set_title("Validation Macro F1", fontweight="bold")
    ax_f1.legend(fontsize=10)
    ax_f1.grid(True, alpha=0.3)
    ax_f1.spines[["top", "right"]].set_visible(False)

    out_path = out_dir / "epoch_selection_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: epoch_selection_curves.png")


def print_summary(avg_df: pd.DataFrame) -> None:
    best_loss_epoch = int(avg_df.loc[avg_df["val_loss_mean"].idxmin(),      "epoch"])
    best_acc_epoch  = int(avg_df.loc[avg_df["val_bal_acc_mean"].idxmax(),   "epoch"])
    best_f1_epoch   = int(avg_df.loc[avg_df["val_macro_f1_mean"].idxmax(),  "epoch"])
    best_val_loss   = avg_df["val_loss_mean"].min()
    best_val_acc    = avg_df["val_bal_acc_mean"].max()
    best_val_f1     = avg_df["val_macro_f1_mean"].max()

    print(f"\n{'='*75}")
    print(f"  Epoch Budget Selection — Summary")
    print(f"{'='*75}")
    print(f"  Total epochs evaluated       : {len(avg_df)}")
    print(f"  Best epoch (min val loss)    : {best_loss_epoch}  "
          f"(val_loss={best_val_loss:.4f})")
    print(f"  Best epoch (max val bal acc) : {best_acc_epoch}  "
          f"(bal_acc={best_val_acc:.4f})")
    print(f"  Best epoch (max val macro F1): {best_f1_epoch}  "
          f"(macro_f1={best_val_f1:.4f})")
    print(f"{'='*75}")

    print(f"\n  {'Epoch':>6}  {'Train loss':>11}  {'Val loss':>9}  "
          f"{'Val bal acc':>12}  {'Val macro F1':>13}")
    print(f"  {'-'*57}")
    for _, row in avg_df.iterrows():
        markers = []
        if int(row["epoch"]) == best_loss_epoch: markers.append("best loss")
        if int(row["epoch"]) == best_acc_epoch:  markers.append("best acc")
        if int(row["epoch"]) == best_f1_epoch:   markers.append("best F1")
        marker = (" ◄ " + ", ".join(markers)) if markers else ""
        print(f"  {int(row['epoch']):>6}  "
              f"{row['train_loss_mean']:>11.4f}  "
              f"{row['val_loss_mean']:>9.4f}  "
              f"{row['val_bal_acc_mean']:>12.4f}  "
              f"{row['val_macro_f1_mean']:>13.4f}"
              f"{marker}")


def main() -> None:
    out_dir = project_root() / "results" / "HAM10000" / "epoch_selection"

    fold_files = sorted(out_dir.glob("fold_*_curves.csv"))
    if not fold_files:
        raise FileNotFoundError(
            f"No fold curve files found in {out_dir}.\n"
            "Run find_epoch_budget.py for all folds first."
        )

    n_found = len(fold_files)
    print(f"Found {n_found}/{FOLDS} fold files:")
    for f in fold_files:
        print(f"  {f.name}")

    if n_found < FOLDS:
        print(f"\nWarning: only {n_found} of {FOLDS} folds present. "
              "Aggregating available folds.")

    # Combine — always overwrite
    curves_df = pd.concat(
        [pd.read_csv(f) for f in fold_files], ignore_index=True
    )
    curves_df.to_csv(out_dir / "curves_per_fold.csv", index=False)
    print(f"\n  Saved: curves_per_fold.csv")

    # Aggregate
    avg_df = (
        curves_df
        .groupby("epoch")
        .agg(
            train_loss_mean=("train_loss",    "mean"),
            train_loss_std=("train_loss",     "std"),
            val_loss_mean=("val_loss",        "mean"),
            val_loss_std=("val_loss",         "std"),
            val_bal_acc_mean=("val_bal_acc",  "mean"),
            val_bal_acc_std=("val_bal_acc",   "std"),
            val_macro_f1_mean=("val_macro_f1", "mean"),
            val_macro_f1_std=("val_macro_f1",  "std"),
        )
        .reset_index()
    )
    avg_df.to_csv(out_dir / "curves_averaged.csv", index=False)
    print(f"  Saved: curves_averaged.csv")

    print_summary(avg_df)
    plot_curves(avg_df, out_dir)
    print(f"\nAll outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()