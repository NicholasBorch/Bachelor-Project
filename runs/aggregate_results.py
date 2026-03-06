# runs/aggregate_results.py
# Aggregates completed training run results across all methods, folds, and tau levels.
# Produces a summary CSV and JSON for use in plotting and tables.
# Run from repo root: python -m runs.aggregate_results

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.io import project_root
from src.common.logging import load_all_results
from src.common.metrics import aggregate_metrics


# Methods and noise types to aggregate — extend as new methods are implemented
METHODS     = ["baseline", "elr", "sce", "asyco"]
NOISE_TYPES = ["standard_idn", "feature_driven_idn"]

SCALAR_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "kappa",
    "auc_macro_ovr",
]


def build_summary_table(results_root: Path) -> pd.DataFrame:
    # Loads all completed runs and builds a flat dataframe with one row per
    # (method, noise_type, tau, fold) combination
    rows = []

    for method in METHODS:
        for noise_type in NOISE_TYPES:
            runs = load_all_results(results_root, method, noise_type)

            if not runs:
                continue

            for run in runs:
                cfg     = run["config"]
                metrics = run["metrics"]

                row = {
                    "method":     cfg["method"],
                    "noise_type": cfg["noise_type"],
                    "tau":        cfg["tau"],
                    "outer_fold": cfg["outer_fold"],
                    "seed":       cfg["seed"],
                    "backbone":   cfg["backbone"],
                    "epochs":     cfg["epochs"],
                }

                # Flatten scalar metrics into columns
                for key in SCALAR_METRICS:
                    row[key] = metrics.get(key, float("nan"))

                # Flatten per-class F1 into separate columns
                for cls, f1 in metrics.get("per_class_f1", {}).items():
                    row[f"f1_{cls}"] = f1

                rows.append(row)

    if not rows:
        print("No completed runs found.")
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["method", "noise_type", "tau", "outer_fold"]
    ).reset_index(drop=True)


def aggregate_over_folds(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregates per-fold rows into mean ± std across folds for each
    # (method, noise_type, tau) combination
    if df.empty:
        return df

    metric_cols = SCALAR_METRICS + [c for c in df.columns if c.startswith("f1_")]
    group_keys  = ["method", "noise_type", "tau"]

    agg_rows = []
    for (method, noise_type, tau), group in df.groupby(group_keys):
        row = {
            "method":     method,
            "noise_type": noise_type,
            "tau":        tau,
            "n_folds":    len(group),
        }
        for col in metric_cols:
            values = group[col].dropna().values
            row[f"{col}_mean"] = float(np.mean(values)) if len(values) > 0 else float("nan")
            row[f"{col}_std"]  = float(np.std(values))  if len(values) > 0 else float("nan")
        agg_rows.append(row)

    return pd.DataFrame(agg_rows).sort_values(
        ["method", "noise_type", "tau"]
    ).reset_index(drop=True)


def print_summary(agg_df: pd.DataFrame, noise_type: str = "standard_idn") -> None:
    # Prints a readable console table for a given noise type
    subset = agg_df[agg_df["noise_type"] == noise_type]
    if subset.empty:
        print(f"No results for noise_type={noise_type}")
        return

    print(f"\n{'='*80}")
    print(f"Results summary — {noise_type}")
    print(f"{'='*80}")
    print(f"{'method':<12} {'tau':>6} {'bal_acc':>10} {'macro_f1':>10} {'auc':>10} {'kappa':>10}")
    print(f"{'-'*80}")

    for _, row in subset.iterrows():
        bal_acc  = f"{row['balanced_accuracy_mean']:.4f}±{row['balanced_accuracy_std']:.4f}"
        macro_f1 = f"{row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f}"
        auc      = f"{row['auc_macro_ovr_mean']:.4f}±{row['auc_macro_ovr_std']:.4f}"
        kappa    = f"{row['kappa_mean']:.4f}±{row['kappa_std']:.4f}"
        print(f"{row['method']:<12} {row['tau']:>6.2f} {bal_acc:>10} {macro_f1:>10} {auc:>10} {kappa:>10}")

    print(f"{'='*80}\n")


def main() -> None:
    root         = project_root()
    results_root = root / "results" / "HAM10000"
    out_dir      = results_root / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = build_summary_table(results_root)

    if df.empty:
        print("No results to aggregate. Run some training experiments first.")
        return

    # Save flat per-fold table
    per_fold_path = out_dir / "per_fold_results.csv"
    df.to_csv(per_fold_path, index=False)
    print(f"Saved per-fold results: {per_fold_path}")

    # Aggregate over folds
    agg_df = aggregate_over_folds(df)

    # Save aggregated table
    agg_csv_path  = out_dir / "aggregated_results.csv"
    agg_json_path = out_dir / "aggregated_results.json"
    agg_df.to_csv(agg_csv_path, index=False)
    agg_df.to_json(agg_json_path, orient="records", indent=2)
    print(f"Saved aggregated results: {agg_csv_path}")
    print(f"Saved aggregated results: {agg_json_path}")

    # Print console summary for each noise type
    for noise_type in NOISE_TYPES:
        print_summary(agg_df, noise_type)

    # Print per-class F1 summary for each method at each tau
    f1_cols = [c for c in agg_df.columns if c.startswith("f1_") and c.endswith("_mean")]
    if f1_cols:
        print("\nPer-class F1 means (standard_idn only):")
        subset = agg_df[agg_df["noise_type"] == "standard_idn"]
        class_names = [c.replace("f1_", "").replace("_mean", "") for c in f1_cols]
        header = f"{'method':<12} {'tau':>6} " + " ".join(f"{c:>8}" for c in class_names)
        print(header)
        print("-" * len(header))
        for _, row in subset.iterrows():
            values = " ".join(f"{row[c]:>8.4f}" for c in f1_cols)
            print(f"{row['method']:<12} {row['tau']:>6.2f} {values}")

    print("\nDone.")


if __name__ == "__main__":
    main()