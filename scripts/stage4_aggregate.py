"""
Main-experiment aggregation: figures, tables, and statistical tests.

Walks results/main_experiment/training/, aggregates across folds, and writes
aggregated/raw CSVs, Wilcoxon tables (vs baseline and vs clean), and per- and
cross-condition figures under results/main_experiment/figures_and_tables/.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.analysis.aggregate import aggregate_mean_std, load_all_results
from src.analysis.plots import (
    plot_dataset_comparison,
    plot_init_optim_ablation,
    plot_method_comparison_bars,
    plot_metrics_vs_tau,
    plot_noise_label_interaction,
    plot_per_class_f1_heatmap,
)
from src.analysis.stats import (
    apply_multiple_testing_corrections,
    wilcoxon_vs_baseline,
    wilcoxon_vs_clean,
)
from src.utils.io import ensure_dir, project_root
from src.utils.manifest import write_manifest

logger = logging.getLogger(__name__)

# Defaults match the main_experiment run topology
DATASETS: tuple[str, ...] = ("imbalanced",)  # only imbalanced for now
INITS: tuple[str, ...] = ("pretrained",)      # toggle if you run scratch later
OPTIMS: tuple[str, ...] = ("adam",)            # toggle if you run sgd later
TAUS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

# Stats are run per metric; corrections applied within each metric.
STATS_METRICS: tuple[str, ...] = ("balanced_accuracy", "macro_f1")


def _tau_tag(tau: float) -> str:
    return f"tau{int(round(tau * 100)):02d}"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Main Experiment analysis — figures, tables, statistical tests."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output dir. Default: results/main_experiment/figures_and_tables/",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help=f"Datasets to analyze (default: {DATASETS})",
    )
    parser.add_argument(
        "--inits", nargs="*", default=None,
        help=f"Inits to analyze (default: {INITS})",
    )
    parser.add_argument(
        "--optims", nargs="*", default=None,
        help=f"Optimizers to analyze (default: {OPTIMS})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if the output dir already has content.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def _make_per_condition_plots(
    df: pd.DataFrame, output_dir: Path,
    datasets, inits, optims,
) -> None:
    per_condition_dir = output_dir / "per_condition"
    ensure_dir(per_condition_dir)
    for dataset in datasets:
        for init in inits:
            for optim in optims:
                prefix = f"{dataset}_{init}_{optim}"
                plot_metrics_vs_tau(
                    df, dataset, init, optim,
                    per_condition_dir / f"metrics_vs_tau_{prefix}.png",
                )
                plot_noise_label_interaction(
                    df, dataset, init, optim,
                    per_condition_dir / f"noise_label_interaction_{prefix}.png",
                )
                for tau in TAUS:
                    plot_method_comparison_bars(
                        df, dataset, init, optim, tau,
                        per_condition_dir
                        / f"method_comparison_{prefix}_{_tau_tag(tau)}.png",
                    )
                    plot_per_class_f1_heatmap(
                        df, dataset, init, optim, tau,
                        per_condition_dir
                        / f"perclass_f1_{prefix}_{_tau_tag(tau)}.png",
                    )


def _make_cross_condition_plots(
    df: pd.DataFrame, output_dir: Path,
    datasets, inits, optims,
) -> None:
    cross_dir = output_dir / "cross_condition"
    ensure_dir(cross_dir)
    if len(inits) >= 2 or len(optims) >= 2:
        for dataset in datasets:
            plot_init_optim_ablation(
                df, dataset,
                cross_dir / f"init_optim_ablation_{dataset}.png",
            )
    if len(datasets) >= 2:
        for init in inits:
            for optim in optims:
                plot_dataset_comparison(
                    df, init, optim,
                    cross_dir / f"dataset_comparison_{init}_{optim}.png",
                )


def _build_statistical_tables(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vs_baseline_parts: list[pd.DataFrame] = []
    vs_clean_parts: list[pd.DataFrame] = []
    for metric in STATS_METRICS:
        b = wilcoxon_vs_baseline(df, metric=metric)
        b["metric"] = metric
        vs_baseline_parts.append(b)
        c = wilcoxon_vs_clean(df, metric=metric)
        c["metric"] = metric
        vs_clean_parts.append(c)

    vs_baseline = (
        pd.concat(vs_baseline_parts, ignore_index=True)
        if vs_baseline_parts else pd.DataFrame()
    )
    vs_clean = (
        pd.concat(vs_clean_parts, ignore_index=True)
        if vs_clean_parts else pd.DataFrame()
    )
    vs_baseline = apply_multiple_testing_corrections(
        vs_baseline, p_col="p_value", family_cols=["metric"],
    )
    vs_clean = apply_multiple_testing_corrections(
        vs_clean, p_col="p_value", family_cols=["metric"],
    )
    return vs_baseline, vs_clean


def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)

    root = project_root()
    results_dir = root / "results" / "main_experiment"

    if not (results_dir / "training").exists():
        logger.error(
            "Results dir %s does not exist. Run final_experiment_train.py first.",
            results_dir / "training",
        )
        return 2

    output_dir = (
        args.output_dir or (results_dir / "figures_and_tables")
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

    # Determine which conditions to analyze
    datasets = args.datasets if args.datasets else list(DATASETS)
    inits = args.inits if args.inits else list(INITS)
    optims = args.optims if args.optims else list(OPTIMS)
    logger.info(
        "Analyzing: datasets=%s inits=%s optims=%s",
        datasets, inits, optims,
    )

    # Load all results
    logger.info("Loading results from %s ...", results_dir / "training")
    df = load_all_results(results_dir)
    logger.info("Loaded %d fold-level rows.", len(df))

    if df.empty:
        logger.warning("No results found. Writing empty tables and exiting.")

    # Tables
    aggregated = aggregate_mean_std(df)
    aggregated_path = output_dir / "aggregated_results.csv"
    aggregated.to_csv(aggregated_path, index=False)
    logger.info("Wrote %s (%d rows).", aggregated_path, len(aggregated))

    raw_path = output_dir / "raw_fold_results.csv"
    df.drop(columns=["confusion_matrix"], errors="ignore").to_csv(
        raw_path, index=False,
    )
    logger.info("Wrote %s (%d rows).", raw_path, len(df))

    # Statistical tests
    stats_vs_baseline, stats_vs_clean = _build_statistical_tables(df)
    sb_path = output_dir / "statistical_tests_vs_baseline.csv"
    sb_path.write_bytes(
        stats_vs_baseline.to_csv(index=False).encode("utf-8")
    )
    logger.info("Wrote %s (%d rows).", sb_path, len(stats_vs_baseline))

    sc_path = output_dir / "statistical_tests_noise_sensitivity.csv"
    sc_path.write_bytes(
        stats_vs_clean.to_csv(index=False).encode("utf-8")
    )
    logger.info("Wrote %s (%d rows).", sc_path, len(stats_vs_clean))

    # Figures
    if not df.empty:
        _make_per_condition_plots(df, output_dir, datasets, inits, optims)
        _make_cross_condition_plots(df, output_dir, datasets, inits, optims)
    plots_written = len(list(output_dir.rglob("*.png")))
    logger.info("Wrote %d figure files.", plots_written)

    # Manifest
    manifests_dir = root / "results" / "manifests"
    ensure_dir(manifests_dir)
    manifest_path = manifests_dir / "final_experiment_analyze.json"
    write_manifest(
        manifest_path,
        stage="final_experiment_analyze",
        params={
            "results_dir": str(results_dir),
            "output_dir": str(output_dir),
            "datasets": datasets,
            "inits": inits,
            "optims": optims,
            "stats_metrics": list(STATS_METRICS),
        },
        outputs=[
            str(aggregated_path), str(raw_path),
            str(sb_path), str(sc_path),
        ],
        extra={
            "n_fold_rows": int(len(df)),
            "n_aggregated_rows": int(len(aggregated)),
            "n_stats_vs_baseline_rows": int(len(stats_vs_baseline)),
            "n_stats_vs_clean_rows": int(len(stats_vs_clean)),
            "n_plots_written": int(plots_written),
        },
    )
    logger.info("Wrote manifest %s.", manifest_path)
    logger.info("Main experiment analysis complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())