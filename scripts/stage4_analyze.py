"""Stage 4 — analysis entry point.

Walks the Stage 3 results tree, produces every figure, table, and
statistical test that goes into the thesis, and writes a manifest summarising
what was produced.

Designed to be:
    - **Read-only** with respect to Stage 3 outputs.
    - **Resilient to partial results** (Stage 3 may still be running).
    - **Fast** (< 5 minutes on a laptop, no GPU required).

Two families of paired Wilcoxon tests are emitted (see
PROJECT_DOCUMENTATION §6 Stage 4 and §2.4):

    statistical_tests_vs_baseline.csv
        Method vs baseline at each (dataset, init, optim, τ). Run for both
        balanced_accuracy and macro_f1, Bonferroni + Holm corrected within
        each metric.

    statistical_tests_noise_sensitivity.csv
        Each method's τ>0 performance vs its own τ=0 performance. Run for
        both balanced_accuracy and macro_f1, Bonferroni + Holm corrected
        within each metric.

Usage::

    python -m scripts.stage4_analyze
    python -m scripts.stage4_analyze --output-dir results/final_figures
    python -m scripts.stage4_analyze --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.analysis.aggregate import (
    aggregate_mean_std,
    load_all_results,
    load_selected_budgets,
)
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
from src.utils.io import ensure_dir, load_config, project_root
from src.utils.manifest import write_manifest

logger = logging.getLogger(__name__)

DATASETS: tuple[str, ...] = ("balanced", "imbalanced")
INITS: tuple[str, ...] = ("pretrained", "scratch")
OPTIMS: tuple[str, ...] = ("sgd", "adam")
TAUS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

# Metrics on which the full statistical-test suite is run. Correction is
# applied within each metric (so the two metrics are independent families).
STATS_METRICS: tuple[str, ...] = ("balanced_accuracy", "macro_f1")


def _tau_tag(tau: float) -> str:
    """Directory-safe tag for a τ value: ``0.1 → 'tau10'``."""
    return f"tau{int(round(tau * 100)):02d}"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4 — aggregate Stage 3 results, make figures and tables."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Directory for figures and tables. Defaults to "
            "<results>/final_figures based on configs/base.yaml."
        ),
    )
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Override results root (useful for testing). Defaults to <project>/results.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if the output dir already exists.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


def _make_per_condition_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Produce per-(dataset, init, optim) and per-(dataset, init, optim, tau) figures."""
    per_condition_dir = output_dir / "per_condition"
    ensure_dir(per_condition_dir)

    for dataset in DATASETS:
        for init in INITS:
            for optim in OPTIMS:
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


def _make_cross_condition_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Produce the overview figures that span multiple conditions."""
    cross_dir = output_dir / "cross_condition"
    ensure_dir(cross_dir)

    for dataset in DATASETS:
        plot_init_optim_ablation(
            df, dataset, cross_dir / f"init_optim_ablation_{dataset}.png"
        )

    for init in INITS:
        for optim in OPTIMS:
            plot_dataset_comparison(
                df, init, optim,
                cross_dir / f"dataset_comparison_{init}_{optim}.png",
            )


def _build_statistical_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both test families on both metrics and concatenate.

    Each family's table carries a ``metric`` column. Multiple-testing
    corrections are applied WITHIN each metric (``family_cols=["metric"]``)
    so the BA tests and the Macro F1 tests are independent families and do
    not penalize each other.

    Returns:
        (stats_vs_baseline_df, stats_vs_clean_df)
    """
    vs_baseline_parts: list[pd.DataFrame] = []
    vs_clean_parts: list[pd.DataFrame] = []

    for metric in STATS_METRICS:
        b = wilcoxon_vs_baseline(df, metric=metric)
        b["metric"] = metric
        vs_baseline_parts.append(b)

        c = wilcoxon_vs_clean(df, metric=metric)
        c["metric"] = metric
        vs_clean_parts.append(c)

    vs_baseline = pd.concat(vs_baseline_parts, ignore_index=True) if vs_baseline_parts else pd.DataFrame()
    vs_clean = pd.concat(vs_clean_parts, ignore_index=True) if vs_clean_parts else pd.DataFrame()

    vs_baseline = apply_multiple_testing_corrections(
        vs_baseline, p_col="p_value", family_cols=["metric"]
    )
    vs_clean = apply_multiple_testing_corrections(
        vs_clean, p_col="p_value", family_cols=["metric"]
    )
    return vs_baseline, vs_clean


def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)

    cfg = load_config("base.yaml")
    root = project_root()
    results_dir = args.results_dir or (root / cfg["paths"]["results"])
    results_dir = Path(results_dir)

    if not results_dir.exists():
        logger.error(
            "Results dir %s does not exist. Run Stage 3 first (or pass --results-dir).",
            results_dir,
        )
        return 2

    output_dir = args.output_dir or (results_dir / "final_figures")
    output_dir = Path(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        logger.error(
            "Output dir %s already has content. Re-run with --force to overwrite.",
            output_dir,
        )
        return 3

    ensure_dir(output_dir)

    # --- 1. Load and aggregate -----------------------------------------------
    logger.info("Loading Stage 3 results from %s ...", results_dir / "training")
    df = load_all_results(results_dir)
    logger.info("Loaded %d fold-level rows.", len(df))

    if df.empty:
        logger.warning(
            "No Stage 3 results were found. Writing empty tables and a manifest and exiting."
        )

    logger.info("Loading selected epoch budgets ...")
    budgets_df = load_selected_budgets(results_dir)
    logger.info("Loaded %d selected_budget rows.", len(budgets_df))

    # --- 2. Tables -----------------------------------------------------------
    aggregated = aggregate_mean_std(df)
    aggregated_path = output_dir / "aggregated_results.csv"
    aggregated.to_csv(aggregated_path, index=False)
    logger.info("Wrote %s (%d rows).", aggregated_path, len(aggregated))

    budgets_path = output_dir / "selected_budgets.csv"
    budgets_df.to_csv(budgets_path, index=False)
    logger.info("Wrote %s (%d rows).", budgets_path, len(budgets_df))

    raw_path = output_dir / "raw_fold_results.csv"
    df.drop(columns=["confusion_matrix"], errors="ignore").to_csv(raw_path, index=False)
    logger.info("Wrote %s (%d rows).", raw_path, len(df))

    # --- 3. Statistical tests ------------------------------------------------
    stats_vs_baseline, stats_vs_clean = _build_statistical_tables(df)

    stats_vs_baseline_path = output_dir / "statistical_tests_vs_baseline.csv"
    stats_vs_baseline.to_csv(stats_vs_baseline_path, index=False)
    logger.info(
        "Wrote %s (%d rows across %d metrics).",
        stats_vs_baseline_path, len(stats_vs_baseline), len(STATS_METRICS),
    )

    stats_vs_clean_path = output_dir / "statistical_tests_noise_sensitivity.csv"
    stats_vs_clean.to_csv(stats_vs_clean_path, index=False)
    logger.info(
        "Wrote %s (%d rows across %d metrics).",
        stats_vs_clean_path, len(stats_vs_clean), len(STATS_METRICS),
    )

    # --- 4. Figures ----------------------------------------------------------
    if not df.empty:
        _make_per_condition_plots(df, output_dir)
        _make_cross_condition_plots(df, output_dir)
    else:
        logger.warning("Skipping plots — no fold-level data loaded.")
    plots_written = len(list(output_dir.rglob("*.png")))
    logger.info("Wrote %d figure files.", plots_written)

    # --- 5. Manifest ---------------------------------------------------------
    manifests_dir = root / cfg["paths"]["manifests"]
    ensure_dir(manifests_dir)
    manifest_path = manifests_dir / "stage4_analyze.json"
    write_manifest(
        manifest_path,
        stage="stage4_analyze",
        params={
            "results_dir": str(results_dir),
            "output_dir": str(output_dir),
            "force": bool(args.force),
            "stats_metrics": list(STATS_METRICS),
        },
        outputs=[
            str(aggregated_path),
            str(budgets_path),
            str(stats_vs_baseline_path),
            str(stats_vs_clean_path),
            str(raw_path),
        ],
        extra={
            "n_fold_rows": int(len(df)),
            "n_aggregated_rows": int(len(aggregated)),
            "n_budget_rows": int(len(budgets_df)),
            "n_stats_vs_baseline_rows": int(len(stats_vs_baseline)),
            "n_stats_vs_clean_rows": int(len(stats_vs_clean)),
            "n_plots_written": int(plots_written),
        },
    )
    logger.info("Wrote manifest %s.", manifest_path)
    logger.info("Stage 4 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
