"""Analysis utilities for Stage 4 (aggregation, plotting, statistics).

This package walks the Stage 3 results tree and produces the final figures,
tables, and statistical tests that go into the thesis. It is read-only with
respect to the training results and can be re-run freely.
"""
from __future__ import annotations

from src.analysis.aggregate import load_all_results, load_selected_budgets
from src.analysis.stats import bootstrap_ci, wilcoxon_vs_baseline

__all__ = [
    "bootstrap_ci",
    "load_all_results",
    "load_selected_budgets",
    "wilcoxon_vs_baseline",
]
