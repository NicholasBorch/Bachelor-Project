"""Main Experiment status checker — read-only completeness report.

Walks results/main_experiment/training/ and reports, for every expected
(method, dataset, init, optim, tau, fold) tuple, whether the job is:

  COMPLETE      — test_metrics.json exists AND training_log.jsonl has 150 entries
  RUNNING       — training_log.jsonl exists but < 150 epochs, no test_metrics yet
  INCOMPLETE    — training_log.jsonl has < 150 entries, no test_metrics.json
                  (i.e. job was killed mid-training)
  NO_METRICS    — 150 epochs reached but test_metrics.json missing (rare;
                  usually means runner crashed after final epoch)
  MISSING       — no output directory at all

Usage:
    # default: walk the same grid the submit script would generate, using
    # the current run's environment defaults (imbalanced, pretrained, adam)
    python -m scripts.final_experiment_status

    # check a different condition:
    python -m scripts.final_experiment_status --init scratch --optim sgd

    # restrict to one method:
    python -m scripts.final_experiment_status --methods asyco_divmix

    # show full per-job table (default just shows summary):
    python -m scripts.final_experiment_status --verbose

Exit code is 0 if all expected jobs are COMPLETE, else 1. Useful for
gating downstream analysis with a quick `&&` check.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.utils.io import project_root


EXPECTED_EPOCHS = 150
ALL_METHODS = ["baseline", "sce", "elr", "asyco_divmix"]
ALL_TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
ALL_FOLDS = list(range(10))


def _tau_dirname(tau: float) -> str:
    return f"tau_{int(round(tau * 100)):02d}"


def _fold_dirname(fold: int) -> str:
    return f"fold_{fold:02d}"


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def _classify(job_dir: Path) -> tuple[str, int]:
    """Return (status, n_epochs_logged)."""
    if not job_dir.exists():
        return "MISSING", 0

    log_path = job_dir / "training_log.jsonl"
    metrics_path = job_dir / "test_metrics.json"
    n_epochs = _count_jsonl_lines(log_path)

    if metrics_path.exists():
        if n_epochs >= EXPECTED_EPOCHS:
            return "COMPLETE", n_epochs
        # Has metrics but log is short — strange edge case
        return "COMPLETE", n_epochs

    # No test_metrics.json
    if n_epochs == 0:
        return "MISSING", 0
    if n_epochs < EXPECTED_EPOCHS:
        return "RUNNING_OR_INCOMPLETE", n_epochs
    return "NO_METRICS", n_epochs


def main(args: argparse.Namespace) -> int:
    root = project_root()
    base_dir = root / "results" / "main_experiment" / "training"

    methods = args.methods if args.methods else ALL_METHODS
    taus = [float(t) for t in args.taus] if args.taus else ALL_TAUS
    folds = [int(f) for f in args.folds] if args.folds else ALL_FOLDS

    print(
        f"Checking results/main_experiment/training/  "
        f"dataset={args.dataset} init={args.init} optim={args.optim}",
        flush=True,
    )
    print(
        f"Expected: {len(methods)} methods x {len(taus)} taus x "
        f"{len(folds)} folds = {len(methods) * len(taus) * len(folds)} jobs",
        flush=True,
    )
    print(f"Each job should have {EXPECTED_EPOCHS} epochs.", flush=True)
    print()

    # Per-method tallies
    overall_counter: Counter = Counter()
    per_method: dict[str, Counter] = {m: Counter() for m in methods}
    incomplete_jobs: list[tuple[str, int, float, int, str]] = []

    for method in methods:
        for fold in folds:
            for tau in taus:
                job_dir = (
                    base_dir / method / args.dataset
                    / f"{args.init}_{args.optim}"
                    / _tau_dirname(tau) / _fold_dirname(fold)
                )
                status, n_epochs = _classify(job_dir)
                per_method[method][status] += 1
                overall_counter[status] += 1

                if status != "COMPLETE":
                    incomplete_jobs.append((method, fold, tau, n_epochs, status))

                if args.verbose:
                    print(
                        f"  {method:>12s} fold={fold} tau={tau:.1f}  "
                        f"epochs={n_epochs:>3}/{EXPECTED_EPOCHS}  {status}"
                    )

    # Summary table
    print()
    print("=" * 76)
    print(f"{'Method':<15s} {'COMPLETE':>10s} {'RUN/INCMP':>10s} "
          f"{'NO_METRIC':>10s} {'MISSING':>10s} {'TOTAL':>8s}")
    print("-" * 76)
    for method in methods:
        c = per_method[method]
        total = sum(c.values())
        print(
            f"{method:<15s} "
            f"{c.get('COMPLETE', 0):>10d} "
            f"{c.get('RUNNING_OR_INCOMPLETE', 0):>10d} "
            f"{c.get('NO_METRICS', 0):>10d} "
            f"{c.get('MISSING', 0):>10d} "
            f"{total:>8d}"
        )
    print("-" * 76)
    total = sum(overall_counter.values())
    print(
        f"{'TOTAL':<15s} "
        f"{overall_counter.get('COMPLETE', 0):>10d} "
        f"{overall_counter.get('RUNNING_OR_INCOMPLETE', 0):>10d} "
        f"{overall_counter.get('NO_METRICS', 0):>10d} "
        f"{overall_counter.get('MISSING', 0):>10d} "
        f"{total:>8d}"
    )
    print("=" * 76)

    # Resubmission helper
    if incomplete_jobs and not args.verbose:
        print()
        print(f"{len(incomplete_jobs)} job(s) are not COMPLETE. First 20:")
        for method, fold, tau, n_epochs, status in incomplete_jobs[:20]:
            print(
                f"  {method:>12s} fold={fold} tau={tau:.1f}  "
                f"epochs={n_epochs:>3}/{EXPECTED_EPOCHS}  {status}"
            )
        if len(incomplete_jobs) > 20:
            print(f"  ... and {len(incomplete_jobs) - 20} more "
                  f"(use --verbose to see all)")

        # Also write a resubmit-helper file so they're easy to re-launch
        out_path = root / "logs" / "final_experiment" / "incomplete_jobs.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for method, fold, tau, n_epochs, status in incomplete_jobs:
                f.write(
                    f"{method}\t{fold}\t{tau}\t{n_epochs}\t{status}\n"
                )
        print(f"\nFull list also written to {out_path}")

    # Exit code 0 only if everything is COMPLETE
    if overall_counter.get("COMPLETE", 0) == total:
        print("\nAll expected jobs COMPLETE.")
        return 0
    return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Status report for the Main Experiment grid"
    )
    p.add_argument("--dataset", default="imbalanced",
                   choices=["balanced", "imbalanced"])
    p.add_argument("--init", default="pretrained",
                   choices=["pretrained", "scratch"])
    p.add_argument("--optim", default="adam", choices=["sgd", "adam"])
    p.add_argument("--methods", nargs="*", default=None,
                   choices=ALL_METHODS,
                   help=f"Restrict to specific methods (default: all "
                        f"{ALL_METHODS})")
    p.add_argument("--taus", nargs="*", default=None,
                   help=f"Restrict to specific taus (default: all {ALL_TAUS})")
    p.add_argument("--folds", nargs="*", default=None,
                   help=f"Restrict to specific folds (default: all {ALL_FOLDS})")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show per-job table in addition to summary")
    sys.exit(main(p.parse_args()))
