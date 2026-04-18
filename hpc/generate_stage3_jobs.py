"""Enumerate the Stage 3 experiment grid and emit LSF ``bsub`` commands.

The full Stage 3 grid is 1,920 jobs::

    4 methods × 2 datasets × 2 init × 2 optim × 6 tau × 10 folds = 1920

This script emits one ``bsub`` command per job to stdout (or ``--output-file``).
It is filterable on every axis, which is essential for resubmission: if
AsyCo fails on the imbalanced dataset, you can generate just those 240 jobs
without rebuilding the whole pipeline by hand.

Usage::

    # All 1920 jobs:
    python -m hpc.generate_stage3_jobs > stage3_jobs.txt

    # Only AsyCo on imbalanced:
    python -m hpc.generate_stage3_jobs --method asyco --dataset imbalanced

    # A single job (sanity check):
    python -m hpc.generate_stage3_jobs \\
        --method elr --dataset balanced --init pretrained --optim sgd \\
        --tau 0.3 --fold 7

    # Then actually submit:
    bash stage3_jobs.txt
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import yaml

METHODS: tuple[str, ...] = ("baseline", "sce", "elr", "asyco")
DATASETS: tuple[str, ...] = ("balanced", "imbalanced")
INITS: tuple[str, ...] = ("pretrained", "scratch")
OPTIMS: tuple[str, ...] = ("sgd", "adam")
TAUS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
FOLDS: tuple[int, ...] = tuple(range(10))

EXPECTED_TOTAL: int = (
    len(METHODS) * len(DATASETS) * len(INITS) * len(OPTIMS) * len(TAUS) * len(FOLDS)
)
assert EXPECTED_TOTAL == 1920, "Grid math drifted — check the axes."

_HPC_DIR = Path(__file__).resolve().parent
_DEFAULT_LSF_YAML = _HPC_DIR / "lsf_defaults.yaml"


def load_defaults(path: Path) -> dict:
    """Load the LSF defaults YAML.

    Args:
        path: Path to ``lsf_defaults.yaml``.

    Returns:
        Parsed YAML as a dict. Raises ``FileNotFoundError`` if missing — the
        campaign depends on these settings, we do not silently fall back.
    """
    if not path.exists():
        raise FileNotFoundError(f"LSF defaults not found at {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def _tau_tag(tau: float) -> str:
    """Tag used in job names/log filenames: ``0.1 → 'tau10'``."""
    return f"tau{int(round(tau * 100)):02d}"


def _closest_tau(target: float) -> float:
    """Snap a user-supplied τ to the nearest grid point.

    This avoids silent mismatches when the user types ``--tau 0.30000001``.
    """
    return min(TAUS, key=lambda t: abs(t - target))


def build_command(
    method: str,
    dataset: str,
    init: str,
    optim: str,
    tau: float,
    fold: int,
    defaults: dict,
) -> str:
    """Build a single ``bsub`` command for one (method, dataset, init, optim, tau, fold).

    Args:
        method, dataset, init, optim, tau, fold: Grid coordinates.
        defaults: LSF defaults dict from :func:`load_defaults`.

    Returns:
        A single-line shell command ready to be piped to bash.
    """
    queue = defaults["queue"]
    walltime = defaults["walltime"]
    memory_gb = defaults["memory_gb"]
    gpu_spec = defaults["gpu_spec"]
    log_dir = defaults["log_dir"]
    job_prefix = defaults["job_prefix"]

    tau_str = _tau_tag(tau)
    job_name = (
        f"{job_prefix}_{method}_{dataset}_{init}_{optim}_{tau_str}_fold{fold:02d}"
    )
    log_stem = (
        f"{log_dir}/stage3_{method}_{dataset}_{init}_{optim}_{tau_str}_fold{fold:02d}"
    )

    inner = (
        f"python -m scripts.stage3_train "
        f"--method {method} --dataset {dataset} "
        f"--init {init} --optim {optim} "
        f"--tau {tau} --fold {fold}"
    )

    return (
        f'bsub -q {queue} -W {walltime} '
        f'-R "rusage[mem={memory_gb}GB]" '
        f'-gpu "{gpu_spec}" '
        f'-J "{job_name}" '
        f'-o "{log_stem}.out" -e "{log_stem}.err" '
        f'"{inner}"'
    )


def enumerate_grid(
    method: str | None = None,
    dataset: str | None = None,
    init: str | None = None,
    optim: str | None = None,
    tau: float | None = None,
    fold: int | None = None,
) -> list[tuple[str, str, str, str, float, int]]:
    """Enumerate the filtered subset of the Stage 3 grid.

    Args:
        method, dataset, init, optim, tau, fold: If given, restrict to that value.

    Returns:
        List of (method, dataset, init, optim, tau, fold) tuples.
    """
    methods = (method,) if method else METHODS
    datasets = (dataset,) if dataset else DATASETS
    inits = (init,) if init else INITS
    optims = (optim,) if optim else OPTIMS
    taus = (_closest_tau(tau),) if tau is not None else TAUS
    folds = (fold,) if fold is not None else FOLDS

    return list(itertools.product(methods, datasets, inits, optims, taus, folds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate the Stage 3 training grid and emit bsub commands. "
            f"Full grid is {EXPECTED_TOTAL} jobs."
        )
    )
    parser.add_argument("--method", choices=METHODS, help="Filter by method.")
    parser.add_argument("--dataset", choices=DATASETS, help="Filter by dataset.")
    parser.add_argument("--init", choices=INITS, help="Filter by initialization.")
    parser.add_argument("--optim", choices=OPTIMS, help="Filter by optimizer.")
    parser.add_argument("--tau", type=float, help="Filter by noise rate (snapped to grid).")
    parser.add_argument("--fold", type=int, choices=list(FOLDS), help="Filter by fold.")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Write commands to this file. Default: stdout.",
    )
    parser.add_argument(
        "--lsf-defaults",
        type=Path,
        default=_DEFAULT_LSF_YAML,
        help="Path to LSF defaults YAML.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Print only the count of matching jobs and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    defaults = load_defaults(args.lsf_defaults)

    grid = enumerate_grid(
        method=args.method,
        dataset=args.dataset,
        init=args.init,
        optim=args.optim,
        tau=args.tau,
        fold=args.fold,
    )

    if args.count_only:
        print(len(grid))
        return 0

    lines = [build_command(*coords, defaults=defaults) for coords in grid]

    # Sanity check: with no filters, we must emit exactly 1920.
    if (
        args.method is None
        and args.dataset is None
        and args.init is None
        and args.optim is None
        and args.tau is None
        and args.fold is None
    ):
        assert len(lines) == EXPECTED_TOTAL, (
            f"Expected {EXPECTED_TOTAL} jobs but enumerated {len(lines)}. "
            "Check the axes in hpc/generate_stage3_jobs.py."
        )

    if args.output_file is None:
        print("\n".join(lines))
    else:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text("\n".join(lines) + "\n")
        # Also report to stderr so users see it even when redirecting stdout
        est_hours = math.ceil(len(lines) * 24 / 10)  # crude: 24h max / 10 concurrent
        sys.stderr.write(
            f"Wrote {len(lines)} bsub commands to {args.output_file} "
            f"(≤{est_hours}h wall-clock at 10 concurrent jobs).\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
