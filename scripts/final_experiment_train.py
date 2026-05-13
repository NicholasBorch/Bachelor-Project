"""Main Experiment training entry point — uses TUNED hyperparameters.

One invocation == one job == one (method, dataset, init, optim, tau, fold)
tuple. The grid is (by default) 4 methods x 1 dataset x 1 init x 1 optim x 6 tau
x 10 folds = 240 jobs, but every dimension is toggleable from the submit script.

DIFFERENCES vs scripts/stage3_train.py:
  - Reads tuned hyperparameters from results/optuna_final/.../best_config.yaml
    for SCE, ELR, AsyCo (paper version). Hard error if the tuned config is
    missing for a method that requires tuning.
  - Baseline uses configs/method/baseline.yaml unchanged (no tunable params).
  - asyco (Eq.5 only) is DROPPED. Only asyco_divmix is included.
  - Always 150 epochs (matches the Optuna trial budget — no Stage 2 lookup).
  - Output tree: results/main_experiment/training/...
  - Idempotent: skips if test_metrics.json exists, unless --force.

TUNED CONFIG RESOLUTION:
  results/optuna_final/{method}/{dataset}/{optim}_resnet34_{init}/
      tau_{tuning_tau*100:02d}/fold_{tuning_fold:02d}/best_config.yaml

  --tuning-fold (default 5) and --tuning-tau (default 0.2) parametrize the
  lookup so future re-tunes (at e.g. sgd_resnet34_scratch) can be wired in
  without script changes.

Run:
    # baseline doesn't need tuning:
    python -m scripts.final_experiment_train \\
        --method baseline --dataset imbalanced --init pretrained \\
        --optim adam --tau 0.2 --fold 0

    # methods that use tuned configs:
    python -m scripts.final_experiment_train \\
        --method elr --dataset imbalanced --init pretrained \\
        --optim adam --tau 0.2 --fold 0

    # if you've later re-tuned on a different setup:
    python -m scripts.final_experiment_train \\
        --method asyco_divmix --dataset imbalanced --init scratch \\
        --optim sgd --tau 0.3 --fold 7 \\
        --tuning-fold 5 --tuning-tau 0.2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.training.runner import run_training
from src.utils.io import load_config, project_root
from src.utils.manifest import write_manifest
from src.utils.seed import fold_seed


# asyco intentionally dropped — only paper-faithful asyco_divmix is included.
METHOD_CHOICES = ["baseline", "sce", "elr", "asyco_divmix"]

# Fixed epoch budget for the main experiment. Matches Optuna trial budget;
# no Stage 2 epoch-selection lookup is performed.
EPOCHS = 150

# Methods that require a tuned best_config.yaml. Baseline has no tunable
# hyperparameters and is excluded from the tuning lookup.
METHODS_REQUIRING_TUNED_CONFIG = {"sce", "elr", "asyco_divmix"}


def _tau_dirname(tau: float) -> str:
    """tau=0.1 -> 'tau_10'. Matches scripts/stage1c_inject_noise.py."""
    return f"tau_{int(round(tau * 100)):02d}"


def _fold_dirname(fold: int) -> str:
    return f"fold_{fold:02d}"


def _load_csvs(
    cfg: dict, dataset: str, tau: float, fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = project_root()
    fold_dir = (
        root / cfg["paths"]["cv_folds"] / dataset
        / "feature_driven" / _tau_dirname(tau) / _fold_dirname(fold)
    )
    train_path = fold_dir / "train_noisy.csv"
    test_path = fold_dir / "test_clean.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Stage 1c outputs not found in {fold_dir}. "
            f"Run `scripts.stage1c_inject_noise --dataset {dataset} "
            f"--noise-type feature_driven --fold {fold}` first."
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def _tuned_config_path(
    method: str, dataset: str, init: str, optim: str,
    tuning_tau: float, tuning_fold: int,
) -> Path:
    """Resolve the path to the tuned best_config.yaml for this method.

    Format mirrors what scripts/optuna_search_final.py writes:
        results/optuna_final/{method}/{dataset}/{optim}_resnet34_{init}/
            tau_{XX}/fold_{YY}/best_config.yaml
    """
    root = project_root()
    return (
        root / "results" / "optuna_final"
        / method / dataset
        / f"{optim}_resnet34_{init}"
        / _tau_dirname(tuning_tau)
        / _fold_dirname(tuning_fold)
        / "best_config.yaml"
    )


def _apply_tuned_config(
    cfg: dict, method: str, dataset: str, init: str, optim: str,
    tuning_tau: float, tuning_fold: int,
) -> Path | None:
    """Overlay tuned hyperparameters onto cfg['method'] in-place.

    Returns the path to the tuned config that was loaded, or None if the
    method doesn't require tuning (baseline).

    Raises FileNotFoundError if the tuned config is required but missing.
    """
    if method not in METHODS_REQUIRING_TUNED_CONFIG:
        return None

    path = _tuned_config_path(method, dataset, init, optim, tuning_tau, tuning_fold)
    if not path.exists():
        raise FileNotFoundError(
            f"\n=== TUNED CONFIG MISSING ===\n"
            f"Method '{method}' requires a tuned best_config.yaml from the "
            f"FINAL Optuna search.\n"
            f"Expected at: {path}\n"
            f"\n"
            f"To produce it, run the FINAL Optuna search for "
            f"({method}, {dataset}, {init}, {optim}) at "
            f"tau={tuning_tau}, fold={tuning_fold}, then run "
            f"scripts.optuna_analyze_final.\n"
        )

    with path.open() as f:
        tuned = yaml.safe_load(f) or {}

    # Drop the provenance block — it's metadata, not method hyperparameters.
    provenance = tuned.pop("_optuna_provenance", None)

    # Replace cfg['method'] entirely. The tuned best_config.yaml started its
    # life as a copy of configs/method/{method}.yaml, then had the best
    # Optuna trial's parameters overlaid. So `tuned` is a complete drop-in
    # method config.
    cfg["method"] = tuned

    print(
        f"[final_experiment] loaded tuned config from {path}",
        flush=True,
    )
    if provenance is not None:
        print(
            f"[final_experiment] provenance: "
            f"trial {provenance.get('best_trial_number')}, "
            f"best value {provenance.get('best_value'):.4f} "
            f"({provenance.get('n_trials_completed')} completed trials)",
            flush=True,
        )
    return path


def _maybe_float(x) -> float | None:
    """NaN -> None for cleaner JSON manifests."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def main(args: argparse.Namespace) -> int:
    valid_taus = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    if not any(abs(args.tau - t) < 1e-6 for t in valid_taus):
        print(
            f"ERROR: --tau must be one of {valid_taus}, got {args.tau}",
            file=sys.stderr,
        )
        return 2

    # Load the BASE config the way Stage 3 does — gives us the resolved
    # dataset/model/optim/method/noise stack.
    cfg = load_config(
        "base.yaml",
        f"data/{args.dataset}.yaml",
        f"model/resnet34_{args.init}.yaml",
        f"optim/{args.optim}.yaml",
        f"method/{args.method}.yaml",
        "noise/feature_driven.yaml",
    )
    root = project_root()

    # Output directory — checked for idempotency BEFORE any expensive work.
    out_dir = (
        root / "results" / "main_experiment" / "training"
        / args.method / args.dataset / f"{args.init}_{args.optim}"
        / _tau_dirname(args.tau) / _fold_dirname(args.fold)
    )
    if (out_dir / "test_metrics.json").exists() and not args.force:
        print(
            f"[final_experiment] {out_dir} already has test_metrics.json — "
            f"skipping. Pass --force to overwrite.",
            flush=True,
        )
        return 0

    # Overlay tuned hyperparameters onto cfg['method']. Hard error if a
    # required tuned config is missing.
    tuned_config_used = _apply_tuned_config(
        cfg,
        method=args.method,
        dataset=args.dataset,
        init=args.init,
        optim=args.optim,
        tuning_tau=float(args.tuning_tau),
        tuning_fold=int(args.tuning_fold),
    )

    # Stage 1c CSVs — train at the requested tau, test on clean labels.
    train_df, test_df = _load_csvs(cfg, args.dataset, args.tau, args.fold)

    images_dir = root / cfg["paths"]["images"]

    print(
        f"[final_experiment] method={args.method} dataset={args.dataset} "
        f"init={args.init} optim={args.optim} tau={args.tau:.2f} "
        f"fold={args.fold} epochs={EPOCHS} "
        f"train={len(train_df)} test={len(test_df)}",
        flush=True,
    )

    test_metrics = run_training(
        cfg=cfg,
        train_df=train_df,
        test_df=test_df,
        images_dir=images_dir,
        method_name=args.method,
        total_epochs=int(EPOCHS),
        output_dir=out_dir,
        val_df=None,  # main experiment: train + clean test only, no val split
        seed=fold_seed(int(cfg["seed"]), int(args.fold)),
    )

    # Per-job manifest entry — records what was actually run.
    manifest_path = (
        root / cfg["paths"]["manifests"]
        / f"main_experiment_{args.method}_{args.dataset}_{args.init}_"
          f"{args.optim}_{_tau_dirname(args.tau)}_"
          f"{_fold_dirname(args.fold)}.json"
    )
    write_manifest(
        manifest_path,
        stage="final_experiment_train",
        params={
            "method": args.method,
            "dataset": args.dataset,
            "init": args.init,
            "optim": args.optim,
            "tau": float(args.tau),
            "fold": int(args.fold),
            "total_epochs": int(EPOCHS),
            "tuning_fold": int(args.tuning_fold),
            "tuning_tau": float(args.tuning_tau),
            "tuned_config": str(tuned_config_used) if tuned_config_used else None,
        },
        outputs=[
            str(out_dir / "config.yaml"),
            str(out_dir / "training_log.jsonl"),
            str(out_dir / "test_metrics.json"),
        ],
        extra={
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "macro_auc": float(test_metrics["macro_auc"]),
            "nta": _maybe_float(test_metrics.get("nta")),
            "lnmr": _maybe_float(test_metrics.get("lnmr")),
            "n_flipped": test_metrics.get("n_flipped"),
        },
    )
    print(
        f"[final_experiment] DONE -> {out_dir}  "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Main experiment: train one config with TUNED hyperparameters"
    )
    p.add_argument("--method", required=True, choices=METHOD_CHOICES)
    p.add_argument("--dataset", required=True, choices=["balanced", "imbalanced"])
    p.add_argument("--init", required=True, choices=["pretrained", "scratch"])
    p.add_argument("--optim", required=True, choices=["sgd", "adam"])
    p.add_argument("--tau", required=True, type=float,
                   help="noise rate in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}")
    p.add_argument("--fold", required=True, type=int, help="fold id, 0..9")
    p.add_argument("--tuning-fold", default=5, type=int,
                   help="Which fold the FINAL Optuna search was performed on (default 5)")
    p.add_argument("--tuning-tau", default=0.2, type=float,
                   help="Which tau the FINAL Optuna search was performed at (default 0.2)")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing test_metrics.json")
    sys.exit(main(p.parse_args()))
