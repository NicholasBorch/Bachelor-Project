"""Stage 3: the main training entry point.

One invocation == one job == one (method, dataset, init, optim, tau, fold)
tuple. The grid is 4 × 2 × 2 × 2 × 6 × 10 = 1,920 jobs total, as enumerated
by `hpc/generate_stage3_jobs.py`.

Prerequisites validated at startup:
  - Stage 1c has produced `train_noisy.csv` and `test_clean.csv` for
    (dataset, feature_driven, tau, fold).
  - Stage 2 has produced `selected_budget.json` for (dataset, method).

Idempotent by default: if `test_metrics.json` already exists in the target
output directory, the script refuses to overwrite unless `--force` is passed.

Noise type is locked to `feature_driven` (standard and normalized IDN are
characterized in Stage 1 for the methodology section only — see
PROJECT_DOCUMENTATION §2.1).

At the end of each Stage 3 run the model is also evaluated on the training
set with test-time transforms to compute NTA and LNMR (see
PROJECT_DOCUMENTATION §2.4 for definitions). These are saved in
`test_metrics.json` alongside the test-set metrics.

Run:
    python -m scripts.stage3_train \\
        --method elr --dataset imbalanced --init pretrained \\
        --optim sgd --tau 0.2 --fold 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from src.training.runner import run_training
from src.utils.io import load_config, project_root
from src.utils.manifest import write_manifest
from src.utils.seed import fold_seed


def _tau_dirname(tau: float) -> str:
    """Matches scripts/stage1c_inject_noise.py exactly. tau=0.1 -> 'tau_10'."""
    return f"tau_{int(round(tau * 100)):02d}"


def _fold_dirname(fold: int) -> str:
    return f"fold_{fold:02d}"


def _load_selected_epochs(cfg: dict, dataset: str, method: str) -> int:
    root = project_root()
    path = (
        root / cfg["paths"]["results"] / "epoch_selection"
        / dataset / method / "selected_budget.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 2 selected budget not found at {path}. "
            f"Run `scripts.stage2_select_epoch_budget` for every fold "
            f"and then `scripts.stage2_aggregate_epoch_budget` "
            f"for ({dataset}, {method})."
        )
    with open(path, "r") as f:
        data = json.load(f)
    se = int(data["selected_epochs"])
    cap = int(cfg.get("epoch_cap", 100))
    return min(se, cap)


def _load_csvs(cfg: dict, dataset: str, tau: float, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _maybe_float(x) -> float | None:
    """Coerce a possibly-None / NaN field into a float suitable for the manifest.

    NTA/LNMR are NaN at τ=0 and None if dx_clean wasn't available. We store
    NaN as ``None`` in the manifest (cleaner JSON) and preserve real numbers.
    """
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
    # Validate tau is one of the pre-registered values
    valid_taus = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    if not any(abs(args.tau - t) < 1e-6 for t in valid_taus):
        print(f"ERROR: --tau must be one of {valid_taus}, got {args.tau}", file=sys.stderr)
        return 2

    cfg = load_config(
        "base.yaml",
        f"data/{args.dataset}.yaml",
        f"model/resnet34_{args.init}.yaml",
        f"optim/{args.optim}.yaml",
        f"method/{args.method}.yaml",
        "noise/feature_driven.yaml",
    )
    root = project_root()

    # Output directory first — needed for idempotency check before we do
    # any expensive work.
    out_dir = (
        root / cfg["paths"]["results"] / "training"
        / args.method / args.dataset / f"{args.init}_{args.optim}"
        / _tau_dirname(args.tau) / _fold_dirname(args.fold)
    )
    if (out_dir / "test_metrics.json").exists() and not args.force:
        print(
            f"[stage3] {out_dir} already has test_metrics.json — skipping. "
            f"Pass --force to overwrite."
        )
        return 0

    # Prerequisite: Stage 1c CSVs
    train_df, test_df = _load_csvs(cfg, args.dataset, args.tau, args.fold)

    # Prerequisite: Stage 2 selected budget
    total_epochs = _load_selected_epochs(cfg, args.dataset, args.method)

    images_dir = root / cfg["paths"]["images"]

    print(
        f"[stage3] method={args.method} dataset={args.dataset} "
        f"init={args.init} optim={args.optim} tau={args.tau:.2f} "
        f"fold={args.fold} epochs={total_epochs} "
        f"train={len(train_df)} test={len(test_df)}"
    )

    test_metrics = run_training(
        cfg=cfg,
        train_df=train_df,
        test_df=test_df,
        images_dir=images_dir,
        method_name=args.method,
        total_epochs=int(total_epochs),
        output_dir=out_dir,
        val_df=None,  # Stage 3: train + test only, no validation
        seed=fold_seed(int(cfg["seed"]), int(args.fold)),
    )

    # Per-job manifest entry
    manifest_path = (
        root / cfg["paths"]["manifests"]
        / f"stage3_{args.method}_{args.dataset}_{args.init}_{args.optim}_"
          f"{_tau_dirname(args.tau)}_{_fold_dirname(args.fold)}.json"
    )
    write_manifest(
        manifest_path,
        stage="stage3_train",
        params={
            "method": args.method,
            "dataset": args.dataset,
            "init": args.init,
            "optim": args.optim,
            "tau": float(args.tau),
            "fold": int(args.fold),
            "total_epochs": int(total_epochs),
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
            # Noise-label interaction (NaN at τ=0, persisted as null in manifest)
            "nta": _maybe_float(test_metrics.get("nta")),
            "lnmr": _maybe_float(test_metrics.get("lnmr")),
            "n_flipped": test_metrics.get("n_flipped"),
        },
    )
    print(
        f"[stage3] DONE -> {out_dir}  "
        f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Stage 3: train one (method, dataset, init, optim, tau, fold) config"
    )
    p.add_argument("--method", required=True, choices=["baseline", "sce", "elr", "asyco"])
    p.add_argument("--dataset", required=True, choices=["balanced", "imbalanced"])
    p.add_argument("--init", required=True, choices=["pretrained", "scratch"])
    p.add_argument("--optim", required=True, choices=["sgd", "adam"])
    p.add_argument("--tau", required=True, type=float,
                   help="noise rate in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}")
    p.add_argument("--fold", required=True, type=int, help="fold id, 0..9")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing test_metrics.json")
    sys.exit(main(p.parse_args()))
