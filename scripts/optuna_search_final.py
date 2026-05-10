"""FINAL Optuna search runner — for the locked-in tuning experiment.

Distinguished from scripts/optuna_search.py by:
  - Three methods supported (elr, sce, asyco_divmix)
  - Validation labels are CLEAN (loaded from tau=0 fold by image_id matching)
  - Output tree: results/optuna_final/{method}/...
  - Study names suffixed _FINAL
  - Loads search spaces from configs/optuna_search_spaces_final.py

Split protocol:
  1. Load tau=0.0 train CSV (clean labels for stratification + val).
  2. Load tau=0.2 train CSV (noisy labels for training).
  3. Carve a 15% stratified slice by image_id, stratifying on clean labels.
  4. The 85% train portion uses NOISY labels from tau=0.2.
  5. The 15% val portion uses CLEAN labels from tau=0.0.

This matches the experimental protocol where the model trains on noisy data
but is validated on clean ground-truth labels.

Usage:
  python -m scripts.optuna_search_final --method elr --fold 9 --n-trials 50

  # Resume an existing study (chunk 2+):
  python -m scripts.optuna_search_final --method elr --fold 9 --n-trials 50 --resume
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedShuffleSplit

from configs.optuna_search_spaces_final import sample as sample_hyperparams
from src.training.runner import run_training, StopTraining
from src.utils.io import load_config, project_root
from src.utils.seed import fold_seed


def _apply_hyperparams_to_cfg(cfg: dict, method: str, hp: dict) -> dict:
    cfg = dict(cfg)
    cfg["method"] = dict(cfg["method"])
    for k, v in hp.items():
        cfg["method"][k] = v
    return cfg


def _tau_dirname(tau: float) -> str:
    return f"tau_{int(round(tau * 100)):02d}"


def _load_train_csv(cfg: dict, dataset: str, tau: float, fold: int) -> pd.DataFrame:
    root = project_root()
    path = (
        root / cfg["paths"]["cv_folds"] / dataset
        / "feature_driven" / _tau_dirname(tau)
        / f"fold_{fold:02d}" / "train_noisy.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Training CSV not found at {path}. "
            f"Run stage1c (feature_driven, tau={tau}, fold {fold}) first."
        )
    return pd.read_csv(path)


def _carve_train_val_with_clean_val(
    train_noisy_df: pd.DataFrame,
    train_clean_df: pd.DataFrame,
    seed: int,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a stratified 85/15 split with clean val labels.

    The 85% train portion uses NOISY labels from train_noisy_df (tau=0.2).
    The 15% val portion uses CLEAN labels from train_clean_df (tau=0.0).
    Stratification is on clean labels (more reliable than noisy).

    Both DataFrames must contain the same image_id values.
    """
    noisy_ids = set(train_noisy_df["image_id"])
    clean_ids = set(train_clean_df["image_id"])
    if noisy_ids != clean_ids:
        raise ValueError(
            f"Image ID mismatch between tau=0.2 ({len(noisy_ids)}) and "
            f"tau=0.0 ({len(clean_ids)}) CSVs. "
            f"Intersection {len(noisy_ids & clean_ids)}, "
            f"only in noisy {len(noisy_ids - clean_ids)}, "
            f"only in clean {len(clean_ids - noisy_ids)}. "
            f"Verify both CSVs are from the same fold."
        )

    # Reorder train_clean_df rows to match train_noisy_df's image_id order
    clean_aligned = (
        train_clean_df.set_index("image_id")
        .loc[train_noisy_df["image_id"].values]
        .reset_index()
    )

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=int(seed),
    )
    train_idx, val_idx = next(
        splitter.split(clean_aligned["image_id"], clean_aligned["dx"])
    )

    # Train: noisy labels from tau=0.2
    train_part = train_noisy_df.iloc[train_idx].reset_index(drop=True)
    # Val: clean labels from tau=0.0
    val_part = clean_aligned.iloc[val_idx].reset_index(drop=True)
    return train_part, val_part


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _peak_val_balanced_accuracy(
    log_path: Path, smooth_window: int = 5,
) -> tuple[float, int]:
    records = _read_jsonl(log_path)
    if not records:
        return float("-inf"), 0
    ba = np.array(
        [r.get("val_balanced_accuracy", float("nan")) for r in records],
        dtype=np.float64,
    )
    if np.isnan(ba).all():
        return float("-inf"), 0
    smoothed = np.empty_like(ba)
    csum = np.cumsum(np.nan_to_num(ba, nan=0.0))
    valid = np.cumsum(~np.isnan(ba))
    for i in range(len(ba)):
        lo = max(0, i - smooth_window + 1)
        n = valid[i] - (valid[lo - 1] if lo > 0 else 0)
        if n == 0:
            smoothed[i] = float("-inf")
        else:
            s = csum[i] - (csum[lo - 1] if lo > 0 else 0.0)
            smoothed[i] = s / n
    best_idx = int(np.argmax(smoothed))
    return float(smoothed[best_idx]), best_idx


def _make_pruning_callback(
    trial: optuna.Trial,
    smooth_window: int = 5,
    min_epoch_for_pruning: int = 0,
):
    """Build an epoch_callback that reports smoothed val BA to Optuna and
    raises StopTraining if Optuna's pruner says so.
    """
    history: list[float] = []

    def _callback(epoch: int, record: dict) -> None:
        ba = record.get("val_balanced_accuracy")
        if ba is None or not (isinstance(ba, (int, float)) and ba == ba):
            return
        history.append(float(ba))
        lo = max(0, len(history) - smooth_window)
        smoothed = sum(history[lo:]) / max(len(history) - lo, 1)
        trial.report(smoothed, step=int(epoch))
        if int(epoch) < int(min_epoch_for_pruning):
            return
        if trial.should_prune():
            raise StopTraining(
                f"Optuna pruning at epoch {epoch} "
                f"(smoothed val BA = {smoothed:.4f})"
            )

    return _callback


def make_objective(
    args: argparse.Namespace,
    base_cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    images_dir: Path,
    out_dir: Path,
):
    def objective(trial: optuna.Trial) -> float:
        trial_dir = out_dir / "per_trial" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        hp = sample_hyperparams(args.method, trial)
        cfg = _apply_hyperparams_to_cfg(base_cfg, args.method, hp)

        pruning_cb = _make_pruning_callback(
            trial,
            smooth_window=int(args.objective_smooth_window),
            min_epoch_for_pruning=int(args.pruner_n_warmup_steps),
        )

        was_pruned = False
        try:
            run_training(
                cfg=cfg,
                train_df=train_df,
                test_df=None,
                images_dir=images_dir,
                method_name=args.method,
                total_epochs=int(args.trial_epochs),
                output_dir=trial_dir,
                val_df=val_df,
                seed=fold_seed(int(cfg["seed"]), int(args.fold)),
                epoch_callback=pruning_cb,
            )
        except StopTraining as e:
            print(f"[trial {trial.number}] PRUNED: {e}", flush=True)
            was_pruned = True
        except Exception as e:
            print(
                f"[trial {trial.number}] FAILED: {type(e).__name__}: {e}",
                file=sys.stderr, flush=True,
            )
            with (trial_dir / "trial_summary.json").open("w") as f:
                json.dump(
                    {"failed": True, "error": str(e), "params": hp},
                    f, indent=2,
                )
            return float("-inf")

        log_path = trial_dir / "training_log.jsonl"
        peak_ba, peak_epoch = _peak_val_balanced_accuracy(log_path)

        summary = {
            "trial_number": int(trial.number),
            "params": hp,
            "peak_val_balanced_accuracy": float(peak_ba),
            "peak_epoch": int(peak_epoch),
            "trial_epochs": int(args.trial_epochs),
            "pruned": bool(was_pruned),
        }
        with (trial_dir / "trial_summary.json").open("w") as f:
            json.dump(summary, f, indent=2, default=float)

        if was_pruned:
            raise optuna.TrialPruned()
        return peak_ba

    return objective


def main(args: argparse.Namespace) -> int:
    base_cfg = load_config(
        "base.yaml",
        f"data/{args.dataset}.yaml",
        f"method/{args.method}.yaml",
        f"optim/{args.optim}.yaml",
        f"model/{args.model}.yaml",
        "noise/feature_driven.yaml",
    )
    root = project_root()
    images_dir = root / base_cfg["paths"]["images"]

    # Load both noisy and clean training CSVs for the SAME fold.
    train_noisy_df = _load_train_csv(base_cfg, args.dataset, args.tau, args.fold)
    train_clean_df = _load_train_csv(base_cfg, args.dataset, 0.0, args.fold)

    train_df, val_df = _carve_train_val_with_clean_val(
        train_noisy_df, train_clean_df,
        seed=int(base_cfg["seed"]),
        val_frac=0.15,
    )

    # Output tree: results/optuna_final/{method}/{dataset}/{optim}_{model}/tau_XX/fold_XX/
    out_dir = (
        root / base_cfg["paths"]["results"] / "optuna_final"
        / args.method / args.dataset
        / f"{args.optim}_{args.model}"
        / _tau_dirname(args.tau) / f"fold_{args.fold:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    search_config = {
        "protocol": "FINAL",
        "method": args.method,
        "dataset": args.dataset,
        "optim": args.optim,
        "model": args.model,
        "tau_train": float(args.tau),
        "tau_val": 0.0,
        "fold": int(args.fold),
        "fold_selection": "random.Random(10).randint(0, 9) -> 9",
        "n_trials": int(args.n_trials),
        "trial_epochs": int(args.trial_epochs),
        "tpe_seed": int(args.tpe_seed),
        "tpe_n_startup_trials": int(args.tpe_n_startup_trials),
        "pruner": "MedianPruner",
        "pruner_n_startup_trials": int(args.pruner_n_startup_trials),
        "pruner_n_warmup_steps": int(args.pruner_n_warmup_steps),
        "val_fraction": 0.15,
        "val_labels": "clean (loaded from tau=0.0 by image_id match)",
        "objective": "peak_smoothed_val_balanced_accuracy",
        "objective_smooth_window": int(args.objective_smooth_window),
        "pruning_enabled": True,
    }
    with (out_dir / "search_config.json").open("w") as f:
        json.dump(search_config, f, indent=2)

    print(
        f"[optuna FINAL] method={args.method} dataset={args.dataset} "
        f"fold={args.fold}", flush=True,
    )
    print(
        f"[optuna FINAL] train(noisy tau={args.tau})={len(train_df)}  "
        f"val(clean tau=0.0)={len(val_df)}  "
        f"n_trials={args.n_trials}  epochs/trial={args.trial_epochs}",
        flush=True,
    )
    print(f"[optuna FINAL] output -> {out_dir}", flush=True)

    storage_url = f"sqlite:///{out_dir / 'study.db'}"
    sampler = TPESampler(
        seed=int(args.tpe_seed),
        multivariate=True,
        n_startup_trials=int(args.tpe_n_startup_trials),
    )
    pruner = MedianPruner(
        n_startup_trials=int(args.pruner_n_startup_trials),
        n_warmup_steps=int(args.pruner_n_warmup_steps),
    )
    study = optuna.create_study(
        study_name=f"{args.method}_fold{args.fold:02d}_FINAL",
        storage=storage_url,
        load_if_exists=bool(args.resume),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(args, base_cfg, train_df, val_df, images_dir, out_dir)

    t_start = time.time()
    study.optimize(objective, n_trials=int(args.n_trials), gc_after_trial=True)
    elapsed = time.time() - t_start

    print(f"\n[optuna FINAL] DONE in {elapsed/3600:.2f} h", flush=True)
    try:
        print(f"[optuna FINAL] best value: {study.best_value:.4f}", flush=True)
        print(f"[optuna FINAL] best params:", flush=True)
        for k, v in study.best_params.items():
            print(f"          {k}: {v}", flush=True)
    except ValueError:
        print("[optuna FINAL] no completed trials", flush=True)

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="FINAL Optuna search for noisy-label methods (ELR, SCE, AsyCo)",
    )
    p.add_argument("--method", required=True,
                   choices=["elr", "sce", "asyco_divmix"])
    p.add_argument("--dataset", default="imbalanced",
                   choices=["balanced", "imbalanced"])
    p.add_argument("--optim", default="adam", choices=["sgd", "adam"])
    p.add_argument("--model", default="resnet34_pretrained",
                   choices=["resnet34_pretrained", "resnet34_scratch"])
    p.add_argument("--tau", default=0.2, type=float,
                   help="Training noise rate. Val always uses tau=0.0.")
    p.add_argument("--fold", default=9, type=int,
                   help="Tuning fold (default 9 from random.Random(10)).")
    p.add_argument("--n-trials", default=50, type=int)
    p.add_argument("--trial-epochs", default=150, type=int)
    p.add_argument("--tpe-seed", default=42, type=int)
    p.add_argument("--tpe-n-startup-trials", default=15, type=int)
    p.add_argument("--pruner-n-startup-trials", default=10, type=int)
    p.add_argument("--pruner-n-warmup-steps", default=30, type=int)
    p.add_argument("--objective-smooth-window", default=5, type=int)
    p.add_argument("--resume", action="store_true")
    sys.exit(main(p.parse_args()))
