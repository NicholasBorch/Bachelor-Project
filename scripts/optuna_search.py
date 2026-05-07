"""Optuna hyperparameter search for noisy-label methods.

Runs N trials of TPE-Bayesian optimization on a single fold, searching the
method's hyperparameter space (defined in configs/optuna_search_spaces.py)
to maximize validation balanced accuracy on a held-out 15% slice of the
fold's train set.

Search protocol decisions (defended in the thesis):

  - Search on noisy data (default: tau=0.2, feature_driven). Searching
    on clean data would optimize for the regime where noise robustness
    is irrelevant, which is the wrong question for our methods.

  - Single fold (default: 5). Standard nested-CV practice: hyperparameter
    selection on one fold; lock the chosen hyperparameters; final
    evaluation in Stage 3 uses all 10 folds with locked hyperparameters.
    This keeps tuning data and evaluation data disjoint.

  - Reduced epoch budget per trial (default: 150 epochs). Full Stage 2
    cap is 300 epochs but trials only need enough to distinguish
    convergent from non-convergent configurations. Median pruning kicks
    in at epoch 30, killing trials that are already clearly subpar.

  - TPE sampler with multivariate=True. Bayesian optimization with
    a tree-structured Parzen estimator surrogate; multivariate lets it
    model parameter interactions (important for AsyCo where lambda_u
    and temperature interact strongly).

  - Persistent SQLite storage. Search survives crashes, can be resumed,
    and post-search analysis (optuna_analyze.py) reads the same DB.

Outputs:

  results/optuna/{method}/fold_{NN}/
    ├── study.db                 # SQLite Optuna storage
    ├── search_config.json       # the search configuration that was run
    └── per_trial/
        ├── trial_0000/
        │   ├── config.yaml      # full merged config used for this trial
        │   ├── training_log.jsonl
        │   └── trial_summary.json   # final metrics + sampled hyperparams
        ├── trial_0001/
        │   ...

Usage:
  python -m scripts.optuna_search \
      --method asyco_divmix --fold 5 --n-trials 100

  # Resume an interrupted search:
  python -m scripts.optuna_search \
      --method asyco_divmix --fold 5 --n-trials 100 --resume
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

# IMPORTANT: this import path assumes you've placed the search-space
# definitions at configs/optuna_search_spaces.py (see file 1 of this patch).
from configs.optuna_search_spaces import sample as sample_hyperparams
from src.training.runner import run_training, StopTraining
from src.utils.io import load_config, project_root
from src.utils.seed import fold_seed


# Map (method, parameter) → which key in the merged cfg dict it lands under.
# All current parameters (ELR + asyco_divmix) live under cfg["method"][...],
# so the mapping is trivial; keeping the helper makes future extensions
# (e.g. tuning optim hyperparameters) explicit rather than implicit.
def _apply_hyperparams_to_cfg(cfg: dict, method: str, hp: dict) -> dict:
    """Merge sampled hyperparameters into the config dict, in-place safe."""
    cfg = dict(cfg)
    cfg["method"] = dict(cfg["method"])
    for k, v in hp.items():
        cfg["method"][k] = v
    return cfg


def _tau_dirname(tau: float) -> str:
    """Mirror scripts/stage1c_inject_noise.py's directory naming."""
    return f"tau_{int(round(tau * 100)):02d}"


def _load_train_csv(cfg: dict, dataset: str, tau: float, fold: int) -> pd.DataFrame:
    """Load the Stage 1c train_noisy.csv for the chosen tau, fold,
    feature_driven noise type."""
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


def _carve_val_split(
    train_df: pd.DataFrame, seed: int, val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified val carve from the fold's train set. Mirrors stage 2's
    protocol so the search uses the same data partitioning convention."""
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=int(seed),
    )
    train_idx, val_idx = next(
        splitter.split(train_df["image_id"], train_df["dx"])
    )
    return (
        train_df.iloc[train_idx].reset_index(drop=True),
        train_df.iloc[val_idx].reset_index(drop=True),
    )


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _peak_val_balanced_accuracy(log_path: Path, smooth_window: int = 5) -> tuple[float, int]:
    """Returns (peak smoothed val BA, epoch at which it was achieved).

    Smoothing matters because the per-epoch validation curves in your
    runs are noisy by ±0.02-0.05; a single high-water mark can be a
    lucky fold rather than a real peak. We use a trailing moving average
    (consistent with stage2_aggregate_epoch_budget.py).
    """
    records = _read_jsonl(log_path)
    if not records:
        return float("-inf"), 0

    ba = np.array(
        [r.get("val_balanced_accuracy", float("nan")) for r in records],
        dtype=np.float64,
    )
    if np.isnan(ba).all():
        return float("-inf"), 0

    # Trailing moving average; for the first (window-1) epochs use the
    # expanding mean of what's available so we don't drop them entirely.
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

    Args:
        trial: the active Optuna trial. We call ``trial.report`` and
            ``trial.should_prune`` on it once per epoch.
        smooth_window: trailing moving-average window applied to
            ``val_balanced_accuracy`` before reporting. Matches the same
            smoothing used to compute the final objective.
        min_epoch_for_pruning: don't allow pruning before this epoch.
            Useful for protecting AsyCo's warmup → post-warmup transition,
            since the BA reading at epoch 0-15 is on a fundamentally
            different model than what we'd get post-rampup.

    Returns:
        A callable ``(epoch, record) -> None`` suitable for
        ``run_training(epoch_callback=...)``.
    """
    history: list[float] = []

    def _callback(epoch: int, record: dict) -> None:
        # Skip if val metrics weren't computed this epoch (shouldn't happen
        # in practice for the search since we always pass val_df, but be
        # defensive).
        ba = record.get("val_balanced_accuracy")
        if ba is None or not (isinstance(ba, (int, float)) and ba == ba):  # NaN check
            return

        history.append(float(ba))
        # Trailing moving average; consistent with how the final objective
        # is computed by _peak_val_balanced_accuracy.
        lo = max(0, len(history) - smooth_window)
        smoothed = sum(history[lo:]) / max(len(history) - lo, 1)

        # Report to Optuna. The "step" is the epoch index — Optuna's
        # MedianPruner uses these to compare across trials at matched steps.
        trial.report(smoothed, step=int(epoch))

        # Don't allow pruning before the configured minimum epoch.
        if int(epoch) < int(min_epoch_for_pruning):
            return

        if trial.should_prune():
            # Raise StopTraining to gracefully stop the runner. The runner
            # catches this at the loop boundary and returns. Then we re-raise
            # optuna.TrialPruned in the objective wrapper so Optuna correctly
            # records this trial as pruned (not failed, not completed).
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
    """Build the Optuna objective callable. Closes over the args/data
    so optuna.optimize() can call it without extra arguments."""

    def objective(trial: optuna.Trial) -> float:
        trial_dir = out_dir / "per_trial" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Sample hyperparameters and merge into a fresh cfg copy.
        hp = sample_hyperparams(args.method, trial)
        cfg = _apply_hyperparams_to_cfg(base_cfg, args.method, hp)

        # Build the per-epoch pruning callback. The runner calls this
        # after each epoch's metrics are appended to training_log.jsonl.
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
                test_df=None,        # search doesn't touch test set
                images_dir=images_dir,
                method_name=args.method,
                total_epochs=int(args.trial_epochs),
                output_dir=trial_dir,
                val_df=val_df,
                seed=fold_seed(int(cfg["seed"]), int(args.fold)),
                epoch_callback=pruning_cb,
            )
        except StopTraining as e:
            # Pruning fired. Mark this trial as pruned so Optuna's bookkeeping
            # is correct (pruned trials inform TPE's surrogate model differently
            # than failed/completed trials).
            print(f"[trial {trial.number}] PRUNED: {e}", flush=True)
            was_pruned = True
        except Exception as e:
            print(f"[trial {trial.number}] FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            with (trial_dir / "trial_summary.json").open("w") as f:
                json.dump({"failed": True, "error": str(e), "params": hp}, f, indent=2)
            return float("-inf")

        # Post-training: extract peak smoothed validation balanced accuracy.
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
            # Re-raise so Optuna marks this trial state as PRUNED. Note that
            # Optuna's pruner will still have access to the intermediate
            # values reported via trial.report() — the surrogate uses them.
            raise optuna.TrialPruned()

        return peak_ba

    return objective


def main(args: argparse.Namespace) -> int:
    # Compose the base config exactly the way Stage 2 v2 does, so we
    # inherit the optimizer, model, and noise settings consistently.
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

    # Load training CSV and carve val split (same protocol as stage 2).
    train_full = _load_train_csv(base_cfg, args.dataset, args.tau, args.fold)
    train_df, val_df = _carve_val_split(
        train_full, seed=int(base_cfg["seed"]), val_frac=0.15,
    )

    # Output directory: results/optuna/{method}/{dataset}/{optim}_{model}/tau_XX/fold_XX/
    out_dir = (
        root / base_cfg["paths"]["results"] / "optuna"
        / args.method / args.dataset
        / f"{args.optim}_{args.model}"
        / _tau_dirname(args.tau) / f"fold_{args.fold:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist the search configuration up front so it's clear later what
    # protocol was used.
    search_config = {
        "method": args.method,
        "dataset": args.dataset,
        "optim": args.optim,
        "model": args.model,
        "tau": float(args.tau),
        "fold": int(args.fold),
        "n_trials": int(args.n_trials),
        "trial_epochs": int(args.trial_epochs),
        "tpe_seed": int(args.tpe_seed),
        "tpe_n_startup_trials": int(args.tpe_n_startup_trials),
        "pruner": "MedianPruner",
        "pruner_n_startup_trials": int(args.pruner_n_startup_trials),
        "pruner_n_warmup_steps": int(args.pruner_n_warmup_steps),
        "val_fraction": 0.15,
        "objective": "peak_smoothed_val_balanced_accuracy",
        "objective_smooth_window": int(args.objective_smooth_window),
        "pruning_enabled": True,
    }
    with (out_dir / "search_config.json").open("w") as f:
        json.dump(search_config, f, indent=2)

    print(f"[optuna] method={args.method} dataset={args.dataset} "
          f"tau={args.tau} fold={args.fold}", flush=True)
    print(f"[optuna] train={len(train_df)} val={len(val_df)} "
          f"n_trials={args.n_trials} epochs/trial={args.trial_epochs} "
          f"pruning=enabled (warmup={args.pruner_n_warmup_steps} epochs)", flush=True)
    print(f"[optuna] output -> {out_dir}", flush=True)

    # Persistent SQLite storage so we can resume after preemption.
    storage_url = f"sqlite:///{out_dir / 'study.db'}"
    sampler = TPESampler(
        seed=int(args.tpe_seed),
        multivariate=True,
        n_startup_trials=int(args.tpe_n_startup_trials),
        # consider 'group' if you add conditional parameters later
    )
    pruner = MedianPruner(
        n_startup_trials=int(args.pruner_n_startup_trials),
        n_warmup_steps=int(args.pruner_n_warmup_steps),
    )
    study = optuna.create_study(
        study_name=f"{args.method}_fold{args.fold:02d}_tau{int(args.tau*100):02d}",
        storage=storage_url,
        load_if_exists=bool(args.resume),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(args, base_cfg, train_df, val_df, images_dir, out_dir)

    t_start = time.time()
    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        gc_after_trial=True,
        show_progress_bar=False,  # we have our own per-trial print
    )
    elapsed = time.time() - t_start

    print(f"\n[optuna] DONE in {elapsed/3600:.2f} h", flush=True)
    print(f"[optuna] best value: {study.best_value:.4f}", flush=True)
    print(f"[optuna] best params:", flush=True)
    for k, v in study.best_params.items():
        print(f"          {k}: {v}", flush=True)

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for noisy-label methods"
    )
    p.add_argument("--method", required=True, choices=["elr", "asyco_divmix"])
    p.add_argument("--dataset", default="imbalanced", choices=["balanced", "imbalanced"])
    p.add_argument("--optim", default="adam", choices=["sgd", "adam"])
    p.add_argument("--model", default="resnet34_pretrained",
                   choices=["resnet34_pretrained", "resnet34_scratch"])
    p.add_argument("--tau", default=0.2, type=float,
                   help="Noise rate for the search. Default 0.2 = realistic noise regime.")
    p.add_argument("--fold", default=5, type=int, help="Fold to search on (default 5).")
    p.add_argument("--n-trials", default=100, type=int)
    p.add_argument("--trial-epochs", default=150, type=int,
                   help="Epoch budget per trial. 150 < full Stage 2 cap (300) for speed.")
    p.add_argument("--tpe-seed", default=42, type=int)
    p.add_argument("--tpe-n-startup-trials", default=15, type=int,
                   help="Random sampling for the first N trials before TPE engages.")
    p.add_argument("--pruner-n-startup-trials", default=10, type=int,
                   help="Trials before pruning is allowed to fire at all.")
    p.add_argument("--pruner-n-warmup-steps", default=40, type=int,
                   help="Within a trial, epochs before that trial can be pruned.")
    p.add_argument("--objective-smooth-window", default=5, type=int,
                   help="Trailing moving-average window for val BA, used for "
                        "both intermediate-value reports (pruning) and the "
                        "final objective. Keep these matched.")
    p.add_argument("--resume", action="store_true",
                   help="Resume an existing study (load study from disk if present).")
    sys.exit(main(p.parse_args()))


# ---------------------------------------------------------------------------
# Mid-trial pruning is wired through src/training/runner.py's epoch_callback
# parameter (added in this patch). The callback computes a trailing-mean of
# val_balanced_accuracy on each epoch, reports it to Optuna via trial.report,
# and raises StopTraining (caught by the runner) if Optuna's pruner says so.
# In the objective wrapper above, StopTraining is converted to optuna.TrialPruned
# so the Optuna study correctly distinguishes pruned trials from completed/failed.
#
# Keep the value of --objective-smooth-window matched to the value used by
# _peak_val_balanced_accuracy for the final objective — otherwise pruning
# decisions would be made on a different metric than the trial is ultimately
# scored on, which gives perverse results.
# ---------------------------------------------------------------------------
