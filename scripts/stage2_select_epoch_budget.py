"""Stage 2: per-fold epoch budget selection on clean data.

For a single (dataset, method, fold) triple, train on the clean (τ=0) training
split — with a 15% stratified validation slice carved off — for up to
`cfg["epoch_cap"]` epochs, logging per-epoch validation balanced accuracy.
The convergence epoch is later extracted by `stage2_aggregate_epoch_budget.py`,
which takes the median over the 10 folds.

Hyperparameters locked for Stage 2 (PROJECT_DOCUMENTATION §6 Stage 2):
    - SGD (lr=0.02, momentum=0.9, weight_decay=5e-4, cosine annealing)
    - ImageNet-pretrained ResNet-34
    - Clean (τ=0) feature-driven fold CSVs
Both choices are intentional: init and optim are Stage 3's concern, so we
pin Stage 2 to one configuration so that all five methods are budgeted
under the same selection conditions.

The fold's TEST set is NEVER touched in Stage 2 — this is enforced by
passing ``test_df=None`` to ``run_training``, which then skips the final
test evaluation and does not write ``test_metrics.json``.
See PROJECT_DOCUMENTATION §10 item 4 for the leakage-prevention rationale.

Run:
    python -m scripts.stage2_select_epoch_budget \\
        --dataset imbalanced --method elr --fold 0
    python -m scripts.stage2_select_epoch_budget \\
        --dataset imbalanced --method asyco_divmix --fold 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.training.runner import run_training
from src.utils.io import load_config, project_root
from src.utils.manifest import write_manifest
from src.utils.seed import fold_seed


METHOD_CHOICES = ["baseline", "sce", "elr", "asyco", "asyco_divmix"]


def _tau_dirname(tau: float) -> str:
    """Matches scripts/stage1c_inject_noise.py exactly."""
    return f"tau_{int(round(tau * 100)):02d}"


def _load_clean_train(cfg: dict, dataset: str, fold: int) -> pd.DataFrame:
    """Load the Stage 1c train_noisy.csv for τ=0 feature_driven, which by
    short-circuit construction contains the clean labels."""
    root = project_root()
    tau_dir = _tau_dirname(0.0)
    path = (
        root / cfg["paths"]["cv_folds"] / dataset
        / "feature_driven" / tau_dir / f"fold_{fold:02d}" / "train_noisy.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Clean train CSV not found at {path}. "
            f"Run stage1c (feature_driven, tau=0.0, fold {fold}) first."
        )
    return pd.read_csv(path)


def _carve_val_split(train_df: pd.DataFrame, seed: int, val_frac: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 15% validation split from the fold's TRAIN set.

    The TEST set is never touched; this split is purely internal to the
    train set for epoch selection.
    """
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=int(seed))
    train_idx, val_idx = next(splitter.split(train_df["image_id"], train_df["dx"]))
    train_part = train_df.iloc[train_idx].reset_index(drop=True)
    val_part = train_df.iloc[val_idx].reset_index(drop=True)
    return train_part, val_part


def main(args: argparse.Namespace) -> int:
    # Stage 2 is pinned to SGD + pretrained ResNet-34 (see module docstring).
    cfg = load_config(
        "base.yaml",
        f"data/{args.dataset}.yaml",
        f"method/{args.method}.yaml",
        "optim/sgd.yaml",
        "model/resnet34_pretrained.yaml",
        "noise/feature_driven.yaml",
    )
    root = project_root()

    # Load ONLY the train split for this fold at τ=0 (clean labels).
    # The fold's test set is deliberately not loaded — see module docstring
    # and PROJECT_DOCUMENTATION §10 item 4.
    train_full = _load_clean_train(cfg, args.dataset, args.fold)

    # Carve the 15% stratified validation slice from train only
    train_df, val_df = _carve_val_split(
        train_full, seed=int(cfg["seed"]), val_frac=0.15,
    )
    print(
        f"[stage2] {args.dataset}/{args.method}/fold{args.fold}: "
        f"train={len(train_df)} val={len(val_df)} "
        f"(cap={cfg['epoch_cap']} epochs; fold test set not touched)"
    )

    images_dir = root / cfg["paths"]["images"]
    out_dir = (
        root / cfg["paths"]["results"] / "epoch_selection"
        / args.dataset / args.method / f"fold_{args.fold:02d}"
    )

    # test_df=None → runner skips the final test evaluation entirely.
    run_training(
        cfg=cfg,
        train_df=train_df,
        test_df=None,
        images_dir=images_dir,
        method_name=args.method,
        total_epochs=int(cfg["epoch_cap"]),
        output_dir=out_dir,
        val_df=val_df,
        seed=fold_seed(int(cfg["seed"]), int(args.fold)),
    )

    manifest_path = (
        root / cfg["paths"]["manifests"]
        / f"stage2_select_{args.dataset}_{args.method}_fold{args.fold:02d}.json"
    )
    write_manifest(
        manifest_path,
        stage="stage2_select",
        params={
            "dataset": args.dataset,
            "method": args.method,
            "fold": int(args.fold),
            "epoch_cap": int(cfg["epoch_cap"]),
            "val_fraction": 0.15,
        },
        outputs=[
            str(out_dir / "config.yaml"),
            str(out_dir / "training_log.jsonl"),
        ],
    )
    print(f"[stage2] DONE -> {out_dir}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Stage 2: per-fold epoch budget selection (clean data)"
    )
    p.add_argument("--dataset", required=True, choices=["balanced", "imbalanced"])
    p.add_argument("--method", required=True, choices=METHOD_CHOICES)
    p.add_argument("--fold", required=True, type=int, help="fold id, 0..9")
    sys.exit(main(p.parse_args()))
