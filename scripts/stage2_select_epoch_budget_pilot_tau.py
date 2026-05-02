from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.training.runner import run_training
from src.utils.io import load_config, project_root
from src.utils.manifest import write_manifest
from src.utils.seed import fold_seed


METHOD_CHOICES = ["elr", "asyco_divmix"]


def _tau_dirname(tau: float) -> str:
    return f"tau_{int(round(tau * 100)):02d}"


def _load_train(cfg: dict, dataset: str, tau: float, fold: int) -> pd.DataFrame:
    root = project_root()
    tau_dir = _tau_dirname(tau)

    path = (
        root / cfg["paths"]["cv_folds"] / dataset
        / "feature_driven" / tau_dir / f"fold_{fold:02d}" / "train_noisy.csv"
    )

    if not path.exists():
        raise FileNotFoundError(f"Missing train CSV: {path}")

    return pd.read_csv(path)


def _carve_val_split(
    train_df: pd.DataFrame,
    seed: int,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_frac,
        random_state=int(seed),
    )

    train_idx, val_idx = next(splitter.split(train_df["image_id"], train_df["dx"]))

    train_part = train_df.iloc[train_idx].reset_index(drop=True)
    val_part = train_df.iloc[val_idx].reset_index(drop=True)

    return train_part, val_part


def main(args: argparse.Namespace) -> int:
    cfg = load_config(
        "base.yaml",
        f"data/{args.dataset}.yaml",
        f"method/{args.method}.yaml",
        f"optim/{args.optim}.yaml",
        f"model/{args.model}.yaml",
        "noise/feature_driven.yaml",
    )
    root = project_root()

    train_full = _load_train(cfg, args.dataset, args.tau, args.fold)

    train_df, val_df = _carve_val_split(
        train_full,
        seed=int(cfg["seed"]),
        val_frac=0.15,
    )

    # Training uses the noisy label column `dx`.
    # Validation must use clean labels, so overwrite val_df["dx"] with dx_clean.
    if "dx_clean" not in val_df.columns:
        raise ValueError("Expected dx_clean column in validation dataframe.")

    val_df = val_df.copy()
    val_df["dx"] = val_df["dx_clean"]

    tau_dir = _tau_dirname(args.tau)

    print(
        f"[stage2-pilot-tau] dataset={args.dataset} "
        f"method={args.method} optim={args.optim} model={args.model} "
        f"tau={args.tau:.2f} fold={args.fold} "
        f"train={len(train_df)} val={len(val_df)} "
        f"cap={cfg['epoch_cap']} validation_labels=clean"
    )

    images_dir = root / cfg["paths"]["images"]

    out_dir = (
        root / cfg["paths"]["results"] / "pilot_stage2_tau_fold5"
        / tau_dir / args.dataset / args.method / args.optim / args.model
        / f"fold_{args.fold:02d}"
    )

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
        / (
            f"pilot_stage2_tau_fold5_{tau_dir}_{args.dataset}_{args.method}_"
            f"{args.optim}_{args.model}_fold{args.fold:02d}.json"
        )
    )

    write_manifest(
        manifest_path,
        stage="pilot_stage2_tau_fold5",
        params={
            "dataset": args.dataset,
            "method": args.method,
            "optim": args.optim,
            "model": args.model,
            "tau": float(args.tau),
            "fold": int(args.fold),
            "epoch_cap": int(cfg["epoch_cap"]),
            "val_fraction": 0.15,
            "validation_labels": "clean",
        },
        outputs=[
            str(out_dir / "config.yaml"),
            str(out_dir / "training_log.jsonl"),
        ],
    )

    print(f"[stage2-pilot-tau] DONE -> {out_dir}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Stage 2 pilot: fold 5 tau=0 vs tau=0.2 with clean validation labels"
    )
    p.add_argument("--dataset", required=True, choices=["balanced", "imbalanced"])
    p.add_argument("--method", required=True, choices=METHOD_CHOICES)
    p.add_argument("--optim", required=True, choices=["sgd", "adam"])
    p.add_argument(
        "--model",
        required=True,
        choices=["resnet34_pretrained", "resnet34_scratch"],
    )
    p.add_argument("--tau", required=True, type=float, choices=[0.0, 0.2])
    p.add_argument("--fold", type=int, default=5)

    sys.exit(main(p.parse_args()))