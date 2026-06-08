"""
Stage 1c: inject instance-dependent label noise into train folds.

For each tau in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5} and each fold:
    - compute train set (folds != test_fold) and test set (fold == test_fold)
    - at tau > 0, apply the requested noise function to train labels
    - at tau = 0, pass train labels through unchanged (short-circuit)
    - test labels are NEVER altered (clean test set always)

Per-fold seed: global_seed * 10_000 + fold_id.

Run:
    python -m scripts.stage1c_inject_noise \\
        --dataset imbalanced --noise-type feature_driven --fold 0
or, for all folds sequentially:
    python -m scripts.stage1c_inject_noise \\
        --dataset imbalanced --noise-type feature_driven --all-folds

Outputs for each (dataset, noise_type, tau, fold):
    data/processed/HAM10000/cv_folds/{dataset}/{noise_type}/tau_NN/fold_NN/
        train_noisy.csv   # noisy labels in 'dx' column, clean in 'dx_clean'
        test_clean.csv    # original labels (test never gets noise)
        report.json       # empirical flip rates
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.folds import load_fold_assignments, split_train_test_by_fold
from src.noise.idn_feature_driven import generate_feature_driven_idn
from src.noise.idn_xia import generate_xia_idn
from src.utils.io import ensure_dir, load_config, project_root
from src.utils.manifest import write_manifest
from src.utils.seed import fold_seed


def _tau_dirname(tau: float) -> str:
    return f"tau_{int(round(tau * 100)):02d}"


def _inject(
    noise_type: str,
    train_df: pd.DataFrame,
    tau: float,
    seed: int,
    images_dir: Path | None,
    oof_probs_subset: np.ndarray | None,
    sigma: float,
    image_size: int,
):
    if noise_type == "standard":
        assert images_dir is not None
        return generate_xia_idn(
            train_df, images_dir=images_dir, tau=tau, seed=seed,
            normalize=False, sigma=sigma, image_size=image_size,
        )
    if noise_type == "normalized":
        assert images_dir is not None
        return generate_xia_idn(
            train_df, images_dir=images_dir, tau=tau, seed=seed,
            normalize=True, sigma=sigma, image_size=image_size,
        )
    if noise_type == "feature_driven":
        assert oof_probs_subset is not None
        return generate_feature_driven_idn(
            train_df, oof_probs=oof_probs_subset, tau=tau, seed=seed, sigma=sigma,
        )
    raise ValueError(f"unknown noise_type: {noise_type}")


def _process_fold(
    cfg: dict,
    dataset: str,
    noise_type: str,
    fold: int,
    metadata: pd.DataFrame,
    folds_df: pd.DataFrame,
    images_dir: Path | None,
    oof_probs_full: np.ndarray | None,
) -> None:
    root = project_root()
    train_df, test_df = split_train_test_by_fold(metadata, folds_df, test_fold=fold)
    print(f"[stage1c] {dataset}/{noise_type}/fold{fold}: train={len(train_df)}, test={len(test_df)}")

    # Slice OOF probs to the train rows in the same order as train_df.image_id.
    oof_subset = None
    if noise_type == "feature_driven":
        if oof_probs_full is None:
            raise RuntimeError("feature_driven requires oof_probs_full")
        id_to_row = {iid: i for i, iid in enumerate(metadata["image_id"].tolist())}
        idxs = [id_to_row[iid] for iid in train_df["image_id"].tolist()]
        oof_subset = oof_probs_full[idxs]

    for tau in cfg["noise_rates"]:
        seed = fold_seed(cfg["seed"], fold)
        sigma = float(cfg["noise_sigma"])
        noisy_df, report = _inject(
            noise_type=noise_type,
            train_df=train_df,
            tau=float(tau),
            seed=seed,
            images_dir=images_dir,
            oof_probs_subset=oof_subset,
            sigma=sigma,
            image_size=cfg["image_size"],
        )

        # test set is always clean
        test_out = test_df.copy()
        if "dx_clean" not in test_out.columns:
            test_out["dx_clean"] = test_out["dx"].values
        test_out["flipped"] = False

        out_dir = ensure_dir(
            root / cfg["paths"]["cv_folds"]
            / dataset / noise_type / _tau_dirname(float(tau))
            / f"fold_{fold:02d}"
        )
        noisy_df.to_csv(out_dir / "train_noisy.csv", index=False)
        test_out.to_csv(out_dir / "test_clean.csv", index=False)
        with open(out_dir / "report.json", "w") as f:
            json.dump(asdict(report), f, indent=2)

        # sanity: tau=0 must not flip anything
        if float(tau) == 0.0 and report.n_flipped != 0:
            raise AssertionError(
                f"tau=0 produced {report.n_flipped} flips — short-circuit failure!"
            )
        print(
            f"[stage1c]   tau={tau:.2f}  flipped={report.n_flipped}/{report.n_total}"
            f"  empirical={report.empirical_rate:.4f}"
        )


def main(args: argparse.Namespace) -> int:
    cfg = load_config("base.yaml", f"data/{args.dataset}.yaml")
    root = project_root()

    fa_path = root / cfg["paths"]["cv_folds"] / args.dataset / "fold_assignments.csv"
    folds_df = load_fold_assignments(fa_path)

    metadata_path = (
        root / cfg["paths"]["data_processed"]
        / "one_image_per_lesion"
        / cfg["data"]["metadata_file"]
    )
    metadata = pd.read_csv(metadata_path)
    images_dir = root / cfg["paths"]["images"]

    oof_probs_full = None
    if args.noise_type == "feature_driven":
        oof_path = root / cfg["paths"]["cv_folds"] / args.dataset / "oof_probs" / "oof_probs_full.npy"
        if not oof_path.exists():
            print(f"ERROR: {oof_path} not found. Run stage1b first.", file=sys.stderr)
            return 1
        oof_probs_full = np.load(oof_path)
        if oof_probs_full.shape[0] != len(metadata):
            print(
                f"ERROR: oof_probs_full size {oof_probs_full.shape[0]} != metadata size {len(metadata)}",
                file=sys.stderr,
            )
            return 1

    folds_to_run = list(range(int(cfg["folds"]))) if args.all_folds else [args.fold]
    if args.fold is not None and args.all_folds:
        print("WARNING: --fold ignored because --all-folds was passed", file=sys.stderr)

    for f in folds_to_run:
        if f is None:
            print("ERROR: must pass either --fold or --all-folds", file=sys.stderr)
            return 1
        _process_fold(
            cfg, args.dataset, args.noise_type, f,
            metadata, folds_df, images_dir, oof_probs_full,
        )

    manifest_path = (
        root / cfg["paths"]["manifests"]
        / f"stage1c_{args.dataset}_{args.noise_type}_"
          f"{'all' if args.all_folds else f'fold{args.fold:02d}'}.json"
    )
    write_manifest(
        manifest_path,
        stage="stage1c",
        params={
            "dataset": args.dataset,
            "noise_type": args.noise_type,
            "folds": folds_to_run,
        },
        outputs=[],
        extra={"taus": cfg["noise_rates"]},
    )
    print("[stage1c] DONE")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1c: IDN noise injection")
    p.add_argument("--dataset", required=True, choices=["balanced", "imbalanced"])
    p.add_argument("--noise-type", required=True,
                   choices=["standard", "normalized", "feature_driven"])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold", type=int, default=None, help="single fold id 0..9")
    g.add_argument("--all-folds", action="store_true", help="run all folds sequentially")
    sys.exit(main(p.parse_args()))