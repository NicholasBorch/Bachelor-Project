"""
Stage 1a: create stratified 10-fold assignments for a dataset.

Writes data/processed/.../cv_folds/{dataset}/fold_assignments.csv with columns
[image_id, dx, fold]. Must run before stages 1b, 1c, 2, 3.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.data.folds import create_fold_assignments
from src.utils.io import ensure_dir, load_config, project_root
from src.utils.manifest import write_manifest


def main(args: argparse.Namespace) -> int:
    cfg = load_config("base.yaml", f"data/{args.dataset}.yaml")
    root = project_root()

    metadata_path = (
        root / cfg["paths"]["data_processed"]
        / "one_image_per_lesion"
        / cfg["data"]["metadata_file"]
    )
    if not metadata_path.exists():
        print(f"ERROR: {metadata_path} not found. Run stage0 first.", file=sys.stderr)
        return 1

    metadata = pd.read_csv(metadata_path)
    print(f"[stage1a] loaded {len(metadata)} rows from {metadata_path}")

    expected_n = cfg["data"]["num_samples"]
    if len(metadata) != expected_n:
        print(
            f"ERROR: expected {expected_n} rows for '{args.dataset}', got {len(metadata)}",
            file=sys.stderr,
        )
        return 1

    folds = create_fold_assignments(
        metadata, n_splits=cfg["folds"], seed=cfg["seed"], label_col="dx",
    )

    # Sanity checks
    assert (folds["fold"] >= 0).all() and (folds["fold"] < cfg["folds"]).all()
    # Stratification check: each class is distributed across all folds
    print("[stage1a] per-class per-fold counts:")
    pivot = folds.pivot_table(index="dx", columns="fold", values="image_id", aggfunc="count", fill_value=0)
    print(pivot.to_string())

    out_dir = ensure_dir(root / cfg["paths"]["cv_folds"] / args.dataset)
    out_path = out_dir / "fold_assignments.csv"
    folds.to_csv(out_path, index=False)
    print(f"[stage1a] wrote {out_path}")

    manifest_path = root / cfg["paths"]["manifests"] / f"stage1a_{args.dataset}.json"
    write_manifest(
        manifest_path,
        stage="stage1a",
        params={"dataset": args.dataset},
        outputs=[str(out_path.relative_to(root))],
        extra={
            "n_samples": len(folds),
            "n_folds": int(cfg["folds"]),
            "seed": int(cfg["seed"]),
        },
    )
    print(f"[stage1a] wrote manifest {manifest_path}")
    print("[stage1a] DONE")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1a: create fold assignments")
    p.add_argument("--dataset", required=True, choices=["balanced", "imbalanced"])
    sys.exit(main(p.parse_args()))