"""Stage 1b (merge step): assemble per-fold OOF probabilities into a single
array aligned with the dataset's canonical metadata row order.

Run: python -m scripts.stage1b_merge_oof_probs --dataset imbalanced

Prerequisites: all 10 per-fold OOF files from stage1b_collect_oof_probs.py.

Output: data/processed/HAM10000/cv_folds/{dataset}/oof_probs/oof_probs_full.npy
        with shape (N_samples, 7), ordered to match the dataset's metadata CSV.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ham10000 import NUM_CLASSES
from src.utils.io import load_config, project_root
from src.utils.manifest import write_manifest


def main(args: argparse.Namespace) -> int:
    cfg = load_config("base.yaml", f"data/{args.dataset}.yaml")
    root = project_root()

    metadata_path = (
        root / cfg["paths"]["data_processed"]
        / "one_image_per_lesion"
        / cfg["data"]["metadata_file"]
    )
    metadata = pd.read_csv(metadata_path)
    n = len(metadata)

    oof_dir = root / cfg["paths"]["cv_folds"] / args.dataset / "oof_probs"
    if not oof_dir.exists():
        print(f"ERROR: {oof_dir} not found. Run stage1b per-fold collection first.", file=sys.stderr)
        return 1

    merged = np.full((n, NUM_CLASSES), np.nan, dtype=np.float32)
    id_to_row = {iid: i for i, iid in enumerate(metadata["image_id"].tolist())}

    n_folds = int(cfg["folds"])
    for fold in range(n_folds):
        npy = oof_dir / f"fold_{fold:02d}.npy"
        ids = oof_dir / f"fold_{fold:02d}_ids.csv"
        if not npy.exists() or not ids.exists():
            print(f"ERROR: missing fold {fold} ({npy} or {ids})", file=sys.stderr)
            return 1
        probs = np.load(npy)
        id_list = pd.read_csv(ids)["image_id"].tolist()
        if probs.shape[0] != len(id_list):
            print(f"ERROR: fold {fold} probs/ids size mismatch", file=sys.stderr)
            return 1
        for row_idx, iid in enumerate(id_list):
            if iid not in id_to_row:
                print(f"ERROR: image {iid} from fold {fold} not in metadata", file=sys.stderr)
                return 1
            merged[id_to_row[iid]] = probs[row_idx]

    if np.isnan(merged).any():
        missing = int(np.isnan(merged).any(axis=1).sum())
        print(f"ERROR: {missing} samples have no OOF probability assigned.", file=sys.stderr)
        return 1

    out_path = oof_dir / "oof_probs_full.npy"
    np.save(out_path, merged)
    print(f"[stage1b-merge] wrote {out_path} (shape {merged.shape})")

    # Sanity: each row sums to ~1
    row_sums = merged.sum(axis=1)
    print(f"[stage1b-merge] row sum stats: min={row_sums.min():.4f}, "
          f"max={row_sums.max():.4f}, mean={row_sums.mean():.4f}")

    manifest_path = root / cfg["paths"]["manifests"] / f"stage1b_merge_{args.dataset}.json"
    write_manifest(
        manifest_path,
        stage="stage1b_merge",
        params={"dataset": args.dataset},
        outputs=[str(out_path.relative_to(root))],
        extra={"n_samples": n, "n_folds": n_folds},
    )
    print("[stage1b-merge] DONE")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage 1b merge: assemble full OOF probability array")
    p.add_argument("--dataset", required=True, choices=["balanced", "imbalanced"])
    sys.exit(main(p.parse_args()))
