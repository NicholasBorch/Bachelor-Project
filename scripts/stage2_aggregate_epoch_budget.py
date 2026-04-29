"""Stage 2 aggregation: convert 10 per-fold validation logs into one
`selected_budget.json` for the given (dataset, method).

Procedure (PROJECT_DOCUMENTATION §6 Stage 2 aggregation):

1. Read `training_log.jsonl` for each fold 0..9.
2. Extract the `val_balanced_accuracy` series per fold.
3. Smooth with a trailing moving average (window=5).
4. Find the convergence epoch: smallest epoch such that the smoothed
   validation balanced accuracy has not improved over the previous best
   for `patience=10` consecutive epochs. If no such epoch exists within
   100 epochs, the convergence epoch is clamped to 100.
5. Take the median of the 10 per-fold convergence epochs.
6. Save `selected_budget.json` + a convergence overlay plot.

Why median, not mean: median is stable against outlier folds that converge
unusually late; mean gets pulled by any single such fold.

Run:
    python -m scripts.stage2_aggregate_epoch_budget \\
        --dataset imbalanced --method elr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402 - must happen before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.io import load_config, project_root  # noqa: E402
from src.utils.manifest import write_manifest  # noqa: E402


METHOD_CHOICES = ["baseline", "sce", "elr", "asyco", "asyco_divmix"]


def _load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    records: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average. For indices < window-1, uses the expanding
    mean of what's available so we don't emit NaNs at the start."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    csum = np.cumsum(x)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        n = i - lo + 1
        if lo == 0:
            out[i] = csum[i] / n
        else:
            out[i] = (csum[i] - csum[lo - 1]) / n
    return out


def _convergence_epoch(
    val_ba: np.ndarray,
    smoothing_window: int = 5,
    patience: int = 10,
    cap: int = 100,
) -> int:
    """Smallest epoch where smoothed val BA has not improved for `patience`
    epochs. Returns 0-indexed convergence epoch. Clamps to cap.
    """
    if len(val_ba) == 0:
        return cap
    s = _moving_average(val_ba, window=smoothing_window)
    best = -np.inf
    bad = 0
    eps = 1e-6  # ignore trivial improvements below this threshold
    for i, v in enumerate(s):
        if v > best + eps:
            best = float(v)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                return int(min(i, cap))
    return int(min(len(val_ba) - 1, cap))


def _plot_convergence(
    fold_curves: list[np.ndarray],
    per_fold_convergence: list[int],
    selected: int,
    output_path: Path,
    title: str,
    smoothing_window: int = 5,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    max_len = max(len(c) for c in fold_curves)
    for fold_id, curve in enumerate(fold_curves):
        # s = _moving_average(curve, window=smoothing_window)
        s = curve
        xs = np.arange(len(s))
        ax.plot(xs, s, alpha=0.5, linewidth=1.0,
                label=f"fold {fold_id} (conv {per_fold_convergence[fold_id]})")
    # Median curve: pad curves to max_len with NaN, then nanmedian
    padded = np.full((len(fold_curves), max_len), np.nan, dtype=np.float64)
    for i, c in enumerate(fold_curves):
        s = _moving_average(c, window=smoothing_window)
        padded[i, : len(s)] = s
    med = np.nanmedian(padded, axis=0)
    ax.plot(np.arange(max_len), med, color="black", linewidth=2.2, label="median (smoothed)")
    ax.axvline(selected, color="red", linestyle="--",
               label=f"selected budget = {selected}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val balanced accuracy (smoothed)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(args: argparse.Namespace) -> int:
    cfg = load_config("base.yaml", f"data/{args.dataset}.yaml")
    root = project_root()

    per_fold_dir = (
        root / cfg["paths"]["results"] / "epoch_selection"
        / args.dataset / args.method
    )
    n_folds = int(cfg["folds"])
    cap = int(cfg["epoch_cap"])

    # Collect per-fold val BA curves
    fold_curves: list[np.ndarray] = []
    per_fold_convergence: list[int] = []
    missing: list[int] = []
    for f in range(n_folds):
        log_path = per_fold_dir / f"fold_{f:02d}" / "training_log.jsonl"
        if not log_path.exists():
            missing.append(f)
            continue
        records = _load_jsonl(log_path)
        vba = np.array(
            [float(r.get("val_balanced_accuracy", float("nan"))) for r in records],
            dtype=np.float64,
        )
        if np.isnan(vba).all():
            print(
                f"ERROR: no val_balanced_accuracy in {log_path} — "
                f"was this fold run with a validation split?",
                file=sys.stderr,
            )
            return 1
        fold_curves.append(vba)
        # +1 converts 0-indexed epoch to "number of epochs trained"
        per_fold_convergence.append(int(_convergence_epoch(vba, cap=cap)) + 1)

    if missing:
        print(
            f"ERROR: missing folds for {args.dataset}/{args.method}: {missing}. "
            f"Run stage2_select_epoch_budget for them first.",
            file=sys.stderr,
        )
        return 1

    per_fold_arr = np.asarray(per_fold_convergence, dtype=np.int64)
    selected = int(np.median(per_fold_arr))
    selected = int(min(selected, cap))
    mean = float(per_fold_arr.mean())
    std = float(per_fold_arr.std(ddof=0))

    out = {
        "dataset": args.dataset,
        "method": args.method,
        "selected_epochs": int(selected),
        "per_fold_convergence": [int(x) for x in per_fold_arr.tolist()],
        "median": int(np.median(per_fold_arr)),
        "mean": mean,
        "std": std,
        "cap": cap,
        "smoothing_window": 5,
        "patience": 10,
    }
    out_json_path = per_fold_dir / "selected_budget.json"
    with open(out_json_path, "w") as f:
        json.dump(out, f, indent=2)

    plot_path = per_fold_dir / "convergence.png"
    _plot_convergence(
        fold_curves=fold_curves,
        per_fold_convergence=per_fold_convergence,
        selected=selected,
        output_path=plot_path,
        title=f"{args.dataset} / {args.method}  —  median = {selected}",
    )

    manifest_path = (
        root / cfg["paths"]["manifests"]
        / f"stage2_aggregate_{args.dataset}_{args.method}.json"
    )
    write_manifest(
        manifest_path,
        stage="stage2_aggregate",
        params={"dataset": args.dataset, "method": args.method},
        outputs=[str(out_json_path), str(plot_path)],
        extra={"selected_epochs": selected, "per_fold_convergence": per_fold_convergence},
    )
    print(
        f"[stage2-agg] {args.dataset}/{args.method} "
        f"-> selected={selected} (median of {per_fold_convergence}, "
        f"mean={mean:.1f}, std={std:.1f})"
    )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Stage 2 aggregation: median epoch budget from 10 per-fold logs"
    )
    p.add_argument("--dataset", required=True, choices=["balanced", "imbalanced"])
    p.add_argument("--method", required=True, choices=METHOD_CHOICES)
    sys.exit(main(p.parse_args()))
