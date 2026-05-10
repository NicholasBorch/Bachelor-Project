"""Post-search analysis for the FINAL Optuna study.

Same outputs as scripts/optuna_analyze.py but reads from
results/optuna_final/ and supports SCE in addition to ELR and AsyCo.

Outputs (per method, in the study's output directory):
  summary.txt        — top-K trials, best params, runtime stats
  all_trials.csv     — every trial's params + objective for spreadsheet inspection
  best_config.yaml   — drop-in tuned config for configs/method/<method>_tuned.yaml
  plots/
    optimization_history.png
    param_importances.png
    parallel_coordinate.png
    slice.png
    contour_pairs.png
    trial_validation_curves.png

Usage:
  python -m scripts.optuna_analyze_final --method elr --fold 9
  python -m scripts.optuna_analyze_final --method sce --fold 9
  python -m scripts.optuna_analyze_final --method asyco_divmix --fold 9
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.utils.io import load_config, project_root  # noqa: E402


def _tau_dirname(tau: float) -> str:
    return f"tau_{int(round(tau * 100)):02d}"


def _trial_smoothed_curve(log_path: Path, smooth_window: int = 5):
    if not log_path.exists():
        return None
    rows = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return None
    ba = np.array(
        [r.get("val_balanced_accuracy", float("nan")) for r in rows],
        dtype=np.float64,
    )
    if np.isnan(ba).all():
        return None
    smoothed = np.empty_like(ba)
    csum = np.cumsum(np.nan_to_num(ba, nan=0.0))
    valid = np.cumsum(~np.isnan(ba))
    for i in range(len(ba)):
        lo = max(0, i - smooth_window + 1)
        n = valid[i] - (valid[lo - 1] if lo > 0 else 0)
        s = csum[i] - (csum[lo - 1] if lo > 0 else 0.0)
        smoothed[i] = s / n if n > 0 else float("nan")
    return smoothed


def _summary_table(study: optuna.Study, top_k: int) -> str:
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in study.trials
              if t.state == optuna.trial.TrialState.FAIL]
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]

    lines = [
        f"Study: {study.study_name}",
        f"Direction: {study.direction.name}",
        f"Total trials: {len(study.trials)}",
        f"  Completed: {len(completed)}",
        f"  Failed:    {len(failed)}",
        f"  Pruned:    {len(pruned)}",
        "",
    ]

    if not completed:
        lines.append("(No completed trials.)")
        return "\n".join(lines)

    sorted_completed = sorted(completed, key=lambda t: t.value, reverse=True)
    lines.append(f"Best value: {sorted_completed[0].value:.6f}")
    lines.append(f"Best trial number: {sorted_completed[0].number}")
    lines.append("Best params:")
    for k, v in sorted_completed[0].params.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append(f"Top {top_k} trials:")
    header_keys = list(sorted_completed[0].params.keys())
    lines.append(
        "  " + "  ".join(
            f"{c:>10s}" for c in (["trial", "value"] + header_keys)
        )
    )
    for t in sorted_completed[:top_k]:
        cells = [f"{t.number:>10d}", f"{t.value:>10.4f}"]
        for k in header_keys:
            v = t.params.get(k)
            if isinstance(v, (int, float)):
                cells.append(f"{v:>10.4g}")
            else:
                cells.append(f"{v!s:>10s}")
        lines.append("  " + "  ".join(cells))

    return "\n".join(lines)


def _all_trials_csv(study: optuna.Study) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        row = {
            "trial": t.number,
            "state": t.state.name,
            "value": t.value if t.value is not None else float("nan"),
            "duration_s": (
                t.duration.total_seconds() if t.duration else float("nan")
            ),
        }
        for k, v in t.params.items():
            row[f"param_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _save_optuna_visualizations(study: optuna.Study, plot_dir: Path) -> None:
    import optuna.visualization.matplotlib as ovm

    plot_dir.mkdir(parents=True, exist_ok=True)
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 2:
        print(f"[analyze] Only {len(completed)} completed trials; "
              f"skipping built-in plots.")
        return

    plots = [
        ("optimization_history",
         lambda: ovm.plot_optimization_history(study)),
        ("parallel_coordinate",
         lambda: ovm.plot_parallel_coordinate(study)),
        ("slice", lambda: ovm.plot_slice(study)),
    ]
    if len(completed) >= 5:
        plots.append(
            ("param_importances",
             lambda: ovm.plot_param_importances(study))
        )

    for name, fn in plots:
        try:
            ax = fn()
            fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
            fig.tight_layout()
            fig.savefig(plot_dir / f"{name}.png", dpi=150)
            plt.close(fig)
            print(f"[analyze] saved {plot_dir / (name + '.png')}")
        except Exception as e:
            print(f"[analyze] skipped {name}: {type(e).__name__}: {e}")

    if len(completed) >= 5:
        try:
            importances = optuna.importance.get_param_importances(study)
            top_two = list(importances.keys())[:2]
            if len(top_two) == 2:
                ax = ovm.plot_contour(study, params=top_two)
                fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
                fig.tight_layout()
                fig.savefig(plot_dir / "contour_pairs.png", dpi=150)
                plt.close(fig)
                print(f"[analyze] saved {plot_dir / 'contour_pairs.png'} "
                      f"for {top_two}")
        except Exception as e:
            print(f"[analyze] skipped contour: {type(e).__name__}: {e}")


def _plot_validation_curves(
    study: optuna.Study, out_dir: Path, plot_path: Path, top_k: int = 5,
) -> None:
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return

    sorted_completed = sorted(completed, key=lambda t: t.value, reverse=True)
    top_trial_numbers = {t.number for t in sorted_completed[:top_k]}

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    for t in completed:
        if t.number in top_trial_numbers:
            continue
        log_path = (
            out_dir / "per_trial" / f"trial_{t.number:04d}" / "training_log.jsonl"
        )
        curve = _trial_smoothed_curve(log_path)
        if curve is not None:
            ax.plot(curve, color="0.85", linewidth=0.6, alpha=0.7, zorder=1)

    for i, t in enumerate(sorted_completed[:top_k]):
        log_path = (
            out_dir / "per_trial" / f"trial_{t.number:04d}" / "training_log.jsonl"
        )
        curve = _trial_smoothed_curve(log_path)
        if curve is None:
            continue
        ax.plot(
            curve,
            color=cmap(i % 10),
            linewidth=1.8,
            label=f"trial {t.number} (best={t.value:.3f})",
            zorder=3,
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel("smoothed validation balanced accuracy (CLEAN labels)")
    ax.set_title(f"{study.study_name}: top-{top_k} trials over all completed")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[analyze] saved {plot_path}")


def _write_best_config_yaml(
    method: str, study: optuna.Study, path: Path,
) -> None:
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return
    best = max(completed, key=lambda t: t.value)

    root = project_root()
    base_path = root / "configs" / "method" / f"{method}.yaml"
    if base_path.exists():
        with base_path.open() as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {"name": method}

    cfg.update(best.params)
    cfg["name"] = method
    cfg["_optuna_provenance"] = {
        "protocol": "FINAL",
        "study_name": study.study_name,
        "best_trial_number": int(best.number),
        "best_value": float(best.value),
        "n_trials_completed": int(len(completed)),
    }

    with path.open("w") as f:
        f.write(
            "# Hyperparameters chosen by FINAL Optuna search.\n"
            f"# See {path.parent / 'summary.txt'} for the search summary.\n"
        )
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[analyze] wrote {path}")


def main(args: argparse.Namespace) -> int:
    base_cfg = load_config(
        "base.yaml",
        f"data/{args.dataset}.yaml",
        f"method/{args.method}.yaml",
    )
    root = project_root()
    out_dir = (
        root / base_cfg["paths"]["results"] / "optuna_final"
        / args.method / args.dataset
        / f"{args.optim}_{args.model}"
        / _tau_dirname(args.tau) / f"fold_{args.fold:02d}"
    )
    if not out_dir.exists():
        print(
            f"ERROR: {out_dir} does not exist. Run optuna_search_final.py first.",
            file=sys.stderr,
        )
        return 1
    db_path = out_dir / "study.db"
    if not db_path.exists():
        print(f"ERROR: {db_path} does not exist.", file=sys.stderr)
        return 1

    storage_url = f"sqlite:///{db_path}"
    study_name = f"{args.method}_fold{args.fold:02d}_FINAL"
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    summary = _summary_table(study, top_k=int(args.top_k))
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary + "\n")
    print(summary)
    print(f"\n[analyze] wrote {summary_path}")

    df = _all_trials_csv(study)
    csv_path = out_dir / "all_trials.csv"
    df.to_csv(csv_path, index=False)
    print(f"[analyze] wrote {csv_path} ({len(df)} rows)")

    _write_best_config_yaml(args.method, study, out_dir / "best_config.yaml")

    plot_dir = out_dir / "plots"
    _save_optuna_visualizations(study, plot_dir)
    _plot_validation_curves(
        study, out_dir, plot_dir / "trial_validation_curves.png",
        top_k=int(args.top_k),
    )

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Post-search analysis for the FINAL Optuna study",
    )
    p.add_argument("--method", required=True,
                   choices=["elr", "sce", "asyco_divmix"])
    p.add_argument("--dataset", default="imbalanced",
                   choices=["balanced", "imbalanced"])
    p.add_argument("--optim", default="adam", choices=["sgd", "adam"])
    p.add_argument("--model", default="resnet34_pretrained",
                   choices=["resnet34_pretrained", "resnet34_scratch"])
    p.add_argument("--tau", default=0.2, type=float)
    p.add_argument("--fold", default=9, type=int)
    p.add_argument("--top-k", default=5, type=int)
    sys.exit(main(p.parse_args()))
