"""
Class-level confusability evidence for Feature-Driven IDN, computed directly from
the saved out-of-fold (OOF) softmax.

Reads oof_probs_full.npy (and per-fold files when present, for a 10-fold CI) and
reports per true class the OOF confidence, the dominant confusion target, the
argmax-to-target rate, and the masked+renormalised flip mass, into
results/oof_confusion/{dataset}/ (CSVs, two heatmaps, a LaTeX table, manifest).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure style consistent with the rest of the Results scripts.
_PLT_STYLE = {
    "font.family":        "serif",
    "font.serif":         ["Palatino", "Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "axes.unicode_minus": False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
}

# Frequency class order used everywhere else in the thesis (nv first, df last).
_CLASS_ORDER_FREQ = ("nv", "bkl", "mel", "bcc", "akiec", "vasc", "df")

# Bootstrap settings: identical to the rest of the thesis (percentile, 10000, seed 10).
_N_BOOT = 10_000
_BOOT_SEED = 10
_CI = 0.95


# Numeric core
def _mask_renorm(row: np.ndarray, true_idx: int) -> np.ndarray:
    """Mask the true-class entry and renormalise over the remaining classes."""
    p = row.astype(np.float64).copy()
    p[true_idx] = 0.0
    s = p.sum()
    if s <= 0.0:
        C = p.shape[0]
        p = np.ones(C) / (C - 1)
        p[true_idx] = 0.0
        return p
    return p / s


def softmax_confusion(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """(C, C) mean un-masked OOF softmax per true class (rows sum to ~1)."""
    M = np.full((n_classes, n_classes), np.nan)
    for i in range(n_classes):
        sel = labels == i
        if sel.any():
            M[i, :] = probs[sel].mean(axis=0)
    return M


def argmax_confusion(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """(C, C) hard argmax confusion per true class (rows sum to 1)."""
    pred = probs.argmax(axis=1)
    M = np.full((n_classes, n_classes), np.nan)
    for i in range(n_classes):
        sel = labels == i
        if sel.any():
            counts = np.bincount(pred[sel], minlength=n_classes).astype(np.float64)
            M[i, :] = counts / counts.sum()
    return M


def flip_target_confusion(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """(C, C) masked+renormalised flip-target distribution per true class (diag 0)."""
    M = np.full((n_classes, n_classes), np.nan)
    for i in range(n_classes):
        sel = labels == i
        if not sel.any():
            continue
        rows = np.vstack([_mask_renorm(probs[k], i) for k in np.flatnonzero(sel)])
        M[i, :] = rows.mean(axis=0)
    return M


def _boot_ci(values: np.ndarray) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean; returns (mean, lo, hi)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(_BOOT_SEED)
    boot = rng.choice(v, size=(_N_BOOT, v.size), replace=True).mean(axis=1)
    return (float(v.mean()),
            float(np.percentile(boot, 100 * (1 - _CI) / 2)),
            float(np.percentile(boot, 100 * (1 + _CI) / 2)))


def per_fold_pair_stats(
    fold_to_probs: dict[int, np.ndarray],
    fold_to_labels: dict[int, np.ndarray],
    src_idx: int,
    tgt_idx: int,
) -> dict:
    """Per-fold stats for a (src -> tgt) confusion pair with a 10-fold bootstrap CI."""
    conf, abs_tgt, argmax_tgt, renorm_tgt = [], [], [], []
    for k in sorted(fold_to_probs):
        P, y = fold_to_probs[k], fold_to_labels[k]
        sel = y == src_idx
        if not sel.any():
            continue
        Psel = P[sel]
        conf.append(float(Psel[:, src_idx].mean()))
        abs_tgt.append(float(Psel[:, tgt_idx].mean()))
        argmax_tgt.append(float((Psel.argmax(axis=1) == tgt_idx).mean()))
        renorm_rows = np.vstack([_mask_renorm(row, src_idx) for row in Psel])
        renorm_tgt.append(float(renorm_rows[:, tgt_idx].mean()))

    def pack(name, arr):
        m, lo, hi = _boot_ci(np.asarray(arr))
        return {f"{name}_mean": m, f"{name}_ci_lo": lo, f"{name}_ci_hi": hi}

    out = {"n_folds": len(conf)}
    out.update(pack("confidence", conf))
    out.update(pack("abs_target", abs_tgt))
    out.update(pack("argmax_target", argmax_tgt))
    out.update(pack("renorm_target", renorm_tgt))
    return out


def confusability_summary(
    soft: np.ndarray, hard: np.ndarray, flip: np.ndarray, class_names: list[str]
) -> pd.DataFrame:
    """Per-true-class headline table (pooled over all samples)."""
    C = len(class_names)
    rows = []
    for i, ci in enumerate(class_names):
        off = soft[i].copy()
        off[i] = -np.inf  # find the top confusion target (excluding the true class)
        j = int(np.argmax(off))
        rows.append(dict(
            true_class=ci,
            confidence_mean_p_true=float(soft[i, i]),
            argmax_accuracy=float(hard[i, i]),
            top_confusion_target=class_names[j],
            abs_mean_p_target=float(soft[i, j]),       # ABSOLUTE, un-masked
            argmax_rate_to_target=float(hard[i, j]),   # hard confusion
            renorm_flip_mass_to_target=float(flip[i, j]),  # masked+renorm flip target
        ))
    return pd.DataFrame(rows)


# heatmap (Purples, matching the Part 5 confusion-matrix style)
def _heatmap(M: np.ndarray, classes: list[str], title: str, out_png: Path,
             diag_is_blank: bool = False) -> None:
    plt.rcParams.update(_PLT_STYLE)
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(M, cmap="Purples", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted / target class"); ax.set_ylabel("True class")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    for x in np.arange(0.5, len(classes) - 1 + 1e-9):
        ax.axvline(x, color="white", linewidth=3.0, zorder=2)
    for y in np.arange(0.5, len(classes) - 1 + 1e-9):
        ax.axhline(y, color="white", linewidth=3.0, zorder=2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v) or (diag_is_blank and i == j):
                continue
            tc = "white" if v > 0.6 else "0.15"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=tc)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[fig] wrote {out_png}")


def _emit_body_table(summary: pd.DataFrame, out_tex: Path) -> None:
    """Compact body table: OOF confidence + dominant confusion target per class."""
    lines = [
        r"\begin{table}[h!]", r"\centering",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"True class & OOF conf.\ $\overline{P(\text{true})}$ & Top confusion target "
        r"& $\overline{P(\text{target})}$ \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{r.true_class} & {r.confidence_mean_p_true:.3f} & "
            f"{r.top_confusion_target} & {r.abs_mean_p_target:.3f} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Out-of-fold ResNet-18 confusability per true class, computed "
        r"from the un-masked OOF softmax that the Feature-Driven noise model is "
        r"built on. $\overline{P(\text{true})}$ is the mean softmax probability the "
        r"OOF model assigns to the true class (its confidence); the dominant "
        r"confusion target is the off-diagonal class with the highest mean "
        r"probability, reported with that absolute (un-masked) mean probability "
        r"$\overline{P(\text{target})}$. Because these are absolute probabilities, "
        r"a non-trivial $\overline{P(\text{target})}$ is direct evidence of genuine "
        r"visual confusability, independent of the masking and renormalisation used "
        r"to build the flip targets.}",
        r"\label{tab:oof-confusability}", r"\end{table}",
    ]
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"[tab] wrote {out_tex}")


# I/O + orchestration (project imports live here so the core stays portable)
def _load_inputs(dataset: str):
    """Return (probs_full, labels_int, class_names, per_fold or None, out_dir, root)."""
    from src.data.ham10000 import CLASS_NAMES, NUM_CLASSES, class_to_index
    from src.utils.io import ensure_dir, load_config, project_root

    cfg = load_config("base.yaml", f"data/{dataset}.yaml")
    root = project_root()

    oof_dir = root / cfg["paths"]["cv_folds"] / dataset / "oof_probs"
    full_path = oof_dir / "oof_probs_full.npy"
    if not full_path.exists():
        raise FileNotFoundError(
            f"{full_path} not found. Run stage1b_collect_oof_probs + "
            f"stage1b_merge_oof_probs first (this is the same artefact "
            f"feature-driven IDN consumes)."
        )

    metadata_path = (
        root / cfg["paths"]["data_processed"]
        / "one_image_per_lesion" / cfg["data"]["metadata_file"]
    )
    metadata = pd.read_csv(metadata_path)
    probs_full = np.load(full_path).astype(np.float64)
    if probs_full.shape != (len(metadata), NUM_CLASSES):
        raise ValueError(
            f"oof_probs_full shape {probs_full.shape} != "
            f"(metadata={len(metadata)}, classes={NUM_CLASSES})."
        )
    labels = np.array([class_to_index(c) for c in metadata["dx"]], dtype=np.int64)

    # Optional per-fold reconstruction for a 10-fold CI on the headline pair.
    id_to_row = {iid: i for i, iid in enumerate(metadata["image_id"].tolist())}
    per_fold = {"probs": {}, "labels": {}}
    n_folds = int(cfg["folds"])
    for k in range(n_folds):
        npy, ids = oof_dir / f"fold_{k:02d}.npy", oof_dir / f"fold_{k:02d}_ids.csv"
        if not (npy.exists() and ids.exists()):
            per_fold = None
            break
        pk = np.load(npy).astype(np.float64)
        idk = pd.read_csv(ids)["image_id"].tolist()
        rows = [id_to_row[i] for i in idk]
        per_fold["probs"][k] = pk
        per_fold["labels"][k] = labels[rows]

    out_dir = ensure_dir(root / cfg["paths"]["results"] / "oof_confusion" / dataset)
    return probs_full, labels, list(CLASS_NAMES), per_fold, out_dir, root, cfg


def _reorder(M: np.ndarray, class_names: list[str], order: list[str]) -> tuple[np.ndarray, list[str]]:
    idx = [class_names.index(c) for c in order if c in class_names]
    return M[np.ix_(idx, idx)], [class_names[i] for i in idx]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="OOF confusability analysis for Feature-Driven IDN.")
    p.add_argument("--dataset", default="imbalanced")
    p.add_argument("--pair", nargs=2, metavar=("SRC", "TGT"), default=["nv", "mel"],
                   help="Spotlight (true -> target) pair for the per-fold CI (default: nv mel).")
    args = p.parse_args(argv)

    probs, labels, class_names, per_fold, out_dir, root, cfg = _load_inputs(args.dataset)
    C = len(class_names)

    soft = softmax_confusion(probs, labels, C)
    hard = argmax_confusion(probs, labels, C)
    flip = flip_target_confusion(probs, labels, C)

    order = [c for c in _CLASS_ORDER_FREQ if c in class_names]
    soft_o, order_lbls = _reorder(soft, class_names, order)
    hard_o, _ = _reorder(hard, class_names, order)
    flip_o, _ = _reorder(flip, class_names, order)

    pd.DataFrame(soft_o, index=order_lbls, columns=order_lbls).to_csv(
        out_dir / "oof_confusion_softmax_rownorm.csv")
    pd.DataFrame(hard_o, index=order_lbls, columns=order_lbls).to_csv(
        out_dir / "oof_confusion_argmax.csv")
    pd.DataFrame(flip_o, index=order_lbls, columns=order_lbls).to_csv(
        out_dir / "oof_flip_target_renorm.csv")

    summary = confusability_summary(soft, hard, flip, class_names)
    summary = summary.set_index("true_class").loc[order].reset_index()

    src, tgt = args.pair
    if per_fold is not None and src in class_names and tgt in class_names:
        si, ti = class_names.index(src), class_names.index(tgt)
        pair_stats = per_fold_pair_stats(per_fold["probs"], per_fold["labels"], si, ti)
        pd.DataFrame([{"src": src, "tgt": tgt, **pair_stats}]).to_csv(
            out_dir / f"oof_pair_{src}_to_{tgt}_perfold_ci.csv", index=False)
    else:
        pair_stats = None
        if per_fold is None:
            print("[note] per-fold OOF files not found; pooled stats only (no 10-fold CI).")

    summary.to_csv(out_dir / "oof_confusability_summary.csv", index=False)

    _heatmap(soft_o, order_lbls,
             "OOF softmax confusion (row-normalised, un-masked)",
             out_dir / "oof_confusion_softmax_rownorm.png")
    _heatmap(hard_o, order_lbls,
             "OOF argmax confusion (hard, row-normalised)",
             out_dir / "oof_confusion_argmax.png")
    _emit_body_table(summary, out_dir / "tab_oof_confusability.tex")

    # console summary
    print("\n" + "=" * 74)
    print("OOF CONFUSABILITY (per true class, pooled over all samples)")
    print("=" * 74)
    print(summary.to_string(index=False,
          formatters={c: (lambda x: f"{x:.3f}") for c in summary.columns
                      if summary[c].dtype.kind == "f"}))
    if pair_stats is not None:
        print("\n" + "-" * 74)
        print(f"SPOTLIGHT PAIR  true={src} -> target={tgt}   "
              f"(per-fold mean [95% CI], n={pair_stats['n_folds']} folds)")
        print("-" * 74)
        def show(lbl, key):
            print(f"  {lbl:<34s} {pair_stats[key+'_mean']:.3f} "
                  f"[{pair_stats[key+'_ci_lo']:.3f}, {pair_stats[key+'_ci_hi']:.3f}]")
        show(f"OOF confidence  P({src})", "confidence")
        show(f"ABS  P({tgt})  [un-masked]", "abs_target")
        show(f"argmax-> {tgt} rate  [hard]", "argmax_target")
        show(f"renorm flip mass -> {tgt}", "renorm_target")

    # manifest
    from src.utils.manifest import write_manifest
    manifest_path = root / cfg["paths"]["manifests"] / f"analyze_oof_confusion_{args.dataset}.json"
    write_manifest(
        manifest_path, stage="analyze_oof_confusion",
        params={"dataset": args.dataset, "pair": [src, tgt],
                "n_boot": _N_BOOT, "boot_seed": _BOOT_SEED, "ci": _CI,
                "source_artifact": "oof_probs/oof_probs_full.npy",
                "note": "reads saved OOF softmax; no retraining, no re-injection"},
        outputs=[str((out_dir / f).relative_to(root)) for f in (
            "oof_confusion_softmax_rownorm.csv", "oof_confusion_argmax.csv",
            "oof_flip_target_renorm.csv", "oof_confusability_summary.csv",
            "oof_confusion_softmax_rownorm.png", "oof_confusion_argmax.png",
            "tab_oof_confusability.tex")],
    )
    print(f"\n[done] outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())