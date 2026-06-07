#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Results.5 - Mechanism analysis (NTA / LNMR / per-class F1 / confusion).

DESCRIPTIVE ONLY. No statistical tests. The single concession to inference is a
95% bootstrap CI band on the aggregate NTA/LNMR curves (for visual consistency
with the rest of the chapter); everything else is fold-averaged texture.

OUTPUTS (into results/mechanism/<protocol>/, one subfolder per artifact type)
  nta_lnmr/        aggregate NTA & LNMR vs tau, one line per method, CI bands,
                   tau=0 excluded (undefined).
  perclass_f1/     per-class F1 heatmap (methods x classes), one per tau.
  perclass_lnmr/   per-class LNMR-by-clean heatmap, one per tau (tau>0).
  perclass_nta/    per-class NTA-by-clean heatmap, one per tau (tau>0).
  confusion/       summed confusion matrices per (method, tau): row-normalized
                   AND raw-count versions.

COLOUR CONVENTION
  Green = good, red = bad, always. For high-is-good metrics (NTA, F1) green is
  the high end; for low-is-good metrics (LNMR) the colormap is reversed so green
  is the low end. Confusion matrices use a neutral sequential map (counts/rates
  are not "good/bad").

DATA SOURCES
  * raw_fold_results.csv (per fold: nta, lnmr, per_class_f1_<cls>) for the
    NTA/LNMR curves and the per-class F1 heatmap.
  * per-run test_metrics.json for the confusion matrices and the per-class
    NTA/LNMR-by-clean arrays (summed / averaged over folds here).

CONFIG: only the block below. Protocol selection is by (init, optim); extend
PROTOCOLS as the other runs land.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


@dataclass
class Config:
    # --- where the data is ---------------------------------------------------
    EXPERIMENT_ROOT: Path = Path("./results/main_experiment")
    # protocol code -> (init, optim, folder under EXPERIMENT_ROOT)
    PROTOCOLS: dict = field(default_factory=lambda: {
        "AP": ("pretrained", "adam", "pretrained_adam"),
        # "A":  ("scratch",   "adam", "scratch_adam"),
        # "SP": ("pretrained","sgd",  "pretrained_sgd"),
        # "S":  ("scratch",   "sgd",  "scratch_sgd"),
    })
    PROTOCOLS_TO_RUN: tuple = ("AP",)
    RAW_FOLD_CSV: str = "figures_and_tables/raw_fold_results.csv"
    TRAINING_SUBDIR: str = "training"
    METRICS_FILENAME: str = "test_metrics.json"
    TAU_DIR_FMT: str = "tau_{tt:02d}"
    FOLD_DIR_FMT: str = "fold_{ff:02d}"
    DATASET: str = "imbalanced"

    # --- experimental design -------------------------------------------------
    METHODS: tuple = ("baseline", "sce", "elr", "asyco_divmix")
    METHOD_LABELS: dict = field(default_factory=lambda: {
        "baseline": "Baseline", "sce": "SCE", "elr": "ELR", "asyco_divmix": "AsyCo",
    })
    TAUS: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    N_FOLDS: int = 10

    # class order: "freq" (nv first, df last), "alpha", or an explicit list
    CLASS_ORDER_MODE: str = "freq"
    CLASSES_ALPHA: tuple = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    CLASSES_FREQ: tuple = ("nv", "bkl", "mel", "bcc", "akiec", "vasc", "df")

    # --- stats (only the NTA/LNMR CI band) -----------------------------------
    N_BOOT: int = 10000
    CI: float = 0.95
    SEED: int = 10

    # --- palette for the line plot -------------------------------------------
    PALETTE: dict = field(default_factory=lambda: {
        "baseline": "#9ec9e2", "sce": "#2a9d8f", "elr": "#e07a3f", "asyco_divmix": "#7b5cb8",
    })

    OUT_ROOT: Path = Path("./results/mechanism")
    FIG_DPI: int = 200
    SAVE_PNG: bool = True
    SAVE_PDF: bool = True


CFG = Config()


def _classes():
    if CFG.CLASS_ORDER_MODE == "freq":
        return list(CFG.CLASSES_FREQ)
    if CFG.CLASS_ORDER_MODE == "alpha":
        return list(CFG.CLASSES_ALPHA)
    return list(CFG.CLASS_ORDER_MODE)


def _out(protocol: str, sub: str) -> Path:
    d = CFG.OUT_ROOT / protocol / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# green=good colormaps
# ---------------------------------------------------------------------------
# Red -> yellow -> green (bad -> good). For high-is-good we map value directly;
# for low-is-good we reverse so green sits at the low end.
_RYG = LinearSegmentedColormap.from_list(
    "ryg", ["#c0392b", "#e67e22", "#f1c40f", "#7dcea0", "#1e8449"])
_GYR = _RYG.reversed()


def good_cmap(low_is_good: bool):
    """Colormap so that 'good' is green. If low values are good, reverse."""
    return _GYR if low_is_good else _RYG


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def _raw_fold(protocol: str) -> pd.DataFrame:
    init, optim, folder = CFG.PROTOCOLS[protocol]
    fp = CFG.EXPERIMENT_ROOT / folder / CFG.RAW_FOLD_CSV
    if not fp.exists():
        # fall back to a single combined csv that carries init/optim columns
        raise FileNotFoundError(f"raw fold csv not found: {fp}")
    df = pd.read_csv(fp)
    # filter to this protocol if the columns exist (combined files carry them)
    for col, val in (("init", init), ("optim", optim), ("dataset", CFG.DATASET)):
        if col in df.columns:
            df = df[df[col] == val]
    return df.reset_index(drop=True)


def _read_run_json(protocol, method, tau, fold):
    _, _, folder = CFG.PROTOCOLS[protocol]
    tt = int(round(tau * 100))
    fp = (CFG.EXPERIMENT_ROOT / folder / CFG.TRAINING_SUBDIR / method
          / CFG.TAU_DIR_FMT.format(tt=tt) / CFG.FOLD_DIR_FMT.format(ff=fold)
          / CFG.METRICS_FILENAME)
    if not fp.exists():
        return None
    with open(fp) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# bootstrap (only used for the NTA/LNMR band)
# ---------------------------------------------------------------------------
def _boot_ci(values):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    if v.size == 1:
        return (float(v[0]), float(v[0]), float(v[0]))
    rng = np.random.default_rng(CFG.SEED)
    boot = rng.choice(v, size=(CFG.N_BOOT, v.size), replace=True).mean(axis=1)
    return (float(v.mean()),
            float(np.percentile(boot, 100 * (1 - CFG.CI) / 2)),
            float(np.percentile(boot, 100 * (1 + CFG.CI) / 2)))


def _style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Palatino", "Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
        "mathtext.fontset":   "cm",
        "axes.unicode_minus": False,
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#cccccc",
        "axes.grid": True, "grid.alpha": 0.25,
        "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def _save(fig, outdir, stem):
    if CFG.SAVE_PDF:
        fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    if CFG.SAVE_PNG:
        fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. aggregate NTA / LNMR vs tau, with bootstrap CI bands
# ---------------------------------------------------------------------------
def fig_nta_lnmr(protocol, raw):
    _style()
    outdir = _out(protocol, "nta_lnmr")
    taus_nz = [t for t in CFG.TAUS if t > 0]
    panels = [("nta", "NTA"), ("lnmr", "LNMR")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, (col, title) in zip(axes, panels):
        for method in CFG.METHODS:
            means, los, his = [], [], []
            for t in taus_nz:
                vals = raw[(raw.method == method) & (np.isclose(raw.tau, t))][col].values
                m, lo, hi = _boot_ci(vals)
                means.append(m); los.append(lo); his.append(hi)
            x = np.array(taus_nz)
            ax.plot(x, means, "-o", color=CFG.PALETTE.get(method), markersize=4,
                    label=CFG.METHOD_LABELS.get(method, method), zorder=3)
            ax.fill_between(x, los, his, color=CFG.PALETTE.get(method), alpha=0.18,
                            linewidth=0, zorder=2)
        ax.set_xlabel(r"Noise rate $\tau$"); ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(taus_nz)
        ax.set_xticklabels([f"{t:.1f}" for t in taus_nz])
        ax.set_xlim(left=taus_nz[0])
        ax.spines["left"].set_position(("data", taus_nz[0]))
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, ncol=2, loc="best")
    fig.suptitle(f"Memorization diagnostics across noise - protocol {protocol}",
                 y=1.02, fontsize=12.5)
    fig.tight_layout()
    _save(fig, outdir, f"nta_lnmr_{protocol}")
    print(f"[fig] {outdir}/nta_lnmr_{protocol}.(pdf|png)")


# ---------------------------------------------------------------------------
# 2. per-class F1 heatmap (methods x classes), one per tau   (green=high)
# ---------------------------------------------------------------------------
def _white_cell_gaps(ax, nrows, ncols, lw=3.0):
    """Draw white separators between imshow cells as explicit high-zorder lines
    so they sit cleanly on top of the cells but below the text labels."""
    for x in np.arange(0.5, ncols - 1 + 1e-9, 1):
        ax.axvline(x, color="white", linewidth=lw, zorder=2)
    for y in np.arange(0.5, nrows - 1 + 1e-9, 1):
        ax.axhline(y, color="white", linewidth=lw, zorder=2)


def _heatmap(matrix, row_labels, col_labels, title, outdir, stem,
             low_is_good=False, vmin=0.0, vmax=1.0, fmt="{:.2f}"):
    _style()
    fig, ax = plt.subplots(figsize=(1.0 + 0.95 * len(col_labels),
                                    1.0 + 0.5 * len(row_labels)))
    im = ax.imshow(matrix, cmap=good_cmap(low_is_good), vmin=vmin, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=0)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    _white_cell_gaps(ax, matrix.shape[0], matrix.shape[1])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                # text colour: dark on light cells, white on saturated ends
                freq = (v - vmin) / (vmax - vmin + 1e-9)
                tc = "white" if (freq < 0.18 or freq > 0.82) else "0.1"
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=8, color=tc)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, outdir, stem)


def fig_perclass_f1(protocol, raw):
    outdir = _out(protocol, "perclass_f1")
    classes = _classes()
    for tau in CFG.TAUS:
        M = np.full((len(CFG.METHODS), len(classes)), np.nan)
        for i, method in enumerate(CFG.METHODS):
            sub = raw[(raw.method == method) & (np.isclose(raw.tau, tau))]
            for j, c in enumerate(classes):
                col = f"per_class_f1_{c}"
                if col in sub.columns and len(sub):
                    M[i, j] = np.nanmean(sub[col].values)
        _heatmap(M, [CFG.METHOD_LABELS.get(m, m) for m in CFG.METHODS], classes,
                 f"Per-class F1 - protocol {protocol}, $\\tau={tau:.2f}$",
                 outdir, f"perclass_f1_{protocol}_tau{int(round(tau*100)):02d}",
                 low_is_good=False, vmin=0.0, vmax=1.0)
    print(f"[fig] {outdir}/perclass_f1_{protocol}_tau*.png  (6 files)")


# ---------------------------------------------------------------------------
# 3+4. per-class LNMR / NTA by-clean heatmaps (from json), one per tau>0
#       LNMR: low is good (green=low). NTA: high is good (green=high).
# ---------------------------------------------------------------------------
def _perclass_byclean_mean(protocol, method, tau, key):
    """Fold-average of a per-class array key from the run jsons."""
    rows = []
    for fold in range(CFG.N_FOLDS):
        d = _read_run_json(protocol, method, tau, fold)
        if d is None or key not in d or d[key] is None:
            continue
        rows.append(np.asarray(d[key], float))
    if not rows:
        return None
    return np.nanmean(np.vstack(rows), axis=0)   # length-7, in alpha order


def fig_perclass_byclean(protocol, key, sub, low_is_good, label):
    outdir = _out(protocol, sub)
    classes = _classes()
    # json arrays are in ALPHA order; map to chosen order
    alpha = list(CFG.CLASSES_ALPHA)
    order_idx = [alpha.index(c) for c in classes]
    for tau in [t for t in CFG.TAUS if t > 0]:
        M = np.full((len(CFG.METHODS), len(classes)), np.nan)
        for i, method in enumerate(CFG.METHODS):
            arr = _perclass_byclean_mean(protocol, method, tau, key)
            if arr is not None:
                M[i, :] = arr[order_idx]
        _heatmap(M, [CFG.METHOD_LABELS.get(m, m) for m in CFG.METHODS], classes,
                 f"{label} - protocol {protocol}, $\\tau={tau:.2f}$",
                 outdir, f"{sub}_{protocol}_tau{int(round(tau*100)):02d}",
                 low_is_good=low_is_good, vmin=0.0, vmax=1.0)
    print(f"[fig] {outdir}/{sub}_{protocol}_tau*.png")


# ---------------------------------------------------------------------------
# 5. summed confusion matrices per (method, tau): normalized + counts
# ---------------------------------------------------------------------------
def _summed_confusion(protocol, method, tau):
    acc = None
    for fold in range(CFG.N_FOLDS):
        d = _read_run_json(protocol, method, tau, fold)
        if d is None or "confusion_matrix" not in d:
            continue
        cm = np.asarray(d["confusion_matrix"], float)
        acc = cm if acc is None else acc + cm
    return acc


def fig_confusion(protocol):
    outdir = _out(protocol, "confusion")
    classes = _classes()
    alpha = list(CFG.CLASSES_ALPHA)
    order_idx = [alpha.index(c) for c in classes]
    for method in CFG.METHODS:
        for tau in CFG.TAUS:
            cm = _summed_confusion(protocol, method, tau)
            if cm is None:
                continue
            cm = cm[np.ix_(order_idx, order_idx)]   # reorder rows+cols
            tt = int(round(tau * 100))
            mlab = CFG.METHOD_LABELS.get(method, method)
            # raw counts
            _confusion_panel(cm, classes,
                             f"{mlab} confusion (counts) - $\\tau={tau:.2f}$",
                             outdir, f"confusion_counts_{method}_tau{tt:02d}",
                             normalize=False)
            # row-normalized
            rs = cm.sum(1, keepdims=True); rs[rs == 0] = 1
            _confusion_panel(cm / rs, classes,
                             f"{mlab} confusion (row-normalized) - $\\tau={tau:.2f}$",
                             outdir, f"confusion_norm_{method}_tau{tt:02d}",
                             normalize=True)
    print(f"[fig] {outdir}/confusion_(counts|norm)_*_tau*.png")


def _confusion_panel(M, classes, title, outdir, stem, normalize):
    _style()
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(M, cmap="Greens", vmin=0, vmax=(1.0 if normalize else None),
                   aspect="auto")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    _white_cell_gaps(ax, M.shape[0], M.shape[1])
    mx = np.nanmax(M) if M.size else 1
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = (f"{v:.2f}" if normalize else f"{int(round(v))}")
            tc = "white" if v > 0.6 * mx else "0.15"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=tc)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, outdir, stem)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    for protocol in CFG.PROTOCOLS_TO_RUN:
        if protocol not in CFG.PROTOCOLS:
            print(f"[skip] {protocol}: not defined in CONFIG.PROTOCOLS.")
            continue
        print(f"\n=== protocol {protocol} ===")
        try:
            raw = _raw_fold(protocol)
        except FileNotFoundError as e:
            print(f"[skip] {protocol}: {e}")
            continue
        raw = raw[raw.method.isin(CFG.METHODS)]
        print(f"[load] {len(raw)} fold-rows; methods={sorted(raw.method.unique())}")

        fig_nta_lnmr(protocol, raw)                # CI bands here only
        fig_perclass_f1(protocol, raw)
        fig_perclass_byclean(protocol, "per_class_lnmr_by_clean", "perclass_lnmr",
                             low_is_good=True,  label="Per-class LNMR (by true class)")
        fig_perclass_byclean(protocol, "per_class_nta_by_clean", "perclass_nta",
                             low_is_good=False, label="Per-class NTA (by true class)")
        fig_confusion(protocol)
    print("\nDone.")


if __name__ == "__main__":
    main()