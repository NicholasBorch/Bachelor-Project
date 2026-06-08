"""
Results.5 - memorization dynamics over training (per-epoch NTA / LNMR).

Reads the per-epoch train_diagnostics logged in each fold's training_log.jsonl
(at checkpoint epochs) and writes, under epoch_trajectory/:
  nta_lnmr_epochs_<P>_tau20.(pdf|png)   single-tau (0.20) NTA/LNMR vs epoch, CI bands.
  grid_nta_epochs_<P>.(pdf|png)         small-multiples (panels = tau) NTA vs epoch.
  grid_lnmr_epochs_<P>.(pdf|png)        same for LNMR.
  _epoch_trajectory_<P>.csv             tidy per (method, tau, epoch) means + CIs.

Descriptive; the only inferential element is the 95% bootstrap CI over folds.
Epoch 0 is the untrained model (NTA/LNMR near chance), not resistance.
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
from matplotlib.patches import Patch


@dataclass
class Config:
    EXPERIMENT_ROOT: Path = Path("./results/main_experiment")
    PROTOCOLS: dict = field(default_factory=lambda: {
        "AP": ("pretrained", "adam", "pretrained_adam"),
        # "A":  ("scratch",   "adam", "scratch_adam"),
        # "SP": ("pretrained","sgd",  "pretrained_sgd"),
        # "S":  ("scratch",   "sgd",  "scratch_sgd"),
    })
    PROTOCOLS_TO_RUN: tuple = ("AP",)
    TRAINING_SUBDIR: str = "training"
    LOG_FILENAME: str = "training_log.jsonl"
    DIAG_KEY: str = "train_diagnostics"
    TAU_DIR_FMT: str = "tau_{tt:02d}"
    FOLD_DIR_FMT: str = "fold_{ff:02d}"

    METHODS: tuple = ("baseline", "sce", "elr", "asyco_divmix")
    METHOD_LABELS: dict = field(default_factory=lambda: {
        "baseline": "Baseline", "sce": "SCE", "elr": "ELR", "asyco_divmix": "AsyCo",
    })
    TAUS: tuple = (0.1, 0.2, 0.3, 0.4, 0.5)   # tau>0 (epoch traj undefined at tau0)
    FOCUS_TAU: float = 0.20
    N_FOLDS: int = 10

    N_BOOT: int = 10000
    CI: float = 0.95
    SEED: int = 10

    PALETTE: dict = field(default_factory=lambda: {
        "baseline": "#9ec9e2", "sce": "#2a9d8f", "elr": "#e07a3f", "asyco_divmix": "#7b5cb8",
    })
    OUT_ROOT: Path = Path("./results/mechanism")
    SUBDIR: str = "epoch_trajectory"
    FIG_DPI: int = 200
    SAVE_PNG: bool = True
    SAVE_PDF: bool = True


CFG = Config()


def _out(protocol: str) -> Path:
    d = CFG.OUT_ROOT / protocol / CFG.SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# Load per-epoch diagnostics from training_log.jsonl across folds
def _read_fold_log(protocol, method, tau, fold):
    """Return [(epoch, nta, lnmr)] for checkpoint epochs in this fold's log, or []."""
    _, _, folder = CFG.PROTOCOLS[protocol]
    tt = int(round(tau * 100))
    fp = (CFG.EXPERIMENT_ROOT / folder / CFG.TRAINING_SUBDIR / method
          / CFG.TAU_DIR_FMT.format(tt=tt) / CFG.FOLD_DIR_FMT.format(ff=fold)
          / CFG.LOG_FILENAME)
    if not fp.exists():
        return []
    out = []
    with open(fp) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            diag = rec.get(CFG.DIAG_KEY)
            if isinstance(diag, dict) and diag.get("nta") is not None:
                out.append((int(rec["epoch"]),
                            float(diag["nta"]), float(diag["lnmr"])))
    return out


def load_trajectory(protocol):
    """Long DataFrame: method, tau, fold, epoch, nta, lnmr."""
    rows = []
    for method in CFG.METHODS:
        for tau in CFG.TAUS:
            for fold in range(CFG.N_FOLDS):
                for ep, nta, lnmr in _read_fold_log(protocol, method, tau, fold):
                    rows.append(dict(method=method, tau=float(tau), fold=fold,
                                     epoch=ep, nta=nta, lnmr=lnmr))
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"No per-epoch diagnostics found for protocol {protocol}. "
            f"Expected {CFG.DIAG_KEY} entries in {CFG.LOG_FILENAME} files.")
    return df


def _boot_ci(values):
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    if v.size == 1:
        return (float(v[0]),) * 3
    rng = np.random.default_rng(CFG.SEED)
    boot = rng.choice(v, size=(CFG.N_BOOT, v.size), replace=True).mean(axis=1)
    return (float(v.mean()),
            float(np.percentile(boot, 100 * (1 - CFG.CI) / 2)),
            float(np.percentile(boot, 100 * (1 + CFG.CI) / 2)))


def summarise(df):
    """Per (method, tau, epoch): mean + bootstrap CI for nta and lnmr."""
    recs = []
    for (method, tau, epoch), g in df.groupby(["method", "tau", "epoch"]):
        n_m, n_lo, n_hi = _boot_ci(g["nta"].values)
        l_m, l_lo, l_hi = _boot_ci(g["lnmr"].values)
        recs.append(dict(method=method, tau=float(tau), epoch=int(epoch),
                         nta=n_m, nta_lo=n_lo, nta_hi=n_hi,
                         lnmr=l_m, lnmr_lo=l_lo, lnmr_hi=l_hi,
                         n=int(g["nta"].notna().sum())))
    return pd.DataFrame(recs).sort_values(["method", "tau", "epoch"])


# Styling / saving
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
    print(f"[fig] {outdir}/{stem}.(pdf|png)")


def _plot_one(ax, summ, tau, metric, lo_key, hi_key, ylabel):
    sub = summ[np.isclose(summ.tau, tau)]
    epochs_present = sorted(sub["epoch"].unique())
    for method in CFG.METHODS:
        s = sub[sub.method == method].sort_values("epoch")
        if s.empty:
            continue
        ax.plot(s["epoch"], s[metric], "-o", color=CFG.PALETTE.get(method),
                markersize=4, label=CFG.METHOD_LABELS.get(method, method), zorder=3)
        ax.fill_between(s["epoch"], s[lo_key], s[hi_key],
                        color=CFG.PALETTE.get(method), alpha=0.18, linewidth=0, zorder=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    # x-ticks only at the recorded checkpoint epochs (no in-between)
    if epochs_present:
        ax.set_xticks(epochs_present)
        ax.set_xticklabels([str(int(e)) for e in epochs_present])
        # pin the y-axis spine at the first epoch (no left padding)
        ax.set_xlim(left=epochs_present[0])
        ax.spines["left"].set_position(("data", epochs_present[0]))
    ax.set_ylim(bottom=0)


# 1. Single-tau two-panel trajectory (with CI bands)
def fig_focus(summ, protocol):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    _plot_one(axes[0], summ, CFG.FOCUS_TAU, "nta", "nta_lo", "nta_hi", "NTA")
    _plot_one(axes[1], summ, CFG.FOCUS_TAU, "lnmr", "lnmr_lo", "lnmr_hi", "LNMR")
    axes[0].set_title("NTA")
    axes[1].set_title("LNMR")
    axes[0].legend(frameon=False, ncol=2, loc="best")
    # flag epoch 0 as near-chance (untrained)
    for ax in axes:
        ax.axvspan(-2, 2, color="0.5", alpha=0.06, zorder=0)
    fig.suptitle(f"Memorization dynamics over training "
                 f"($\\tau = {CFG.FOCUS_TAU:.2f}$) - protocol {protocol}",
                 y=1.02, fontsize=12.5)
    fig.tight_layout()
    _save(fig, _out(protocol), f"nta_lnmr_epochs_{protocol}_tau{int(round(CFG.FOCUS_TAU*100)):02d}")


# 2. Small-multiples grid: panels = tau, lines = method
def fig_grid(summ, protocol, metric, lo_key, hi_key, label):
    _style()
    taus = list(CFG.TAUS)
    ncol = 3
    nrow = int(np.ceil(len(taus) / ncol))
    n_last = len(taus) - ncol * (nrow - 1)   # panels in the final row

    # 2x-fine column grid so a partially-filled last row can be centered
    fig = plt.figure(figsize=(5.0 * ncol, 3.6 * nrow))
    gs = fig.add_gridspec(nrow, 2 * ncol)
    axes = []
    first_ax = None
    for idx, tau in enumerate(taus):
        row = idx // ncol
        col_in_row = idx % ncol
        is_last_row = (row == nrow - 1)
        offset = (ncol - n_last) if (is_last_row and n_last < ncol) else 0
        c0 = 2 * col_in_row + offset
        ax = fig.add_subplot(gs[row, c0:c0 + 2],
                             sharey=first_ax if first_ax is not None else None)
        if first_ax is None:
            first_ax = ax
        _plot_one(ax, summ, tau, metric, lo_key, hi_key, label)
        ax.set_title(f"$\\tau = {tau:.2f}$")
        axes.append(ax)

    handles = [Patch(facecolor=CFG.PALETTE.get(m), label=CFG.METHOD_LABELS.get(m, m))
               for m in CFG.METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=len(CFG.METHODS),
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"{label} over training across noise rates - protocol {protocol}",
                 y=1.0, fontsize=12.5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _save(fig, _out(protocol), f"grid_{metric}_epochs_{protocol}")


def main():
    for protocol in CFG.PROTOCOLS_TO_RUN:
        if protocol not in CFG.PROTOCOLS:
            print(f"[skip] {protocol}: not in CONFIG.PROTOCOLS"); continue
        print(f"\n=== protocol {protocol} ===")
        try:
            df = load_trajectory(protocol)
        except FileNotFoundError as e:
            print(f"[skip] {protocol}: {e}"); continue
        eps = sorted(df.epoch.unique())
        print(f"[load] {len(df)} rows; checkpoint epochs = {eps}; "
              f"methods = {sorted(df.method.unique())}")
        summ = summarise(df)
        summ.to_csv(_out(protocol) / f"_epoch_trajectory_{protocol}.csv", index=False)
        print(f"[csv] {_out(protocol)}/_epoch_trajectory_{protocol}.csv")
        fig_focus(summ, protocol)
        fig_grid(summ, protocol, "nta", "nta_lo", "nta_hi", "NTA")
        fig_grid(summ, protocol, "lnmr", "lnmr_lo", "lnmr_hi", "LNMR")
    print("\nDone.")


if __name__ == "__main__":
    main()