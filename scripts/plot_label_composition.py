#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In-flip label composition under label noise.

For each noise rate tau, one subplot; within a subplot, one column per class.
Each column is the set of training examples whose *observed* (possibly noisy)
label is that class, split into:
  - the genuinely-correct part  (true label == observed label), and
  - the noisy part              (intruders that flipped IN from other classes).

Columns are proportions (sum to 1), so the height of the noisy segment is
exactly 1 - purity for that observed class. Because the dataset is imbalanced,
rare classes show a large noisy segment even at modest tau (a small flip rate
out of the majority class can outnumber a rare class's own retained members),
while the majority class stays almost fully correct. That contrast is the point.

WHAT IT PRODUCES  (under RESULTS_ROOT / ANALYSIS_DIR, e.g. results/label_noise_composition/)
  fig_label_composition_inflip_<noise_model>.pdf / .png
  label_composition_inflip_<noise_model>.csv   (tidy: tau, class, n_true, n_noisy, total, purity)

============================================================================
EDIT ONLY THE CONFIG BLOCK.
============================================================================
The composition is a property of the noise process, not of any model, so the
script needs one of:

  SOURCE = "matrix": a row-stochastic transition matrix T per tau
                     (T[i, j] = P(observed = j | true = i); rows/cols ordered
                     exactly like CONFIG.CLASSES) plus a class-count vector.
                     Exact, clean, recommended if you have the matrices.
  SOURCE = "labels": per-tau arrays of (y_true, y_noisy) over the dataset
                     (folds pooled). Empirical; use if you only saved the
                     realised noisy labels.
  SOURCE = "csv"   : a precomputed tidy CSV with the columns above.

Everything else (the math, the styling) needs no change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================================
# CONFIG
# ============================================================================
@dataclass
class Config:
    # ---- data source -------------------------------------------------------
    # "folds"  : your cv tree, read realised noisy labels from train_noisy.csv  (default)
    # "matrix" : a transition matrix per tau + class counts
    # "labels" : per-tau (y_true, y_noisy) files, folds pooled
    # "csv"    : a precomputed tidy file
    SOURCE: str = "folds"

    # --- SOURCE="folds": your layout ---------------------------------------
    #   {CV_ROOT}/{SAMPLING}/{NOISE_MODEL}/tau_{NN}/fold_{NN}/{TRAIN_FILE}
    # tau_NN is tau*100 (tau_00=0.0 ... tau_50=0.5); balanced is ignored.
    CV_ROOT: Path = Path("./data/processed/HAM10000/cv_folds")
    SAMPLING: str = "imbalanced"               # NOT "balanced"
    NOISE_MODEL: str = "feature_driven"        # e.g. swap to "normalized" for the other model
    FOLD: int = 0                              # used only when AGGREGATE_FOLDS is False
    AGGREGATE_FOLDS: bool = True               # average the composition over all N_FOLDS folds
    N_FOLDS: int = 10
    TAU_DIR_FMT: str = "tau_{tt:02d}"
    FOLD_DIR_FMT: str = "fold_{ff:02d}"
    TRAIN_FILE: str = "train_noisy.csv"
    # column mapping for train_noisy.csv. Your files store the (noisy) working
    # label in 'dx', the preserved truth in 'dx_clean', and a flip flag in
    # 'flipped'; with the flag present the truth column is not strictly needed.
    NOISY_LABEL_COL: str = "dx"
    TRUE_LABEL_COL: str = "dx_clean"
    IS_NOISY_COL: str = "flipped"

    # SOURCE="matrix": one matrix file per tau + a counts file.
    #   matrices: 7x7, rows = true class, cols = observed class, rows sum to 1,
    #   axis order MUST match CONFIG.CLASSES. .npy or .csv (no header) accepted.
    T_PATH_TEMPLATE: str = "./noise/transition_tau{tau}.npy"
    CLASS_COUNTS_PATH: str = "./noise/class_counts.json"   # {"akiec": 327, ...}

    # SOURCE="labels": per-tau label files (folds pooled). Each file holds two
    #   integer/string columns: true label and noisy label. .npz (keys
    #   y_true,y_noisy), .npy (Nx2), or .csv (cols named below) accepted.
    LABELS_PATH_TEMPLATE: str = "./noise/labels_tau{tau}.csv"
    LABELS_TRUE_COL: str = "y_true"
    LABELS_NOISY_COL: str = "y_noisy"

    # SOURCE="csv": precomputed tidy file with columns
    #   tau, class, n_true, n_noisy  (total/purity derived if absent)
    TIDY_CSV_PATH: str = "./noise/label_composition_inflip.csv"

    # ---- design ------------------------------------------------------------
    CLASSES: tuple = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    TAUS: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    ORDER_BY_PREVALENCE: bool = True           # x-axis: largest class first
    AS_PROPORTION: bool = False                # False: counts (shows the distribution); True: 100% columns
    ANNOTATE_N: bool = True                    # print each column's total above it (key for the rare classes)

    NROWS: int = 2
    NCOLS: int = 3
    XTICK_ROTATION: int = 0                     # 0 so the per-class total sits on a clean second line

    # ---- colours (true = solid teal, noisy = semi-transparent amber) -------
    # B&W is not a constraint, so transparency does the work and the hatch is
    # dropped. Teal/amber is a colourblind-safe blue/orange pairing.
    TRUE_COLOR: str = "#1F7A8C"
    NOISE_COLOR: str = "#E4A552"
    TRUE_ALPHA: float = 0.95
    NOISE_ALPHA: float = 0.60
    NOISE_HATCH: str = ""                       # set e.g. "////" to re-enable a hatch
    BAR_EDGE: str = "white"
    BAR_EDGE_LW: float = 0.5
    TRUE_LABEL: str = "Correctly labelled"
    NOISE_LABEL: str = "Noisy (flipped in)"
    # count-label colours: match each bar (amber darkened a touch so it stays legible)
    TRUE_TEXT_COLOR: str = "#1F7A8C"
    NOISE_TEXT_COLOR: str = "#B8791F"

    # ---- output ------------------------------------------------------------
    # Everything is written to RESULTS_ROOT / ANALYSIS_DIR, a self-describing
    # subfolder of your results/ folder. The noise model is appended to the
    # filenames so feature_driven and normalized runs don't overwrite each other.
    RESULTS_ROOT: Path = Path("./results")
    ANALYSIS_DIR: str = "label_noise_composition"
    FIG_STEM: str = "fig_label_composition_inflip"
    DPI: int = 200
    SAVE_PNG: bool = True
    SAVE_PDF: bool = True


CFG = Config()


# ============================================================================
# data loading -> tidy DataFrame  [tau, class, n_true, n_noisy, total, purity, prior_n]
# ============================================================================
def _read_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        T = np.load(path)
    else:
        T = np.loadtxt(path, delimiter=",")
    T = np.asarray(T, float)
    k = len(CFG.CLASSES)
    if T.shape != (k, k):
        raise ValueError(f"{path}: expected {k}x{k} matrix, got {T.shape}.")
    rs = T.sum(axis=1, keepdims=True)
    if not np.allclose(rs, 1.0, atol=1e-3):
        print(f"[warn] {path.name}: rows not normalised (max dev "
              f"{float(np.abs(rs - 1).max()):.3g}); renormalising.")
        T = T / np.where(rs == 0, 1, rs)
    return T


def _read_counts() -> np.ndarray:
    p = Path(CFG.CLASS_COUNTS_PATH)
    if p.suffix == ".json":
        d = json.loads(p.read_text())
    else:  # csv with columns class,count
        df = pd.read_csv(p)
        d = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(float)))
    return np.array([float(d[c]) for c in CFG.CLASSES], float)


def _inflip_from_matrix(T: np.ndarray, n: np.ndarray):
    """Return (n_true_per_obs, n_noisy_per_obs) for observed classes.
    retained_j = n_j T[j,j]; intruders_j = sum_{i!=j} n_i T[i,j]."""
    contrib = n[:, None] * T            # contrib[i, j] = expected count true=i, obs=j
    total = contrib.sum(axis=0)         # observed-class totals
    retained = np.diag(contrib)         # true == observed
    intruders = total - retained
    return retained, intruders


def _load_matrix_rows():
    n = _read_counts()
    rows = []
    for tau in CFG.TAUS:
        T = _read_matrix(Path(CFG.T_PATH_TEMPLATE.format(tau=tau)))
        retained, intruders = _inflip_from_matrix(T, n)
        for j, c in enumerate(CFG.CLASSES):
            rows.append(dict(tau=tau, **{"class": c},
                             n_true=float(retained[j]), n_noisy=float(intruders[j])))
    return rows


def _load_labels_rows():
    rows = []
    for tau in CFG.TAUS:
        yt, yn = _read_labels(Path(CFG.LABELS_PATH_TEMPLATE.format(tau=tau)))
        yt, yn = np.asarray(yt).astype(str), np.asarray(yn).astype(str)
        for c in CFG.CLASSES:
            obs = (yn == c)
            rows.append(dict(tau=tau, **{"class": c},
                             n_true=int(np.sum(obs & (yt == c))),
                             n_noisy=int(np.sum(obs & (yt != c)))))
    return rows


# ---- SOURCE="folds": read realised noisy labels from the cv tree -----------
def _fold_train_path(tau: float, fold: int) -> Path:
    tt = int(round(tau * 100))
    return (CFG.CV_ROOT / CFG.SAMPLING / CFG.NOISE_MODEL
            / CFG.TAU_DIR_FMT.format(tt=tt) / CFG.FOLD_DIR_FMT.format(ff=fold)
            / CFG.TRAIN_FILE)


def _detect_columns(df: pd.DataFrame):
    low = {c.lower(): c for c in df.columns}

    def pick(cands, override):
        if override:
            return override
        for c in cands:
            if c in low:
                return low[c]
        return None

    noisy = pick(["noisy_label", "label_noisy", "y_noisy", "noisy_dx",
                  "observed_label", "obs_label", "noisy_target"], CFG.NOISY_LABEL_COL)
    true_ = pick(["true_label", "label_true", "y_true", "clean_label", "true_dx",
                  "orig_label", "original_label", "dx", "target"], CFG.TRUE_LABEL_COL)
    flag = pick(["is_noisy", "is_flipped", "flipped", "label_changed",
                 "noisy_flag", "is_noise"], CFG.IS_NOISY_COL)
    if noisy is None and "label" in low:
        noisy = low["label"]
    return noisy, true_, flag


def _as_bool(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.to_numpy()
    return (s.astype(str).str.strip().str.lower()
            .isin(["true", "1", "yes", "y", "t"]).to_numpy())


def _resolve_classes(observed_sorted: list[str]) -> list[str]:
    if set(observed_sorted) == set(CFG.CLASSES):
        return list(CFG.CLASSES)
    print(f"[warn] labels in data {observed_sorted} differ from CONFIG.CLASSES "
          f"{list(CFG.CLASSES)}; using the data's labels.")
    return observed_sorted


def _compose_one(df, noisy, true_, flag, classes):
    obs = df[noisy].astype(str)
    out = {}
    if flag is not None:
        f = _as_bool(df[flag])
        for c in classes:
            m = (obs == c).to_numpy()
            out[c] = (int((m & ~f).sum()), int((m & f).sum()))
    else:
        tr = df[true_].astype(str)
        for c in classes:
            m = (obs == c)
            out[c] = (int((m & (tr == c)).sum()), int((m & (tr != c)).sum()))
    return out


def _load_folds():
    folds = list(range(CFG.N_FOLDS)) if CFG.AGGREGATE_FOLDS else [CFG.FOLD]
    classes, detected, rows = None, False, []
    for tau in CFG.TAUS:
        per_fold = []
        for fold in folds:
            path = _fold_train_path(tau, fold)
            if not path.exists():
                print(f"[warn] missing {path}")
                continue
            df = pd.read_csv(path)
            noisy, true_, flag = _detect_columns(df)
            if not detected:
                print(f"[folds] reading e.g. {path}")
                print(f"[folds] columns present: {list(df.columns)}")
                print(f"[folds] -> observed/noisy label: {noisy!r}; "
                      f"true label: {true_!r}; is-noisy flag: {flag!r}")
                if noisy is None:
                    raise ValueError("Could not find the observed/noisy-label "
                                     "column; set CONFIG.NOISY_LABEL_COL.")
                if true_ is None and flag is None:
                    raise ValueError("train_noisy.csv needs a true-label column "
                                     "OR an is-noisy flag; found neither. Set "
                                     "CONFIG.TRUE_LABEL_COL or CONFIG.IS_NOISY_COL.")
                detected = True
            if classes is None:
                vals = set(df[noisy].astype(str).unique())
                if true_ is not None:
                    vals |= set(df[true_].astype(str).unique())
                classes = _resolve_classes(sorted(vals))
            per_fold.append(_compose_one(df, noisy, true_, flag, classes))
        if not per_fold:
            raise FileNotFoundError(
                f"No '{CFG.TRAIN_FILE}' found for tau={tau} under "
                f"{CFG.CV_ROOT / CFG.SAMPLING / CFG.NOISE_MODEL}. "
                "Check CV_ROOT / SAMPLING / NOISE_MODEL / the tau_NN naming.")
        for c in classes:
            rows.append(dict(tau=tau, **{"class": c},
                             n_true=float(np.mean([p[c][0] for p in per_fold])),
                             n_noisy=float(np.mean([p[c][1] for p in per_fold]))))
    where = "averaged over all folds" if CFG.AGGREGATE_FOLDS else f"fold {CFG.FOLD}"
    print(f"[folds] composition computed from {CFG.NOISE_MODEL}/{CFG.SAMPLING} "
          f"train_noisy labels ({where}).")
    return rows


def load_composition() -> pd.DataFrame:
    if CFG.SOURCE == "csv":
        df = pd.read_csv(CFG.TIDY_CSV_PATH)
    else:
        dispatch = {"folds": _load_folds, "matrix": _load_matrix_rows,
                    "labels": _load_labels_rows}
        if CFG.SOURCE not in dispatch:
            raise ValueError(f"Unknown SOURCE={CFG.SOURCE!r}")
        df = pd.DataFrame(dispatch[CFG.SOURCE]())

    if "total" not in df:
        df["total"] = df["n_true"] + df["n_noisy"]
    if "purity" not in df:
        df["purity"] = np.where(df["total"] > 0, df["n_true"] / df["total"], np.nan)
    tau0 = min(CFG.TAUS)
    prior = df[np.isclose(df["tau"], tau0)].set_index("class")["total"]
    df["prior_n"] = df["class"].map(prior)
    return df


def _read_labels(path: Path):
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        return np.asarray(z[CFG.LABELS_TRUE_COL]), np.asarray(z[CFG.LABELS_NOISY_COL])
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        return np.asarray(arr[:, 0]), np.asarray(arr[:, 1])
    df = pd.read_csv(path)
    return df[CFG.LABELS_TRUE_COL].to_numpy(), df[CFG.LABELS_NOISY_COL].to_numpy()


# ============================================================================
# plotting
# ============================================================================
def _class_order(df: pd.DataFrame) -> list[str]:
    present = list(df["class"].unique())
    if CFG.ORDER_BY_PREVALENCE:
        prior = df.drop_duplicates("class").set_index("class")["prior_n"]
        return list(prior.sort_values(ascending=False).index)
    return ([c for c in CFG.CLASSES if c in present]
            + [c for c in present if c not in CFG.CLASSES])


def _style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Palatino", "Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
        "mathtext.fontset":   "cm",      # serif math, e.g. $\tau$, matches body text
        "axes.unicode_minus": False,
        "figure.dpi":         150,
        "savefig.dpi": CFG.DPI,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,         # grid behind bars
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def plot(df: pd.DataFrame):
    _style()
    order = _class_order(df)
    x = np.arange(len(order))
    fig, axes = plt.subplots(CFG.NROWS, CFG.NCOLS,
                             figsize=(4.0 * CFG.NCOLS, 3.2 * CFG.NROWS),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()

    ymax_count = 0.0
    if not CFG.AS_PROPORTION:
        ymax_count = df["total"].max()

    for ax, tau in zip(axes, CFG.TAUS):
        sub = (df[np.isclose(df["tau"], tau)]
               .set_index("class").reindex(order).reset_index())
        if CFG.AS_PROPORTION:
            true_h = sub["purity"].fillna(0).to_numpy()
            noisy_h = (1 - sub["purity"].fillna(0)).to_numpy()
        else:
            true_h = sub["n_true"].fillna(0).to_numpy()
            noisy_h = sub["n_noisy"].fillna(0).to_numpy()

        off = 0.20                                  # sub-bar offset; group spans ~0.8
        bw = 0.38                                   # sub-bar width
        totals = true_h + noisy_h
        ax.bar(x - off, true_h, width=bw, color=CFG.TRUE_COLOR, alpha=CFG.TRUE_ALPHA,
               edgecolor=CFG.BAR_EDGE, linewidth=CFG.BAR_EDGE_LW, zorder=3)
        ax.bar(x + off, noisy_h, width=bw, color=CFG.NOISE_COLOR, alpha=CFG.NOISE_ALPHA,
               hatch=CFG.NOISE_HATCH, edgecolor=CFG.BAR_EDGE, linewidth=CFG.BAR_EDGE_LW,
               zorder=3)

        ax.set_title(rf"$\tau = {tau:g}$")
        ax.set_xticks(x)
        if CFG.AS_PROPORTION:
            xlabels = list(order)
        else:
            # class name with the class total on a second line
            xlabels = [f"{c}\n({int(round(t))})" for c, t in zip(order, totals)]
        ax.set_xticklabels(xlabels, rotation=CFG.XTICK_ROTATION,
                           ha="right" if CFG.XTICK_ROTATION else "center")
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
        if CFG.AS_PROPORTION:
            ax.set_ylim(0, 1)
        elif ymax_count > 0:
            ax.set_ylim(0, ymax_count * 1.10)

        if CFG.ANNOTATE_N:
            if CFG.AS_PROPORTION:
                for xi in x:
                    ax.text(xi - off, true_h[xi] + 0.015, f"{true_h[xi]:.2f}", ha="center",
                            va="bottom", fontsize=6.5, color=CFG.TRUE_TEXT_COLOR)
                    ax.text(xi + off, noisy_h[xi] + 0.015, f"{noisy_h[xi]:.2f}", ha="center",
                            va="bottom", fontsize=6.5, color=CFG.NOISE_TEXT_COLOR)
            else:
                pad = 0.006 * ymax_count
                for xi in x:
                    t, n = true_h[xi], noisy_h[xi]
                    ax.text(xi - off, t + pad, f"{int(round(t))}", ha="center",
                            va="bottom", fontsize=6.5, color=CFG.TRUE_TEXT_COLOR)
                    if n > 0:
                        ax.text(xi + off, n + pad, f"{int(round(n))}", ha="center",
                                va="bottom", fontsize=6.5, color=CFG.NOISE_TEXT_COLOR)

    # hide any unused panels
    for ax in axes[len(CFG.TAUS):]:
        ax.set_visible(False)

    # shared y label and legend
    ylab = "Share of labels" if CFG.AS_PROPORTION else "Number of samples"
    for r in range(CFG.NROWS):
        axes[r * CFG.NCOLS].set_ylabel(ylab)

    handles = [
        Patch(facecolor=CFG.TRUE_COLOR, alpha=CFG.TRUE_ALPHA,
              edgecolor=CFG.BAR_EDGE, label=CFG.TRUE_LABEL),
        Patch(facecolor=CFG.NOISE_COLOR, alpha=CFG.NOISE_ALPHA,
              hatch=CFG.NOISE_HATCH, edgecolor=CFG.BAR_EDGE, label=CFG.NOISE_LABEL),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Composition of observed (noisy) labels per class - Training set only", y=1.0,
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_dir = CFG.RESULTS_ROOT / CFG.ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{CFG.FIG_STEM}_{CFG.NOISE_MODEL}"
    if CFG.SAVE_PDF:
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    if CFG.SAVE_PNG:
        fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out_dir / stem}.(pdf|png)")


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 64)
    print("In-flip purity (true share of each observed class)")
    print("=" * 64)
    piv = (df.pivot_table(index="class", columns="tau", values="purity")
           .reindex(_class_order(df)))
    with pd.option_context("display.float_format", lambda v: f"{v:.2f}"):
        print(piv.to_string())
    print("\nRare-class contamination is read off the high-tau columns: a value "
          "near 0.5 means about half of that class's training labels are wrong.\n")


def main():
    print(f"Loading composition (SOURCE={CFG.SOURCE}) ...")
    df = load_composition()
    plot(df)
    out_dir = CFG.RESULTS_ROOT / CFG.ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"label_composition_inflip_{CFG.NOISE_MODEL}.csv"
    df.sort_values(["tau", "class"]).to_csv(out, index=False)
    print(f"[csv] wrote {out}")
    print_summary(df)
    print("Done.")


if __name__ == "__main__":
    main()