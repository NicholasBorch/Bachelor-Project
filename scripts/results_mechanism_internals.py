#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
results_mechanism_internals.py
==============================================================================
Selection- and loss-component diagnostics from the per-epoch training logs.

This script turns three mechanism claims that the thesis currently states as
*interpretation* into *evidenced* findings, using fields that are logged once
per epoch in every

    results/main_experiment/{protocol_dir}/training/{method_dir}/
        tau_{tt:02d}/fold_{ff:02d}/training_log.jsonl

It reads:
  - AsyCo (asyco_divmix):  n_clean / n_noisy / n_discard      -> co-divide label
                                                                 partition sizes
  - ELR  (elr):            loss_components.{ce, reg}           -> CE vs the ELR
                                                                 regularisation term
  - SCE  (sce):            loss_components.{ce, rce}           -> CE vs the reverse
                                                                 cross-entropy term

It produces (Adam/pretrained = AP is primary; the other three protocols are
shown only where they deviate):

  FIGURES  (results/mechanism_internals/figures/)
    p5_asyco_selection_partition.png
        (a) labeled/unlabeled/discarded fraction over training (AP, focus tau)
        (b) post-warmup labeled fraction vs tau, all four protocols
    p5_loss_component_shares.png
        (a) ELR: CE vs reg contribution over training (AP, focus tau)
        (b) SCE: CE vs RCE contribution over training (AP, focus tau)

  TABLES   (results/mechanism_internals/tables/)
    p5_asyco_selection_summary.tex     body  : partition at focus tau, 4 protocols
    p5_loss_component_summary.tex      body  : CE/aux/share at focus tau, 4 protocols
    app_asyco_selection_full.tex       appx  : partition, every protocol x tau
    app_loss_component_full.tex        appx  : CE/aux/share, every protocol x tau

  A "PROSE NUMBERS" block is printed to stdout so the few inline numbers in the
  Results/Discussion text can be confirmed against the actual data.

------------------------------------------------------------------------------
SCHEMA ROBUSTNESS
------------------------------------------------------------------------------
The uploaded sample log was the *baseline* (loss_components = {"ce": ...} only),
so the exact key placement for AsyCo's partition counts and ELR/SCE's auxiliary
terms could not be inspected directly. The extractors therefore search for the
keys recursively at any depth and accept a few aliases (see KEY_ALIASES). If a
method's expected fields are not found, the script prints the union of keys it
*did* see for that method so the aliases can be adjusted in one place.

    python results_mechanism_internals.py --probe
        -> only prints, per method, the keys found in the logs (no outputs).

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python results_mechanism_internals.py
    python results_mechanism_internals.py --root ./results/main_experiment
    python results_mechanism_internals.py --focus-tau 0.20 --last-k 10
    python results_mechanism_internals.py --probe

Only matplotlib, numpy and pandas are required.
==============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# configuration  (mirrors results_part4_protocol_sensitivity.py conventions)
# =============================================================================
@dataclass
class Config:
    EXPERIMENT_ROOT: Path = Path("./results/main_experiment")
    TRAINING_LOG_FILENAME: str = "training_log.jsonl"
    TAU_DIR_FMT: str = "tau_{tt:02d}"
    FOLD_DIR_FMT: str = "fold_{ff:02d}"

    PROTOCOL_DIRS: dict = field(default_factory=lambda: {
        "S":  "scratch_sgd",
        "SP": "pretrained_sgd",
        "A":  "scratch_adam",
        "AP": "pretrained_adam",
    })
    PROTOCOL_LABELS: dict = field(default_factory=lambda: {
        "S":  "SGD / scratch",
        "SP": "SGD / pretrained",
        "A":  "Adam / scratch",
        "AP": "Adam / pretrained",
    })
    PROTOCOLS_TO_RUN: tuple = ("S", "SP", "A", "AP")
    ANCHOR_PROTOCOL: str = "AP"

    METHOD_DIRS: dict = field(default_factory=lambda: {
        "baseline": "baseline",
        "SCE":      "sce",
        "ELR":      "elr",
        "AsyCo":    "asyco_divmix",
    })
    METHOD_LABELS: dict = field(default_factory=lambda: {
        "baseline": "Baseline",
        "SCE":      "SCE",
        "ELR":      "ELR",
        "AsyCo":    "AsyCo",
    })

    TAUS: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    N_FOLDS: int = 10
    FOCUS_TAU: float = 0.20

    # Steady-state window: mean over the last LAST_K logged epochs of each fold.
    LAST_K: int = 10
    # None -> auto-detect AsyCo warm-up as the leading epochs with no active
    # partition (n_noisy == 0 and n_discard == 0). Override with an int if known.
    WARMUP_EPOCHS: Optional[int] = None

    # statistics
    N_BOOT: int = 10_000
    CI: float = 0.95
    SEED: int = 10

    # outputs
    OUT_ROOT: Path = Path("./results/mechanism_internals")
    FIG_DPI: int = 300
    SAVE_PNG: bool = True

    # If set, final PNGs are also copied here (e.g. a thesis Figures/results/p5).
    THESIS_FIG_DIR: Optional[Path] = None

    # method palette (identical to Part 3/4)
    PALETTE: dict = field(default_factory=lambda: {
        "baseline": "#9ec9e2",
        "SCE":      "#2a9d8f",
        "ELR":      "#e07a3f",
        "AsyCo":    "#7b5cb8",
    })
    PROTOCOL_PALETTE: dict = field(default_factory=lambda: {
        "S":  "#4c78a8",
        "SP": "#72b7b2",
        "A":  "#f58518",
        "AP": "#e45756",
    })
    PROTOCOL_LINESTYLES: dict = field(default_factory=lambda: {
        "S": "-", "SP": "--", "A": "-.", "AP": ":",
    })
    # partition encoding for the AsyCo figure
    PART_COLORS: dict = field(default_factory=lambda: {
        "clean":   "#2a9d8f",   # labeled set
        "noisy":   "#e07a3f",   # unlabeled (pseudo-labeled) set
        "discard": "#9e9e9e",   # discarded
    })
    REFERENCE_GREY: str = "#777777"


CFG = Config()

LATEX_PREAMBLE = (
    r"% Preamble: \usepackage{booktabs,makecell,multirow,graphicx,longtable,amsmath}"
)

# key aliases, tried in order. Exact thesis names are first.
KEY_ALIASES = {
    "n_clean":   ["n_clean", "num_clean", "n_labeled", "n_label", "clean"],
    "n_noisy":   ["n_noisy", "num_noisy", "n_unlabeled", "n_unlabel", "noisy"],
    "n_discard": ["n_discard", "num_discard", "n_discarded", "n_drop", "n_dropped", "discard"],
    "ce":  ["ce", "cross_entropy", "ce_loss", "loss_ce"],
    "reg": ["reg", "elr_reg", "reg_loss", "regularization", "regularisation", "elr"],
    "rce": ["rce", "reverse_ce", "rce_loss", "reverse_cross_entropy", "loss_rce"],
}


# =============================================================================
# paths and tiny helpers
# =============================================================================
def _protocol_root(protocol: str) -> Path:
    return CFG.EXPERIMENT_ROOT / CFG.PROTOCOL_DIRS[protocol]


def _method_root(protocol: str, method: str) -> Path:
    return _protocol_root(protocol) / "training" / CFG.METHOD_DIRS[method]


def _tau_dir(tau: float) -> str:
    return CFG.TAU_DIR_FMT.format(tt=int(round(tau * 100)))


def _fold_dir(fold: int) -> str:
    return CFG.FOLD_DIR_FMT.format(ff=int(fold))


def _log_path(protocol: str, method: str, tau: float, fold: int) -> Path:
    return _method_root(protocol, method) / _tau_dir(tau) / _fold_dir(fold) / CFG.TRAINING_LOG_FILENAME


def _seed(*parts) -> int:
    s = "|".join(map(str, (CFG.SEED, *parts)))
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def _boot_ci(values, seed: int) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI over a 1-D sample (matches Part 4)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    if v.size == 1:
        x = float(v[0])
        return x, x, x
    rng = np.random.default_rng(seed)
    boot = rng.choice(v, size=(CFG.N_BOOT, v.size), replace=True).mean(axis=1)
    alpha = 1.0 - CFG.CI
    return (float(v.mean()),
            float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def _find_num(obj, names: list[str]):
    """Breadth-first search for the first key in `names` that maps to a finite
    number, anywhere in a nested dict/list. Returns float or np.nan."""
    q = deque([obj])
    while q:
        cur = q.popleft()
        if isinstance(cur, dict):
            for name in names:
                if name in cur:
                    val = cur[name]
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        return float(val)
            q.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            q.extend(cur)
    return np.nan


def _all_keys(obj, prefix: str = "") -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(prefix + str(k))
            out |= _all_keys(v, prefix + str(k) + ".")
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= _all_keys(v, prefix)
    return out


def _fmt(x, nd: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{float(x):.{nd}f}"


def _fmt_pct(x, nd: int = 1) -> str:
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * float(x):.{nd}f}"


# =============================================================================
# raw log reading
# =============================================================================
def _read_log(protocol: str, method: str, tau: float, fold: int) -> list[dict]:
    fp = _log_path(protocol, method, tau, fold)
    if not fp.exists():
        return []
    rows = []
    with open(fp, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def probe() -> None:
    """Print, per method, the union of record keys found across a few folds."""
    print("\n=== schema probe (union of keys seen per method) ===")
    for method in CFG.METHOD_DIRS:
        seen: set[str] = set()
        n_files = 0
        for protocol in CFG.PROTOCOLS_TO_RUN:
            for fold in range(CFG.N_FOLDS):
                recs = _read_log(protocol, method, CFG.FOCUS_TAU, fold)
                if recs:
                    n_files += 1
                    for r in recs[:1] + recs[-1:]:
                        seen |= _all_keys(r)
                if n_files >= 4:
                    break
            if n_files >= 4:
                break
        rel = sorted(k for k in seen if any(
            t in k.lower() for t in ("clean", "noisy", "discard", "ce", "reg", "rce", "loss")))
        print(f"\n[{method:>9}]  files inspected: {n_files}")
        if not seen:
            print("    (no logs found under this method)")
            continue
        print("    relevant keys:", rel if rel else "(none matched clean/noisy/discard/ce/reg/rce)")
    print("\nAdjust KEY_ALIASES at the top of the script if a needed key is not matched.\n")


# =============================================================================
# loaders -> long DataFrames
# =============================================================================
def load_selection_long() -> pd.DataFrame:
    """AsyCo co-divide partition sizes, one row per (protocol, tau, fold, epoch)."""
    rows = []
    method = "AsyCo"
    for protocol in CFG.PROTOCOLS_TO_RUN:
        for tau in CFG.TAUS:
            for fold in range(CFG.N_FOLDS):
                for rec in _read_log(protocol, method, tau, fold):
                    if rec.get("epoch") is None:
                        continue
                    nc = _find_num(rec, KEY_ALIASES["n_clean"])
                    nn = _find_num(rec, KEY_ALIASES["n_noisy"])
                    nd = _find_num(rec, KEY_ALIASES["n_discard"])
                    if not (np.isfinite(nc) or np.isfinite(nn) or np.isfinite(nd)):
                        continue
                    nc = 0.0 if not np.isfinite(nc) else nc
                    nn = 0.0 if not np.isfinite(nn) else nn
                    nd = 0.0 if not np.isfinite(nd) else nd
                    total = nc + nn + nd
                    if total <= 0:
                        continue
                    rows.append(dict(
                        protocol=protocol, tau=float(tau), fold=int(fold),
                        epoch=int(rec["epoch"]),
                        n_clean=nc, n_noisy=nn, n_discard=nd, total=total,
                        frac_clean=nc / total, frac_noisy=nn / total,
                        frac_discard=nd / total))
    df = pd.DataFrame(rows)
    if df.empty:
        print("[selection] no AsyCo partition counts found "
              "(checked keys: n_clean/n_noisy/n_discard and aliases).")
    else:
        print(f"[selection] loaded {len(df)} AsyCo epoch-rows "
              f"across {df[['protocol','tau','fold']].drop_duplicates().shape[0]} runs.")
    return df.sort_values(["protocol", "tau", "fold", "epoch"]).reset_index(drop=True) if not df.empty else df


def load_loss_long() -> pd.DataFrame:
    """CE and the method-specific auxiliary term, one row per epoch.
    ELR aux = reg, SCE aux = rce. Baseline/AsyCo skipped (no auxiliary term)."""
    aux_for = {"ELR": "reg", "SCE": "rce"}
    rows = []
    for method, aux_key in aux_for.items():
        for protocol in CFG.PROTOCOLS_TO_RUN:
            for tau in CFG.TAUS:
                for fold in range(CFG.N_FOLDS):
                    for rec in _read_log(protocol, method, tau, fold):
                        if rec.get("epoch") is None:
                            continue
                        ce = _find_num(rec, KEY_ALIASES["ce"])
                        aux = _find_num(rec, KEY_ALIASES[aux_key])
                        if not (np.isfinite(ce) and np.isfinite(aux)):
                            continue
                        train_loss = rec.get("train_loss", np.nan)
                        train_loss = float(train_loss) if isinstance(train_loss, (int, float)) else np.nan
                        denom = ce + aux  # signed sum, used for the train_loss check
                        mag = abs(ce) + abs(aux)
                        # magnitude share: well-defined when the two terms have
                        # opposite signs (ELR's reg term is negative by construction,
                        # so a signed aux/(ce+aux) ratio crosses zero and is unusable).
                        share = abs(aux) / mag if mag > 1e-12 else np.nan
                        rows.append(dict(
                            protocol=protocol, method=method, tau=float(tau),
                            fold=int(fold), epoch=int(rec["epoch"]),
                            ce=ce, aux=aux, aux_name=aux_key, share=share,
                            train_loss=train_loss, comp_sum=denom))
    df = pd.DataFrame(rows)
    if df.empty:
        print("[loss] no ELR/SCE loss components found "
              "(checked loss_components.ce with reg/rce and aliases).")
        return df
    # sanity: do the two logged components sum to train_loss?
    chk = df.dropna(subset=["train_loss"]).copy()
    if not chk.empty:
        resid = (chk["comp_sum"] - chk["train_loss"]).abs()
        frac_match = float((resid <= 1e-3 * (1 + chk["train_loss"].abs())).mean())
        print(f"[loss] loaded {len(df)} ELR/SCE epoch-rows. "
              f"ce+aux == train_loss in {100*frac_match:.0f}% of rows "
              f"(if low, the logged terms may be raw/unweighted; shares are still "
              f"computed relative to ce+aux).")
    return df.sort_values(["method", "protocol", "tau", "fold", "epoch"]).reset_index(drop=True)


# =============================================================================
# aggregation
# =============================================================================
def _detect_warmup(g: pd.DataFrame) -> int:
    """First epoch of a single (protocol,tau,fold) run with an active partition."""
    if CFG.WARMUP_EPOCHS is not None:
        return int(CFG.WARMUP_EPOCHS)
    active = g[(g["n_noisy"] > 0) | (g["n_discard"] > 0)]
    return int(active["epoch"].min()) if not active.empty else int(g["epoch"].max())


def _last_k_mean(g: pd.DataFrame, col: str) -> float:
    gg = g.sort_values("epoch")
    return float(gg[col].tail(CFG.LAST_K).mean())


def selection_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (protocol, tau): warm-up length and steady-state partition fractions."""
    if df.empty:
        return df
    rows = []
    for (protocol, tau), g in df.groupby(["protocol", "tau"]):
        warmups, fc, fn, fd = [], [], [], []
        for _, gf in g.groupby("fold"):
            warmups.append(_detect_warmup(gf))
            fc.append(_last_k_mean(gf, "frac_clean"))
            fn.append(_last_k_mean(gf, "frac_noisy"))
            fd.append(_last_k_mean(gf, "frac_discard"))
        rec = dict(protocol=protocol, tau=float(tau),
                   n_folds=int(g["fold"].nunique()),
                   warmup=float(np.median(warmups)))
        for name, vals in (("clean", fc), ("noisy", fn), ("discard", fd)):
            m, lo, hi = _boot_ci(vals, _seed("sel", protocol, tau, name))
            rec[name] = m
            rec[f"{name}_lo"] = lo
            rec[f"{name}_hi"] = hi
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["protocol", "tau"]).reset_index(drop=True)


def loss_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (method, protocol, tau): steady-state CE, aux and aux share."""
    if df.empty:
        return df
    rows = []
    for (method, protocol, tau), g in df.groupby(["method", "protocol", "tau"]):
        ce_v, aux_v, sh_v = [], [], []
        for _, gf in g.groupby("fold"):
            ce_v.append(_last_k_mean(gf, "ce"))
            aux_v.append(_last_k_mean(gf, "aux"))
            sh_v.append(_last_k_mean(gf, "share"))
        rec = dict(method=method, protocol=protocol, tau=float(tau),
                   aux_name=g["aux_name"].iloc[0], n_folds=int(g["fold"].nunique()))
        for name, vals in (("ce", ce_v), ("aux", aux_v), ("share", sh_v)):
            m, lo, hi = _boot_ci(vals, _seed("loss", method, protocol, tau, name))
            rec[name] = m
            rec[f"{name}_lo"] = lo
            rec[f"{name}_hi"] = hi
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["method", "protocol", "tau"]).reset_index(drop=True)


def _epoch_curve(df: pd.DataFrame, protocol: str, tau: float, value_cols: list[str],
                 method: Optional[str] = None) -> pd.DataFrame:
    """Per-epoch mean and bootstrap CI across folds for one (protocol, tau)."""
    sub = df[(df["protocol"] == protocol) & (np.isclose(df["tau"], tau))]
    if method is not None:
        sub = sub[sub["method"] == method]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for epoch, g in sub.groupby("epoch"):
        rec = dict(epoch=int(epoch), n=int(g["fold"].nunique()))
        for col in value_cols:
            m, lo, hi = _boot_ci(g[col].values, _seed("curve", protocol, tau, method, epoch, col))
            rec[col] = m
            rec[f"{col}_lo"] = lo
            rec[f"{col}_hi"] = hi
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


# =============================================================================
# plotting
# =============================================================================
def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino", "Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "figure.dpi": 150, "savefig.dpi": CFG.FIG_DPI, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.24, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#cccccc", "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def _fig_dir() -> Path:
    d = CFG.OUT_ROOT / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tab_dir() -> Path:
    d = CFG.OUT_ROOT / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _savefig(fig, stem: str) -> None:
    out = _fig_dir()
    if CFG.SAVE_PNG:
        fp = out / f"{stem}.png"
        fig.savefig(fp)
        print(f"[fig] wrote {fp}")
        if CFG.THESIS_FIG_DIR is not None:
            CFG.THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(fp, CFG.THESIS_FIG_DIR / f"{stem}.png")
            print(f"[fig] copied -> {CFG.THESIS_FIG_DIR / stem}.png")
    plt.close(fig)


def _write_tex(stem: str, body: str) -> None:
    fp = _tab_dir() / f"{stem}.tex"
    fp.write_text(LATEX_PREAMBLE + "\n" + body.rstrip() + "\n")
    print(f"[tab] wrote {fp}")


def fig_asyco_partition(df_sel: pd.DataFrame, summ: pd.DataFrame) -> None:
    if df_sel.empty:
        print("[fig] AsyCo partition figure skipped (no data).")
        return
    ap = CFG.ANCHOR_PROTOCOL
    tau = CFG.FOCUS_TAU
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # ---- (a) trajectory at AP, focus tau --------------------------------
    cur = _epoch_curve(df_sel, ap, tau, ["frac_clean", "frac_noisy", "frac_discard"])
    if not cur.empty:
        order = [("frac_clean", "Labeled (clean)", "clean"),
                 ("frac_noisy", "Unlabeled (noisy)", "noisy"),
                 ("frac_discard", "Discarded", "discard")]
        for col, lbl, ckey in order:
            c = CFG.PART_COLORS[ckey]
            axL.plot(cur["epoch"], cur[col], color=c, lw=1.8, label=lbl)
            axL.fill_between(cur["epoch"], cur[f"{col}_lo"], cur[f"{col}_hi"],
                             color=c, alpha=0.15, linewidth=0)
        # reference: clean fraction of a perfect filter = 1 - tau
        axL.axhline(1.0 - tau, color=CFG.REFERENCE_GREY, lw=1.1, ls=(0, (4, 3)))
        axL.text(cur["epoch"].max(), 1.0 - tau, r" $1-\tau$ (clean fraction)",
                 color=CFG.REFERENCE_GREY, va="center", ha="left", fontsize=8.5)
        # warm-up shading: AsyCo logs the partition only after warm-up, so the
        # curves start at the first post-warm-up epoch. Begin the axis at 0 and
        # shade [0, warm-up] so the warm-up phase is visible and explains the
        # offset start of the curves.
        wrow = summ[(summ["protocol"] == ap) & (np.isclose(summ["tau"], tau))]
        w = float(wrow["warmup"].iloc[0]) if not wrow.empty else 0.0
        x_lo = 0
        if w > x_lo:
            axL.axvspan(x_lo, w, color="#000000", alpha=0.06, linewidth=0)
            axL.text(w / 2, 0.04, "warm-up", fontsize=8.5, color="0.45",
                     ha="center", va="bottom")
        axL.set_xlim(x_lo, cur["epoch"].max())
    axL.set_ylim(-0.02, 1.04)
    axL.set_xlabel("epoch")
    axL.set_ylabel("fraction of training samples")
    axL.set_title(rf"AsyCo label partition over training "
                  rf"({CFG.PROTOCOL_LABELS[ap]}, $\tau={tau:.2f}$)")
    axL.legend(loc="center right", frameon=False)

    # ---- (b) post-warmup labeled fraction vs tau, all protocols ----------
    taus_sorted = sorted(summ["tau"].unique())
    for protocol in CFG.PROTOCOLS_TO_RUN:
        s = summ[summ["protocol"] == protocol].sort_values("tau")
        if s.empty:
            continue
        c = CFG.PROTOCOL_PALETTE[protocol]
        ls = CFG.PROTOCOL_LINESTYLES[protocol]
        axR.plot(s["tau"], s["clean"], color=c, ls=ls, lw=1.8, marker="o", ms=4,
                 label=CFG.PROTOCOL_LABELS[protocol])
        axR.fill_between(s["tau"], s["clean_lo"], s["clean_hi"], color=c, alpha=0.12, linewidth=0)
    # perfect-filter reference 1 - tau
    tt = np.array(taus_sorted, dtype=float)
    axR.plot(tt, 1.0 - tt, color=CFG.REFERENCE_GREY, lw=1.1, ls=(0, (4, 3)),
             label=r"$1-\tau$ (perfect filter)")
    axR.set_xlabel(r"noise rate $\tau$")
    axR.set_ylabel("labeled fraction after warm-up")
    axR.set_title("Labeled fraction vs noise rate")
    axR.set_ylim(0.0, 1.04)
    axR.legend(loc="lower left", frameon=False)

    fig.suptitle("AsyCo co-divide label partition",
                 y=1.00, fontsize=13)
    fig.tight_layout()
    _savefig(fig, "p5_asyco_selection_partition")


def fig_loss_components(df_loss: pd.DataFrame, summ: pd.DataFrame) -> None:
    if df_loss.empty:
        print("[fig] loss-component figure skipped (no data).")
        return
    ap = CFG.ANCHOR_PROTOCOL
    tau = CFG.FOCUS_TAU
    panels = [("ELR", "reg", "regularisation term"),
              ("SCE", "rce", "reverse CE term")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)

    # common y-limit across both panels
    lo, hi = np.inf, -np.inf
    curves = {}
    for method, _, _ in panels:
        cur = _epoch_curve(df_loss, ap, tau, ["ce", "aux"], method=method)
        curves[method] = cur
        if not cur.empty:
            vals = np.concatenate([cur["ce_lo"].values, cur["ce_hi"].values,
                                   cur["aux_lo"].values, cur["aux_hi"].values])
            vals = vals[np.isfinite(vals)]
            if vals.size:
                lo = min(lo, float(vals.min()))
                hi = max(hi, float(vals.max()))
    if not np.isfinite(lo):
        lo, hi = 0.0, 1.0
    span = max(hi - lo, 0.05)

    for ax, (method, aux_key, aux_desc) in zip(axes, panels):
        cur = curves[method]
        if cur.empty:
            ax.set_axis_off()
            ax.set_title(f"{method}: no data")
            continue
        cmeth = CFG.PALETTE[method]
        # CE in neutral grey, auxiliary term in the method colour
        ax.plot(cur["epoch"], cur["ce"], color="0.45", lw=1.8, label="CE term")
        ax.fill_between(cur["epoch"], cur["ce_lo"], cur["ce_hi"], color="0.45",
                        alpha=0.13, linewidth=0)
        aux_label = "reg term" if aux_key == "reg" else "RCE term"
        ax.plot(cur["epoch"], cur["aux"], color=cmeth, lw=1.8, label=aux_label)
        ax.fill_between(cur["epoch"], cur["aux_lo"], cur["aux_hi"], color=cmeth,
                        alpha=0.18, linewidth=0)
        ax.axhline(0.0, color="#cccccc", lw=0.8, zorder=0)
        # steady-state magnitude-share annotation
        srow = summ[(summ["method"] == method) & (summ["protocol"] == ap)
                    & (np.isclose(summ["tau"], tau))]
        if not srow.empty and np.isfinite(srow["share"].iloc[0]):
            sh = float(srow["share"].iloc[0])
            ax.text(0.97, 0.06, rf"{aux_label}: $|{{\cdot}}|$ share $\approx$ {sh:.2f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                    color=cmeth,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=cmeth, alpha=0.9))
        ax.set_xlim(cur["epoch"].min(), cur["epoch"].max())
        ax.set_xlabel("epoch")
        ax.set_title(rf"{CFG.METHOD_LABELS[method]} loss components "
                     rf"({CFG.PROTOCOL_LABELS[ap]}, $\tau={tau:.2f}$)")
    axes[0].set_ylabel("contribution to training loss")
    axes[0].set_ylim(lo - 0.10 * span, hi + 0.12 * span)

    # shared, frameless, horizontal legend pinned bottom-centre (house style)
    handles = [
        Line2D([0], [0], color="0.45", lw=1.8, label="CE term"),
        Line2D([0], [0], color=CFG.PALETTE["ELR"], lw=1.8, label="reg term"),
        Line2D([0], [0], color=CFG.PALETTE["SCE"], lw=1.8, label="RCE term"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("Per-epoch loss components for ELR and SCE",
                 y=1.00, fontsize=13)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _savefig(fig, "p5_loss_component_shares")


# =============================================================================
# tables
# =============================================================================
def tab_asyco_selection_body(summ: pd.DataFrame) -> None:
    if summ.empty:
        return
    tau = CFG.FOCUS_TAU
    s = summ[np.isclose(summ["tau"], tau)].set_index("protocol")
    lines = [
        r"\begin{table}[h!]", r"\centering",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Protocol & Warm-up & Labeled & Unlabeled & Discarded & $1-\tau$ \\",
        r" & (epochs) & (\%) & (\%) & (\%) & (\%) \\", r"\midrule",
    ]
    for protocol in CFG.PROTOCOLS_TO_RUN:
        if protocol not in s.index:
            continue
        r = s.loc[protocol]
        lines.append(
            f"{CFG.PROTOCOL_LABELS[protocol]} & {int(round(r['warmup']))} & "
            f"{_fmt_pct(r['clean'])} & {_fmt_pct(r['noisy'])} & "
            f"{_fmt_pct(r['discard'])} & {_fmt_pct(1.0 - tau)} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        (r"\caption{AsyCo co-divide label partition at $\tau=" f"{tau:.2f}" r"$, by "
         r"training protocol. Each cell is the mean over the final "
         f"{CFG.LAST_K}" r" epochs and ten folds. \emph{Labeled} is the set AsyCo "
         r"trains on as clean, \emph{Unlabeled} the pseudo-labeled set and "
         r"\emph{Discarded} the dropped fraction; warm-up is the median epoch at "
         r"which the partition first activates. The final column is the labeled "
         r"fraction a perfectly selective filter would retain.}"),
        r"\label{tab:mech-asyco-selection}", r"\end{table}",
    ]
    _write_tex("p5_asyco_selection_summary", "\n".join(lines))


def tab_loss_components_body(summ: pd.DataFrame) -> None:
    if summ.empty:
        return
    tau = CFG.FOCUS_TAU
    def cell(method, protocol, col):
        r = summ[(summ["method"] == method) & (summ["protocol"] == protocol)
                 & (np.isclose(summ["tau"], tau))]
        if r.empty:
            return "--"
        return _fmt(r[col].iloc[0]) if col != "share" else _fmt(r["share"].iloc[0])
    lines = [
        r"\begin{table}[h!]", r"\centering",
        r"\begin{tabular}{l*{3}{c}*{3}{c}}", r"\toprule",
        r"\multirow{2}{*}{Protocol} & \multicolumn{3}{c}{ELR} & \multicolumn{3}{c}{SCE} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r" & CE & reg & reg share & CE & RCE & RCE share \\", r"\midrule",
    ]
    for protocol in CFG.PROTOCOLS_TO_RUN:
        lines.append(
            f"{CFG.PROTOCOL_LABELS[protocol]} & "
            f"{cell('ELR', protocol, 'ce')} & {cell('ELR', protocol, 'aux')} & "
            f"{cell('ELR', protocol, 'share')} & "
            f"{cell('SCE', protocol, 'ce')} & {cell('SCE', protocol, 'aux')} & "
            f"{cell('SCE', protocol, 'share')} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        (r"\caption{Loss-component magnitudes at $\tau=" f"{tau:.2f}" r"$, by training "
         r"protocol. Each cell is the mean over the final "
         f"{CFG.LAST_K}" r" epochs and ten folds. \emph{share} is the auxiliary term "
         r"as a fraction of the total loss magnitude, "
         r"$|\mathrm{aux}|/(|\mathrm{CE}|+|\mathrm{aux}|)$. ELR's \emph{reg} term "
         r"($\lambda\log(1-\langle p,t\rangle)$) is negative by construction, so its "
         r"magnitude rather than its signed value is what the share reflects; SCE's "
         r"\emph{RCE} is the reverse cross-entropy term.}"),
        r"\label{tab:mech-loss-components}", r"\end{table}",
    ]
    _write_tex("p5_loss_component_summary", "\n".join(lines))


def tab_asyco_selection_appendix(summ: pd.DataFrame) -> None:
    if summ.empty:
        return
    lines = [
        r"\begin{table}[h!]", r"\centering",
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"Protocol & $\tau$ & Warm-up & Labeled (\%) & Unlabeled (\%) & Discarded (\%) \\",
        r"\midrule",
    ]
    for protocol in CFG.PROTOCOLS_TO_RUN:
        block = summ[summ["protocol"] == protocol].sort_values("tau")
        if block.empty:
            continue
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{CFG.PROTOCOL_LABELS[protocol]}}}}} \\")
        for _, r in block.iterrows():
            lines.append(
                f" & {r['tau']:.2f} & {int(round(r['warmup']))} & "
                f"{_fmt_pct(r['clean'])} & {_fmt_pct(r['noisy'])} & {_fmt_pct(r['discard'])} \\\\")
        lines.append(r"\addlinespace")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines += [
        r"\bottomrule", r"\end{tabular}",
        (r"\caption{AsyCo co-divide partition for every protocol and noise rate. "
         r"Means over the final " f"{CFG.LAST_K}" r" epochs and ten folds.}"),
        r"\label{tab:app-mech-asyco-selection}", r"\end{table}",
    ]
    _write_tex("app_asyco_selection_full", "\n".join(lines))


def tab_loss_components_appendix(summ: pd.DataFrame) -> None:
    if summ.empty:
        return
    lines = [
        r"\begin{table}[h!]", r"\centering",
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"Method & Protocol & $\tau$ & CE & aux & aux share \\", r"\midrule",
    ]
    for method in ("ELR", "SCE"):
        block = summ[summ["method"] == method].sort_values(["protocol", "tau"])
        if block.empty:
            continue
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{method}}}}} \\")
        for _, r in block.iterrows():
            lines.append(
                f" & {CFG.PROTOCOL_LABELS[r['protocol']]} & {r['tau']:.2f} & "
                f"{_fmt(r['ce'])} & {_fmt(r['aux'])} & {_fmt(r['share'])} \\\\")
        lines.append(r"\addlinespace")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines += [
        r"\bottomrule", r"\end{tabular}",
        (r"\caption{CE and auxiliary-term magnitudes (ELR \emph{reg}, SCE "
         r"\emph{RCE}) for every protocol and noise rate. Means over the final "
         f"{CFG.LAST_K}" r" epochs and ten folds; \emph{aux share} is the magnitude "
         r"share $|\mathrm{aux}|/(|\mathrm{CE}|+|\mathrm{aux}|)$. ELR's reg term is "
         r"negative by construction (see Table~\ref{tab:mech-loss-components}).}"),
        r"\label{tab:app-mech-loss-components}", r"\end{table}",
    ]
    _write_tex("app_loss_component_full", "\n".join(lines))


# =============================================================================
# prose numbers
# =============================================================================
def print_prose_numbers(sel_summ: pd.DataFrame, loss_summ: pd.DataFrame) -> None:
    ap, tau = CFG.ANCHOR_PROTOCOL, CFG.FOCUS_TAU
    print("\n" + "=" * 72)
    print(f"PROSE NUMBERS  (primary protocol {ap}, tau = {tau:.2f})")
    print("=" * 72)
    if not sel_summ.empty:
        r = sel_summ[(sel_summ["protocol"] == ap) & (np.isclose(sel_summ["tau"], tau))]
        if not r.empty:
            r = r.iloc[0]
            print(f"AsyCo labeled (clean) fraction : {100*r['clean']:.1f}%  "
                  f"[{100*r['clean_lo']:.1f}, {100*r['clean_hi']:.1f}]")
            print(f"      unlabeled (noisy) fraction: {100*r['noisy']:.1f}%")
            print(f"      discarded fraction        : {100*r['discard']:.1f}%")
            print(f"      perfect-filter clean (1-tau): {100*(1-tau):.1f}%   "
                  f"warm-up (median): {int(round(r['warmup']))} epochs")
            verdict = ("near-complete -> selection largely INACTIVE; corrupted "
                       "labels enter the labeled batch"
                       if r['clean'] > (1 - tau) + 0.05 else
                       "close to the perfect-filter value -> partition SIZE is "
                       "selective (membership not testable from counts)")
            print(f"      -> labeled set is {verdict}.")
        # protocol spread of labeled fraction
        allp = sel_summ[np.isclose(sel_summ["tau"], tau)]
        if not allp.empty:
            print("      labeled fraction by protocol: " +
                  ", ".join(f"{p}={100*allp[allp['protocol']==p]['clean'].iloc[0]:.1f}%"
                            for p in CFG.PROTOCOLS_TO_RUN if p in set(allp['protocol'])))
    if not loss_summ.empty:
        for method, aux in (("ELR", "reg"), ("SCE", "RCE")):
            r = loss_summ[(loss_summ["method"] == method) & (loss_summ["protocol"] == ap)
                          & (np.isclose(loss_summ["tau"], tau))]
            if not r.empty:
                r = r.iloc[0]
                print(f"{method}: CE={r['ce']:.3f}, {aux}={r['aux']:.3f}, "
                      f"|{aux}| share={r['share']:.3f} "
                      f"[{r['share_lo']:.3f}, {r['share_hi']:.3f}]")
    print("=" * 72 + "\n")


# =============================================================================
# main
# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None, help="experiment root")
    ap.add_argument("--out", type=Path, default=None, help="output root")
    ap.add_argument("--thesis-fig-dir", type=Path, default=None,
                    help="optional dir to also copy final PNGs into")
    ap.add_argument("--focus-tau", type=float, default=None)
    ap.add_argument("--last-k", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None,
                    help="fixed AsyCo warm-up epochs (default: auto-detect)")
    ap.add_argument("--probe", action="store_true",
                    help="only print the keys found per method, then exit")
    args = ap.parse_args(argv)

    if args.root is not None:
        CFG.EXPERIMENT_ROOT = args.root
    if args.out is not None:
        CFG.OUT_ROOT = args.out
    if args.thesis_fig_dir is not None:
        CFG.THESIS_FIG_DIR = args.thesis_fig_dir
    if args.focus_tau is not None:
        CFG.FOCUS_TAU = args.focus_tau
    if args.last_k is not None:
        CFG.LAST_K = args.last_k
    if args.warmup is not None:
        CFG.WARMUP_EPOCHS = args.warmup

    print(f"[cfg] experiment root: {CFG.EXPERIMENT_ROOT.resolve()}")
    if not CFG.EXPERIMENT_ROOT.exists():
        print(f"[error] experiment root not found: {CFG.EXPERIMENT_ROOT}")
        return 2

    if args.probe:
        probe()
        return 0

    _style()

    df_sel = load_selection_long()
    df_loss = load_loss_long()
    if df_sel.empty and df_loss.empty:
        print("[error] no usable selection or loss-component data found. "
              "Run with --probe to inspect the keys present in the logs.")
        probe()
        return 1

    sel_summ = selection_summary(df_sel)
    loss_summ = loss_summary(df_loss)

    fig_asyco_partition(df_sel, sel_summ)
    fig_loss_components(df_loss, loss_summ)

    tab_asyco_selection_body(sel_summ)
    tab_loss_components_body(loss_summ)
    tab_asyco_selection_appendix(sel_summ)
    tab_loss_components_appendix(loss_summ)

    print_prose_numbers(sel_summ, loss_summ)
    print(f"[done] outputs under {CFG.OUT_ROOT.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())