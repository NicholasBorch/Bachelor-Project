"""
Results Part 4 - protocol sensitivity analysis (RQ3).

Standalone fold-level analysis of whether the noise-robust-method conclusions
depend on the training protocol. Reads the raw per-fold test_metrics.json
directly (and, when present, raw_fold_results.csv and per-epoch
training_log.jsonl), and writes a self-contained package into
results/protocol_sensitivity/ (figures/, tables/, data/, manifest.json):
aggregate performance figures and tables, ranking stability, exploratory
best-vs-next comparisons, cross-protocol difference-of-differences
interactions, and optional mechanism / epoch-trajectory / matrix diagnostics.
Edit the CONFIG block only; optional CLI overrides are available
(--protocols, --methods, --focus-tau, --skip-mechanism/-epoch/-matrices).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

try:
    import scripts.thesis_paired_stats as TPS
except ModuleNotFoundError:  
    import thesis_paired_stats as TPS


# CONFIG - edit only this block
@dataclass
class Config:
    # raw experiment tree
    EXPERIMENT_ROOT: Path = Path("./results/main_experiment")
    TRAINING_SUBDIR: str = "training"
    METRICS_FILENAME: str = "test_metrics.json"
    TRAINING_LOG_FILENAME: str = "training_log.jsonl"
    TRAIN_DIAG_KEY: str = "train_diagnostics"
    RAW_FOLD_CSV: str = "figures_and_tables/raw_fold_results.csv"
    TAU_DIR_FMT: str = "tau_{tt:02d}"
    FOLD_DIR_FMT: str = "fold_{ff:02d}"
    DATASET: str = "imbalanced"
    BUILD_THESIS_SPLIT: bool = True
    THESIS_SUBDIR: str = "THESIS"

    # protocol code -> folder under EXPERIMENT_ROOT
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
    # Toggle protocols here. Missing protocols are reported and skipped.
    PROTOCOLS_TO_RUN: tuple = ("S", "SP", "A", "AP")
    ANCHOR_PROTOCOL: str = "AP"

    # logical method -> folder under each protocol's training directory
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
    # Toggle methods here; baseline is auto re-added if omitted
    METHODS_TO_RUN: tuple = ("baseline", "SCE", "ELR", "AsyCo")
    BASELINE: str = "baseline"

    TAUS: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    N_FOLDS: int = 10
    FOCUS_TAU: float = 0.20
    AVG_INCLUDE_CLEAN: bool = True

    # metric aliases are tried in order, including inside the optional nests
    METRIC_KEYS: dict = field(default_factory=lambda: {
        "BA":       ["balanced_accuracy", "bacc", "balanced_acc", "BA", "bal_acc"],
        "MacroF1":  ["macro_f1", "f1_macro", "macro_F1", "f1macro", "f1_macro_avg"],
        "MacroAUC": ["macro_auc", "auc_macro", "macro_AUC", "roc_auc_macro", "auroc_macro"],
    })
    METRIC_NEST_KEYS: tuple = ("", "test", "metrics", "test_metrics")
    METRIC_DISPLAY: dict = field(default_factory=lambda: {
        "BA":       ("Balanced accuracy", "Balanced accuracy", 0.0, 1.0),
        "MacroF1":  ("Macro F1",          "Macro F1",          0.0, 1.0),
        "MacroAUC": ("Macro AUC",         "Macro AUC",         0.0, 1.0),
    })
    FIG_METRICS: tuple = ("BA", "MacroF1", "MacroAUC")
    TABLE_METRICS: tuple = ("BA", "MacroF1", "MacroAUC")

    # statistical settings
    N_BOOT: int = 10_000
    CI: float = 0.95
    SEED: int = 10
    HOLM_ALPHA: float = 0.05
    SIG_USES_CORRECTED: bool = True
    SHOW_NS_IN_FIG: bool = False
    NS_SYMBOL: str = "n.s."

    # interaction pairs (direction P1 - P2)
    INTERACTION_CONTRASTS: tuple = (("S", "AP"), ("SP", "AP"), ("A", "AP"))
    ASSUME_IDENTICAL_FOLD_SPLITS: bool = True
    MIN_PAIRED_FOLDS_FOR_TEST: int = 2

    # include baseline in ranking; best-vs-next is exploratory
    RANKING_INCLUDE_BASELINE: bool = True

    # optional mechanism analysis
    RUN_FINAL_EPOCH_MECHANISM: bool = True
    RUN_EPOCH_TRAJECTORIES: bool = True
    # taus read for the epoch grids (cross-protocol limited to those below)
    EPOCH_TAUS: tuple = (0.10, 0.20, 0.30, 0.40, 0.50)
    EPOCH_PROTOCOL_COMPARISON_TAUS: tuple = (0.20,)

    # protocol-resolved matrices / per-class diagnostics
    RUN_MATRIX_DIAGNOSTICS: bool = True
    CLASS_ORDER_MODE: str = "freq"  # "freq", "alpha", or an explicit tuple/list
    CLASSES_ALPHA: tuple = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    CLASSES_FREQ: tuple = ("nv", "bkl", "mel", "bcc", "akiec", "vasc", "df")

    # remove only old flat files in figures/tables/data; nested folders kept
    CLEAN_LEGACY_FLAT_OUTPUTS: bool = True
    # remove stale PDFs under OUT_ROOT (figures are PNG-only)
    CLEAN_REDUNDANT_PDFS: bool = True

    # outputs
    OUT_ROOT: Path = Path("./results/protocol_sensitivity")
    SAVE_PNG: bool = True
    FIG_DPI: int = 300

    # Part 3 method palette
    PALETTE: dict = field(default_factory=lambda: {
        "baseline": "#9ec9e2",
        "SCE":      "#2a9d8f",
        "ELR":      "#e07a3f",
        "AsyCo":    "#7b5cb8",
    })
    # protocol encoding for plots whose lines are protocols
    PROTOCOL_PALETTE: dict = field(default_factory=lambda: {
        "S":  "#4c78a8",
        "SP": "#72b7b2",
        "A":  "#f58518",
        "AP": "#e45756",
    })
    PROTOCOL_LINESTYLES: dict = field(default_factory=lambda: {
        "S": "-", "SP": "--", "A": "-.", "AP": ":",
    })


CFG = Config()


# paths, formatting, small helpers
LATEX_PREAMBLE = r"""% Preamble: \usepackage{booktabs,makecell,multirow,graphicx,longtable,amsmath,pdflscape}
"""


def _fig_dir(*parts: str) -> Path:
    d = CFG.OUT_ROOT / "figures"
    for part in parts:
        d = d / part
    d.mkdir(parents=True, exist_ok=True)
    return d


def _table_dir(*parts: str) -> Path:
    d = CFG.OUT_ROOT / "tables"
    for part in parts:
        d = d / part
    d.mkdir(parents=True, exist_ok=True)
    return d


def _data_dir(*parts: str) -> Path:
    d = CFG.OUT_ROOT / "data"
    for part in parts:
        d = d / part
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cleanup_legacy_flat_outputs() -> None:
    """Remove old direct artifacts so the nested layout is not mixed with stale files."""
    if not CFG.CLEAN_LEGACY_FLAT_OUTPUTS:
        return
    removed = []
    for base in (CFG.OUT_ROOT / "figures", CFG.OUT_ROOT / "tables", CFG.OUT_ROOT / "data"):
        if not base.exists():
            continue
        for fp in base.iterdir():
            if fp.is_file():
                fp.unlink()
                removed.append(fp)
    if removed:
        print(f"[output] removed {len(removed)} legacy flat artifact(s) before rebuilding nested folders.")
    if CFG.CLEAN_REDUNDANT_PDFS and CFG.OUT_ROOT.exists():
        stale_pdfs = list(CFG.OUT_ROOT.rglob("*.pdf"))
        for fp in stale_pdfs:
            fp.unlink()
        if stale_pdfs:
            print(f"[output] removed {len(stale_pdfs)} stale PDF artifact(s); figures are PNG-only.")


def _ensure_out_tree() -> None:
    CFG.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _fig_dir(); _table_dir(); _data_dir()
    _cleanup_legacy_flat_outputs()
    # create the output folder tree up front
    for parts in (
        ("performance", "grouped_bars"),
        ("performance", "protocol_lines"),
        ("performance", "baseline_vs_tau", "across_protocols"),
        ("performance", "baseline_vs_tau", "by_protocol"),
        ("performance", "method_advantage_focus"),
        ("performance", "combined_by_protocol"),
        ("mechanism", "across_tau"),
        ("mechanism", "focus_tau"),
        ("mechanism", "epoch_trajectories"),
        ("mechanism", "by_protocol", "nta_lnmr"),
        ("mechanism", "epoch_by_protocol", "focus_tau"),
        ("mechanism", "epoch_by_protocol", "grids"),
        ("matrices", "confusion"),
        ("matrices", "perclass_f1"),
        ("matrices", "perclass_nta"),
        ("matrices", "perclass_lnmr"),
        ("matrices", "confusion_delta"),
        ("matrices", "confusion_grid"),
    ):
        _fig_dir(*parts)
    for parts in (
        ("performance", "body"), ("performance", "deltas"),
        ("stats",), ("mechanism",), ("appendix", "performance"),
        ("appendix", "mechanism"), ("matrices",),
    ):
        _table_dir(*parts)
    for parts in (("performance",), ("stats",), ("mechanism",), ("epoch",), ("matrices",), ("diagnostics",)):
        _data_dir(*parts)


def _table_subdir(stem: str) -> tuple[str, ...]:
    if stem.startswith("tab_app_mechanism"):
        return ("appendix", "mechanism")
    if stem.startswith("tab_app_"):
        return ("appendix", "performance")
    if "mechanism" in stem:
        return ("mechanism",)
    if any(token in stem for token in ("ranking", "best_vs_next", "interaction")):
        return ("stats",)
    if "delta" in stem:
        return ("performance", "deltas")
    return ("performance", "body")


def _figure_subdir(stem: str) -> tuple[str, ...]:
    if stem.startswith("fig_combined_body_") or stem.startswith("fig_combined_all_"):
        return ("performance", "combined_by_protocol")
    if stem.startswith("fig_nta_lnmr_"):
        return ("mechanism", "by_protocol", "nta_lnmr")
    if stem.startswith("fig_epoch_focus_"):
        return ("mechanism", "epoch_by_protocol", "focus_tau")
    if stem.startswith("fig_epoch_grid_"):
        return ("mechanism", "epoch_by_protocol", "grids")
    if stem.startswith("fig_protocol_lines"):
        return ("performance", "protocol_lines")
    if stem.startswith("fig_protocol_"):
        return ("performance", "grouped_bars")
    if stem.startswith("fig_advantage"):
        return ("performance", "method_advantage_focus")
    if stem.startswith("fig_baseline_protocol_overlay"):
        return ("performance", "baseline_vs_tau", "across_protocols")
    if stem.startswith("fig_baseline_metrics_"):
        return ("performance", "baseline_vs_tau", "by_protocol")
    if stem.startswith("fig_mechanism_protocol"):
        return ("mechanism", "across_tau")
    if stem.startswith("fig_mechanism_focus"):
        return ("mechanism", "focus_tau")
    if stem.startswith("fig_epoch_protocol"):
        return ("mechanism", "epoch_trajectories")
    return ()


def _data_subdir(filename: str) -> tuple[str, ...]:
    low = filename.lower()
    if any(token in low for token in ("confusion", "perclass", "matrix")):
        return ("matrices",)
    if "epoch" in low:
        return ("epoch",)
    if "mechanism" in low:
        return ("mechanism",)
    if any(token in low for token in ("interaction", "ranking", "best_vs_next")):
        return ("stats",)
    if "completeness" in low:
        return ("diagnostics",)
    return ("performance",)


def _write_csv(df: pd.DataFrame, filename: str, subdir: Optional[tuple[str, ...]] = None) -> Path:
    out = _data_dir(*(subdir if subdir is not None else _data_subdir(filename))) / filename
    df.to_csv(out, index=False)
    print(f"[csv] wrote {out}")
    return out


def _classes() -> list[str]:
    if CFG.CLASS_ORDER_MODE == "freq":
        return list(CFG.CLASSES_FREQ)
    if CFG.CLASS_ORDER_MODE == "alpha":
        return list(CFG.CLASSES_ALPHA)
    return list(CFG.CLASS_ORDER_MODE)

def _seed_for(*parts) -> int:
    h = hashlib.sha256(("|".join(map(str, parts))).encode()).hexdigest()
    return (CFG.SEED + int(h[:8], 16)) % (2**32 - 1)


def _active_protocols() -> list[str]:
    return list(CFG.PROTOCOLS_TO_RUN)


def _active_methods() -> list[str]:
    methods = list(dict.fromkeys(CFG.METHODS_TO_RUN))
    if CFG.BASELINE not in methods:
        print(f"[config] '{CFG.BASELINE}' was omitted from METHODS_TO_RUN; re-adding it as the reference.")
        methods.insert(0, CFG.BASELINE)
        # persist the normalized selection (printed once)
        CFG.METHODS_TO_RUN = tuple(methods)
    return methods


def _robust_methods() -> list[str]:
    return [m for m in _active_methods() if m != CFG.BASELINE]


def _protocol_root(protocol: str) -> Path:
    return CFG.EXPERIMENT_ROOT / CFG.PROTOCOL_DIRS[protocol]


def _training_root(protocol: str) -> Path:
    base = _protocol_root(protocol)
    return base / CFG.TRAINING_SUBDIR if CFG.TRAINING_SUBDIR else base


def _method_root(protocol: str, method: str) -> Path:
    return _training_root(protocol) / CFG.METHOD_DIRS[method]


def _tau_dir(tau: float) -> str:
    return CFG.TAU_DIR_FMT.format(tt=int(round(float(tau) * 100)))


def _fold_dir(fold: int) -> str:
    return CFG.FOLD_DIR_FMT.format(ff=int(fold))


def _metrics_path(protocol: str, method: str, tau: float, fold: int) -> Path:
    return _method_root(protocol, method) / _tau_dir(tau) / _fold_dir(fold) / CFG.METRICS_FILENAME


def _log_path(protocol: str, method: str, tau: float, fold: int) -> Path:
    return _method_root(protocol, method) / _tau_dir(tau) / _fold_dir(fold) / CFG.TRAINING_LOG_FILENAME


def _read_json(fp: Path) -> dict:
    with open(fp, "r") as fh:
        return json.load(fh)


def _flatten(d: dict, prefix: str) -> dict:
    if prefix and isinstance(d.get(prefix), dict):
        return d[prefix]
    return d


def _extract_metric(d: dict, aliases: Iterable[str]) -> Optional[float]:
    for nest in CFG.METRIC_NEST_KEYS:
        scope = _flatten(d, nest)
        if not isinstance(scope, dict):
            continue
        for key in aliases:
            if key in scope and scope[key] is not None:
                try:
                    return float(scope[key])
                except (TypeError, ValueError):
                    pass
    return None


def _all_keys(d: dict) -> list[str]:
    keys = list(d.keys())
    for nest in CFG.METRIC_NEST_KEYS:
        if nest and isinstance(d.get(nest), dict):
            keys.extend(f"{nest}.{k}" for k in d[nest].keys())
    return sorted(set(keys))


def _boot_ci(values, seed: int) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
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


def _fmt_metric(x, nd: int = 3) -> str:
    return "--" if x is None or pd.isna(x) else f"{float(x):.{nd}f}"


def _fmt_signed(x, nd: int = 3) -> str:
    if x is None or pd.isna(x):
        return "--"
    x = float(x)
    if abs(x) < 0.5 * 10 ** (-nd):
        x = 0.0
    return f"{x:+.{nd}f}"


def _fmt_pct(x, nd: int = 1) -> str:
    return "--" if x is None or pd.isna(x) else f"{float(x):+.{nd}f}\\%"


def _fmt_p(p) -> str:
    if p is None or pd.isna(p):
        return "--"
    return r"$<0.001$" if float(p) < 0.001 else f"{float(p):.3f}"


def _fmt_W(w) -> str:
    if w is None or pd.isna(w):
        return "--"
    w = float(w)
    return f"{w:.0f}" if abs(w - round(w)) < 1e-9 else f"{w:.1f}"


def _fmt_ci(lo, hi, signed: bool = True) -> str:
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return "--"
    if signed:
        return f"[{float(lo):+.3f},\\,{float(hi):+.3f}]"
    return f"[{float(lo):.3f},\\,{float(hi):.3f}]"


def _sig_symbol(p, ns: bool = True) -> str:
    return TPS.sig_code(p, ns=CFG.NS_SYMBOL if ns else "")


def _write_tex(stem: str, body: str, subdir: Optional[tuple[str, ...]] = None) -> Path:
    out = _table_dir(*(subdir if subdir is not None else _table_subdir(stem)))
    fp = out / f"{stem}.tex"
    fp.write_text(LATEX_PREAMBLE + "\n" + body.rstrip() + "\n")
    print(f"[tab] wrote {fp}")
    return fp


def _style() -> None:
    """Thesis plotting convention with safe save margins."""
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Palatino", "Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
        "mathtext.fontset":   "cm",
        "axes.unicode_minus": False,
        "figure.dpi": 150, "savefig.dpi": CFG.FIG_DPI, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.24, "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#cccccc", "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def _savefig(fig, stem: str, subdir: Optional[tuple[str, ...]] = None) -> None:
    """Write the publication figure as PNG only."""
    out = _fig_dir(*(subdir if subdir is not None else _figure_subdir(stem)))
    if CFG.SAVE_PNG:
        fig.savefig(out / f"{stem}.png")
    plt.close(fig)
    print(f"[fig] wrote {out / stem}.png")


def _finite_bounds(*arrays) -> tuple[float, float]:
    chunks = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float).ravel()
        chunks.append(a[np.isfinite(a)])
    vals = np.concatenate([x for x in chunks if x.size]) if any(x.size for x in chunks) else np.array([])
    if vals.size == 0:
        return np.nan, np.nan
    return float(vals.min()), float(vals.max())


def _set_headroom_ylim(ax, lows, highs, *, floor: Optional[float] = None,
                       ceiling: Optional[float] = None, pad_frac: float = 0.10) -> None:
    lo, hi = _finite_bounds(lows, highs)
    if not np.isfinite(hi):
        return
    if floor is not None:
        lo = min(lo, floor)
    span = max(hi - lo, 0.05)
    lo2 = lo - pad_frac * span
    hi2 = hi + pad_frac * span
    if floor is not None:
        lo2 = max(floor, lo2)
    if ceiling is not None:
        hi2 = min(ceiling, hi2)
    if hi2 <= lo2:
        hi2 = lo2 + 0.05
    ax.set_ylim(lo2, hi2)


# Red -> yellow -> green: high-is-good. Reverse for LNMR where low is good.
_RYG = LinearSegmentedColormap.from_list(
    "ryg", ["#c0392b", "#e67e22", "#f1c40f", "#7dcea0", "#1e8449"])
_GYR = _RYG.reversed()


def _good_cmap(low_is_good: bool):
    return _GYR if low_is_good else _RYG

def _empty_paired_record(**extra) -> dict:
    rec = dict(
        n=0, delta=np.nan, delta_ci_lo=np.nan, delta_ci_hi=np.nan,
        W=np.nan, p_wilcoxon=np.nan, p_perm=np.nan, r_rb=np.nan,
        direction=0, n_boot=CFG.N_BOOT, perm_exact=True,
    )
    rec.update(extra)
    return rec


def _finish_holm(block: list[dict]) -> list[dict]:
    TPS.add_holm_and_flags(block, alpha=CFG.HOLM_ALPHA)
    for rec in block:
        rec["p_raw"] = rec.get("p_wilcoxon", np.nan)
        rec["p_holm"] = rec.get("p_wilcoxon_holm", np.nan)
        rec["mean_delta"] = rec.get("delta", np.nan)
    return block


def _validate_config() -> None:
    unknown_p = [p for p in _active_protocols() if p not in CFG.PROTOCOL_DIRS]
    unknown_m = [m for m in _active_methods() if m not in CFG.METHOD_DIRS]
    if unknown_p:
        raise ValueError(f"Unknown protocol code(s) in PROTOCOLS_TO_RUN: {unknown_p}")
    if unknown_m:
        raise ValueError(f"Unknown method(s) in METHODS_TO_RUN: {unknown_m}")
    if CFG.BASELINE not in CFG.METHOD_DIRS:
        raise ValueError(f"BASELINE='{CFG.BASELINE}' is not in METHOD_DIRS")
    if CFG.ANCHOR_PROTOCOL not in CFG.PROTOCOL_DIRS:
        raise ValueError(f"ANCHOR_PROTOCOL='{CFG.ANCHOR_PROTOCOL}' is not in PROTOCOL_DIRS")


# aggregate metric loading and summaries
def load_metric_long() -> pd.DataFrame:
    rows: list[dict] = []
    first_payload: Optional[dict] = None
    first_path: Optional[Path] = None
    methods = _active_methods()

    for protocol in _active_protocols():
        proot = _training_root(protocol)
        if not proot.exists():
            print(f"[skip] protocol {protocol}: training directory not found: {proot}")
            continue
        present = sorted(p.name for p in proot.iterdir() if p.is_dir())
        print(f"[scan] protocol {protocol} ({proot})")
        print(f"[scan]   method dirs on disk = {present}")
        configured_dirs = {CFG.METHOD_DIRS[m] for m in methods}
        ignored = [m for m in present if m not in configured_dirs and m != "figures_and_tables"]
        if ignored:
            print(f"[scan]   not selected / unmapped, ignored = {ignored}")

        for method in methods:
            mroot = _method_root(protocol, method)
            if not mroot.exists():
                print(f"[warn]   {protocol}/{method}: method directory missing: {mroot}")
                continue
            for tau in CFG.TAUS:
                for fold in range(CFG.N_FOLDS):
                    fp = _metrics_path(protocol, method, tau, fold)
                    if not fp.exists():
                        continue
                    try:
                        payload = _read_json(fp)
                    except (json.JSONDecodeError, OSError) as exc:
                        print(f"[warn]   could not read {fp}: {exc}")
                        continue
                    if first_payload is None:
                        first_payload, first_path = payload, fp
                    rec = dict(protocol=protocol, method=method,
                               tau=float(tau), fold=int(fold), source_file=str(fp))
                    for logical, aliases in CFG.METRIC_KEYS.items():
                        rec[logical] = _extract_metric(payload, aliases)
                    rows.append(rec)

    if not rows:
        raise FileNotFoundError(
            "No test_metrics.json files were found for the selected protocols and methods. "
            "Check EXPERIMENT_ROOT, PROTOCOL_DIRS, METHOD_DIRS, tau/fold naming, and which runs have completed.")

    if first_payload is not None:
        print(f"\n[schema] first metrics file: {first_path}")
        print(f"[schema] keys present: {_all_keys(first_payload)}")
        resolved = {name: _extract_metric(first_payload, aliases)
                    for name, aliases in CFG.METRIC_KEYS.items()}
        print("[schema] resolved -> " + ", ".join(
            f"{name}={'OK' if value is not None else 'MISSING'}" for name, value in resolved.items()))

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["protocol", "method", "tau", "fold"])
    df = df.sort_values(["protocol", "method", "tau", "fold"]).reset_index(drop=True)
    for metric in ("BA", "MacroF1"):
        if metric not in df.columns or df[metric].isna().all():
            raise ValueError(
                f"Required metric '{metric}' was not resolved in any JSON file. "
                f"Update CONFIG.METRIC_KEYS after reviewing the printed schema.")
    return df


def completeness_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol in _active_protocols():
        for method in _active_methods():
            for tau in CFG.TAUS:
                cell = df[(df.protocol == protocol) & (df.method == method) & np.isclose(df.tau, tau)]
                folds = sorted(int(x) for x in cell.fold.unique()) if not cell.empty else []
                missing = sorted(set(range(CFG.N_FOLDS)) - set(folds))
                rows.append(dict(protocol=protocol, method=method, tau=float(tau),
                                 n_folds=len(folds), expected_folds=CFG.N_FOLDS,
                                 missing_folds=",".join(map(str, missing)),
                                 complete=(len(folds) == CFG.N_FOLDS)))
    return pd.DataFrame(rows)


def print_completeness(comp: pd.DataFrame) -> None:
    print("\n[completeness] expected folds per protocol x method x tau =", CFG.N_FOLDS)
    incomplete = comp[~comp.complete]
    if incomplete.empty:
        print("[completeness] OK - every selected cell has the full fold set.")
        return
    for _, row in incomplete.iterrows():
        print(f"   ! {row.protocol:3s} {row.method:9s} tau={row.tau:.1f}: "
              f"{int(row.n_folds)}/{CFG.N_FOLDS} folds (missing [{row.missing_folds}])")
    print("[completeness] Partial cells are retained descriptively; paired tests use only aligned non-missing folds.")


def available_protocols(df: pd.DataFrame) -> list[str]:
    present = set(df.protocol.unique())
    return [p for p in _active_protocols() if p in present]


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol in available_protocols(df):
        for metric in CFG.TABLE_METRICS:
            if metric not in df.columns:
                continue
            for method in _active_methods():
                for tau in CFG.TAUS:
                    vals = df[(df.protocol == protocol) & (df.method == method)
                              & np.isclose(df.tau, tau)][metric].values
                    mean, lo, hi = _boot_ci(vals, _seed_for("summary", protocol, metric, method, tau))
                    rows.append(dict(protocol=protocol, metric=metric, method=method,
                                     tau=float(tau), mean=mean, lo=lo, hi=hi,
                                     n=int(np.sum(~pd.isna(vals)))))
    return pd.DataFrame(rows)


def _wide_fold(df: pd.DataFrame, protocol: str, metric: str, tau: float) -> pd.DataFrame:
    sub = df[(df.protocol == protocol) & np.isclose(df.tau, tau)]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="fold", columns="method", values=metric, aggfunc="first")


def method_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol in available_protocols(df):
        for metric in CFG.TABLE_METRICS:
            if metric not in df.columns:
                continue
            for method in _robust_methods():
                block = []
                for tau in CFG.TAUS:
                    wide = _wide_fold(df, protocol, metric, tau)
                    if CFG.BASELINE in wide.columns and method in wide.columns:
                        pair = wide[[method, CFG.BASELINE]].dropna()
                        if len(pair) >= CFG.MIN_PAIRED_FOLDS_FOR_TEST:
                            d = pair[method].values - pair[CFG.BASELINE].values
                            result = TPS.paired_compare(
                                d, n_boot=CFG.N_BOOT,
                                boot_seed=_seed_for("mvb", protocol, metric, method, tau))
                            rec = dict(protocol=protocol, metric=metric, method=method,
                                       tau=float(tau), folds=",".join(map(str, pair.index.tolist())),
                                       **result.as_dict())
                        else:
                            rec = _empty_paired_record(protocol=protocol, metric=metric,
                                                       method=method, tau=float(tau), folds="")
                    else:
                        rec = _empty_paired_record(protocol=protocol, metric=metric,
                                                   method=method, tau=float(tau), folds="")
                    block.append(rec)
                rows.extend(_finish_holm(block))
    return pd.DataFrame(rows)


def method_vs_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Noise sensitivity: each (protocol, method) at tau>0 vs the same at tau=0, paired by fold."""
    rows = []
    for protocol in available_protocols(df):
        for metric in CFG.TABLE_METRICS:
            if metric not in df.columns:
                continue
            clean = _wide_fold(df, protocol, metric, 0.0)   # index=fold, columns=method
            for method in _active_methods():                # all methods, incl. baseline
                block = []
                for tau in CFG.TAUS:
                    if np.isclose(tau, 0.0):
                        continue
                    noisy = _wide_fold(df, protocol, metric, tau)
                    if method in clean.columns and method in noisy.columns:
                        pair = pd.concat(
                            [noisy[method].rename("noisy"), clean[method].rename("clean")],
                            axis=1).dropna()
                        if len(pair) >= CFG.MIN_PAIRED_FOLDS_FOR_TEST:
                            d = pair["noisy"].to_numpy() - pair["clean"].to_numpy()
                            result = TPS.paired_compare(
                                d, n_boot=CFG.N_BOOT,
                                boot_seed=_seed_for("mvc", protocol, metric, method, tau))
                            rec = dict(protocol=protocol, metric=metric, method=method,
                                       tau=float(tau),
                                       folds=",".join(map(str, pair.index.tolist())),
                                       **result.as_dict())
                        else:
                            rec = _empty_paired_record(protocol=protocol, metric=metric,
                                                       method=method, tau=float(tau), folds="")
                    else:
                        rec = _empty_paired_record(protocol=protocol, metric=metric,
                                                   method=method, tau=float(tau), folds="")
                    block.append(rec)
                rows.extend(_finish_holm(block))
    return pd.DataFrame(rows)


# ranking stability and exploratory best-vs-next-best
def build_ranking_stability(summary: pd.DataFrame) -> pd.DataFrame:
    methods = _active_methods() if CFG.RANKING_INCLUDE_BASELINE else _robust_methods()
    rows = []
    for protocol in summary.protocol.unique():
        for metric in CFG.TABLE_METRICS:
            for tau in CFG.TAUS:
                cell = summary[(summary.protocol == protocol) & (summary.metric == metric)
                               & np.isclose(summary.tau, tau) & summary.method.isin(methods)].copy()
                cell = cell.dropna(subset=["mean"]).sort_values("mean", ascending=False)
                for rank, (_, rec) in enumerate(cell.iterrows(), start=1):
                    next_mean = cell.iloc[rank]["mean"] if rank < len(cell) else np.nan
                    rows.append(dict(protocol=protocol, metric=metric, tau=float(tau),
                                     rank=rank, method=rec.method, mean=float(rec["mean"]),
                                     lo=float(rec.lo), hi=float(rec.hi), n=int(rec.n),
                                     gap_to_next=(float(rec["mean"] - next_mean)
                                                  if not pd.isna(next_mean) else np.nan)))
    return pd.DataFrame(rows)


def best_vs_next(df: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol in available_protocols(df):
        for metric in CFG.TABLE_METRICS:
            block = []
            for tau in CFG.TAUS:
                cell = ranking[(ranking.protocol == protocol) & (ranking.metric == metric)
                               & np.isclose(ranking.tau, tau)].sort_values("rank")
                if len(cell) < 2:
                    block.append(_empty_paired_record(
                        protocol=protocol, metric=metric, tau=float(tau),
                        best_method="--", next_method="--", folds="", exploratory=True))
                    continue
                best_method = str(cell.iloc[0].method)
                next_method = str(cell.iloc[1].method)
                wide = _wide_fold(df, protocol, metric, tau)
                if best_method in wide.columns and next_method in wide.columns:
                    pair = wide[[best_method, next_method]].dropna()
                else:
                    pair = pd.DataFrame()
                if len(pair) >= CFG.MIN_PAIRED_FOLDS_FOR_TEST:
                    d = pair[best_method].values - pair[next_method].values
                    result = TPS.paired_compare(
                        d, n_boot=CFG.N_BOOT,
                        boot_seed=_seed_for("best-next", protocol, metric, tau,
                                            best_method, next_method))
                    rec = dict(protocol=protocol, metric=metric, tau=float(tau),
                               best_method=best_method, next_method=next_method,
                               folds=",".join(map(str, pair.index.tolist())), exploratory=True,
                               **result.as_dict())
                else:
                    rec = _empty_paired_record(
                        protocol=protocol, metric=metric, tau=float(tau),
                        best_method=best_method, next_method=next_method,
                        folds="", exploratory=True)
                block.append(rec)
            rows.extend(_finish_holm(block))
    return pd.DataFrame(rows)


# cross-protocol difference-of-differences interactions
def _aligned_advantages(df: pd.DataFrame, p1: str, p2: str, metric: str,
                        method: str, tau: float) -> tuple[pd.DataFrame, dict]:
    """Fold-aligned own-baseline advantages in two protocols."""
    w1 = _wide_fold(df, p1, metric, tau)
    w2 = _wide_fold(df, p2, metric, tau)
    required = [method, CFG.BASELINE]
    if any(col not in w1.columns for col in required) or any(col not in w2.columns for col in required):
        return pd.DataFrame(), dict(p1_folds="", p2_folds="", aligned_folds="",
                                    same_fold_ids=False, n_aligned=0)
    a1 = (w1[method] - w1[CFG.BASELINE]).rename("adv_p1")
    a2 = (w2[method] - w2[CFG.BASELINE]).rename("adv_p2")
    p1_ids = sorted(int(x) for x in a1.dropna().index.tolist())
    p2_ids = sorted(int(x) for x in a2.dropna().index.tolist())
    paired = pd.concat([a1, a2], axis=1).dropna()
    aligned_ids = sorted(int(x) for x in paired.index.tolist())
    info = dict(
        p1_folds=",".join(map(str, p1_ids)),
        p2_folds=",".join(map(str, p2_ids)),
        aligned_folds=",".join(map(str, aligned_ids)),
        same_fold_ids=(p1_ids == p2_ids),
        n_aligned=len(aligned_ids),
    )
    return paired, info


def interaction_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict] = []
    alignment_rows: list[dict] = []
    present = set(available_protocols(df))
    if not CFG.ASSUME_IDENTICAL_FOLD_SPLITS:
        print("[interaction] disabled: ASSUME_IDENTICAL_FOLD_SPLITS=False")
        return pd.DataFrame(), pd.DataFrame()

    for p1, p2 in CFG.INTERACTION_CONTRASTS:
        if p1 not in present or p2 not in present:
            print(f"[interaction] skip {p1} - {p2}: one or both protocols are unavailable.")
            continue
        for metric in CFG.TABLE_METRICS:
            for method in _robust_methods():
                block = []
                for tau in CFG.TAUS:
                    paired, info = _aligned_advantages(df, p1, p2, metric, method, tau)
                    alignment_rows.append(dict(protocol_1=p1, protocol_2=p2,
                                               metric=metric, method=method, tau=float(tau), **info))
                    if len(paired) >= CFG.MIN_PAIRED_FOLDS_FOR_TEST and info["same_fold_ids"]:
                        interaction = paired.adv_p1.values - paired.adv_p2.values
                        result = TPS.paired_compare(
                            interaction, n_boot=CFG.N_BOOT,
                            boot_seed=_seed_for("interaction", p1, p2, metric, method, tau))
                        rec = dict(protocol_1=p1, protocol_2=p2, metric=metric,
                                   method=method, tau=float(tau),
                                   advantage_p1=float(paired.adv_p1.mean()),
                                   advantage_p2=float(paired.adv_p2.mean()),
                                   folds=info["aligned_folds"], fold_ids_aligned=True,
                                   **result.as_dict())
                    else:
                        if len(paired) and not info["same_fold_ids"]:
                            print(f"[interaction] skip {p1}-{p2}/{metric}/{method}/tau={tau:.1f}: "
                                  "available fold-ID sets differ.")
                        rec = _empty_paired_record(
                            protocol_1=p1, protocol_2=p2, metric=metric, method=method,
                            tau=float(tau), advantage_p1=(float(paired.adv_p1.mean()) if len(paired) else np.nan),
                            advantage_p2=(float(paired.adv_p2.mean()) if len(paired) else np.nan),
                            folds=info["aligned_folds"], fold_ids_aligned=bool(info["same_fold_ids"]))
                    block.append(rec)
                result_rows.extend(_finish_holm(block))
    return pd.DataFrame(result_rows), pd.DataFrame(alignment_rows)


# cross-protocol comparison: which protocol is best for a fixed method
def protocol_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare each method's per-fold scores between every protocol pair (paired by fold)."""
    import itertools
    protocols = available_protocols(df)
    if len(protocols) < 2:
        print("[protocol-comparison] fewer than two protocols loaded; "
              "skipping (uncomment more protocols in PROTOCOLS_TO_RUN to enable).")
        return pd.DataFrame()
    pairs = list(itertools.combinations(protocols, 2))
    rows = []
    for method in _active_methods():
        for metric in CFG.TABLE_METRICS:
            if metric not in df.columns:
                continue
            for tau in CFG.TAUS:
                block = []
                for p1, p2 in pairs:
                    w1 = _wide_fold(df, p1, metric, tau)
                    w2 = _wide_fold(df, p2, metric, tau)
                    if (method in getattr(w1, "columns", []) and
                            method in getattr(w2, "columns", [])):
                        paired = pd.concat(
                            [w1[method].rename("p1"), w2[method].rename("p2")],
                            axis=1).dropna()
                    else:
                        paired = pd.DataFrame()
                    if len(paired) >= CFG.MIN_PAIRED_FOLDS_FOR_TEST:
                        d = paired["p1"].values - paired["p2"].values  # P1 - P2
                        result = TPS.paired_compare(
                            d, n_boot=CFG.N_BOOT,
                            boot_seed=_seed_for("protocol-comp", method, metric,
                                                tau, p1, p2))
                        rec = dict(method=method, metric=metric, tau=float(tau),
                                   protocol_1=p1, protocol_2=p2,
                                   folds=",".join(map(str, paired.index.tolist())),
                                   **result.as_dict())
                    else:
                        rec = _empty_paired_record(
                            method=method, metric=metric, tau=float(tau),
                            protocol_1=p1, protocol_2=p2, folds="")
                    block.append(rec)
                rows.extend(_finish_holm(block))
    return pd.DataFrame(rows)


# own-baseline delta data
def delta_long(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_rows = []
    for protocol in summary.protocol.unique():
        for metric in CFG.TABLE_METRICS:
            for method in _robust_methods():
                for tau in CFG.TAUS:
                    mr = summary[(summary.protocol == protocol) & (summary.metric == metric)
                                 & (summary.method == method) & np.isclose(summary.tau, tau)]
                    br = summary[(summary.protocol == protocol) & (summary.metric == metric)
                                 & (summary.method == CFG.BASELINE) & np.isclose(summary.tau, tau)]
                    m = float(mr.iloc[0]["mean"]) if len(mr) else np.nan
                    b = float(br.iloc[0]["mean"]) if len(br) else np.nan
                    d = m - b if not (pd.isna(m) or pd.isna(b)) else np.nan
                    rel = 100.0 * d / b if not pd.isna(d) and b != 0 else np.nan
                    full_rows.append(dict(protocol=protocol, metric=metric, method=method,
                                          tau=float(tau), method_mean=m, baseline_mean=b,
                                          delta=d, rel_pct=rel))
    full = pd.DataFrame(full_rows)
    focus = full[np.isclose(full.tau, CFG.FOCUS_TAU)].copy()
    avg_source = full if CFG.AVG_INCLUDE_CLEAN else full[full.tau > 0]
    avg = (avg_source.groupby(["protocol", "metric", "method"], as_index=False)
           .agg(delta=("delta", "mean"), rel_pct=("rel_pct", "mean")))
    return full, focus, avg


# aggregate performance figures
def _yerr(rows: pd.DataFrame) -> np.ndarray:
    means = rows["mean"].to_numpy(dtype=float)
    lo = rows["lo"].to_numpy(dtype=float)
    hi = rows["hi"].to_numpy(dtype=float)
    return np.vstack([np.clip(means - lo, 0, None), np.clip(hi - means, 0, None)])


def _metric_dynamic_upper(summary: pd.DataFrame, metric: str, extra: float = 0.07) -> float:
    cell = summary[summary.metric == metric]
    if cell.empty:
        return 1.0
    hi = cell["hi"].to_numpy(dtype=float)
    hi = hi[np.isfinite(hi)]
    if hi.size == 0:
        return 1.0
    return min(1.0, max(0.2, float(hi.max()) + extra))


def _grouped_bar_panel(ax, summary: pd.DataFrame, mvb: pd.DataFrame,
                       metric: str, protocol: str, ylim=(0.0, 1.0)) -> None:
    """Part-3 visual grammar with additional star headroom."""
    taus = list(CFG.TAUS)
    methods = _active_methods()
    x = np.arange(len(taus))
    width = 0.8 / max(len(methods), 1)
    metric_sum = summary[(summary.metric == metric) & (summary.protocol == protocol)]

    for j, method in enumerate(methods):
        rows = (metric_sum[metric_sum.method == method]
                .set_index("tau").reindex(taus).reset_index())
        offs = (j - (len(methods) - 1) / 2) * width
        ax.bar(x + offs, rows["mean"].values, width=width, yerr=_yerr(rows),
               color=CFG.PALETTE.get(method), edgecolor="white", linewidth=0.6,
               capsize=2.5, error_kw=dict(elinewidth=0.9, alpha=0.85),
               label=CFG.METHOD_LABELS.get(method, method), zorder=3)
        if method == CFG.BASELINE or mvb.empty:
            continue
        st = mvb[(mvb.protocol == protocol) & (mvb.metric == metric)
                 & (mvb.method == method)].set_index("tau")
        pcol = "p_holm" if CFG.SIG_USES_CORRECTED else "p_raw"
        for xi, tau in zip(x, taus):
            if tau not in st.index:
                continue
            sym = _sig_symbol(st.loc[tau, pcol], ns=CFG.SHOW_NS_IN_FIG)
            if not sym:
                continue
            if sym != CFG.NS_SYMBOL:
                direction = st.loc[tau, "direction"]
                if direction > 0:
                    sym = "+" + sym
                elif direction < 0:
                    sym = "-" + sym
            high = rows.loc[np.isclose(rows.tau, tau), "hi"].values
            if len(high) and not pd.isna(high[0]):
                color, fs = (("0.45", 7) if sym == CFG.NS_SYMBOL else ("0.15", 8))
                ax.text(xi + offs, float(high[0]) + 0.012, sym, color=color,
                        fontsize=fs, ha="center", va="bottom", zorder=5)

    _, ylabel, _, _ = CFG.METRIC_DISPLAY[metric]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1f}" for t in taus])
    ax.set_xlabel(r"Noise rate $\tau$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(CFG.PROTOCOL_LABELS.get(protocol, protocol) + f"  ({protocol})")



def _fig_combined_metrics_by_protocol(summary: pd.DataFrame, mvb: pd.DataFrame,
                                      protocol: str, metrics: list[str], stem: str,
                                      title_suffix: str) -> None:
    """Part-3-style side-by-side grouped bars for one protocol."""
    metrics = [m for m in metrics if m in set(summary.metric.unique())]
    if not metrics:
        return
    _style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.9), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        _grouped_bar_panel(ax, summary, mvb, metric, protocol, ylim=(0.0, 1.0))
        ax.set_title(CFG.METRIC_DISPLAY[metric][0])
    axes[0].set_ylabel("Score")
    for ax in axes[1:]:
        ax.set_ylabel("")
    handles = [Patch(facecolor=CFG.PALETTE.get(m), edgecolor="white",
                     label=CFG.METHOD_LABELS.get(m, m)) for m in _active_methods()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(f"Method comparison under label noise - {CFG.PROTOCOL_LABELS.get(protocol, protocol)} ({protocol}){title_suffix}",
                 y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0.075, 1, 0.955])
    _savefig(fig, stem)


def fig_combined_metrics_by_protocol(summary: pd.DataFrame, mvb: pd.DataFrame,
                                     protocol: str) -> None:
    """Emit both AP-body-style and complete three-metric Part-3 figures."""
    _fig_combined_metrics_by_protocol(
        summary, mvb, protocol, ["BA", "MacroF1"],
        f"fig_combined_body_{protocol}", "")
    _fig_combined_metrics_by_protocol(
        summary, mvb, protocol, ["BA", "MacroF1", "MacroAUC"],
        f"fig_combined_all_{protocol}", " - all metrics")

def fig_protocol_bars(summary: pd.DataFrame, mvb: pd.DataFrame, metric: str) -> None:
    protocols = [p for p in _active_protocols() if p in set(summary.protocol.unique())]
    if not protocols:
        return
    _style()
    ncol = min(2, len(protocols)); nrow = int(np.ceil(len(protocols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 5.1 * nrow),
                             squeeze=False, sharey=True)
    upper = _metric_dynamic_upper(summary, metric, extra=0.075)
    for idx, protocol in enumerate(protocols):
        _grouped_bar_panel(axes[idx // ncol, idx % ncol], summary, mvb, metric, protocol, ylim=(0.0, upper))
    for idx in range(len(protocols), nrow * ncol):
        axes[idx // ncol, idx % ncol].axis("off")
    handles = [Patch(facecolor=CFG.PALETTE.get(m), edgecolor="white",
                     label=CFG.METHOD_LABELS.get(m, m)) for m in _active_methods()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(f"{CFG.METRIC_DISPLAY[metric][0]} across training protocols",
                 y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0.075, 1, 0.955])
    _savefig(fig, f"fig_protocol_{metric}")


def fig_protocol_lines(summary: pd.DataFrame, mvc: pd.DataFrame, metric: str) -> None:
    protocols = [p for p in _active_protocols() if p in set(summary.protocol.unique())]
    methods = _active_methods()
    if not protocols or not methods:
        return
    _style()
    ncol = min(2, len(methods)); nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 4.9 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    metric_cell = summary[summary.metric == metric]
    ymin, ymax = _finite_bounds(metric_cell["lo"], metric_cell["hi"])
    pad = max(0.025, 0.10 * max(ymax - ymin, 0.05))
    ylim = (max(0.0, ymin - pad), min(1.0, ymax + pad))
    pcol = "p_holm" if CFG.SIG_USES_CORRECTED else "p_raw"
    for idx, method in enumerate(methods):
        ax = axes[idx // ncol, idx % ncol]
        for protocol in protocols:
            cell = summary[(summary.protocol == protocol) & (summary.metric == metric)
                           & (summary.method == method)].sort_values("tau")
            if cell.empty:
                continue
            x = cell.tau.to_numpy(dtype=float)
            ax.plot(x, cell["mean"], marker="o", color=CFG.PROTOCOL_PALETTE.get(protocol),
                    linestyle=CFG.PROTOCOL_LINESTYLES.get(protocol, "-"), markersize=4,
                    label=CFG.PROTOCOL_LABELS.get(protocol, protocol), zorder=3)
            ax.fill_between(x, cell.lo.to_numpy(dtype=float), cell.hi.to_numpy(dtype=float),
                            color=CFG.PROTOCOL_PALETTE.get(protocol), alpha=0.13, linewidth=0, zorder=2)
            # vs-clean significance stars at each tau for this protocol line
            st = (mvc[(mvc.protocol == protocol) & (mvc.metric == metric)
                      & (mvc.method == method)].set_index("tau"))
            hi_by_tau = dict(zip(cell.tau.to_numpy(dtype=float),
                                 cell.hi.to_numpy(dtype=float)))
            line_col = CFG.PROTOCOL_PALETTE.get(protocol)
            for tau in CFG.TAUS:
                if tau not in st.index or tau not in hi_by_tau:   # tau=0 absent -> skipped
                    continue
                sym = _sig_symbol(st.loc[tau, pcol], ns=CFG.SHOW_NS_IN_FIG)
                if not sym:
                    continue
                if sym != CFG.NS_SYMBOL:
                    direction = st.loc[tau, "direction"]
                    if direction > 0:
                        sym = "+" + sym
                    elif direction < 0:
                        sym = "-" + sym
                hi = hi_by_tau[tau]
                if pd.isna(hi):
                    continue
                tcol, fs = (("0.5", 7) if sym == CFG.NS_SYMBOL else (line_col, 8))
                ax.text(float(tau), float(hi) + 0.02, sym, color=tcol,
                        fontsize=fs, ha="center", va="bottom", zorder=5)
        ax.set_title(CFG.METHOD_LABELS.get(method, method))
        ax.set_xlabel(r"Noise rate $\tau$")
        ax.set_ylabel("Score")
        ax.set_xticks(CFG.TAUS)
        ax.set_xticklabels([f"{t:.1f}" for t in CFG.TAUS])
        ax.set_ylim(*ylim)
    for idx in range(len(methods), nrow * ncol):
        axes[idx // ncol, idx % ncol].axis("off")
    handles = [plt.Line2D([0], [0], color=CFG.PROTOCOL_PALETTE.get(p),
                          linestyle=CFG.PROTOCOL_LINESTYLES.get(p, "-"), marker="o",
                          markersize=4, label=CFG.PROTOCOL_LABELS.get(p, p))
               for p in protocols]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(f"Protocol sensitivity of {CFG.METRIC_DISPLAY[metric][0]}",
                 y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0.075, 1, 0.955])
    _savefig(fig, f"fig_protocol_lines_{metric}")


def fig_baseline_protocol_overlay(summary: pd.DataFrame, metric: str) -> None:
    """One baseline-only line per protocol for a single metric."""
    cell = summary[(summary.method == CFG.BASELINE) & (summary.metric == metric)]
    protocols = [p for p in _active_protocols() if p in set(cell.protocol.unique())]
    if cell.empty or not protocols:
        return
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    lows, highs = [], []
    for protocol in protocols:
        line = cell[cell.protocol == protocol].sort_values("tau")
        x = line.tau.to_numpy(dtype=float)
        ax.plot(x, line["mean"], marker="o", markersize=5, linewidth=2,
                color=CFG.PROTOCOL_PALETTE.get(protocol),
                linestyle=CFG.PROTOCOL_LINESTYLES.get(protocol, "-"),
                label=CFG.PROTOCOL_LABELS.get(protocol, protocol), zorder=3)
        ax.fill_between(x, line.lo.to_numpy(dtype=float), line.hi.to_numpy(dtype=float),
                        color=CFG.PROTOCOL_PALETTE.get(protocol), alpha=0.13, linewidth=0)
        lows.extend(line.lo.tolist()); highs.extend(line.hi.tolist())
    ax.set_title(f"Baseline {CFG.METRIC_DISPLAY[metric][0]} across training protocols")
    ax.set_xlabel(r"Noise rate $\tau$"); ax.set_ylabel(CFG.METRIC_DISPLAY[metric][1])
    ax.set_xticks(CFG.TAUS); ax.set_xticklabels([f"{t:.1f}" for t in CFG.TAUS])
    _set_headroom_ylim(ax, lows, highs, floor=0.0, ceiling=1.0, pad_frac=0.10)
    ax.legend(frameon=False, ncol=min(2, len(protocols)), loc="best")
    fig.tight_layout()
    _savefig(fig, f"fig_baseline_protocol_overlay_{metric}")


def fig_baseline_metrics_by_protocol(summary: pd.DataFrame, protocol: str) -> None:
    """AP-style baseline degradation plot: three metric lines for one protocol."""
    cell = summary[(summary.method == CFG.BASELINE) & (summary.protocol == protocol)]
    metrics = [m for m in CFG.FIG_METRICS if m in set(cell.metric.unique())]
    if cell.empty or not metrics:
        return
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    metric_colors = {"BA": "#4c78a8", "MacroF1": "#f58518", "MacroAUC": "#54a24b"}
    lows, highs = [], []
    for metric in metrics:
        line = cell[cell.metric == metric].sort_values("tau")
        x = line.tau.to_numpy(dtype=float)
        ax.plot(x, line["mean"], marker="o", markersize=5, linewidth=2,
                color=metric_colors.get(metric), label=CFG.METRIC_DISPLAY[metric][0], zorder=3)
        ax.fill_between(x, line.lo.to_numpy(dtype=float), line.hi.to_numpy(dtype=float),
                        color=metric_colors.get(metric), alpha=0.13, linewidth=0)
        lows.extend(line.lo.tolist()); highs.extend(line.hi.tolist())
    ax.set_title(f"Baseline degradation - {CFG.PROTOCOL_LABELS.get(protocol, protocol)} ({protocol})")
    ax.set_xlabel(r"Noise rate $\tau$"); ax.set_ylabel("Score")
    ax.set_xticks(CFG.TAUS); ax.set_xticklabels([f"{t:.1f}" for t in CFG.TAUS])
    _set_headroom_ylim(ax, lows, highs, floor=0.0, ceiling=1.0, pad_frac=0.10)
    ax.legend(frameon=False, ncol=1, loc="best")
    fig.tight_layout()
    _savefig(fig, f"fig_baseline_metrics_{protocol}")


def fig_advantage_focus(mvb: pd.DataFrame) -> None:
    if mvb.empty:
        return
    protocols = [p for p in _active_protocols() if p in set(mvb.protocol.unique())]
    methods = _robust_methods()
    metrics = [m for m in CFG.FIG_METRICS if m in set(mvb.metric.unique())]
    if not protocols or not methods or not metrics:
        return
    _style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.0 * len(metrics), 5.2), squeeze=False)
    axes = axes[0]
    x = np.arange(len(protocols))
    for ax, metric in zip(axes, metrics):
        lows, highs = [], []
        for method in methods:
            cell = mvb[(mvb.metric == metric) & (mvb.method == method)
                       & np.isclose(mvb.tau, CFG.FOCUS_TAU)].set_index("protocol").reindex(protocols)
            mean = cell.delta.to_numpy(dtype=float)
            lo = cell.delta_ci_lo.to_numpy(dtype=float)
            hi = cell.delta_ci_hi.to_numpy(dtype=float)
            lows.extend(lo[np.isfinite(lo)]); highs.extend(hi[np.isfinite(hi)])
            yerr = np.vstack([np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)])
            ax.errorbar(x, mean, yerr=yerr, marker="o", markersize=5,
                        color=CFG.PALETTE.get(method), capsize=3, linewidth=1.8,
                        label=CFG.METHOD_LABELS.get(method, method), zorder=3)
        ax.axhline(0, color="0.35", linewidth=0.9, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels(protocols)
        ax.set_xlabel("Training protocol"); ax.set_ylabel(r"$\Delta$ vs. own baseline")
        ax.set_title(CFG.METRIC_DISPLAY[metric][0])
        _set_headroom_ylim(ax, lows + [0.0], highs + [0.0], pad_frac=0.15)
    handles = [plt.Line2D([0], [0], color=CFG.PALETTE.get(m), marker="o",
                          label=CFG.METHOD_LABELS.get(m, m)) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(rf"Method advantage across protocols ($\tau = {CFG.FOCUS_TAU:.2f}$)",
                 y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0.08, 1, 0.955])
    _savefig(fig, f"fig_advantage_focus_tau{int(round(CFG.FOCUS_TAU * 100)):02d}")

# LaTeX tables: aggregate, delta, ranking, best-next, interaction
def _summary_cell(summary: pd.DataFrame, mvb: pd.DataFrame, protocol: str,
                  metric: str, method: str, tau: float, best_method: str) -> str:
    row = summary[(summary.protocol == protocol) & (summary.metric == metric)
                  & (summary.method == method) & np.isclose(summary.tau, tau)]
    if row.empty or pd.isna(row.iloc[0]["mean"]):
        return r"\makecell{--}"
    r = row.iloc[0]
    sig = ""
    if method != CFG.BASELINE and not mvb.empty:
        st = mvb[(mvb.protocol == protocol) & (mvb.metric == metric)
                 & (mvb.method == method) & np.isclose(mvb.tau, tau)]
        if len(st):
            sig = str(st.iloc[0].sig)
            if sig == CFG.NS_SYMBOL:
                sig = ""
    mean = _fmt_metric(r["mean"])
    value = rf"\mathbf{{{mean}}}" if method == best_method else mean
    sup = rf"^{{{sig}}}" if sig else ""
    return rf"\makecell{{${value}{sup}$\\{{\scriptsize $({_fmt_metric(r.lo)},\,{_fmt_metric(r.hi)})$}}}}"


def emit_protocol_body_table(summary: pd.DataFrame, mvb: pd.DataFrame, metric: str) -> None:
    """Emit a compact focus-rate body table and a complete appendix table."""
    protocols = [p for p in _active_protocols() if p in set(summary.protocol.unique())]
    methods = _active_methods()

    def _best_at(protocol: str, tau: float) -> str:
        cell = summary[(summary.protocol == protocol) & (summary.metric == metric)
                       & np.isclose(summary.tau, tau) & summary.method.isin(methods)].dropna(subset=["mean"])
        return str(cell.loc[cell["mean"].idxmax(), "method"]) if len(cell) else ""

    # Compact focus-rate body table.
    tau = CFG.FOCUS_TAU
    best = {p: _best_at(p, tau) for p in protocols}
    rows = []
    for method in methods:
        cells = [_summary_cell(summary, mvb, p, metric, method, tau, best[p]) for p in protocols]
        rows.append(" & ".join([CFG.METHOD_LABELS.get(method, method), *cells]) + r" \\")
    caption = (
        f"{CFG.METRIC_DISPLAY[metric][0]} across training protocols at $\\tau={tau:.2f}$. Cells give "
        f"the mean over folds with the 95\\% bootstrap confidence interval below. Stars mark a "
        f"significant method-vs-own-protocol-baseline difference (paired Wilcoxon by fold, "
        f"Holm-corrected across $\\tau$ within each method and protocol: $^{{*}}p<.05$, "
        f"$^{{**}}p<.01$, $^{{***}}p<.001$; no star = n.s.). The best included method per "
        f"protocol is in bold. The complete noise-rate grid is reported in the appendix.")
    tex = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}",
           rf"\label{{tab:protocol-body-{metric.lower()}}}", r"\resizebox{\textwidth}{!}{%",
           rf"\begin{{tabular}}{{l{'c' * len(protocols)}}}", r"\toprule",
           " & ".join(["Method"] + [CFG.PROTOCOL_LABELS.get(p, p) for p in protocols]) + r" \\",
           r"\midrule", *rows, r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _write_tex(f"tab_protocol_body_{metric}", "\n".join(tex))

    # Complete appendix table: all noise rates, breakable over pages.
    rows = []
    for tau in CFG.TAUS:
        best = {p: _best_at(p, tau) for p in protocols}
        for idx, method in enumerate(methods):
            cells = [_summary_cell(summary, mvb, p, metric, method, tau, best[p]) for p in protocols]
            rows.append(" & ".join([f"{tau:.1f}" if idx == 0 else "",
                                     CFG.METHOD_LABELS.get(method, method), *cells]) + r" \\")
        rows.append(r"\addlinespace")
    caption = (
        f"Complete {CFG.METRIC_DISPLAY[metric][0].lower()} grid across training protocols. Cells give "
        f"the fold mean with the 95\\% bootstrap confidence interval below. Stars mark the "
        f"Holm-corrected method-vs-own-protocol-baseline result; the best included method per "
        f"protocol and $\\tau$ is in bold.")
    header = " & ".join([r"$\tau$", "Method"] +
                        [CFG.PROTOCOL_LABELS.get(p, p) for p in protocols]) + r" \\"
    tex = [r"\begin{small}", r"\setlength{\LTcapwidth}{\textwidth}",
           rf"\begin{{longtable}}{{ll{'c' * len(protocols)}}}",
           rf"\caption{{{caption}}}\label{{tab:app-protocol-full-{metric.lower()}}}\\",
           r"\toprule", header, r"\midrule", r"\endfirsthead", r"\toprule", header,
           r"\midrule", r"\endhead", *rows[:-1], r"\bottomrule", r"\end{longtable}", r"\end{small}"]
    _write_tex(f"tab_app_protocol_full_{metric}", "\n".join(tex))

def emit_delta_grid(df: pd.DataFrame, stem: str, caption: str, label: str) -> None:
    protocols = [p for p in _active_protocols() if p in set(df.protocol.unique())]
    rows = []
    for metric in CFG.TABLE_METRICS:
        first = True
        for method in _robust_methods():
            cells = []
            for protocol in protocols:
                cell = df[(df.protocol == protocol) & (df.metric == metric) & (df.method == method)]
                if cell.empty:
                    cells.append("--")
                else:
                    r = cell.iloc[0]
                    cells.append(f"{_fmt_signed(r.delta)} ({_fmt_pct(r.rel_pct)})")
            rows.append(" & ".join([
                CFG.METRIC_DISPLAY[metric][0] if first else "",
                CFG.METHOD_LABELS.get(method, method), *cells]) + r" \\"
            )
            first = False
        rows.append(r"\addlinespace")
    tex = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}",
           rf"\label{{{label}}}", r"\resizebox{\textwidth}{!}{%",
           rf"\begin{{tabular}}{{ll{'c' * len(protocols)}}}",
           r"\toprule", " & ".join(["Metric", "Method"] +
                                     [CFG.PROTOCOL_LABELS.get(p, p) for p in protocols]) + r" \\",
           r"\midrule", *rows[:-1], r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _write_tex(stem, "\n".join(tex))


def emit_delta_full(full: pd.DataFrame) -> None:
    protocols = [p for p in _active_protocols() if p in set(full.protocol.unique())]
    rows = []
    ncols = 2 + len(protocols)
    for metric in CFG.TABLE_METRICS:
        rows.append(r"\addlinespace")
        rows.append(rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{{CFG.METRIC_DISPLAY[metric][0]}}}}} \\")
        for method in _robust_methods():
            first = True
            for tau in CFG.TAUS:
                cells = []
                for p in protocols:
                    cell = full[(full.protocol == p) & (full.metric == metric)
                                & (full.method == method) & np.isclose(full.tau, tau)]
                    cells.append(_fmt_signed(cell.iloc[0].delta) if len(cell) else "--")
                rows.append(" & ".join([CFG.METHOD_LABELS.get(method, method) if first else "",
                                        f"{tau:.1f}", *cells]) + r" \\"
                            )
                first = False
            rows.append(r"\addlinespace")
    tex = [r"\begin{small}", r"\setlength{\LTcapwidth}{\textwidth}",
           rf"\begin{{longtable}}{{ll{'c' * len(protocols)}}}",
           r"\caption{Absolute difference to each protocol's own baseline "
           r"($\Delta=\text{method}-\text{baseline}$), by method, noise rate and training protocol. "
           r"Positive values indicate that the robust method outperforms its baseline. "
           r"Within-protocol significance is reported in the protocol score tables.}"
           r"\label{tab:app-protocol-delta-full}\\",
           r"\toprule", " & ".join(["Method", r"$\tau$"] +
                                      [CFG.PROTOCOL_LABELS.get(p, p) for p in protocols]) + r" \\",
           r"\midrule", r"\endfirsthead", r"\toprule",
           " & ".join(["Method", r"$\tau$"] +
                      [CFG.PROTOCOL_LABELS.get(p, p) for p in protocols]) + r" \\",
           r"\midrule", r"\endhead", *rows, r"\bottomrule", r"\end{longtable}", r"\end{small}"]
    _write_tex("tab_app_delta_full", "\n".join(tex))


def emit_ranking_focus(ranking: pd.DataFrame) -> None:
    focus = ranking[np.isclose(ranking.tau, CFG.FOCUS_TAU)]
    rows = []
    for metric in CFG.TABLE_METRICS:
        rows.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{CFG.METRIC_DISPLAY[metric][0]}}}}} \\")
        for protocol in [p for p in _active_protocols() if p in set(focus.protocol.unique())]:
            cell = focus[(focus.protocol == protocol) & (focus.metric == metric)].sort_values("rank")
            winner = cell.iloc[0] if len(cell) else None
            nxt = cell.iloc[1] if len(cell) > 1 else None
            rows.append(" & ".join([
                CFG.PROTOCOL_LABELS.get(protocol, protocol),
                CFG.METHOD_LABELS.get(winner.method, winner.method) if winner is not None else "--",
                _fmt_metric(winner["mean"]) if winner is not None else "--",
                CFG.METHOD_LABELS.get(nxt.method, nxt.method) if nxt is not None else "--",
                _fmt_metric(nxt["mean"]) if nxt is not None else "--",
            ]) + r" \\"
            )
        rows.append(r"\addlinespace")
    tex = [r"\begin{table}[htbp]", r"\centering",
           rf"\caption{{Ranking stability at $\tau={CFG.FOCUS_TAU:.2f}$. The winner and next-best "
           r"included method are defined descriptively by their mean score across folds. Baseline is "
           r"included in the ranking so the table reveals when no robust method wins.}",
           r"\label{tab:protocol-ranking-focus}", r"\begin{tabular}{lcccc}", r"\toprule",
           r"Protocol & Winner & Mean & Next best & Mean \\", r"\midrule", *rows[:-1],
           r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_tex(f"tab_ranking_focus_tau{int(round(CFG.FOCUS_TAU * 100)):02d}", "\n".join(tex))


def _stats_sig_cell(r) -> str:
    sig = str(r.sig) if hasattr(r, "sig") else CFG.NS_SYMBOL
    return "n.s." if sig == CFG.NS_SYMBOL else sig


def emit_best_next_focus(best_next_df: pd.DataFrame) -> None:
    focus = best_next_df[np.isclose(best_next_df.tau, CFG.FOCUS_TAU)]
    rows = []
    for metric in CFG.TABLE_METRICS:
        rows.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{CFG.METRIC_DISPLAY[metric][0]}}}}} \\")
        for protocol in [p for p in _active_protocols() if p in set(focus.protocol.unique())]:
            cell = focus[(focus.protocol == protocol) & (focus.metric == metric)]
            if cell.empty:
                continue
            r = cell.iloc[0]
            rows.append(" & ".join([
                CFG.PROTOCOL_LABELS.get(protocol, protocol),
                f"{CFG.METHOD_LABELS.get(r.best_method, r.best_method)} vs. {CFG.METHOD_LABELS.get(r.next_method, r.next_method)}",
                _fmt_signed(r.delta), f"${_fmt_ci(r.delta_ci_lo, r.delta_ci_hi)}$",
                (f"{r.r_rb:+.2f}" if not pd.isna(r.r_rb) else "--"), _fmt_W(r.W),
                _fmt_p(r.p_raw), _fmt_p(r.p_holm), _stats_sig_cell(r),
            ]) + r" \\"
            )
        rows.append(r"\addlinespace")
    caption = (
        rf"Exploratory paired comparison of the descriptively best and next-best included method at "
        rf"$\tau={CFG.FOCUS_TAU:.2f}$. $\Delta$ is winner minus runner-up; $p_{{\mathrm{{Holm}}}}$ is "
        rf"corrected across $\tau$ within each protocol and metric. Because the pair is selected from "
        rf"the observed means, use this table to characterize separation, not as a pre-specified "
        rf"confirmatory claim.")
    tex = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}",
           r"\label{tab:best-next-focus}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{llrrrrrrl}", r"\toprule",
           r"Protocol & Pair & $\Delta$ & 95\% CI & $r$ & $W$ & $p_{\mathrm{raw}}$ & $p_{\mathrm{Holm}}$ & sig. \\",
           r"\midrule", *rows[:-1], r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _write_tex(f"tab_best_vs_next_focus_tau{int(round(CFG.FOCUS_TAU * 100)):02d}", "\n".join(tex))


def emit_best_next_full(best_next_df: pd.DataFrame) -> None:
    rows = []
    for metric in CFG.TABLE_METRICS:
        rows.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{CFG.METRIC_DISPLAY[metric][0]}}}}} \\")
        for protocol in [p for p in _active_protocols() if p in set(best_next_df.protocol.unique())]:
            cell = best_next_df[(best_next_df.protocol == protocol) & (best_next_df.metric == metric)].sort_values("tau")
            for _, r in cell.iterrows():
                rows.append(" & ".join([
                    protocol, f"{r.tau:.1f}",
                    f"{CFG.METHOD_LABELS.get(r.best_method, r.best_method)} vs. {CFG.METHOD_LABELS.get(r.next_method, r.next_method)}",
                    _fmt_signed(r.delta), f"${_fmt_ci(r.delta_ci_lo, r.delta_ci_hi)}$",
                    (f"{r.r_rb:+.2f}" if not pd.isna(r.r_rb) else "--"), _fmt_W(r.W),
                    _fmt_p(r.p_raw), _fmt_p(r.p_holm), _stats_sig_cell(r),
                ]) + r" \\"
                )
        rows.append(r"\addlinespace")
    tex = [r"\begin{landscape}", r"\begin{scriptsize}", r"\setlength{\tabcolsep}{3pt}",
           r"\setlength{\LTcapwidth}{\linewidth}", r"\begin{longtable}{lllrrrrrrl}",
           r"\caption{Exploratory paired comparison of the descriptively best and next-best included "
           r"method for every protocol, metric and noise rate. The pair is selected from observed means; "
           r"interpret the tests as separation diagnostics rather than pre-specified confirmatory tests.}"
           r"\label{tab:app-best-next-full}\\", r"\toprule",
           r"Protocol & $\tau$ & Pair & $\Delta$ & 95\% CI & $r$ & $W$ & $p_{\mathrm{raw}}$ & $p_{\mathrm{Holm}}$ & sig. \\",
           r"\midrule", r"\endfirsthead", r"\toprule",
           r"Protocol & $\tau$ & Pair & $\Delta$ & 95\% CI & $r$ & $W$ & $p_{\mathrm{raw}}$ & $p_{\mathrm{Holm}}$ & sig. \\",
           r"\midrule", r"\endhead", *rows, r"\bottomrule", r"\end{longtable}",
           r"\end{scriptsize}", r"\end{landscape}"]
    _write_tex("tab_app_best_vs_next_full", "\n".join(tex))


def emit_interaction_focus(interactions: pd.DataFrame) -> None:
    if interactions.empty:
        return
    focus = interactions[np.isclose(interactions.tau, CFG.FOCUS_TAU)]
    rows = []
    for metric in CFG.TABLE_METRICS:
        rows.append(rf"\multicolumn{{11}}{{l}}{{\textit{{{CFG.METRIC_DISPLAY[metric][0]}}}}} \\")
        sub = focus[focus.metric == metric]
        for _, r in sub.iterrows():
            rows.append(" & ".join([
                f"{r.protocol_1} - {r.protocol_2}", CFG.METHOD_LABELS.get(r.method, r.method),
                _fmt_signed(r.advantage_p1), _fmt_signed(r.advantage_p2),
                _fmt_signed(r.delta), f"${_fmt_ci(r.delta_ci_lo, r.delta_ci_hi)}$",
                (f"{r.r_rb:+.2f}" if not pd.isna(r.r_rb) else "--"), _fmt_W(r.W),
                _fmt_p(r.p_raw), _fmt_p(r.p_holm), _stats_sig_cell(r),
            ]) + r" \\"
            )
        rows.append(r"\addlinespace")
    caption = (
        rf"Cross-protocol difference-of-differences at $\tau={CFG.FOCUS_TAU:.2f}$. For each method, "
        rf"$\Delta_{{P}}$ is method minus the same protocol's baseline and the interaction is "
        rf"$\Delta_{{P_1}}-\Delta_{{P_2}}$. Positive values indicate a larger method advantage in "
        rf"$P_1$. Tests are paired by fold; $p_{{\mathrm{{Holm}}}}$ is corrected across $\tau$ within "
        rf"each protocol-pair, method and metric family.")
    tex = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}",
           r"\label{tab:interaction-focus}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{llrrrrrrrrl}", r"\toprule",
           r"Contrast & Method & $\Delta_{P_1}$ & $\Delta_{P_2}$ & interaction & 95\% CI & $r$ & $W$ & $p_{\mathrm{raw}}$ & $p_{\mathrm{Holm}}$ & sig. \\",
           r"\midrule", *rows[:-1], r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _write_tex(f"tab_interaction_focus_tau{int(round(CFG.FOCUS_TAU * 100)):02d}", "\n".join(tex))


def emit_interaction_full(interactions: pd.DataFrame) -> None:
    if interactions.empty:
        return
    rows = []
    for metric in CFG.TABLE_METRICS:
        rows.append(rf"\multicolumn{{12}}{{l}}{{\textit{{{CFG.METRIC_DISPLAY[metric][0]}}}}} \\")
        sub = interactions[interactions.metric == metric]
        for _, r in sub.sort_values(["protocol_1", "protocol_2", "method", "tau"]).iterrows():
            rows.append(" & ".join([
                f"{r.protocol_1} - {r.protocol_2}", CFG.METHOD_LABELS.get(r.method, r.method),
                f"{r.tau:.1f}", _fmt_signed(r.advantage_p1), _fmt_signed(r.advantage_p2),
                _fmt_signed(r.delta), f"${_fmt_ci(r.delta_ci_lo, r.delta_ci_hi)}$",
                (f"{r.r_rb:+.2f}" if not pd.isna(r.r_rb) else "--"), _fmt_W(r.W),
                _fmt_p(r.p_raw), _fmt_p(r.p_holm), _stats_sig_cell(r),
            ]) + r" \\"
            )
        rows.append(r"\addlinespace")
    tex = [r"\begin{landscape}", r"\begin{scriptsize}", r"\setlength{\tabcolsep}{2.5pt}",
           r"\setlength{\LTcapwidth}{\linewidth}", r"\begin{longtable}{lllrrrrrrrrl}",
           r"\caption{Cross-protocol difference-of-differences tests. For each method, "
           r"$\Delta_P=\text{method}-\text{baseline}$ within protocol $P$ and the interaction is "
           r"$\Delta_{P_1}-\Delta_{P_2}$. Tests are paired by fold; Holm correction is across $\tau$ "
           r"within each protocol-pair, method and metric family.}"
           r"\label{tab:app-interaction-full}\\", r"\toprule",
           r"Contrast & Method & $\tau$ & $\Delta_{P_1}$ & $\Delta_{P_2}$ & interaction & 95\% CI & $r$ & $W$ & $p_{\mathrm{raw}}$ & $p_{\mathrm{Holm}}$ & sig. \\",
           r"\midrule", r"\endfirsthead", r"\toprule",
           r"Contrast & Method & $\tau$ & $\Delta_{P_1}$ & $\Delta_{P_2}$ & interaction & 95\% CI & $r$ & $W$ & $p_{\mathrm{raw}}$ & $p_{\mathrm{Holm}}$ & sig. \\",
           r"\midrule", r"\endhead", *rows, r"\bottomrule", r"\end{longtable}",
           r"\end{scriptsize}", r"\end{landscape}"]
    _write_tex("tab_app_interaction_full", "\n".join(tex))



def emit_protocol_comparison_full(pc: pd.DataFrame) -> None:
    """Full cross-protocol comparison table (landscape longtable)."""
    if pc.empty:
        return
    rows = []
    for metric in CFG.TABLE_METRICS:
        sub = pc[pc.metric == metric]
        if sub.empty:
            continue
        rows.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{CFG.METRIC_DISPLAY[metric][0]}}}}} \\")
        for _, r in sub.sort_values(["method", "tau", "protocol_1", "protocol_2"]).iterrows():
            rows.append(" & ".join([
                CFG.METHOD_LABELS.get(r.method, r.method),
                f"{r.tau:.1f}",
                f"{r.protocol_1} vs.\\ {r.protocol_2}",
                _fmt_signed(r.delta),
                f"${_fmt_ci(r.delta_ci_lo, r.delta_ci_hi)}$",
                (f"{r.r_rb:+.2f}" if not pd.isna(r.r_rb) else "--"),
                _fmt_W(r.W), _fmt_p(r.p_raw), _fmt_p(r.p_holm),
                _stats_sig_cell(r),
            ]) + r" \\")
        rows.append(r"\addlinespace")
    head = (r"Method & $\tau$ & Protocols & $\Delta$ & 95\% CI & $r$ & $W$ & "
            r"$p_{\mathrm{raw}}$ & $p_{\mathrm{Holm}}$ & sig. \\")
    tex = [
        r"\begin{landscape}", r"\begin{scriptsize}",
        r"\setlength{\tabcolsep}{3pt}", r"\setlength{\LTcapwidth}{\linewidth}",
        r"\begin{longtable}{lllrrrrrrl}",
        r"\caption{Cross-protocol comparison of each method's performance. For a "
        r"fixed method, $\Delta=\text{score}(P_1)-\text{score}(P_2)$ is the "
        r"per-fold difference between two training protocols (paired by fold, as "
        r"the folds are identical across protocols). Holm correction is across the "
        r"six protocol-pairs within each method, metric and noise rate. The method "
        r"is pre-specified, so this family is confirmatory.}"
        r"\label{tab:app-protocol-comparison-full}\\",
        r"\toprule", head, r"\midrule", r"\endfirsthead",
        r"\toprule", head, r"\midrule", r"\endhead",
        *rows, r"\bottomrule", r"\end{longtable}",
        r"\end{scriptsize}", r"\end{landscape}",
    ]
    _write_tex("tab_app_protocol_comparison_full", "\n".join(tex))


def emit_protocol_comparison_focus(pc: pd.DataFrame) -> None:
    """Focus table at FOCUS_TAU: all methods and protocol-pairs (one tau)."""
    if pc.empty:
        return
    sub_tau = pc[np.isclose(pc.tau, CFG.FOCUS_TAU)]
    if sub_tau.empty:
        return
    rows = []
    for metric in CFG.TABLE_METRICS:
        sub = sub_tau[sub_tau.metric == metric]
        if sub.empty:
            continue
        rows.append(r"\addlinespace")
        rows.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{CFG.METRIC_DISPLAY[metric][0]}}}}} \\")
        for _, r in sub.sort_values(["method", "protocol_1", "protocol_2"]).iterrows():
            rows.append(" & ".join([
                CFG.METHOD_LABELS.get(r.method, r.method),
                f"{r.protocol_1} vs.\\ {r.protocol_2}",
                _fmt_signed(r.delta),
                f"${_fmt_ci(r.delta_ci_lo, r.delta_ci_hi)}$",
                (f"{r.r_rb:+.2f}" if not pd.isna(r.r_rb) else "--"),
                _fmt_p(r.p_holm), _stats_sig_cell(r),
            ]) + r" \\")
    head = (r"Method & Protocols & $\Delta$ & 95\% CI & $r$ & "
            r"$p_{\mathrm{Holm}}$ & sig. \\")
    caption = (
        f"Cross-protocol comparison at $\\tau={CFG.FOCUS_TAU:.2f}$. For each "
        f"method, the per-fold score is compared between every pair of training "
        f"protocols ($\\Delta=\\text{{score}}(P_1)-\\text{{score}}(P_2)$, paired by "
        f"fold), Holm-corrected across the six protocol-pairs within each method "
        f"and metric. The method is pre-specified, so this family is confirmatory. "
        f"The full sweep over all noise rates is in "
        f"Table~\\ref{{tab:app-protocol-comparison-full}}."
    )
    tex = [
        r"\begin{table}[h!]", r"\centering",
        rf"\caption{{{caption}}}", r"\label{tab:protocol-comparison-focus}",
        r"\begin{tabular}{llrrrrl}", r"\toprule", head, r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    _write_tex("tab_protocol_comparison_focus", "\n".join(tex))


# mechanism sensitivity: final-epoch NTA / LNMR from raw_fold_results.csv
def _raw_method_to_logical(value: str) -> str:
    value = str(value)
    inverse = {folder: logical for logical, folder in CFG.METHOD_DIRS.items()}
    if value in CFG.METHOD_DIRS:
        return value
    return inverse.get(value, value)


def load_mechanism_raw(metric_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol in available_protocols(metric_df):
        fp = _protocol_root(protocol) / CFG.RAW_FOLD_CSV
        if not fp.exists():
            print(f"[mechanism] skip {protocol}: raw fold CSV not found: {fp}")
            continue
        raw = pd.read_csv(fp)
        # Combined files may carry protocol columns. Filter them if present.
        init = "pretrained" if "pretrained" in CFG.PROTOCOL_DIRS[protocol] else "scratch"
        optim = "adam" if "adam" in CFG.PROTOCOL_DIRS[protocol] else "sgd"
        for col, val in (("init", init), ("optim", optim), ("dataset", CFG.DATASET)):
            if col in raw.columns:
                raw = raw[raw[col].astype(str).str.lower() == val]
        if "method" not in raw.columns or "tau" not in raw.columns:
            print(f"[mechanism] skip {protocol}: required columns method/tau absent in {fp}")
            continue
        if "nta" not in raw.columns or "lnmr" not in raw.columns:
            print(f"[mechanism] skip {protocol}: required columns nta/lnmr absent in {fp}")
            continue
        raw = raw.copy()
        raw["method"] = raw["method"].map(_raw_method_to_logical)
        raw = raw[raw.method.isin(_active_methods())]
        raw["protocol"] = protocol
        if "fold" not in raw.columns:
            # synthetic row IDs when fold is absent
            raw["fold"] = raw.groupby(["method", "tau"]).cumcount()
        keep = ["protocol", "method", "tau", "fold", "nta", "lnmr"]
        rows.extend(raw[keep].to_dict("records"))
        print(f"[mechanism] loaded {len(raw)} rows for {protocol} from {fp}")
    if not rows:
        return pd.DataFrame(columns=["protocol", "method", "tau", "fold", "nta", "lnmr"])
    out = pd.DataFrame(rows)
    out["residual"] = 1.0 - out["nta"].astype(float) - out["lnmr"].astype(float)
    return out.sort_values(["protocol", "method", "tau", "fold"]).reset_index(drop=True)


def summarize_mechanism(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if raw.empty:
        return pd.DataFrame()
    for (protocol, method, tau), g in raw.groupby(["protocol", "method", "tau"]):
        rec = dict(protocol=protocol, method=method, tau=float(tau), n=int(len(g)))
        for col in ("nta", "lnmr", "residual"):
            mean, lo, hi = _boot_ci(g[col].values, _seed_for("mech", protocol, method, tau, col))
            rec[col] = mean; rec[f"{col}_lo"] = lo; rec[f"{col}_hi"] = hi
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["protocol", "method", "tau"]).reset_index(drop=True)


def fig_mechanism_protocol(mech: pd.DataFrame, metric: str, label: str) -> None:
    if mech.empty:
        return
    protocols = [p for p in _active_protocols() if p in set(mech.protocol.unique())]
    if not protocols:
        return
    _style()
    ncol = min(2, len(protocols)); nrow = int(np.ceil(len(protocols) / ncol))
    taus = [t for t in CFG.TAUS if t > 0]
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 4.9 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    global_lows = mech.loc[mech.tau.isin(taus), f"{metric}_lo"].to_numpy(dtype=float)
    global_highs = mech.loc[mech.tau.isin(taus), f"{metric}_hi"].to_numpy(dtype=float)
    _, global_hi = _finite_bounds(global_lows, global_highs)
    global_top = max(0.05, global_hi + 0.10 * max(global_hi, 0.05))
    for idx, protocol in enumerate(protocols):
        ax = axes[idx // ncol, idx % ncol]
        for method in _active_methods():
            cell = mech[(mech.protocol == protocol) & (mech.method == method)
                        & mech.tau.isin(taus)].sort_values("tau")
            if cell.empty:
                continue
            x = cell.tau.to_numpy(dtype=float)
            ax.plot(x, cell[metric], "-o", color=CFG.PALETTE.get(method), markersize=4,
                    label=CFG.METHOD_LABELS.get(method, method), zorder=3)
            ax.fill_between(x, cell[f"{metric}_lo"].to_numpy(dtype=float),
                            cell[f"{metric}_hi"].to_numpy(dtype=float),
                            color=CFG.PALETTE.get(method), alpha=0.18, linewidth=0, zorder=2)
        ax.set_title(CFG.PROTOCOL_LABELS.get(protocol, protocol) + f"  ({protocol})")
        ax.set_xlabel(r"Noise rate $\tau$")
        ax.set_ylabel(label)
        ax.set_xticks(taus)
        ax.set_xticklabels([f"{t:.1f}" for t in taus])
        ax.set_ylim(0.0, global_top)
    for idx in range(len(protocols), nrow * ncol):
        axes[idx // ncol, idx % ncol].axis("off")
    handles = [Patch(facecolor=CFG.PALETTE.get(m), label=CFG.METHOD_LABELS.get(m, m))
               for m in _active_methods()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(f"Final-epoch {label} across training protocols", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0.075, 1, 0.955])
    _savefig(fig, f"fig_mechanism_protocol_{metric.upper()}")



def fig_nta_lnmr_by_protocol(mech: pd.DataFrame, protocol: str) -> None:
    """Mirror the Part-5 AP two-panel NTA/LNMR-vs-tau figure for one protocol."""
    cell = mech[mech.protocol == protocol]
    if cell.empty:
        return
    taus = [t for t in CFG.TAUS if t > 0]
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    for ax, metric, label in zip(axes, ("nta", "lnmr"), ("NTA", "LNMR")):
        lows, highs = [], []
        for method in _active_methods():
            line = cell[(cell.method == method) & cell.tau.isin(taus)].sort_values("tau")
            if line.empty:
                continue
            x = line.tau.to_numpy(dtype=float)
            ax.plot(x, line[metric], "-o", color=CFG.PALETTE.get(method), markersize=4,
                    label=CFG.METHOD_LABELS.get(method, method), zorder=3)
            ax.fill_between(x, line[f"{metric}_lo"].to_numpy(dtype=float),
                            line[f"{metric}_hi"].to_numpy(dtype=float),
                            color=CFG.PALETTE.get(method), alpha=0.18, linewidth=0, zorder=2)
            lows.extend(line[f"{metric}_lo"].tolist()); highs.extend(line[f"{metric}_hi"].tolist())
        ax.set_title(label)
        ax.set_xlabel(r"Noise rate $\tau$"); ax.set_ylabel(label)
        ax.set_xticks(taus); ax.set_xticklabels([f"{t:.1f}" for t in taus])
        _set_headroom_ylim(ax, lows, highs, floor=0.0, pad_frac=0.10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)),
               frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(f"Memorization diagnostics across noise - {CFG.PROTOCOL_LABELS.get(protocol, protocol)} ({protocol})",
                 y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0.075, 1, 0.955])
    _savefig(fig, f"fig_nta_lnmr_{protocol}")

def fig_mechanism_focus(mech: pd.DataFrame) -> None:
    focus = mech[np.isclose(mech.tau, CFG.FOCUS_TAU)]
    if focus.empty:
        return
    protocols = [p for p in _active_protocols() if p in set(focus.protocol.unique())]
    methods = _active_methods()
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3))
    x = np.arange(len(protocols))
    for ax, metric, label in zip(axes, ("nta", "lnmr"), ("NTA", "LNMR")):
        axis_lows, axis_highs = [], []
        for method in methods:
            cell = focus[focus.method == method].set_index("protocol").reindex(protocols)
            mean = cell[metric].to_numpy(dtype=float)
            lo = cell[f"{metric}_lo"].to_numpy(dtype=float)
            hi = cell[f"{metric}_hi"].to_numpy(dtype=float)
            axis_lows.extend(lo[np.isfinite(lo)]); axis_highs.extend(hi[np.isfinite(hi)])
            ax.errorbar(x, mean, yerr=np.vstack([np.clip(mean - lo, 0, None),
                                                 np.clip(hi - mean, 0, None)]),
                        marker="o", markersize=4, color=CFG.PALETTE.get(method),
                        capsize=3, linewidth=1.5,
                        label=CFG.METHOD_LABELS.get(method, method), zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(protocols)
        ax.set_xlabel("Training protocol"); ax.set_ylabel(label); ax.set_title(label)
        _set_headroom_ylim(ax, axis_lows, axis_highs, floor=0.0, pad_frac=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=min(4, len(handles)),
               loc="lower center", bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(rf"Final-epoch memorization diagnostics ($\tau={CFG.FOCUS_TAU:.2f}$)",
                 y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0.075, 1, 0.955])
    _savefig(fig, f"fig_mechanism_focus_tau{int(round(CFG.FOCUS_TAU * 100)):02d}")


def emit_mechanism_focus(mech: pd.DataFrame) -> None:
    if mech.empty:
        return
    focus = mech[np.isclose(mech.tau, CFG.FOCUS_TAU)]
    protocols = [p for p in _active_protocols() if p in set(focus.protocol.unique())]
    rows = []
    for metric, label in (("nta", "NTA"), ("lnmr", "LNMR"), ("residual", "Residual")):
        first = True
        for method in _active_methods():
            cells = []
            for protocol in protocols:
                cell = focus[(focus.protocol == protocol) & (focus.method == method)]
                if cell.empty:
                    cells.append("--")
                else:
                    r = cell.iloc[0]
                    cells.append(rf"\makecell{{${_fmt_metric(r[metric])}$\\{{\scriptsize $({_fmt_metric(r[f'{metric}_lo'])},\,{_fmt_metric(r[f'{metric}_hi'])})$}}}}")
            rows.append(" & ".join([label if first else "", CFG.METHOD_LABELS.get(method, method), *cells]) + r" \\"
                        )
            first = False
        rows.append(r"\addlinespace")
    tex = [r"\begin{table}[htbp]", r"\centering",
           rf"\caption{{Final-epoch memorization diagnostics at $\tau={CFG.FOCUS_TAU:.2f}$ across "
           r"training protocols. Cells give the fold mean with 95\% bootstrap CI below. NTA is the "
           r"fraction of flipped samples predicted as their clean class, LNMR the fraction predicted "
           r"as the assigned noisy label, and residual the fraction predicted as neither.}",
           r"\label{tab:mechanism-focus}", r"\resizebox{\textwidth}{!}{%",
           rf"\begin{{tabular}}{{ll{'c' * len(protocols)}}}", r"\toprule",
           " & ".join(["Diagnostic", "Method"] + [CFG.PROTOCOL_LABELS.get(p, p) for p in protocols]) + r" \\",
           r"\midrule", *rows[:-1], r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _write_tex(f"tab_mechanism_focus_tau{int(round(CFG.FOCUS_TAU * 100)):02d}", "\n".join(tex))


def emit_mechanism_full(mech: pd.DataFrame) -> None:
    if mech.empty:
        return
    rows = []
    for protocol in [p for p in _active_protocols() if p in set(mech.protocol.unique())]:
        for method in _active_methods():
            cell = mech[(mech.protocol == protocol) & (mech.method == method) & (mech.tau > 0)].sort_values("tau")
            for _, r in cell.iterrows():
                rows.append(" & ".join([
                    protocol, CFG.METHOD_LABELS.get(method, method), f"{r.tau:.1f}",
                    rf"{_fmt_metric(r.nta)} $({_fmt_metric(r.nta_lo)},\,{_fmt_metric(r.nta_hi)})$",
                    rf"{_fmt_metric(r.lnmr)} $({_fmt_metric(r.lnmr_lo)},\,{_fmt_metric(r.lnmr_hi)})$",
                    rf"{_fmt_metric(r.residual)} $({_fmt_metric(r.residual_lo)},\,{_fmt_metric(r.residual_hi)})$",
                ]) + r" \\"
                )
    tex = [r"\begin{small}", r"\setlength{\LTcapwidth}{\textwidth}",
           r"\begin{longtable}{lllccc}",
           r"\caption{Final-epoch memorization diagnostics across protocols and noise rates. Values are "
           r"fold means with 95\% bootstrap confidence intervals in parentheses.}"
           r"\label{tab:app-mechanism-full}\\", r"\toprule",
           r"Protocol & Method & $\tau$ & NTA & LNMR & Residual \\", r"\midrule", r"\endfirsthead",
           r"\toprule", r"Protocol & Method & $\tau$ & NTA & LNMR & Residual \\", r"\midrule", r"\endhead",
           *rows, r"\bottomrule", r"\end{longtable}", r"\end{small}"]
    _write_tex("tab_app_mechanism_full", "\n".join(tex))


# optional epoch-trajectory analysis
def _read_epoch_log(protocol: str, method: str, tau: float, fold: int) -> list[dict]:
    fp = _log_path(protocol, method, tau, fold)
    if not fp.exists():
        return []
    rows = []
    with open(fp, "r") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            diag = rec.get(CFG.TRAIN_DIAG_KEY)
            if not isinstance(diag, dict) or diag.get("nta") is None or diag.get("lnmr") is None:
                continue
            if rec.get("epoch") is None:
                continue
            rows.append(dict(protocol=protocol, method=method, tau=float(tau), fold=int(fold),
                             epoch=int(rec["epoch"]), nta=float(diag["nta"]),
                             lnmr=float(diag["lnmr"])))
    return rows


def load_epoch_trajectories(metric_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol in available_protocols(metric_df):
        for method in _active_methods():
            for tau in CFG.EPOCH_TAUS:
                for fold in range(CFG.N_FOLDS):
                    rows.extend(_read_epoch_log(protocol, method, tau, fold))
    if not rows:
        print("[epoch] no training-log diagnostics found; epoch-trajectory outputs skipped.")
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    print(f"[epoch] loaded {len(out)} diagnostic checkpoint rows.")
    return out.sort_values(["protocol", "method", "tau", "fold", "epoch"]).reset_index(drop=True)


def summarize_epoch(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if raw.empty:
        return pd.DataFrame()
    for (protocol, method, tau, epoch), g in raw.groupby(["protocol", "method", "tau", "epoch"]):
        rec = dict(protocol=protocol, method=method, tau=float(tau), epoch=int(epoch), n=int(len(g)))
        for col in ("nta", "lnmr"):
            mean, lo, hi = _boot_ci(g[col].values, _seed_for("epoch", protocol, method, tau, epoch, col))
            rec[col] = mean; rec[f"{col}_lo"] = lo; rec[f"{col}_hi"] = hi
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["protocol", "method", "tau", "epoch"]).reset_index(drop=True)


def fig_epoch_protocol(epoch: pd.DataFrame, metric: str, label: str, tau: float) -> None:
    cell = epoch[np.isclose(epoch.tau, tau)]
    if cell.empty:
        return
    protocols = [p for p in _active_protocols() if p in set(cell.protocol.unique())]
    _style()
    ncol = min(2, len(protocols)); nrow = int(np.ceil(len(protocols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 4.9 * nrow), squeeze=False, sharey=True)
    global_lows = cell[f"{metric}_lo"].to_numpy(dtype=float)
    global_highs = cell[f"{metric}_hi"].to_numpy(dtype=float)
    _, global_hi = _finite_bounds(global_lows, global_highs)
    global_top = max(0.05, global_hi + 0.10 * max(global_hi, 0.05))
    for idx, protocol in enumerate(protocols):
        ax = axes[idx // ncol, idx % ncol]
        pcell = cell[cell.protocol == protocol]
        epochs_present = sorted(int(x) for x in pcell.epoch.unique())
        for method in _active_methods():
            line = pcell[pcell.method == method].sort_values("epoch")
            if line.empty:
                continue
            x = line.epoch.to_numpy(dtype=float)
            ax.plot(x, line[metric], "-o", color=CFG.PALETTE.get(method), markersize=4,
                    label=CFG.METHOD_LABELS.get(method, method), zorder=3)
            ax.fill_between(x, line[f"{metric}_lo"].to_numpy(dtype=float),
                            line[f"{metric}_hi"].to_numpy(dtype=float),
                            color=CFG.PALETTE.get(method), alpha=0.18, linewidth=0, zorder=2)
        ax.set_title(CFG.PROTOCOL_LABELS.get(protocol, protocol) + f"  ({protocol})")
        ax.set_xlabel("Epoch"); ax.set_ylabel(label); ax.set_ylim(0.0, global_top)
        if epochs_present:
            ax.set_xticks(epochs_present)
            ax.set_xticklabels([str(x) for x in epochs_present])
            ax.set_xlim(left=epochs_present[0])
            ax.spines["left"].set_position(("data", epochs_present[0]))
    for idx in range(len(protocols), nrow * ncol):
        axes[idx // ncol, idx % ncol].axis("off")
    handles = [Patch(facecolor=CFG.PALETTE.get(m), label=CFG.METHOD_LABELS.get(m, m))
               for m in _active_methods()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(rf"{label} over training across protocols ($\tau={tau:.2f}$)",
                 y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0.075, 1, 0.955])
    _savefig(fig, f"fig_epoch_protocol_{metric.upper()}_tau{int(round(tau * 100)):02d}")



def _plot_epoch_line_panel(ax, epoch: pd.DataFrame, metric: str, label: str) -> None:
    """Shared Part-5-style epoch panel with robust headroom."""
    lows, highs = [], []
    epochs_present = sorted(int(x) for x in epoch.epoch.unique())
    for method in _active_methods():
        line = epoch[epoch.method == method].sort_values("epoch")
        if line.empty:
            continue
        x = line.epoch.to_numpy(dtype=float)
        ax.plot(x, line[metric], "-o", color=CFG.PALETTE.get(method), markersize=4,
                label=CFG.METHOD_LABELS.get(method, method), zorder=3)
        ax.fill_between(x, line[f"{metric}_lo"].to_numpy(dtype=float),
                        line[f"{metric}_hi"].to_numpy(dtype=float),
                        color=CFG.PALETTE.get(method), alpha=0.18, linewidth=0, zorder=2)
        lows.extend(line[f"{metric}_lo"].tolist()); highs.extend(line[f"{metric}_hi"].tolist())
    ax.set_xlabel("Epoch"); ax.set_ylabel(label)
    if epochs_present:
        ax.set_xticks(epochs_present); ax.set_xticklabels([str(x) for x in epochs_present])
        ax.set_xlim(left=epochs_present[0])
        ax.spines["left"].set_position(("data", epochs_present[0]))
    _set_headroom_ylim(ax, lows, highs, floor=0.0, pad_frac=0.10)


def fig_epoch_focus_by_protocol(epoch: pd.DataFrame, protocol: str) -> None:
    """Mirror the Part-5 AP single-tau two-panel epoch trajectory."""
    cell = epoch[(epoch.protocol == protocol) & np.isclose(epoch.tau, CFG.FOCUS_TAU)]
    if cell.empty:
        return
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    _plot_epoch_line_panel(axes[0], cell, "nta", "NTA")
    _plot_epoch_line_panel(axes[1], cell, "lnmr", "LNMR")
    axes[0].set_title("NTA"); axes[1].set_title("LNMR")
    for ax in axes:
        ax.axvspan(-2, 2, color="0.5", alpha=0.06, zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)),
               frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(rf"Memorization dynamics over training ($\tau={CFG.FOCUS_TAU:.2f}$) - "
                 f"{CFG.PROTOCOL_LABELS.get(protocol, protocol)} ({protocol})",
                 y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0.075, 1, 0.955])
    _savefig(fig, f"fig_epoch_focus_{protocol}_tau{int(round(CFG.FOCUS_TAU * 100)):02d}")


def fig_epoch_grid_by_protocol(epoch: pd.DataFrame, protocol: str, metric: str, label: str) -> None:
    """Mirror the Part-5 AP all-tau small-multiples trajectory grid."""
    cell = epoch[epoch.protocol == protocol]
    taus = [t for t in CFG.EPOCH_TAUS if np.isclose(cell.tau, t).any()]
    if cell.empty or not taus:
        return
    _style()
    ncol = 3
    nrow = int(np.ceil(len(taus) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.8 * nrow), squeeze=False)
    for idx, tau in enumerate(taus):
        ax = axes[idx // ncol, idx % ncol]
        _plot_epoch_line_panel(ax, cell[np.isclose(cell.tau, tau)], metric, label)
        ax.set_title(rf"$\tau={tau:.2f}$")
    for idx in range(len(taus), nrow * ncol):
        axes[idx // ncol, idx % ncol].axis("off")
    handles = [Patch(facecolor=CFG.PALETTE.get(m), label=CFG.METHOD_LABELS.get(m, m))
               for m in _active_methods()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(f"{label} over training across noise rates - {CFG.PROTOCOL_LABELS.get(protocol, protocol)} ({protocol})",
                 y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0.075, 1, 0.955])
    _savefig(fig, f"fig_epoch_grid_{metric}_{protocol}")

def epoch_features(epoch: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if epoch.empty:
        return pd.DataFrame()
    for (protocol, method, tau), g in epoch.groupby(["protocol", "method", "tau"]):
        g = g.sort_values("epoch")
        for metric in ("nta", "lnmr"):
            peak_idx = g[metric].idxmax()
            rows.append(dict(protocol=protocol, method=method, tau=float(tau), metric=metric,
                             first_epoch=int(g.iloc[0].epoch), first_value=float(g.iloc[0][metric]),
                             peak_epoch=int(g.loc[peak_idx, "epoch"]), peak_value=float(g.loc[peak_idx, metric]),
                             final_epoch=int(g.iloc[-1].epoch), final_value=float(g.iloc[-1][metric])))
    return pd.DataFrame(rows)



# protocol-resolved matrix and per-class diagnostics
def _read_metrics_payload(protocol: str, method: str, tau: float, fold: int) -> Optional[dict]:
    fp = _metrics_path(protocol, method, tau, fold)
    if not fp.exists():
        return None
    try:
        return _read_json(fp)
    except (json.JSONDecodeError, OSError):
        return None


def _class_vector(payload: dict, key: str) -> Optional[np.ndarray]:
    """Read a seven-class vector stored as a list, dict, or class-suffixed keys."""
    value = payload.get(key)
    if isinstance(value, dict):
        arr = [value.get(c, np.nan) for c in CFG.CLASSES_ALPHA]
        return np.asarray(arr, dtype=float)
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).reshape(-1)
        return arr if arr.size == len(CFG.CLASSES_ALPHA) else None
    suffix_vals = [payload.get(f"{key}_{c}") for c in CFG.CLASSES_ALPHA]
    if any(v is not None for v in suffix_vals):
        return np.asarray([np.nan if v is None else float(v) for v in suffix_vals], dtype=float)
    return None


def _mean_class_vector(protocol: str, method: str, tau: float, key: str) -> Optional[np.ndarray]:
    rows = []
    for fold in range(CFG.N_FOLDS):
        payload = _read_metrics_payload(protocol, method, tau, fold)
        if payload is None:
            continue
        arr = _class_vector(payload, key)
        if arr is not None:
            rows.append(arr)
    return None if not rows else np.nanmean(np.vstack(rows), axis=0)


def _summed_confusion(protocol: str, method: str, tau: float) -> Optional[np.ndarray]:
    acc = None
    for fold in range(CFG.N_FOLDS):
        payload = _read_metrics_payload(protocol, method, tau, fold)
        if payload is None or payload.get("confusion_matrix") is None:
            continue
        cm = np.asarray(payload["confusion_matrix"], dtype=float)
        if cm.shape != (len(CFG.CLASSES_ALPHA), len(CFG.CLASSES_ALPHA)):
            continue
        acc = cm if acc is None else acc + cm
    return acc


def _white_cell_gaps(ax, nrows: int, ncols: int, lw: float = 3.0) -> None:
    for x in np.arange(0.5, ncols - 1 + 1e-9, 1):
        ax.axvline(x, color="white", linewidth=lw, zorder=2)
    for y in np.arange(0.5, nrows - 1 + 1e-9, 1):
        ax.axhline(y, color="white", linewidth=lw, zorder=2)


def _heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str,
             stem: str, subdir: tuple[str, ...], *, low_is_good: bool = False,
             vmin: float = 0.0, vmax: float = 1.0, fmt: str = "{:.2f}") -> None:
    _style()
    fig, ax = plt.subplots(figsize=(1.5 + 0.98 * len(col_labels), 1.5 + 0.58 * len(row_labels)))
    im = ax.imshow(matrix, cmap=_good_cmap(low_is_good), vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.grid(False); ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _white_cell_gaps(ax, matrix.shape[0], matrix.shape[1])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            pos = (value - vmin) / max(vmax - vmin, 1e-9)
            color = "white" if pos < 0.18 or pos > 0.82 else "0.12"
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=8, color=color, zorder=3)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _savefig(fig, stem, subdir=subdir)



def _grid_dims(n_panels, ncols):
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, ncols


def _collect_perclass_matrix(protocol, tau, kind):
    """(methods x classes) fold-mean matrix for one tau."""
    classes = _classes()
    alpha = list(CFG.CLASSES_ALPHA)
    order_idx = [alpha.index(c) for c in classes]
    key = {"f1": "per_class_f1",
           "lnmr": "per_class_lnmr_by_clean",
           "nta": "per_class_nta_by_clean"}[kind]
    methods = _active_methods()
    M = np.full((len(methods), len(classes)), np.nan)
    for i, method in enumerate(methods):
        arr = _mean_class_vector(protocol, method, tau, key)
        if arr is not None:
            M[i, :] = arr[order_idx]
    return M


def fig_perclass_grid(protocol, kind, low_is_good, label, include_clean):
    """One tall figure: rows = tau, each panel a (methods x classes) heatmap."""
    classes = _classes()
    methods = _active_methods()
    mlabels = [CFG.METHOD_LABELS.get(m, m) for m in methods]
    taus = [t for t in CFG.TAUS if (include_clean or t > 0)]
    _style()

    nrows = len(taus)
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(1.4 + 0.95 * len(classes), nrows * (0.6 + 0.42 * len(methods))),
        squeeze=False, constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.12, wspace=0.02)
    axes = axes[:, 0]
    im = None
    for ax, tau in zip(axes, taus):
        M = _collect_perclass_matrix(protocol, tau, kind)
        im = ax.imshow(M, cmap=_good_cmap(low_is_good), vmin=0.0, vmax=1.0,
                       aspect="auto")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(mlabels)
        if tau == taus[-1]:
            ax.set_xticklabels(classes, rotation=0)
            ax.set_xlabel("Class")
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(f"$\\tau={tau:.2f}$")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        _white_cell_gaps(ax, M.shape[0], M.shape[1])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isnan(v):
                    freq = v  # vmin=0, vmax=1
                    tc = "white" if (freq < 0.18 or freq > 0.82) else "0.1"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color=tc)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
    fig.suptitle(f"{label} - protocol {protocol} (all noise rates)", fontsize=12.5)
    _savefig(fig, f"grid_perclass_{kind}_{protocol}",
             subdir=("matrices", f"perclass_{kind}", protocol))


def _confusion_panel(matrix: np.ndarray, classes: list[str], title: str, stem: str,
                     subdir: tuple[str, ...], *, normalized: bool) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 5.7))
    im = ax.imshow(matrix, cmap="Greens", vmin=0.0, vmax=(1.0 if normalized else None), aspect="auto")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.grid(False); ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _white_cell_gaps(ax, matrix.shape[0], matrix.shape[1])
    mx = max(float(np.nanmax(matrix)), 1e-9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            txt = f"{value:.2f}" if normalized else f"{int(round(value))}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color=("white" if value > 0.60 * mx else "0.15"), zorder=3)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _savefig(fig, stem, subdir=subdir)



_DIVERGING = LinearSegmentedColormap.from_list(
    "delta_rwg", ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"])


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    rs = cm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return cm / rs


def _confusion_delta_panel(matrix: np.ndarray, classes: list[str], title: str,
                           stem: str, subdir: tuple[str, ...]) -> None:
    _style()
    vmax = max(float(np.nanmax(np.abs(matrix))), 1e-6)
    fig, ax = plt.subplots(figsize=(6.4, 5.7))
    im = ax.imshow(matrix, cmap=_DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.grid(False); ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _white_cell_gaps(ax, matrix.shape[0], matrix.shape[1])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix[i, j])
            shown = 0.0 if abs(value) < 0.005 else value
            txt = "0.00" if shown == 0.0 else f"{shown:+.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color=("white" if abs(value) > 0.60 * vmax else "0.15"), zorder=3)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _savefig(fig, stem, subdir=subdir)


def emit_confusion_focus_extras(protocol: str, methods: list[str], classes: list[str],
                                order: list[int]) -> list[dict]:
    """Mirror the AP compact confusion grid and delta-vs-clean matrices."""
    tau = CFG.FOCUS_TAU
    tt = int(round(tau * 100))
    rows = []
    # Delta-vs-clean matrix for each method.
    for method in methods:
        cm_clean = _summed_confusion(protocol, method, 0.0)
        cm_noisy = _summed_confusion(protocol, method, tau)
        if cm_clean is None or cm_noisy is None:
            continue
        delta = _row_normalize(cm_noisy)[np.ix_(order, order)] - _row_normalize(cm_clean)[np.ix_(order, order)]
        title = (f"{CFG.METHOD_LABELS.get(method, method)}: $\\Delta$ row-normalized confusion - "
                 f"{CFG.PROTOCOL_LABELS.get(protocol, protocol)} ({protocol}), $\\tau={tau:.2f}$ minus clean")
        _confusion_delta_panel(delta, classes, title,
                               f"confusion_delta_{protocol}_{method}_tau{tt:02d}",
                               ("matrices", "confusion_delta", protocol))
        for i, true_c in enumerate(classes):
            for j, pred_c in enumerate(classes):
                rows.append(dict(protocol=protocol, method=method, tau=float(tau),
                                 true_class=true_c, pred_class=pred_c,
                                 delta_rate=float(delta[i, j])))

    # Compact row-normalized grid at the focus tau.
    available = []
    for method in methods:
        cm = _summed_confusion(protocol, method, tau)
        if cm is not None:
            available.append((method, _row_normalize(cm[np.ix_(order, order)])))
    if available:
        _style()
        ncol = min(2, len(available)); nrow = int(np.ceil(len(available) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 5.2 * nrow), squeeze=False,
                                 constrained_layout=True)
        axes_flat = axes.ravel(); im = None
        for ax, (method, matrix) in zip(axes_flat, available):
            im = ax.imshow(matrix, cmap="Greens", vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
            ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
            ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
            ax.grid(False); ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            _white_cell_gaps(ax, matrix.shape[0], matrix.shape[1])
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = float(matrix[i, j])
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                            color=("white" if value > 0.60 else "0.15"), zorder=3)
            ax.set_title(CFG.METHOD_LABELS.get(method, method))
        for ax in axes_flat[len(available):]:
            ax.axis("off")
        if im is not None:
            fig.colorbar(im, ax=axes_flat.tolist(), fraction=0.046, pad=0.04)
        fig.suptitle(f"Row-normalized confusion matrices - {CFG.PROTOCOL_LABELS.get(protocol, protocol)} ({protocol}), $\\tau={tau:.2f}$",
                     fontsize=14)
        _savefig(fig, f"confusion_norm_grid_{protocol}_tau{tt:02d}",
                 subdir=("matrices", "confusion_grid", protocol))
    return rows



def fig_confusion_grid_all(protocol, include_clean=False):
    """One master figure: rows = tau, cols = method, each a row-normalized 7x7 confusion matrix."""
    classes = _classes()
    alpha = list(CFG.CLASSES_ALPHA)
    order_idx = [alpha.index(c) for c in classes]
    methods = _active_methods()
    taus = [t for t in CFG.TAUS if (include_clean or t > 0)]
    _style()

    nrows, ncols = len(taus), len(methods)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.0 * ncols, 3.0 * nrows),
                             squeeze=False, constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.10, wspace=0.06)
    im = None
    for r, tau in enumerate(taus):
        for c, method in enumerate(methods):
            ax = axes[r, c]
            cm = _summed_confusion(protocol, method, tau)
            if cm is None:
                ax.set_visible(False)
                continue
            cm = cm[np.ix_(order_idx, order_idx)]
            rs = cm.sum(1, keepdims=True); rs[rs == 0] = 1
            M = cm / rs
            im = ax.imshow(M, cmap="Greens", vmin=0, vmax=1.0, aspect="auto")
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            if r == 0:
                ax.set_title(CFG.METHOD_LABELS.get(method, method))
            if r == nrows - 1:
                ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_yticklabels(classes, fontsize=7)
                ax.set_ylabel(f"$\\tau={tau:.2f}$", fontsize=11)
            else:
                ax.set_yticklabels([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)
            _white_cell_gaps(ax, M.shape[0], M.shape[1])
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    tc = "white" if v > 0.6 else "0.15"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.5, color=tc)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    fig.suptitle(f"Row-normalized confusion matrices - protocol {protocol} "
                 f"(all noise rates)", fontsize=13)
    tag = "with_clean" if include_clean else "noisy"
    _savefig(fig, f"confusion_norm_grid_ALL_{protocol}_{tag}",
             subdir=("matrices", "confusion", protocol))




def _emit_perclass_focus_table(perclass: pd.DataFrame, protocol: str, diagnostic: str, tau: float) -> None:
    cell = perclass[(perclass.protocol == protocol) & (perclass.diagnostic == diagnostic)
                    & np.isclose(perclass.tau, tau)]
    if cell.empty:
        return
    classes = _classes()
    methods = _active_methods()
    rows = []
    for method in methods:
        line = cell[cell.method == method].set_index("class_name").reindex(classes)
        vals = ["--" if pd.isna(v) else f"{float(v):.3f}" for v in line.value]
        rows.append(" & ".join([CFG.METHOD_LABELS.get(method, method), *vals]) + r" \\")
    label = {"perclass_f1": "Per-class F1", "perclass_nta": "Per-class NTA", "perclass_lnmr": "Per-class LNMR"}[diagnostic]
    tex = [r"\begin{table}[htbp]", r"\centering",
           rf"\caption{{{label} for protocol {protocol} at $\tau={tau:.2f}$. Values are fold-averaged descriptive diagnostics.}}",
           rf"\label{{tab:{diagnostic}-{protocol.lower()}-tau{int(round(tau*100)):02d}}}",
           r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{l{'c' * len(classes)}}}", r"\toprule",
           "Method & " + " & ".join(classes) + r" \\", r"\midrule", *rows,
           r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    _write_tex(f"tab_{diagnostic}_{protocol}_tau{int(round(tau*100)):02d}", "\n".join(tex), subdir=("matrices", protocol))



def _focus_protocol_grid(protocols, diagnostic, key, low_is_good, tau=None):
    """2x2 per-class grid at one tau (panels are protocols)."""
    if tau is None:
        tau = CFG.FOCUS_TAU
    label = {"perclass_f1": "Per-class F1",
             "perclass_nta": "Per-class NTA (by true class)",
             "perclass_lnmr": "Per-class LNMR (by true class)"}[diagnostic]
    methods = _active_methods()
    classes = _classes()
    alpha = list(CFG.CLASSES_ALPHA)
    order = [alpha.index(c) for c in classes]
    layout = [["SP", "S"], ["AP", "A"]]   # columns: left = pretrained, right = scratch
    present = set(protocols)
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(2.5 + 1.25 * len(classes),
                                            1.4 + 0.95 * len(methods) * 2),
                             constrained_layout=True)
    cmap = _good_cmap(low_is_good)
    im = None
    for r in range(2):
        for c in range(2):
            ax = axes[r, c]
            protocol = layout[r][c]
            M = np.full((len(methods), len(classes)), np.nan)
            if protocol in present:
                for i, method in enumerate(methods):
                    arr = _mean_class_vector(protocol, method, tau, key)
                    if arr is not None:
                        M[i, :] = arr[order]
            if not np.isfinite(M).any():
                ax.axis("off")
                continue
            im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels([CFG.METHOD_LABELS.get(m, m) for m in methods])
            ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
            ax.grid(False); ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            _white_cell_gaps(ax, M.shape[0], M.shape[1])
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    if not np.isnan(v):
                        tc = "white" if (v < 0.18 or v > 0.82) else "0.1"
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=8, color=tc)
            ax.set_title(f"{CFG.PROTOCOL_LABELS.get(protocol, protocol)} ({protocol})")
    fig.suptitle(rf"{label} at $\tau={tau:.2f}$ across training protocols",
                 fontsize=14)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.030, pad=0.02)
    _savefig(fig, f"{diagnostic}_focus_grid_tau{int(round(tau*100)):02d}",
             subdir=("matrices", diagnostic))


def emit_matrix_diagnostics(metric_df: pd.DataFrame) -> None:
    if not CFG.RUN_MATRIX_DIAGNOSTICS:
        return
    protocols = available_protocols(metric_df)
    methods = _active_methods()
    classes = _classes(); alpha = list(CFG.CLASSES_ALPHA)
    order = [alpha.index(c) for c in classes]
    perclass_rows = []
    confusion_rows = []
    confusion_delta_rows = []

    for protocol in protocols:
        print(f"[matrix] building per-class and confusion diagnostics for {protocol} ...")
        for diagnostic, key, low_is_good, taus in (
            ("perclass_f1", "per_class_f1", False, CFG.TAUS),
            ("perclass_nta", "per_class_nta_by_clean", False, [t for t in CFG.TAUS if t > 0]),
            ("perclass_lnmr", "per_class_lnmr_by_clean", True, [t for t in CFG.TAUS if t > 0]),
        ):
            for tau in taus:
                matrix = np.full((len(methods), len(classes)), np.nan)
                for i, method in enumerate(methods):
                    arr = _mean_class_vector(protocol, method, tau, key)
                    if arr is None:
                        continue
                    arr = arr[order]
                    matrix[i, :] = arr
                    for class_name, value in zip(classes, arr):
                        perclass_rows.append(dict(protocol=protocol, diagnostic=diagnostic,
                                                  method=method, tau=float(tau), class_name=class_name,
                                                  value=float(value)))
                # if np.isfinite(matrix).any():
                #     lab = {"perclass_f1": "Per-class F1", "perclass_nta": "Per-class NTA (by true class)",
                #            "perclass_lnmr": "Per-class LNMR (by true class)"}[diagnostic]
                #     _heatmap(matrix, [CFG.METHOD_LABELS.get(m, m) for m in methods], classes,
                #              rf"{lab} - {CFG.PROTOCOL_LABELS.get(protocol, protocol)} ({protocol}), $\tau={tau:.2f}$",
                #              f"{diagnostic}_{protocol}_tau{int(round(tau*100)):02d}",
                #              subdir=("matrices", diagnostic, protocol), low_is_good=low_is_good)
            fig_perclass_grid(
                protocol,
                {"perclass_f1": "f1", "perclass_nta": "nta", "perclass_lnmr": "lnmr"}[diagnostic],
                low_is_good,
                {"perclass_f1": "Per-class F1",
                 "perclass_nta": "Per-class NTA (by true class)",
                 "perclass_lnmr": "Per-class LNMR (by true class)"}[diagnostic],
                include_clean=(diagnostic == "perclass_f1"))

        for method in methods:
            for tau in CFG.TAUS:
                cm = _summed_confusion(protocol, method, tau)
                if cm is None:
                    continue
                cm = cm[np.ix_(order, order)]
                rs = cm.sum(axis=1, keepdims=True); rs[rs == 0] = 1
                norm = cm / rs
                title_base = f"{CFG.METHOD_LABELS.get(method, method)} - {CFG.PROTOCOL_LABELS.get(protocol, protocol)} ({protocol}), $\\tau={tau:.2f}$"
                stem = f"{protocol}_{method}_tau{int(round(tau*100)):02d}"
                _confusion_panel(cm, classes, f"Confusion counts: {title_base}",
                                 f"confusion_counts_{stem}", ("matrices", "confusion", protocol), normalized=False)
                _confusion_panel(norm, classes, f"Row-normalized confusion: {title_base}",
                                 f"confusion_norm_{stem}", ("matrices", "confusion", protocol), normalized=True)
                for i, true_c in enumerate(classes):
                    for j, pred_c in enumerate(classes):
                        confusion_rows.append(dict(protocol=protocol, method=method, tau=float(tau),
                                                   true_class=true_c, pred_class=pred_c,
                                                   count=float(cm[i, j]), rate=float(norm[i, j])))
        confusion_delta_rows.extend(emit_confusion_focus_extras(protocol, methods, classes, order))
        fig_confusion_grid_all(protocol)

    perclass = pd.DataFrame(perclass_rows)
    confusion = pd.DataFrame(confusion_rows)
    confusion_delta = pd.DataFrame(confusion_delta_rows)
    _write_csv(perclass, "perclass_diagnostics.csv", subdir=("matrices",))
    _write_csv(confusion, "confusion_matrices.csv", subdir=("matrices",))
    _write_csv(confusion_delta, "confusion_delta_focus.csv", subdir=("matrices",))
    if not perclass.empty:
        for protocol in protocols:
            for diagnostic in ("perclass_f1", "perclass_nta", "perclass_lnmr"):
                _emit_perclass_focus_table(perclass, protocol, diagnostic, CFG.FOCUS_TAU)

    for diagnostic, key, low_is_good in (
        ("perclass_f1", "per_class_f1", False),
        ("perclass_nta", "per_class_nta_by_clean", False),
        ("perclass_lnmr", "per_class_lnmr_by_clean", True),
    ):
        _focus_protocol_grid(protocols, diagnostic, key, low_is_good, CFG.FOCUS_TAU)
            


def write_output_readme() -> None:
    fp = CFG.OUT_ROOT / "README_outputs.txt"
    fp.write_text(
        "Protocol-sensitivity output navigation\n"
        "======================================\n\n"
        "figures/performance/grouped_bars/                 grouped method bars, panels=protocols\n"
        "figures/performance/protocol_lines/               one panel per method, lines=protocols\n"
        "figures/performance/baseline_vs_tau/across_protocols/ baseline-only overlays by metric\n"
        "figures/performance/baseline_vs_tau/by_protocol/  AP-style baseline degradation plots per protocol\n"
        "figures/performance/method_advantage_focus/       robust-method delta vs own baseline at focus tau\n"
        "figures/performance/combined_by_protocol/           Part-3-style combined body/all-metric plots per protocol\n"
        "figures/mechanism/across_tau/                     final-epoch NTA/LNMR across noise rates\n"
        "figures/mechanism/focus_tau/                      final-epoch focus-tau comparisons\n"
        "figures/mechanism/epoch_trajectories/             cross-protocol NTA/LNMR over epochs\n"
        "figures/mechanism/by_protocol/nta_lnmr/             Part-5-style combined NTA/LNMR plots per protocol\n"
        "figures/mechanism/epoch_by_protocol/focus_tau/      Part-5-style two-panel epoch plot per protocol\n"
        "figures/mechanism/epoch_by_protocol/grids/          all-tau epoch grids per protocol\n"
        "figures/matrices/confusion/<protocol>/            raw and row-normalized confusion matrices\n"
        "figures/matrices/confusion_delta/<protocol>/      focus-tau delta-vs-clean confusion matrices\n"
        "figures/matrices/confusion_grid/<protocol>/       compact focus-tau row-normalized confusion grids\n"
        "figures/matrices/perclass_f1/<protocol>/          per-class F1 heatmaps\n"
        "figures/matrices/perclass_nta/<protocol>/         per-class NTA heatmaps\n"
        "figures/matrices/perclass_lnmr/<protocol>/        per-class LNMR heatmaps\n"
        "tables/performance/body/                          compact body tables\n"
        "tables/performance/deltas/                        baseline-relative delta tables\n"
        "tables/stats/                                     ranking, best-next, and interaction tables\n"
        "tables/appendix/performance/                      full aggregate appendix tables\n"
        "tables/mechanism/ and tables/appendix/mechanism/  mechanism tables\n"
        "tables/matrices/<protocol>/                       focus-tau per-class appendix tables\n"
        "data/                                             tidy CSVs behind outputs, grouped by purpose\n"
    )
    print(f"[output] wrote {fp}")


# prose helper and manifest
def print_prose_helper(summary: pd.DataFrame, ranking: pd.DataFrame,
                       best_next_df: pd.DataFrame, interactions: pd.DataFrame,
                       mechanism: pd.DataFrame, comp: pd.DataFrame) -> None:
    print("\n" + "=" * 82)
    print("PROSE HELPER - Results Part 4: protocol sensitivity")
    print("=" * 82)
    protocols = [p for p in _active_protocols() if p in set(summary.protocol.unique())]
    print(f"protocols loaded: {protocols}; anchor={CFG.ANCHOR_PROTOCOL}; methods={_active_methods()}")
    incomplete = comp[(comp.protocol.isin(protocols)) & (~comp.complete)]
    if not incomplete.empty:
        print(f"CAUTION: {len(incomplete)} aggregate protocol x method x tau cells are incomplete. See data/completeness.csv.")

    print(f"\n[Ranking at tau={CFG.FOCUS_TAU:.2f}]")
    focus_rank = ranking[np.isclose(ranking.tau, CFG.FOCUS_TAU)]
    focus_bn = best_next_df[np.isclose(best_next_df.tau, CFG.FOCUS_TAU)]
    for metric in CFG.TABLE_METRICS:
        print(f"  {CFG.METRIC_DISPLAY[metric][0]}:")
        for protocol in protocols:
            r = focus_rank[(focus_rank.protocol == protocol) & (focus_rank.metric == metric)].sort_values("rank")
            b = focus_bn[(focus_bn.protocol == protocol) & (focus_bn.metric == metric)]
            if len(r) < 2:
                continue
            extra = ""
            if len(b):
                br = b.iloc[0]
                extra = f"; exploratory gap={_fmt_signed(br.delta)}, Holm p={_fmt_p(br.p_holm)}, sig={_stats_sig_cell(br)}"
            print(f"    {protocol:3s}: {r.iloc[0].method} ({r.iloc[0]['mean']:.3f}) > "
                  f"{r.iloc[1].method} ({r.iloc[1]['mean']:.3f}){extra}")

    if not interactions.empty:
        print(f"\n[Cross-protocol interactions at tau={CFG.FOCUS_TAU:.2f}; direction=P1-P2]")
        focus_int = interactions[np.isclose(interactions.tau, CFG.FOCUS_TAU)]
        for _, r in focus_int.iterrows():
            marker = "SIGNIFICANT" if not pd.isna(r.p_holm) and r.p_holm < CFG.HOLM_ALPHA else "n.s."
            print(f"  {r.protocol_1}-{r.protocol_2:2s} {r.metric:8s} {r.method:8s}: "
                  f"interaction={_fmt_signed(r.delta)}, Holm p={_fmt_p(r.p_holm)} ({marker})")
    else:
        print("\n[Cross-protocol interactions] none computed; see console messages and alignment report.")

    if not mechanism.empty:
        print(f"\n[Final-epoch mechanism at tau={CFG.FOCUS_TAU:.2f}]")
        focus = mechanism[np.isclose(mechanism.tau, CFG.FOCUS_TAU)]
        for protocol in protocols:
            print(f"  {protocol}:")
            for method in _active_methods():
                cell = focus[(focus.protocol == protocol) & (focus.method == method)]
                if len(cell):
                    r = cell.iloc[0]
                    print(f"    {method:8s}: NTA={r.nta:.3f}, LNMR={r.lnmr:.3f}, residual={r.residual:.3f}")
    print("=" * 82 + "\n")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(x) for x in value]
    if isinstance(value, list):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def write_manifest() -> None:
    files = sorted(str(p.relative_to(CFG.OUT_ROOT)) for p in CFG.OUT_ROOT.rglob("*") if p.is_file())
    payload = dict(
        generated_at=datetime.now(timezone.utc).isoformat(),
        script="results_part4_protocol_sensitivity.py",
        description="Standalone RQ3 protocol-sensitivity analysis",
        config=_jsonable(asdict(CFG)),
        files=files,
    )
    fp = CFG.OUT_ROOT / "manifest.json"
    fp.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[manifest] wrote {fp}")


# CLI and main
def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Results Part 4 protocol sensitivity analysis")
    parser.add_argument("--protocols", type=str, default=None,
                        help="Comma-separated protocol codes overriding CONFIG.PROTOCOLS_TO_RUN")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated logical method names overriding CONFIG.METHODS_TO_RUN")
    parser.add_argument("--focus-tau", type=float, default=None,
                        help="Override the focus tau used in body tables and focus plots")
    parser.add_argument("--skip-mechanism", action="store_true",
                        help="Skip final-epoch NTA/LNMR protocol analysis")
    parser.add_argument("--skip-epoch", action="store_true",
                        help="Skip training-log epoch trajectories")
    parser.add_argument("--skip-matrices", action="store_true",
                        help="Skip confusion matrices and per-class F1/NTA/LNMR heatmaps")
    return parser.parse_args(argv)


def apply_args(args) -> None:
    if args.protocols:
        CFG.PROTOCOLS_TO_RUN = _split_csv(args.protocols)
    if args.methods:
        CFG.METHODS_TO_RUN = _split_csv(args.methods)
    if args.focus_tau is not None:
        CFG.FOCUS_TAU = float(args.focus_tau)
    if args.skip_mechanism:
        CFG.RUN_FINAL_EPOCH_MECHANISM = False
    if args.skip_epoch:
        CFG.RUN_EPOCH_TRAJECTORIES = False
    if args.skip_matrices:
        CFG.RUN_MATRIX_DIAGNOSTICS = False


import shutil as _shutil

try:
    from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont
    _HAS_PIL = True
except Exception:                                    # pragma: no cover
    _HAS_PIL = False


def _thesis_root() -> Path:
    return CFG.OUT_ROOT / getattr(CFG, "THESIS_SUBDIR", "THESIS")


def _thesis_dest(where: str, kind: str, name: str) -> Path:
    sub = "figures" if kind == "figure" else "tables"
    d = _thesis_root() / where / sub
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def _thesis_font(size: int):
    if not _HAS_PIL:
        return None
    for nm in ("DejaVuSerif.ttf", "Palatino.ttf", "DejaVuSans.ttf"):
        try:
            return _PILFont.truetype(nm, size)
        except OSError:
            continue
    return _PILFont.load_default()


def _thesis_grid(image_paths, out_path: Path, ncols: int,
                 panel_titles=None, suptitle=None, pad: int = 16) -> bool:
    """Stitch existing PNGs into a single grid PNG. Returns True if written."""
    if not _HAS_PIL:
        print("[thesis][grid] Pillow not installed; skipping grid", out_path.name)
        return False
    imgs, titles = [], []
    for i, p in enumerate(image_paths):
        if p and Path(p).exists():
            imgs.append(_PILImage.open(p).convert("RGB"))
            titles.append(panel_titles[i] if panel_titles else None)
        else:
            print(f"[thesis][grid][skip-missing] {p}")
    if not imgs:
        print(f"[thesis][grid][skip] no inputs for {out_path.name}")
        return False
    target_w = min(im.width for im in imgs)
    scaled = []
    for im in imgs:
        if im.width != target_w:
            im = im.resize((target_w, round(im.height * target_w / im.width)), _PILImage.LANCZOS)
        scaled.append(im)
    nrows = (len(scaled) + ncols - 1) // ncols
    row_h = [max(s.height for s in scaled[r * ncols:(r + 1) * ncols]) for r in range(nrows)]
    ptitle_h = 24 if panel_titles else 0
    sup_h = 40 if suptitle else 0
    W = ncols * target_w + (ncols + 1) * pad
    H = sup_h + sum(h + ptitle_h for h in row_h) + (nrows + 1) * pad
    canvas = _PILImage.new("RGB", (W, H), "white")
    draw = _PILDraw.Draw(canvas)
    if suptitle:
        f = _thesis_font(max(18, target_w // 22))
        draw.text(((W - draw.textlength(suptitle, font=f)) / 2, pad // 2),
                  suptitle, fill="black", font=f)
    fp = _thesis_font(max(13, target_w // 30))
    y = sup_h + pad
    for r in range(nrows):
        x = pad
        for s, t in zip(scaled[r * ncols:(r + 1) * ncols], titles[r * ncols:(r + 1) * ncols]):
            if t:
                draw.text((x + (target_w - draw.textlength(t, font=fp)) / 2, y), t,
                          fill="black", font=fp)
            canvas.paste(s, (x, y + ptitle_h))
            x += target_w + pad
        y += row_h[r] + ptitle_h + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[thesis][grid] wrote {out_path.name} ({len(scaled)} panels)")
    return True


def _thesis_copy(src: Path, where: str, kind: str, name: str, rows: list,
                 label: str, caption: str) -> None:
    if not src.exists():
        print(f"[thesis][copy][skip-missing] {src.relative_to(CFG.OUT_ROOT)}")
        return
    dest = _thesis_dest(where, kind, name)
    _shutil.copy2(src, dest)
    print(f"[thesis][copy] {name} -> {where}")
    rows.append((where, kind, label, str(dest.relative_to(_thesis_root())), caption))


def _fig_src(*parts) -> Path:
    return CFG.OUT_ROOT / "figures" / Path(*parts)


def _tab_src(*parts) -> Path:
    return CFG.OUT_ROOT / "tables" / Path(*parts)


def _tt(tau: float) -> str:
    return f"{int(round(tau * 100)):02d}"


def curate_thesis_split(metric_df) -> None:
    """Build OUT_ROOT/THESIS/{body,appendix} from Part 4's own outputs."""
    if not getattr(CFG, "BUILD_THESIS_SPLIT", True):
        return
    print("\n[thesis] curating body/appendix split ...")
    protocols = available_protocols(metric_df)
    methods = _active_methods()
    classes = _classes()
    ft, ftt = CFG.FOCUS_TAU, _tt(CFG.FOCUS_TAU)
    taus_nz = [t for t in CFG.TAUS if t > 0]
    taus_all = list(CFG.TAUS)
    rows: list = []

    # BODY
    # RQ3 headline: grouped bars (BA, F1) + interaction + ranking + body scores.
    for metric, lab in (("BA", "balanced accuracy"), ("MacroF1", "Macro F1")):
        _thesis_copy(_fig_src("performance", "grouped_bars", f"fig_protocol_{metric}.png"),
                     "body", "figure", f"p4_protocol_bars_{metric}.png", rows,
                     f"fig:protocol-bars-{metric.lower()}",
                     f"{lab.capitalize()} by protocol (panels) and method (bars) vs $\\tau$, "
                     f"with within-protocol method-vs-baseline significance.")
    _thesis_copy(_tab_src("stats", f"tab_interaction_focus_tau{ftt}.tex"),
                 "body", "table", "p4_interaction_focus.tex", rows,
                 "tab:interaction-focus",
                 f"Cross-protocol difference-of-differences at $\\tau={ft:.2f}$: whether each "
                 f"method's own-baseline advantage differs between a protocol and AP.")
    _thesis_copy(_tab_src("stats", f"tab_ranking_focus_tau{ftt}.tex"),
                 "body", "table", "p4_ranking_focus.tex", rows,
                 "tab:ranking-focus",
                 f"Winner and next-best method under every protocol at $\\tau={ft:.2f}$.")
    for metric in ("BA", "MacroF1"):
        _thesis_copy(_tab_src("performance", "body", f"tab_protocol_body_{metric}.tex"),
                     "body", "table", f"p4_protocol_body_{metric}.tex", rows,
                     f"tab:protocol-body-{metric.lower()}",
                     f"Per-protocol {metric} scores (mean, 95\\% CI) at the focus rate with "
                     f"method-vs-baseline significance.")

    # APPENDIX
    # Aggregate: AUC bars, line views, full grids, deltas, exploratory best-next.
    _thesis_copy(_fig_src("performance", "grouped_bars", "fig_protocol_MacroAUC.png"),
                 "appendix", "figure", "app_p4_protocol_bars_MacroAUC.png", rows,
                 "fig:app-protocol-auc", "Macro AUC by protocol and method vs $\\tau$.")
    for metric in CFG.FIG_METRICS:
        _thesis_copy(_fig_src("performance", "protocol_lines", f"fig_protocol_lines_{metric}.png"),
                     "appendix", "figure", f"app_p4_protocol_lines_{metric}.png", rows,
                     f"fig:app-protocol-lines-{metric.lower()}",
                     f"{metric} as lines (protocols) per method panel.")
    for metric in CFG.TABLE_METRICS:
        _thesis_copy(_tab_src("appendix", "performance", f"tab_app_protocol_full_{metric}.tex"),
                     "appendix", "table", f"app_p4_protocol_full_{metric}.tex", rows,
                     f"tab:app-protocol-full-{metric.lower()}",
                     f"Complete by-protocol {metric} grid, all $\\tau$, with significance.")
    for stem, name, label, cap in (
        ("tab_app_interaction_full", "app_p4_interaction_full.tex",
         "tab:app-interaction-full", "Full difference-of-differences interaction tests, all $\\tau$."),
        ("tab_app_delta_full", "app_p4_delta_full.tex",
         "tab:app-delta-full", "Full own-baseline advantage (delta) grid, all $\\tau$."),
        ("tab_app_best_vs_next_full", "app_p4_best_vs_next_full.tex",
         "tab:app-best-next-full", "Full exploratory best-vs-next-best comparisons, all $\\tau$."),
    ):
        _thesis_copy(_tab_src("appendix", "performance", f"{stem}.tex"),
                     "appendix", "table", name, rows, label, cap)

    # Mechanism across protocols (this is "Part 5 across protocols").
    for metric in ("NTA", "LNMR"):
        _thesis_copy(_fig_src("mechanism", "across_tau", f"fig_mechanism_protocol_{metric}.png"),
                     "appendix", "figure", f"app_p4_mechanism_{metric}.png", rows,
                     f"fig:app-mech-{metric.lower()}",
                     f"Final-epoch {metric} by protocol across $\\tau$.")
    for proto in protocols:
        _thesis_copy(_fig_src("mechanism", "by_protocol", "nta_lnmr", f"fig_nta_lnmr_{proto}.png"),
                     "appendix", "figure", f"app_p4_nta_lnmr_{proto}.png", rows,
                     f"fig:app-nta-lnmr-{proto}",
                     f"Aggregate NTA and LNMR vs $\\tau$ for Protocol {proto}.")
    _thesis_copy(_tab_src("appendix", "mechanism", "tab_app_mechanism_full.tex"),
                 "appendix", "table", "app_p4_mechanism_full.tex", rows,
                 "tab:app-mech-full", "Full final-epoch NTA/LNMR by protocol and $\\tau$.")

    # Epoch trajectories per protocol.
    for proto in protocols:
        for metric in ("nta", "lnmr"):
            _thesis_copy(_fig_src("mechanism", "epoch_by_protocol", "grids",
                                  f"fig_epoch_grid_{metric}_{proto}.png"),
                         "appendix", "figure", f"app_p4_epoch_grid_{metric}_{proto}.png", rows,
                         f"fig:app-epoch-{metric}-{proto}",
                         f"{metric.upper()} over training across $\\tau$ for Protocol {proto}.")

    # Matrices: across-tau GRIDS, stitched in Python (per protocol).
    for proto in protocols:
        # per-class F1 across all six tau
        _thesis_copy(_fig_src("matrices", "perclass_f1", proto, f"grid_perclass_f1_{proto}.png"),
                     "appendix", "figure", f"app_p4_perclass_f1_grid_{proto}.png", rows,
                     f"fig:app-perclass-f1-{proto}", f"Per-class F1 at every $\\tau$, Protocol {proto}.")
        for diag, kind, lab in (("perclass_nta", "nta", "NTA"), ("perclass_lnmr", "lnmr", "LNMR")):
            _thesis_copy(_fig_src("matrices", diag, proto, f"grid_perclass_{kind}_{proto}.png"),
                         "appendix", "figure", f"app_p4_{diag}_grid_{proto}.png", rows,
                         f"fig:app-{diag}-{proto}", f"Per-class {lab} at every $\\tau>0$, Protocol {proto}.")
        # confusion grid figure, just copy it
        _thesis_copy(_fig_src("matrices", "confusion", proto,
                              f"confusion_norm_grid_ALL_{proto}_noisy.png"),
                     "appendix", "figure", f"app_p4_confusion_grid_{proto}.png", rows,
                     f"fig:app-confusion-{proto}",
                     f"Row-normalized confusion (rows $\\tau$, columns methods), Protocol {proto}.")
        # the compact focus-tau confusion grid Part 4 already builds (per protocol)
        _thesis_copy(_fig_src("matrices", "confusion_grid", proto,
                              f"confusion_norm_grid_{proto}_tau{ftt}.png"),
                     "appendix", "figure", f"app_p4_confusion_focus_grid_{proto}.png", rows,
                     f"fig:app-confusion-focus-{proto}",
                     f"Row-normalized confusion (2x2 methods) at $\\tau={ft:.2f}$, Protocol {proto}.")

    _thesis_write_map(rows)
    print(f"[thesis] split written to {_thesis_root()}")


def _thesis_write_map(rows: list) -> None:
    out = _thesis_root() / "THESIS_MAP.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Part 4 thesis artefact map",
             "",
             "Auto-generated by curate_thesis_split(). Each row: a curated artefact, "
             "where it belongs in the thesis, its intended LaTeX label, and a caption hint.",
             ""]
    for where in ("body", "appendix"):
        sub = [r for r in rows if r[0] == where]
        if not sub:
            continue
        lines += [f"## {where.capitalize()}", "",
                  "| kind | file | `\\label` | caption hint |",
                  "| --- | --- | --- | --- |"]
        for _, kind, label, path, cap in sub:
            lines.append(f"| {kind} | `{path}` | `{label}` | {cap.replace('|', '/')} |")
        lines.append("")
    out.write_text("\n".join(lines) + "\n")
    print(f"[thesis] wrote {out}")


def main(argv=None) -> None:
    apply_args(parse_args(argv))
    _validate_config()
    _ensure_out_tree()

    print("Loading raw fold-level aggregate metrics ...")
    metric_df = load_metric_long()
    _write_csv(metric_df, "aggregate_metrics_per_fold.csv")

    comp = completeness_frame(metric_df)
    _write_csv(comp, "completeness.csv")
    print_completeness(comp)

    print("\nComputing aggregate summaries and within-protocol method-vs-baseline tests ...")
    summary = summarize_metrics(metric_df)
    mvb = method_vs_baseline(metric_df)
    mvc = method_vs_clean(metric_df)
    _write_csv(summary, "aggregate_summary.csv")
    _write_csv(mvb, "method_vs_baseline.csv")

    print("Computing ranking stability and exploratory best-vs-next-best comparisons ...")
    ranking = build_ranking_stability(summary)
    best_next_df = best_vs_next(metric_df, ranking)
    _write_csv(ranking, "ranking_stability.csv")
    _write_csv(best_next_df, "best_vs_next.csv")

    print("Computing own-baseline delta grids ...")
    full_delta, focus_delta, avg_delta = delta_long(summary)
    _write_csv(full_delta, "delta_full.csv")
    _write_csv(focus_delta, f"delta_focus_tau{int(round(CFG.FOCUS_TAU * 100)):02d}.csv")
    _write_csv(avg_delta, "delta_avg.csv")

    print("Computing paired cross-protocol difference-of-differences ...")
    interactions, alignment = interaction_tests(metric_df)
    _write_csv(interactions, "interaction_tests.csv")
    _write_csv(alignment, "interaction_fold_alignment.csv")

    print("Computing cross-protocol comparisons (which protocol is best per method) ...")
    protocol_comp = protocol_comparison(metric_df)
    _write_csv(protocol_comp, "protocol_comparison.csv")

    print("Building aggregate figures ...")
    for metric in CFG.FIG_METRICS:
        if metric in set(summary.metric.unique()):
            fig_protocol_bars(summary, mvb, metric)
            fig_protocol_lines(summary, mvc, metric)
            fig_baseline_protocol_overlay(summary, metric)
    for protocol in available_protocols(metric_df):
        fig_baseline_metrics_by_protocol(summary, protocol)
        fig_combined_metrics_by_protocol(summary, mvb, protocol)
    fig_advantage_focus(mvb)

    print("Building aggregate LaTeX tables ...")
    for metric in CFG.TABLE_METRICS:
        if metric in set(summary.metric.unique()):
            emit_protocol_body_table(summary, mvb, metric)
    emit_delta_grid(
        focus_delta,
        f"tab_delta_focus_tau{int(round(CFG.FOCUS_TAU * 100)):02d}",
        (f"Difference to each training protocol's own baseline at $\\tau={CFG.FOCUS_TAU:.2f}$. "
         f"Cells give the absolute change in metric units with the relative change in parentheses. "
         f"These are descriptive magnitudes; paired within-protocol significance is reported in the "
         f"protocol score tables."),
        "tab:protocol-delta-focus")
    clean_note = "including" if CFG.AVG_INCLUDE_CLEAN else "excluding"
    emit_delta_grid(
        avg_delta, "tab_delta_avg",
        (f"Difference to each training protocol's own baseline averaged over noise rates ({clean_note} "
         f"$\\tau=0$). Cells give the mean absolute change in metric units with the mean relative "
         f"change in parentheses. These are descriptive magnitudes."),
        "tab:protocol-delta-avg")
    emit_delta_full(full_delta)
    emit_ranking_focus(ranking)
    emit_best_next_focus(best_next_df)
    emit_best_next_full(best_next_df)
    emit_interaction_focus(interactions)
    emit_interaction_full(interactions)
    emit_protocol_comparison_focus(protocol_comp)
    emit_protocol_comparison_full(protocol_comp)

    mechanism_summary = pd.DataFrame()
    if CFG.RUN_FINAL_EPOCH_MECHANISM:
        print("\nLoading optional final-epoch NTA/LNMR mechanism diagnostics ...")
        mechanism_raw = load_mechanism_raw(metric_df)
        _write_csv(mechanism_raw, "mechanism_per_fold.csv")
        mechanism_summary = summarize_mechanism(mechanism_raw)
        _write_csv(mechanism_summary, "mechanism_summary.csv")
        if mechanism_summary.empty:
            print("[mechanism] no compatible mechanism diagnostics found; mechanism outputs skipped.")
        else:
            fig_mechanism_protocol(mechanism_summary, "nta", "NTA")
            fig_mechanism_protocol(mechanism_summary, "lnmr", "LNMR")
            fig_mechanism_focus(mechanism_summary)
            for protocol in available_protocols(metric_df):
                fig_nta_lnmr_by_protocol(mechanism_summary, protocol)
            emit_mechanism_focus(mechanism_summary)
            emit_mechanism_full(mechanism_summary)

    if CFG.RUN_EPOCH_TRAJECTORIES:
        print("\nLoading optional per-epoch NTA/LNMR trajectories ...")
        epoch_raw = load_epoch_trajectories(metric_df)
        if not epoch_raw.empty:
            _write_csv(epoch_raw, "epoch_trajectory_per_fold.csv")
            epoch_summary = summarize_epoch(epoch_raw)
            _write_csv(epoch_summary, "epoch_trajectory_summary.csv")
            _write_csv(epoch_features(epoch_summary), "epoch_trajectory_features.csv")
            for tau in CFG.EPOCH_PROTOCOL_COMPARISON_TAUS:
                fig_epoch_protocol(epoch_summary, "nta", "NTA", tau)
                fig_epoch_protocol(epoch_summary, "lnmr", "LNMR", tau)
            for protocol in available_protocols(metric_df):
                fig_epoch_focus_by_protocol(epoch_summary, protocol)
                fig_epoch_grid_by_protocol(epoch_summary, protocol, "nta", "NTA")
                fig_epoch_grid_by_protocol(epoch_summary, protocol, "lnmr", "LNMR")

    if CFG.RUN_MATRIX_DIAGNOSTICS:
        print("\nBuilding optional per-protocol confusion and per-class diagnostics ...")
        emit_matrix_diagnostics(metric_df)

    print_prose_helper(summary, ranking, best_next_df, interactions, mechanism_summary, comp)
    write_output_readme()
    if CFG.BUILD_THESIS_SPLIT:
            curate_thesis_split(metric_df)
    write_manifest()
    print("Done.")


if __name__ == "__main__":
    main()