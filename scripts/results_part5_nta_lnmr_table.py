"""
Results.5 - table of aggregate NTA and LNMR (with bootstrap CIs).

Emits the values plotted in the memorization-diagnostics figure (fig:nta-lnmr):
per method and noise rate, the fold-mean NTA and LNMR with a 95% bootstrap CI
across the ten folds, using the figure's bootstrap settings (same N_BOOT, SEED)
so the table and plot are numerically identical. tau = 0 is omitted (both
quantities undefined with no flipped labels).

Writes, into results/mechanism/<protocol>/nta_lnmr/:
  tab_nta_lnmr_<P>.tex   LaTeX table (methods grouped, rows = tau)
  _nta_lnmr_<P>.csv      tidy values behind the table
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


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
    RAW_FOLD_CSV: str = "figures_and_tables/raw_fold_results.csv"
    DATASET: str = "imbalanced"

    METHODS: tuple = ("baseline", "sce", "elr", "asyco_divmix")
    METHOD_LABELS: dict = field(default_factory=lambda: {
        "baseline": "Baseline", "sce": "SCE", "elr": "ELR", "asyco_divmix": "AsyCo",
    })
    TAUS_NONZERO: tuple = (0.1, 0.2, 0.3, 0.4, 0.5)

    # must match the figure's bootstrap exactly
    N_BOOT: int = 10000
    CI: float = 0.95
    SEED: int = 10

    OUT_ROOT: Path = Path("./results/mechanism")
    SUBDIR: str = "nta_lnmr"


CFG = Config()


def _out(protocol: str) -> Path:
    d = CFG.OUT_ROOT / protocol / CFG.SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _raw_fold(protocol: str) -> pd.DataFrame:
    init, optim, folder = CFG.PROTOCOLS[protocol]
    fp = CFG.EXPERIMENT_ROOT / folder / CFG.RAW_FOLD_CSV
    if not fp.exists():
        raise FileNotFoundError(f"raw fold csv not found: {fp}")
    df = pd.read_csv(fp)
    for col, val in (("init", init), ("optim", optim), ("dataset", CFG.DATASET)):
        if col in df.columns:
            df = df[df[col] == val]
    return df[df.method.isin(CFG.METHODS)].reset_index(drop=True)


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


def build(protocol: str):
    raw = _raw_fold(protocol).copy()
    # Third-class residual, computed per fold so its CI is bootstrapped directly
    raw["residual"] = 1.0 - raw["nta"] - raw["lnmr"]
    rows = []
    for method in CFG.METHODS:
        for tau in CFG.TAUS_NONZERO:
            sub = raw[(raw.method == method) & (np.isclose(raw.tau, tau))]
            n_m, n_lo, n_hi = _boot_ci(sub["nta"].values)
            l_m, l_lo, l_hi = _boot_ci(sub["lnmr"].values)
            r_m, r_lo, r_hi = _boot_ci(sub["residual"].values)
            rows.append(dict(method=method, tau=float(tau),
                             nta=n_m, nta_lo=n_lo, nta_hi=n_hi,
                             lnmr=l_m, lnmr_lo=l_lo, lnmr_hi=l_hi,
                             residual=r_m, residual_lo=r_lo, residual_hi=r_hi,
                             n=int(sub["nta"].notna().sum())))
    return pd.DataFrame(rows)


def _cell(df, metric, method, tau, best_method):
    """A \\makecell value-over-CI cell; bold if this method is best at (metric, tau)."""
    r = df[(df.method == method) & (np.isclose(df.tau, tau))]
    if r.empty or np.isnan(r[metric].values[0]):
        return r"\makecell{--}"
    m = r[metric].values[0]
    lo = r[f"{metric}_lo"].values[0]
    hi = r[f"{metric}_hi"].values[0]
    val = f"\\mathbf{{{m:.3f}}}" if method == best_method else f"{m:.3f}"
    return (r"\makecell{$%s$\\{\scriptsize $(%.3f,\,%.3f)$}}" % (val, lo, hi))


def _best_method(df, metric, tau, low_is_better):
    sub = df[np.isclose(df.tau, tau)]
    if sub.empty or sub[metric].isna().all():
        return None
    idx = sub[metric].idxmin() if low_is_better else sub[metric].idxmax()
    return sub.loc[idx, "method"]


def _value_block_rows(df, metrics_best):
    """Build the per-tau row strings for a set of (metric, low_is_better) cols."""
    methods = list(CFG.METHODS)
    rows = []
    for tau in CFG.TAUS_NONZERO:
        cells = [f"{tau:.1f}"]
        for metric, low_is_better in metrics_best:
            best = (None if low_is_better is None
                    else _best_method(df, metric, tau, low_is_better))
            cells += [_cell(df, metric, m, tau, best) for m in methods]
        rows.append(" & ".join(cells) + r" \\")
    return rows


def emit_table(df: pd.DataFrame, protocol: str):
    lab = CFG.METHOD_LABELS
    methods = list(CFG.METHODS)
    mlabels = [lab.get(m, m) for m in methods]
    nM = len(methods)
    outdir = _out(protocol)

    # Table 1: NTA + LNMR (two blocks)
    rows = _value_block_rows(df, [("nta", False), ("lnmr", True)])
    colspec = "l" + ("*{%d}{c}" % nM) * 2
    header_group = (r" & \multicolumn{%d}{c}{NTA (recovered to true)} "
                    r"& \multicolumn{%d}{c}{LNMR (memorized noisy)} \\" % (nM, nM))
    cmid = (r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}"
            % (1 + nM, 2 + nM, 1 + 2 * nM))
    header_methods = (r"$\tau$ & " + " & ".join(mlabels) + " & "
                      + " & ".join(mlabels) + r" \\")
    tex1 = [
        r"\begin{table}[h!]", r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{%s}" % colspec, r"\toprule",
        header_group, cmid, header_methods, r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}}",
        r"\caption{Aggregate memorization diagnostics under the " + protocol +
        r" protocol by noise rate $\tau$: noisy-true accuracy (NTA, the rate at "
        r"which flipped samples are recovered to their true class) and "
        r"label-noise memorization rate (LNMR, the rate at which they are "
        r"predicted as their assigned noisy label). Values are the mean across "
        r"ten folds with $95\%$ bootstrap confidence intervals below, matching "
        r"Figure~\ref{fig:nta-lnmr}. The best method per row is in bold: highest "
        r"NTA and lowest LNMR (less memorization is better). The residual "
        r"third-class rate is reported in Table~\ref{tab:nta-lnmr-residual-" + protocol + r"}. "
        r"$\tau = 0$ is omitted as the quantities are undefined without flipped "
        r"labels.}",
        r"\label{tab:nta-lnmr-%s}" % protocol,
        r"\end{table}",
    ]
    fp1 = outdir / f"tab_nta_lnmr_{protocol}.tex"
    fp1.write_text("\n".join(tex1) + "\n")
    print(f"[tab] wrote {fp1}")

    # Table 2: residual only
    rows_r = _value_block_rows(df, [("residual", None)])
    colspec_r = "l" + "*{%d}{c}" % nM
    header_r = r"$\tau$ & " + " & ".join(mlabels) + r" \\"
    tex2 = [
        r"\begin{table}[h!]", r"\centering",
        r"\begin{tabular}{%s}" % colspec_r, r"\toprule",
        header_r, r"\midrule", *rows_r,
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Residual third-class rate ($1-\text{NTA}-\text{LNMR}$) under "
        r"the " + protocol + r" protocol by noise rate $\tau$: the fraction of "
        r"flipped samples predicted as neither their true class nor their "
        r"assigned noisy label. Values are the mean across ten folds with $95\%$ "
        r"bootstrap confidence intervals below, computed per fold and "
        r"bootstrapped directly. This complements the NTA and LNMR values in "
        r"Table~\ref{tab:nta-lnmr-" + protocol + r"}; the three quantities sum "
        r"to one. $\tau = 0$ is omitted.}",
        r"\label{tab:nta-lnmr-residual-%s}" % protocol,
        r"\end{table}",
    ]
    fp2 = outdir / f"tab_nta_lnmr_residual_{protocol}.tex"
    fp2.write_text("\n".join(tex2) + "\n")
    print(f"[tab] wrote {fp2}")


def main():
    for protocol in CFG.PROTOCOLS_TO_RUN:
        if protocol not in CFG.PROTOCOLS:
            print(f"[skip] {protocol}: not in CONFIG.PROTOCOLS")
            continue
        try:
            df = build(protocol)
        except FileNotFoundError as e:
            print(f"[skip] {protocol}: {e}")
            continue
        df.to_csv(_out(protocol) / f"_nta_lnmr_{protocol}.csv", index=False)
        print(f"[csv] wrote {_out(protocol) / f'_nta_lnmr_{protocol}.csv'}")
        emit_table(df, protocol)
    print("Done.")


if __name__ == "__main__":
    main()