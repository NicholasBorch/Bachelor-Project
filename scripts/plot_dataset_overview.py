"""
Dataset overview figures for the Methods/Data section.

Produces two figures from the HAM10000 metadata, styled to match the
label-composition plot (same serif font, same teal bars):

  1. class_examples: one seeded example image per diagnostic class, in a row.
  2. data_distribution_dedup: per-class counts after lesion-level
     deduplication, annotated with count and percentage.

Both selections use SEED = 10.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread


# Config
@dataclass
class Config:
    # Inputs
    # HAM10000 metadata: lesion id + diagnosis columns
    METADATA_CSV: Path = Path("./data/raw/HAM10000/HAM10000_metadata.csv")
    LESION_ID_COL: str = "lesion_id"
    IMAGE_ID_COL: str = "image_id"
    DX_COL: str = "dx"

    # Directories searched (recursively) for "<image_id>.jpg".
    IMAGE_DIRS: tuple = (
        Path("./data/raw/HAM10000/HAM10000_images_part_1"),
        Path("./data/raw/HAM10000/HAM10000_images_part_2"),
    )
    IMAGE_EXT: str = ".jpg"

    # Seed (same as the rest of the thesis)
    SEED: int = 10

    # Classes (canonical order and full names)
    CLASS_NAMES: dict = field(default_factory=lambda: {
        "nv":    "Melanocytic nevi",
        "mel":   "Melanoma",
        "bkl":   "Benign keratosis",
        "bcc":   "Basal cell carcinoma",
        "akiec": "Actinic keratoses",
        "vasc":  "Vascular lesions",
        "df":    "Dermatofibroma",
    })
    # Order montage panels by prevalence (largest first); False uses dict order.
    MONTAGE_ORDER_BY_PREVALENCE: bool = True

    # Styling (matched to plot_label_composition.py)
    BAR_COLOR: str = "#1F7A8C"     # composition plot's TRUE_COLOR (teal/blue)
    BAR_ALPHA: float = 0.95
    BAR_EDGE: str = "white"
    BAR_EDGE_LW: float = 0.5
    BAR_TEXT_COLOR: str = "#1F7A8C"

    # Output
    OUT_DIR: Path = Path("./results/dataset_overview")
    EXAMPLES_STEM: str = "class_examples"
    DIST_STEM: str = "data_distribution_dedup"
    DPI: int = 200
    SAVE_PNG: bool = True
    SAVE_PDF: bool = True


CFG = Config()


# Styling (identical block to plot_label_composition.py::_style)
def _style():
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Palatino", "Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
        "mathtext.fontset":   "cm",
        "axes.unicode_minus": False,
        "figure.dpi":         150,
        "savefig.dpi":        CFG.DPI,
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.axisbelow":     True,
        "figure.facecolor":   "white",
        "savefig.facecolor":  "white",
    })


def _save(fig, stem: str):
    CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CFG.SAVE_PDF:
        fig.savefig(CFG.OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    if CFG.SAVE_PNG:
        fig.savefig(CFG.OUT_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {CFG.OUT_DIR / stem}.(pdf|png)")


# Data
def _load_metadata() -> pd.DataFrame:
    if not CFG.METADATA_CSV.exists():
        raise FileNotFoundError(f"metadata CSV not found: {CFG.METADATA_CSV}")
    df = pd.read_csv(CFG.METADATA_CSV)
    for col in (CFG.LESION_ID_COL, CFG.IMAGE_ID_COL, CFG.DX_COL):
        if col not in df.columns:
            raise KeyError(f"column '{col}' absent in {CFG.METADATA_CSV} "
                           f"(have: {list(df.columns)})")
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique lesion_id, chosen at random with SEED."""
    rng = np.random.default_rng(CFG.SEED)
    # Deterministic shuffle, then first row per lesion_id = uniform pick.
    shuffled = df.sample(frac=1.0, random_state=CFG.SEED).reset_index(drop=True)
    dedup = shuffled.drop_duplicates(subset=[CFG.LESION_ID_COL], keep="first")
    return dedup.reset_index(drop=True)


def _class_counts(dedup: pd.DataFrame) -> pd.Series:
    return dedup[CFG.DX_COL].value_counts()


def _find_image(image_id: str) -> Path | None:
    fname = f"{image_id}{CFG.IMAGE_EXT}"
    for d in CFG.IMAGE_DIRS:
        cand = d / fname
        if cand.exists():
            return cand
    # fall back to a recursive search in case of nested layouts
    for d in CFG.IMAGE_DIRS:
        if d.exists():
            hit = next(d.rglob(fname), None)
            if hit is not None:
                return hit
    return None


def _pick_example_per_class(df: pd.DataFrame, order: list[str]) -> dict:
    """One random image_id per class (seeded), resolved to a file path."""
    rng = np.random.default_rng(CFG.SEED)
    picks = {}
    for cls in order:
        rows = df[df[CFG.DX_COL] == cls]
        if rows.empty:
            picks[cls] = None
            continue
        # deterministic shuffle, then walk until an image file is found
        rows = rows.sample(frac=1.0, random_state=CFG.SEED)
        chosen_path = None
        for image_id in rows[CFG.IMAGE_ID_COL]:
            p = _find_image(str(image_id))
            if p is not None:
                chosen_path = p
                break
        picks[cls] = chosen_path
        if chosen_path is None:
            print(f"[warn] no image file found for class '{cls}' "
                  f"(searched {[str(d) for d in CFG.IMAGE_DIRS]})")
    return picks


# Figure 1: one example image per class
def fig_class_examples(df: pd.DataFrame, order: list[str]):
    _style()
    picks = _pick_example_per_class(df, order)
    n = len(order)
    fig, axes = plt.subplots(1, n, figsize=(2.05 * n, 2.5))
    axes = np.atleast_1d(axes).ravel()
    for ax, cls in zip(axes, order):
        path = picks.get(cls)
        if path is not None:
            try:
                ax.imshow(imread(str(path)))
            except Exception as e:
                ax.text(0.5, 0.5, "image\nunreadable", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="0.4")
                print(f"[warn] could not read {path}: {e}")
        else:
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       facecolor="0.92", edgecolor="0.7"))
            ax.text(0.5, 0.5, "no image", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="0.4")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        full = CFG.CLASS_NAMES.get(cls, cls)
        ax.set_title(f"{full}\n(\\texttt{{{cls}}})" if False else f"{full}\n({cls})",
                     fontsize=9.5)
    fig.suptitle("Example dermatoscopic image per diagnostic class",
                 y=1.04, fontsize=12.5)
    fig.tight_layout()
    _save(fig, CFG.EXAMPLES_STEM)


# Figure 2: deduplicated class distribution
def fig_distribution(counts: pd.Series):
    _style()
    order = list(counts.sort_values(ascending=False).index)
    vals = counts.reindex(order).to_numpy(dtype=float)
    total = float(vals.sum())
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x, vals, width=0.74, color=CFG.BAR_COLOR, alpha=CFG.BAR_ALPHA,
           edgecolor=CFG.BAR_EDGE, linewidth=CFG.BAR_EDGE_LW, zorder=3)

    pad = 0.012 * vals.max()
    for xi, v in zip(x, vals):
        pct = 100.0 * v / total if total > 0 else 0.0
        ax.text(xi, v + pad, f"{int(round(v))}\n({pct:.1f}\\%)" if False
                else f"{int(round(v))}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=8, color=CFG.BAR_TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Number of lesions")
    ax.set_ylim(0, vals.max() * 1.16)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.suptitle("HAM10000 class distribution after lesion-level deduplication",
                 y=0.98, fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, CFG.DIST_STEM)


# Main
def main():
    print(f"Loading metadata from {CFG.METADATA_CSV} ...")
    meta = _load_metadata()
    print(f"[load] {len(meta)} rows; "
          f"{meta[CFG.LESION_ID_COL].nunique()} unique lesions; "
          f"classes = {sorted(meta[CFG.DX_COL].unique())}")

    dedup = _deduplicate(meta)
    counts = _class_counts(dedup)
    print(f"[dedup] {len(dedup)} samples after one-per-lesion (seed={CFG.SEED})")
    print("[dedup] per-class counts:")
    for cls, c in counts.sort_values(ascending=False).items():
        print(f"        {cls:6s} {int(c):5d}  ({100.0 * c / len(dedup):.1f}%)")

    # montage order
    if CFG.MONTAGE_ORDER_BY_PREVALENCE:
        montage_order = list(counts.sort_values(ascending=False).index)
    else:
        montage_order = [c for c in CFG.CLASS_NAMES if c in counts.index]

    fig_class_examples(meta, montage_order)
    fig_distribution(counts)
    print("Done.")


if __name__ == "__main__":
    main()