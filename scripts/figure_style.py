"""Shared publication styling for the MAGD-Fraud EAAI manuscript figures.

Imported by every scripts/make_*_figure.py script so all Results figures share
identical typography, spine treatment, and export settings. Analysis/plotting
only -- no modeling or data-processing logic lives here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
FIGURE_DATA_DIR = ROOT / "data/outputs/final_reproducible_run/figure_data"

PROPOSED_COLOR = "#c44e52"  # MAGD-Fraud (proposed method) accent
PROPOSED_DARK = "#7a1f24"
NEUTRAL_COLOR = "#333333"  # other methods / baseline series
ACCENT_BLUE = "#4c72b0"  # reference lines / secondary series / annotations
GRID_GRAY = "#bbbbbb"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.set_axisbelow(True)


def save_fig(fig, stem: str) -> tuple[Path, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{stem}.png"
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path
