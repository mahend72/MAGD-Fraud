"""Publication figures 3 & 4: how the FROZEN MAGD-Fraud assurance model's
predicted risk responds to (a) calibration risk x confidence and (b)
calibration risk x neighbourhood risk. Analysis/visualisation only -- the
model is reconstructed via src/assurance/magd_v2.py's own
fit_final_v2_pipeline (fit once on validation, exactly as the frozen
pipeline does) and verified, before any plotting, to reproduce the
already-frozen `magd_assurance_risk` column in
data/outputs/final_reproducible_run/magd_v2_test_decisions.csv to floating-
point precision (see frozen_v2_model_utils.load_frozen_v2_model). Nothing is
refit, retuned, or rerouted.

Both response surfaces hold the two non-varying assurance signals at their
VALIDATION-DERIVED MEDIANS (distance_uncertainty_norm and, for Figure 3,
neighbor_error_rate_norm; for Figure 4, numerical_confidence). Both figures
share one colour scale so they can be read side by side.

These are partial fitted-model interpretations of the frozen logistic
assurance model -- not causal effects.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from figure_style import FIGURE_DATA_DIR, apply_style, save_fig
from frozen_v2_model_utils import load_frozen_v2_model, predict_risk_grid

GRID_N = 120
CAUTION_NOTE = "Partial fitted-model interpretation (frozen MAGD-Fraud logistic\nassurance model) -- not a causal effect."


def build_calibration_confidence_grid(frozen) -> pd.DataFrame:
    cal = np.linspace(0.0, 1.0, GRID_N)
    conf = np.linspace(0.5, 1.0, GRID_N)
    CAL, CONF = np.meshgrid(cal, conf)
    risk = predict_risk_grid(
        frozen,
        calibration_risk_norm=CAL,
        distance_uncertainty_norm=frozen.medians["distance_uncertainty_norm"],
        neighbor_error_rate_norm=frozen.medians["neighbor_error_rate_norm"],
        numerical_confidence=CONF,
    )
    return pd.DataFrame({"calibration_risk_norm": CAL.ravel(), "numerical_confidence": CONF.ravel(), "predicted_assurance_risk": risk.ravel()}), CAL, CONF, risk


def build_calibration_neighbour_grid(frozen) -> pd.DataFrame:
    cal = np.linspace(0.0, 1.0, GRID_N)
    nbr = np.linspace(0.0, 1.0, GRID_N)
    CAL, NBR = np.meshgrid(cal, nbr)
    risk = predict_risk_grid(
        frozen,
        calibration_risk_norm=CAL,
        distance_uncertainty_norm=frozen.medians["distance_uncertainty_norm"],
        neighbor_error_rate_norm=NBR,
        numerical_confidence=frozen.medians["numerical_confidence"],
    )
    return pd.DataFrame({"calibration_risk_norm": CAL.ravel(), "neighbor_error_rate_norm": NBR.ravel(), "predicted_assurance_risk": risk.ravel()}), CAL, NBR, risk


def make_heatmap(X, Y, Z, *, xlabel: str, ylabel: str, held_note: str, vmin: float, vmax: float, levels) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(3.8, 3.7), dpi=300)
    cf = ax.contourf(X, Y, Z, levels=levels, vmin=vmin, vmax=vmax, cmap="viridis")
    cs = ax.contour(X, Y, Z, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=5.5, fmt="%.2f", colors="white")

    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Predicted MAGD assurance risk", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(labelsize=8)

    fig.subplots_adjust(bottom=0.30)
    fig.text(0.03, 0.13, held_note, fontsize=6, color="#444444", ha="left", va="top", wrap=True)
    fig.text(0.03, 0.055, CAUTION_NOTE, fontsize=6, color="#666666", ha="left", va="top", style="italic")
    return fig


def main() -> None:
    frozen = load_frozen_v2_model()
    print(f"Frozen v2 model reconstruction verified: max abs diff vs magd_v2_test_decisions.csv = {frozen.max_reconstruction_diff:.3e}")
    print(f"Validation medians used as held-fixed values: {frozen.medians}")

    df3, CAL3, CONF3, RISK3 = build_calibration_confidence_grid(frozen)
    df4, CAL4, NBR4, RISK4 = build_calibration_neighbour_grid(frozen)

    vmin = float(min(RISK3.min(), RISK4.min()))
    vmax = float(max(RISK3.max(), RISK4.max()))
    levels = np.linspace(vmin, vmax, 13)

    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df3.to_csv(FIGURE_DATA_DIR / "calibration_confidence_interaction_plot_data.csv", index=False)
    df4.to_csv(FIGURE_DATA_DIR / "calibration_neighbour_interaction_plot_data.csv", index=False)

    held3 = (
        f"Held fixed at validation medians: distance_uncertainty_norm={frozen.medians['distance_uncertainty_norm']:.3f}, "
        f"neighbor_error_rate_norm={frozen.medians['neighbor_error_rate_norm']:.3f}"
    )
    fig3 = make_heatmap(
        CAL3, CONF3, RISK3,
        xlabel="Normalized calibration risk",
        ylabel="Confidence",
        held_note=held3, vmin=vmin, vmax=vmax, levels=levels,
    )
    png3, pdf3 = save_fig(fig3, "calibration_confidence_interaction")
    plt.close(fig3)

    held4 = (
        f"Held fixed at validation medians: distance_uncertainty_norm={frozen.medians['distance_uncertainty_norm']:.3f}, "
        f"confidence={frozen.medians['numerical_confidence']:.3f}"
    )
    fig4 = make_heatmap(
        CAL4, NBR4, RISK4,
        xlabel="Normalized calibration risk",
        ylabel="Normalized neighbourhood risk",
        held_note=held4, vmin=vmin, vmax=vmax, levels=levels,
    )
    png4, pdf4 = save_fig(fig4, "calibration_neighbour_interaction")
    plt.close(fig4)

    print(f"Shared color scale: vmin={vmin:.4f}, vmax={vmax:.4f}")
    print(f"Saved: {png3}\nSaved: {pdf3}")
    print(f"Saved: {png4}\nSaved: {pdf4}")
    print(f"Saved: {FIGURE_DATA_DIR / 'calibration_confidence_interaction_plot_data.csv'}")
    print(f"Saved: {FIGURE_DATA_DIR / 'calibration_neighbour_interaction_plot_data.csv'}")


if __name__ == "__main__":
    main()
