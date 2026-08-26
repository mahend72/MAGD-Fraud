"""Publication figure: operational-loss sensitivity to the reviewer-action cost
c_R, for MAGD-Fraud vs. L2D-Standard. Analysis/visualisation only -- reads
frozen canonical loss and reviewer-action counts, does not retrain, reroute, or
alter any threshold, model, or metric.

L_operational(c_R) = L_fraud + c_R * (N_expert + 5 * N_escalation)

Source of truth: data/outputs/final_reproducible_run/final_canonical_metrics_audit.csv
(cost_sensitive_loss = L_fraud; reviewer_workload_cases = N_expert + 5*N_escalation),
cross-checked against the frozen decision logs (same recipe as Figure 1).

c_R is a relative, loss-unit accounting-sensitivity parameter used to test how
the ranking between methods responds to how expensive reviewer actions are
assumed to be -- it is NOT a monetary estimate of human-review cost.
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
from figure_style import ACCENT_BLUE, FIGURE_DATA_DIR, NEUTRAL_COLOR, PROPOSED_COLOR, PROPOSED_DARK, apply_style, clean_axes, save_fig

CANONICAL_AUDIT = ROOT / "data/outputs/final_reproducible_run/final_canonical_metrics_audit.csv"
DECISION_LOGS = ROOT / "data/outputs/final_reproducible_run/decision_logs"
MAGD_V2_DECISIONS = ROOT / "data/outputs/final_reproducible_run/magd_v2_test_decisions.csv"
PLOT_DATA_CSV = FIGURE_DATA_DIR / "reviewer_cost_sensitivity_plot_data.csv"

EXPECTED = {
    "MAGD-Fraud": {"loss": 1011.48, "actions": 73420},
    "L2D-Standard": {"loss": 1086.61, "actions": 3627},
}
EXPECTED_BREAKEVEN = 1.08e-3


def load_canonical_values() -> dict[str, dict[str, float]]:
    audit = pd.read_csv(CANONICAL_AUDIT)
    magd_row = audit.loc[audit["method"] == "MAGD-v2"].iloc[0]
    l2d_row = audit.loc[audit["method"] == "L2D-Standard"].iloc[0]

    values = {
        "MAGD-Fraud": {
            "loss": float(magd_row["cost_sensitive_loss"]),
            "actions": int(magd_row["reviewer_workload_cases"]),
        },
        "L2D-Standard": {
            "loss": float(l2d_row["cost_sensitive_loss"]),
            "actions": int(l2d_row["reviewer_workload_cases"]),
        },
    }

    # Cross-check against the raw frozen decision logs, same recipe as Figure 1.
    magd_decisions = pd.read_csv(MAGD_V2_DECISIONS)
    magd_escalations = int((magd_decisions["selected_route"] == "Escalate").sum())
    magd_deferrals = int((magd_decisions["selected_route"] == "Human Expert").sum())
    magd_actions_recomputed = magd_deferrals + 5 * magd_escalations

    l2d_decisions = pd.read_csv(DECISION_LOGS / "learning_to_defer_decisions.csv")
    l2d_escalations = int((l2d_decisions["selected_route"] == "Escalate").sum())
    l2d_deferrals = int((l2d_decisions["selected_route"] == "Human Expert").sum())
    l2d_actions_recomputed = l2d_deferrals + 5 * l2d_escalations

    assert magd_actions_recomputed == values["MAGD-Fraud"]["actions"], "MAGD-Fraud reviewer-action count mismatch vs decision log"
    assert l2d_actions_recomputed == values["L2D-Standard"]["actions"], "L2D-Standard reviewer-action count mismatch vs decision log"

    for name, expected in EXPECTED.items():
        assert abs(values[name]["loss"] - expected["loss"]) < 0.02, f"{name}: canonical loss {values[name]['loss']} != expected {expected['loss']}"
        assert values[name]["actions"] == expected["actions"], f"{name}: canonical actions {values[name]['actions']} != expected {expected['actions']}"

    return values


def analytical_breakeven(values: dict[str, dict[str, float]]) -> float:
    """Solve L_MAGD(c_R) = L_L2D(c_R) for c_R analytically from the canonical
    values (never hard-coded)."""
    l_magd, a_magd = values["MAGD-Fraud"]["loss"], values["MAGD-Fraud"]["actions"]
    l_l2d, a_l2d = values["L2D-Standard"]["loss"], values["L2D-Standard"]["actions"]
    # l_magd + c_R*a_magd = l_l2d + c_R*a_l2d  =>  c_R = (l_l2d - l_magd) / (a_magd - a_l2d)
    c_r_star = (l_l2d - l_magd) / (a_magd - a_l2d)
    return c_r_star


def build_curves(values: dict[str, dict[str, float]], c_r_star: float) -> pd.DataFrame:
    c_r_max = 2.5 * c_r_star
    c_r = np.linspace(0.0, c_r_max, 400)
    rows = []
    for name, v in values.items():
        loss = v["loss"] + c_r * v["actions"]
        for cr, l in zip(c_r, loss):
            rows.append({"method": name, "c_R": cr, "operational_loss": l, "L_fraud": v["loss"], "reviewer_actions": v["actions"]})
    return pd.DataFrame(rows)


def make_figure(curves: pd.DataFrame, c_r_star: float, values: dict[str, dict[str, float]]) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(3.6, 3.3), dpi=300)

    # Plot c_R in units of 1e-3 loss-units-per-action so tick labels stay short
    # plain decimals (no separate scientific-notation offset box to collide
    # with the axis label).
    scale = 1e-3
    for name, color in [("MAGD-Fraud", PROPOSED_COLOR), ("L2D-Standard", NEUTRAL_COLOR)]:
        sub = curves[curves["method"] == name]
        ax.plot(
            sub["c_R"] / scale, sub["operational_loss"],
            color=color, linewidth=1.8 if name == "MAGD-Fraud" else 1.4,
            zorder=3,
        )

    loss_at_breakeven = values["MAGD-Fraud"]["loss"] + c_r_star * values["MAGD-Fraud"]["actions"]
    ax.axvline(c_r_star / scale, color=ACCENT_BLUE, linewidth=0.9, linestyle="--", alpha=0.7, zorder=2)
    ax.annotate(
        f"break-even\ncR ≈ {c_r_star:.2e}",
        xy=(c_r_star / scale, loss_at_breakeven), xytext=(10, -30), textcoords="offset points",
        fontsize=7, color=ACCENT_BLUE, ha="left", va="top",
        arrowprops=dict(arrowstyle="-", color=ACCENT_BLUE, alpha=0.6, lw=0.7),
    )

    x_max = curves["c_R"].max() / scale
    end_magd = curves[(curves["method"] == "MAGD-Fraud")].iloc[-1]
    end_l2d = curves[(curves["method"] == "L2D-Standard")].iloc[-1]
    ax.annotate("MAGD-Fraud", xy=(end_magd["c_R"] / scale, end_magd["operational_loss"]), xytext=(-4, 6),
                textcoords="offset points", fontsize=7.5, ha="right", va="bottom", color=PROPOSED_DARK, fontweight="bold")
    ax.annotate("L2D-Standard", xy=(end_l2d["c_R"] / scale, end_l2d["operational_loss"]), xytext=(-4, -8),
                textcoords="offset points", fontsize=7.5, ha="right", va="top", color="#222222")

    ax.set_xlim(0, x_max)
    ax.set_xlabel("Reviewer-action cost, cR (×10⁻³ per action)")
    ax.set_ylabel("Operational loss (loss units)")
    ax.text(
        0.02, 0.02,
        "cR: relative loss-unit sensitivity parameter,\nnot a monetary cost estimate.",
        transform=ax.transAxes, fontsize=6, color="#666666", ha="left", va="bottom", style="italic",
    )
    clean_axes(ax)
    fig.tight_layout(pad=0.6)
    return fig


def main() -> None:
    values = load_canonical_values()
    c_r_star = analytical_breakeven(values)
    assert abs(c_r_star - EXPECTED_BREAKEVEN) / EXPECTED_BREAKEVEN < 0.05, (
        f"Analytical break-even {c_r_star:.4e} deviates >5% from expected {EXPECTED_BREAKEVEN:.2e}"
    )

    curves = build_curves(values, c_r_star)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    curves.to_csv(PLOT_DATA_CSV, index=False)

    fig = make_figure(curves, c_r_star, values)
    png_path, pdf_path = save_fig(fig, "reviewer_cost_sensitivity")
    plt.close(fig)

    print("Canonical values used:")
    for name, v in values.items():
        print(f"  {name}: L_fraud={v['loss']}, reviewer_actions={v['actions']}")
    print(f"Analytical break-even c_R = {c_r_star:.6e} (expected ~{EXPECTED_BREAKEVEN:.2e})")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {PLOT_DATA_CSV}")


if __name__ == "__main__":
    main()
