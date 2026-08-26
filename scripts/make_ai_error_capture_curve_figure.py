"""Publication figure 5 (optional): cumulative AI-error capture as the reviewed
fraction (ranked by the frozen magd_assurance_risk score) increases.
Analysis/visualisation only -- reads the frozen test-set decisions, does not
retrain, reroute, or alter any threshold, model, or metric, and does not smooth
or otherwise manipulate the empirical curve.

Source of truth: data/outputs/final_reproducible_run/magd_v2_test_decisions.csv
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
from figure_style import ACCENT_BLUE, FIGURE_DATA_DIR, NEUTRAL_COLOR, PROPOSED_COLOR, apply_style, clean_axes, save_fig

MAGD_V2_DECISIONS = ROOT / "data/outputs/final_reproducible_run/magd_v2_test_decisions.csv"
PLOT_DATA_CSV = FIGURE_DATA_DIR / "ai_error_capture_curve_plot_data.csv"

EXPECTED_OPERATING_POINTS = {
    1: (305, 21.45),
    5: (605, 42.55),
    10: (656, 46.13),
    20: (686, 48.24),
}


def main() -> None:
    df = pd.read_csv(MAGD_V2_DECISIONS)
    df["ai_wrong"] = (df["ai_pred"].astype(int) != df["y_true"].astype(int)).astype(int)
    n = len(df)
    total_errors = int(df["ai_wrong"].sum())

    ranked = df.sort_values("magd_assurance_risk", ascending=False).reset_index(drop=True)
    cum_errors = ranked["ai_wrong"].cumsum().to_numpy()
    reviewed_pct = (np.arange(1, n + 1) / n) * 100.0
    captured_pct = cum_errors / total_errors * 100.0

    curve = pd.DataFrame({"reviewed_pct": reviewed_pct, "cumulative_ai_errors_captured": cum_errors, "pct_of_all_ai_errors_captured": captured_pct})

    operating_points = []
    for pct, (expected_captured, expected_pct) in EXPECTED_OPERATING_POINTS.items():
        n_top = max(1, int(round(n * pct / 100)))
        captured = int(cum_errors[n_top - 1])
        captured_rate = captured / total_errors * 100.0
        assert captured == expected_captured, f"Top {pct}%: captured {captured} != expected {expected_captured}"
        assert abs(captured_rate - expected_pct) < 0.02, f"Top {pct}%: captured_rate {captured_rate} != expected {expected_pct}"
        operating_points.append({"reviewed_pct": pct, "n_cases": n_top, "ai_errors_captured": captured, "pct_of_all_ai_errors_captured": round(captured_rate, 2)})
    operating_df = pd.DataFrame(operating_points)

    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_csv(PLOT_DATA_CSV, index=False)
    operating_df.to_csv(FIGURE_DATA_DIR / "ai_error_capture_curve_operating_points.csv", index=False)

    apply_style()
    fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=300)

    ax.plot(curve["reviewed_pct"], curve["pct_of_all_ai_errors_captured"], color=PROPOSED_COLOR, linewidth=1.6, zorder=3)
    ax.plot([0, 100], [0, 100], color=NEUTRAL_COLOR, linewidth=0.9, linestyle=":", alpha=0.6, zorder=2)
    ax.annotate("random review", xy=(70, 70), xytext=(6, -10), textcoords="offset points",
                fontsize=6.5, color="#555555", ha="left", va="top", rotation=27)

    label_offsets = {1: (8, -4), 5: (8, -11), 10: (10, 16), 20: (10, -4)}
    for _, row in operating_df.iterrows():
        ax.scatter([row["reviewed_pct"]], [row["pct_of_all_ai_errors_captured"]], s=22, color=ACCENT_BLUE, zorder=4, edgecolor="white", linewidth=0.5)
        dx, dy = label_offsets[int(row["reviewed_pct"])]
        ax.annotate(
            f"{int(row['reviewed_pct'])}%→{row['pct_of_all_ai_errors_captured']:.1f}%",
            xy=(row["reviewed_pct"], row["pct_of_all_ai_errors_captured"]),
            xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color="#274670", ha="left", va="center",
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Reviewed fraction (%, ranked by assurance risk)")
    ax.set_ylabel("Cumulative % of AI errors captured")
    clean_axes(ax)
    fig.tight_layout(pad=0.6)

    png_path, pdf_path = save_fig(fig, "ai_error_capture_curve")
    plt.close(fig)

    print(f"n={n}, total_ai_errors={total_errors}")
    print(operating_df.to_string(index=False))
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {PLOT_DATA_CSV}")
    print(f"Saved: {FIGURE_DATA_DIR / 'ai_error_capture_curve_operating_points.csv'}")


if __name__ == "__main__":
    main()
