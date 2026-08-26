"""Publication figure: F1 vs. reviewer workload for the five final MAGD-Fraud
manuscript methods. Analysis/visualisation only -- reads frozen canonical
results and decision logs, does not retrain, reroute, or alter any threshold,
model, or metric.

Source of truth: data/outputs/final_reproducible_run/final_canonical_metrics_audit.csv
Cross-checked against the raw frozen decision logs in
data/outputs/final_reproducible_run/decision_logs/ and
data/outputs/final_reproducible_run/magd_v2_test_decisions.csv.

Reviewer workload = expert_deferral_rate + 5 * escalation_rate (an escalation
fans out to a panel of 5 reviewers; a direct expert deferral is 1 reviewer),
NOT the simple intervention rate (expert_deferral_rate + escalation_rate).
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from figure_style import ACCENT_BLUE, FIGURE_DATA_DIR, NEUTRAL_COLOR, PROPOSED_COLOR, PROPOSED_DARK, apply_style, clean_axes, save_fig

CANONICAL_AUDIT = ROOT / "data/outputs/final_reproducible_run/final_canonical_metrics_audit.csv"
DECISION_LOGS = ROOT / "data/outputs/final_reproducible_run/decision_logs"
MAGD_V2_DECISIONS = ROOT / "data/outputs/final_reproducible_run/magd_v2_test_decisions.csv"
PLOT_DATA_CSV = FIGURE_DATA_DIR / "f1_vs_reviewer_workload_plot_data.csv"

# manuscript label -> (canonical_audit `method` value, frozen decision-log file)
METHOD_MAP = {
    "AI-only": ("AI-only", DECISION_LOGS / "ai_only_decisions.csv"),
    "L2D-Standard": ("L2D-Standard", DECISION_LOGS / "learning_to_defer_decisions.csv"),
    "MAGD-Additive": ("MAGD-v1-Heuristic", DECISION_LOGS / "magd_heuristic_decisions.csv"),
    "MAGD-Additive-Tuned": ("MAGD-v1-ValidationTuned", DECISION_LOGS / "magd_validation_tuned_decisions.csv"),
    "MAGD-Fraud": ("MAGD-v2", MAGD_V2_DECISIONS),
}

EXPECTED = {
    "AI-only": (0.0718, 0.00),
    "L2D-Standard": (0.3629, 3.75),
    "MAGD-Additive": (0.2305, 97.83),
    "MAGD-Additive-Tuned": (0.0938, 1.95),
    "MAGD-Fraud": (0.4466, 75.81),
}


def cross_check() -> pd.DataFrame:
    audit = pd.read_csv(CANONICAL_AUDIT)
    rows = []
    for label, (audit_method, log_path) in METHOD_MAP.items():
        audit_row = audit.loc[audit["method"] == audit_method].iloc[0]
        f1_audit = float(audit_row["f1"])
        workload_audit = float(audit_row["reviewer_workload_rate"]) * 100.0

        decisions = pd.read_csv(log_path)
        n = len(decisions)
        f1_recomputed = f1_score(decisions["y_true"].astype(int), decisions["final_prediction"].astype(int))
        escalations = int((decisions["selected_route"] == "Escalate").sum())
        expert_deferrals = int((decisions["selected_route"] == "Human Expert").sum())
        workload_cases = expert_deferrals + 5 * escalations
        workload_recomputed = workload_cases / n * 100.0
        naive_intervention_rate = (expert_deferrals + escalations) / n * 100.0

        exp_f1, exp_workload = EXPECTED[label]
        assert abs(f1_audit - exp_f1) < 1e-3, f"{label}: audit F1 {f1_audit} != expected {exp_f1}"
        assert abs(workload_audit - exp_workload) < 0.02, f"{label}: audit workload {workload_audit} != expected {exp_workload}"
        assert abs(f1_recomputed - exp_f1) < 1e-3, f"{label}: recomputed F1 {f1_recomputed} != expected {exp_f1}"
        assert abs(workload_recomputed - exp_workload) < 0.02, f"{label}: recomputed workload {workload_recomputed} != expected {exp_workload}"
        assert abs(workload_audit - workload_recomputed) < 0.02, f"{label}: audit vs decision-log workload mismatch"

        rows.append(
            {
                "method": label,
                "canonical_audit_method_name": audit_method,
                "f1": round(f1_audit, 4),
                "reviewer_workload_pct": round(workload_audit, 2),
                "n_cases": n,
                "expert_deferrals": expert_deferrals,
                "escalations": escalations,
                "reviewer_workload_cases": workload_cases,
                "naive_intervention_rate_pct_DO_NOT_USE": round(naive_intervention_rate, 2),
                "source_decision_log": str(log_path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def pareto_flags(df: pd.DataFrame) -> pd.DataFrame:
    """A point is Pareto-optimal (for max F1, min workload) if no other point
    has workload <= its workload AND f1 >= its f1, with at least one strict."""
    flags = []
    for _, row in df.iterrows():
        dominated = False
        for _, other in df.iterrows():
            if other["method"] == row["method"]:
                continue
            not_worse = other["reviewer_workload_pct"] <= row["reviewer_workload_pct"] and other["f1"] >= row["f1"]
            strictly_better = other["reviewer_workload_pct"] < row["reviewer_workload_pct"] or other["f1"] > row["f1"]
            if not_worse and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    df = df.copy()
    df["on_pareto_frontier"] = flags
    return df


def make_figure(df: pd.DataFrame) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.1), dpi=300)

    # Subtle, non-cluttering indication of the preferred (high-F1, low-workload)
    # corner. Confined to the empty upper-left area well above/left of the
    # actual points, so it does not overlap or imply anything about specific
    # points.
    ax.axhspan(0.40, 0.50, xmin=0.0, xmax=0.10, color=ACCENT_BLUE, alpha=0.07, zorder=0, linewidth=0)
    ax.text(
        1.2, 0.487, "preferred\ndirection", fontsize=6.5, color=ACCENT_BLUE, alpha=0.8,
        ha="left", va="top", style="italic", zorder=1,
    )

    others = df[df["method"] != "MAGD-Fraud"]
    proposed = df[df["method"] == "MAGD-Fraud"]

    ax.scatter(
        others["reviewer_workload_pct"], others["f1"],
        s=32, facecolor="white", edgecolor=NEUTRAL_COLOR, linewidth=1.1, zorder=3,
    )
    ax.scatter(
        proposed["reviewer_workload_pct"], proposed["f1"],
        s=70, marker="*", facecolor=PROPOSED_COLOR, edgecolor=PROPOSED_DARK, linewidth=0.8, zorder=4,
    )

    label_offsets = {
        "AI-only": (6, -3, "left", "top"),
        "MAGD-Additive-Tuned": (6, 6, "left", "bottom"),
        "L2D-Standard": (6, 3, "left", "center"),
        "MAGD-Fraud": (-6, 6, "right", "bottom"),
        "MAGD-Additive": (-6, -3, "right", "top"),
    }
    for _, row in df.iterrows():
        dx, dy, ha, va = label_offsets[row["method"]]
        weight = "bold" if row["method"] == "MAGD-Fraud" else "normal"
        ax.annotate(
            row["method"],
            xy=(row["reviewer_workload_pct"], row["f1"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=7.5, ha=ha, va=va, fontweight=weight,
            color=PROPOSED_DARK if row["method"] == "MAGD-Fraud" else "#222222",
            zorder=5,
        )

    ax.set_xlim(-2, 102)
    ax.set_ylim(0, 0.50)
    ax.set_xlabel("Reviewer workload (% of test-set equivalent)")
    ax.set_ylabel("F1 score")
    clean_axes(ax)
    fig.tight_layout(pad=0.6)
    return fig


def main() -> None:
    df = cross_check()
    df = pareto_flags(df)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PLOT_DATA_CSV, index=False)
    fig = make_figure(df)
    png_path, pdf_path = save_fig(fig, "f1_vs_reviewer_workload")
    plt.close(fig)
    print(df.to_string(index=False))
    print()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {PLOT_DATA_CSV}")


if __name__ == "__main__":
    main()
