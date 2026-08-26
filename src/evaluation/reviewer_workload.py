"""Reviewer workload and reviewer-cost sensitivity, computed ONLY from already-frozen
decision logs (route counts + fraud cost). No case is rerouted, no model is refit, no
threshold is changed - this module is pure arithmetic over quantities that already
exist in the saved decision logs.

    ReviewerWorkload = N_expert + k * N_escalation
    L_total = L_fraud + c_R * (N_expert + k * N_escalation)

k (panel_k) = 5, matching the frozen MAGD escalation panel size used throughout.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

PANEL_K = 5

# Small, pre-specified range of reviewer-cost values to evaluate - not searched or
# tuned to produce any particular outcome; chosen only to span from "reviewer time is
# free" to "reviewer time costs as much as a false negative" relative to the fraud
# cost scale (fp_cost=0.057, fn_cost=1.0).
DEFAULT_REVIEWER_COST_RANGE = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


@dataclass
class ReviewerWorkloadSummary:
    method: str
    n_total: int
    n_expert: int
    n_escalation: int
    reviewer_workload: int
    fraud_loss: float


def compute_reviewer_workload(decisions: pd.DataFrame, *, panel_k: int = PANEL_K) -> tuple[int, int, int]:
    """Returns (n_expert, n_escalation, reviewer_workload) from an already-frozen
    decisions frame's `selected_route` column. Does not reroute anything."""
    route = decisions["selected_route"].astype(str)
    n_expert = int((route == "Human Expert").sum())
    n_escalation = int((route == "Escalate").sum())
    reviewer_workload = n_expert + panel_k * n_escalation
    return n_expert, n_escalation, reviewer_workload


def summarize_reviewer_workload(method: str, decisions: pd.DataFrame, fraud_loss: float, *, panel_k: int = PANEL_K) -> ReviewerWorkloadSummary:
    n_expert, n_escalation, workload = compute_reviewer_workload(decisions, panel_k=panel_k)
    return ReviewerWorkloadSummary(
        method=method, n_total=len(decisions), n_expert=n_expert, n_escalation=n_escalation,
        reviewer_workload=workload, fraud_loss=float(fraud_loss),
    )


def summary_from_canonical_metrics_row(row: pd.Series, *, panel_k: int = PANEL_K) -> ReviewerWorkloadSummary:
    """Builds a ReviewerWorkloadSummary directly from a row of the canonical
    final_canonical_metrics_audit.csv (data/outputs/final_reproducible_run/), which
    already carries a `reviewer_workload_cases` column computed with the same
    N_expert + panel_k * N_escalation formula. This is the preferred source for the
    final paper report: it reads only the frozen canonical audit artifact, never a raw
    decision log, so it cannot drift from what was already reported."""
    n_total = int(row["n"])
    n_expert = int(round(float(row["expert_deferral_rate"]) * n_total))
    n_escalation = int(round(float(row["escalation_rate"]) * n_total))
    return ReviewerWorkloadSummary(
        method=str(row["method"]),
        n_total=n_total,
        n_expert=n_expert,
        n_escalation=n_escalation,
        reviewer_workload=int(row["reviewer_workload_cases"]),
        fraud_loss=float(row["cost_sensitive_loss"]),
    )


def total_cost_at_reviewer_cost(summary: ReviewerWorkloadSummary, reviewer_cost: float) -> float:
    """L_total = L_fraud + c_R * ReviewerWorkload."""
    return summary.fraud_loss + reviewer_cost * summary.reviewer_workload


def cost_sensitivity_table(summaries: list[ReviewerWorkloadSummary], reviewer_costs: tuple[float, ...] = DEFAULT_REVIEWER_COST_RANGE) -> pd.DataFrame:
    rows = []
    for summary in summaries:
        row = {
            "method": summary.method,
            "n_expert": summary.n_expert,
            "n_escalation": summary.n_escalation,
            "reviewer_workload": summary.reviewer_workload,
            "fraud_loss": summary.fraud_loss,
        }
        for c_r in reviewer_costs:
            row[f"total_cost_c_r={c_r}"] = total_cost_at_reviewer_cost(summary, c_r)
        rows.append(row)
    return pd.DataFrame(rows)


def break_even_reviewer_cost(summary_a: ReviewerWorkloadSummary, summary_b: ReviewerWorkloadSummary) -> float | None:
    """Analytic break-even c_R where L_total(a) == L_total(b), solving the two linear
    equations exactly (no search/tuning): c_R* = (L_fraud_b - L_fraud_a) /
    (workload_a - workload_b). Returns None if the two methods' total-cost lines are
    parallel (identical workload) - no finite break-even exists in that case."""
    workload_diff = summary_a.reviewer_workload - summary_b.reviewer_workload
    if workload_diff == 0:
        return None
    return (summary_b.fraud_loss - summary_a.fraud_loss) / workload_diff


def which_is_cheaper_at(summary_a: ReviewerWorkloadSummary, summary_b: ReviewerWorkloadSummary, reviewer_cost: float) -> str:
    cost_a = total_cost_at_reviewer_cost(summary_a, reviewer_cost)
    cost_b = total_cost_at_reviewer_cost(summary_b, reviewer_cost)
    if cost_a < cost_b:
        return summary_a.method
    if cost_b < cost_a:
        return summary_b.method
    return "tie"
