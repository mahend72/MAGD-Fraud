from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.reviewer_workload import (
    PANEL_K,
    ReviewerWorkloadSummary,
    break_even_reviewer_cost,
    compute_reviewer_workload,
    cost_sensitivity_table,
    summarize_reviewer_workload,
    summary_from_canonical_metrics_row,
    total_cost_at_reviewer_cost,
    which_is_cheaper_at,
)

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_METRICS_PATH = ROOT / "data" / "outputs" / "final_reproducible_run" / "final_canonical_metrics_audit.csv"


def test_panel_k_matches_frozen_escalation_config() -> None:
    """panel_k must equal the frozen top_k_for_escalation used by expert_routing.py's
    escalation panel - this is not an independent constant, it must track the same
    value that actually determines how many experts an escalation queries."""
    assert PANEL_K == 5


def test_compute_reviewer_workload_counts_expert_and_escalation_routes() -> None:
    decisions = pd.DataFrame(
        {"selected_route": ["AI", "Human Expert", "Human Expert", "Escalate", "AI"]}
    )
    n_expert, n_escalation, workload = compute_reviewer_workload(decisions, panel_k=5)
    assert n_expert == 2
    assert n_escalation == 1
    assert workload == 2 + 5 * 1


def test_total_cost_formula() -> None:
    summary = ReviewerWorkloadSummary(method="X", n_total=10, n_expert=2, n_escalation=1, reviewer_workload=7, fraud_loss=100.0)
    assert total_cost_at_reviewer_cost(summary, 0.0) == pytest.approx(100.0)
    assert total_cost_at_reviewer_cost(summary, 2.0) == pytest.approx(100.0 + 2.0 * 7)


def test_break_even_reviewer_cost_recovers_analytic_crossing() -> None:
    cheap_fraud_high_workload = ReviewerWorkloadSummary(method="A", n_total=10, n_expert=0, n_escalation=10, reviewer_workload=50, fraud_loss=10.0)
    expensive_fraud_low_workload = ReviewerWorkloadSummary(method="B", n_total=10, n_expert=1, n_escalation=0, reviewer_workload=1, fraud_loss=59.0)
    c_star = break_even_reviewer_cost(cheap_fraud_high_workload, expensive_fraud_low_workload)
    assert c_star == pytest.approx((59.0 - 10.0) / (50 - 1))
    assert total_cost_at_reviewer_cost(cheap_fraud_high_workload, c_star) == pytest.approx(
        total_cost_at_reviewer_cost(expensive_fraud_low_workload, c_star)
    )
    assert which_is_cheaper_at(cheap_fraud_high_workload, expensive_fraud_low_workload, c_star - 0.1) == "A"
    assert which_is_cheaper_at(cheap_fraud_high_workload, expensive_fraud_low_workload, c_star + 0.1) == "B"


def test_break_even_is_none_for_identical_workload() -> None:
    a = ReviewerWorkloadSummary(method="A", n_total=10, n_expert=0, n_escalation=2, reviewer_workload=10, fraud_loss=5.0)
    b = ReviewerWorkloadSummary(method="B", n_total=10, n_expert=10, n_escalation=0, reviewer_workload=10, fraud_loss=8.0)
    assert break_even_reviewer_cost(a, b) is None


def test_cost_sensitivity_table_contains_one_row_per_summary_and_all_requested_costs() -> None:
    summaries = [
        ReviewerWorkloadSummary(method="A", n_total=10, n_expert=1, n_escalation=0, reviewer_workload=1, fraud_loss=10.0),
        ReviewerWorkloadSummary(method="B", n_total=10, n_expert=0, n_escalation=2, reviewer_workload=10, fraud_loss=5.0),
    ]
    table = cost_sensitivity_table(summaries, reviewer_costs=(0.0, 1.0))
    assert len(table) == 2
    assert "total_cost_c_r=0.0" in table.columns
    assert "total_cost_c_r=1.0" in table.columns


def test_summarize_reviewer_workload_from_raw_decisions() -> None:
    decisions = pd.DataFrame({"selected_route": ["AI", "Human Expert", "Escalate"]})
    summary = summarize_reviewer_workload("method-x", decisions, fraud_loss=42.0, panel_k=5)
    assert summary.n_expert == 1
    assert summary.n_escalation == 1
    assert summary.reviewer_workload == 1 + 5 * 1
    assert summary.fraud_loss == 42.0


@pytest.mark.skipif(not CANONICAL_METRICS_PATH.exists(), reason="canonical final_reproducible_run metrics audit not present")
def test_canonical_reviewer_workload_matches_expert_plus_panel_k_escalation() -> None:
    """Regression guard: the reviewer_workload_cases column already saved in the
    canonical metrics audit must be reproducible from expert_deferral_rate and
    escalation_rate via N_expert + panel_k * N_escalation - this is the same formula
    generate_reviewer_cost_sensitivity.py depends on and must not silently drift."""
    metrics = pd.read_csv(CANONICAL_METRICS_PATH)
    for _, row in metrics.iterrows():
        summary = summary_from_canonical_metrics_row(row)
        expected = summary.n_expert + PANEL_K * summary.n_escalation
        assert summary.reviewer_workload == expected, (
            f"{row['method']}: reviewer_workload_cases={summary.reviewer_workload} != "
            f"N_expert + panel_k*N_escalation={expected}"
        )
