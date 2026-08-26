from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.deferral.capacity_assignment import CapacityState
from src.deferral.expert_routing import estimate_expert_reliability, route_cases


def _core() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3"],
            "y_true": [1, 0, 1],
            "ai_score": [0.3, 0.4, 0.6],
            "ai_pred": [0, 1, 1],
            "magd_assurance_risk": [0.6, 0.5, 0.9],
            "risk_category": ["medium", "medium", "high"],
            "adaptive_threshold": [0.4, 0.4, 0.4],
            "passes_threshold": [False, False, False],
            "batch": ["b1", "b1", "b1"],
            "expert_a": [1, 0, 1],
            "expert_b": [1, None, 0],
        }
    )


def _expert_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expert": ["expert_a", "expert_b"],
            "accuracy": [0.9, 0.8],
            "false_positive_rate": [0.1, 0.2],
            "false_negative_rate": [0.1, 0.2],
            "cost_sensitive_loss": [1.0, 2.0],
            "groupwise_false_positive_rate": [0.1, 0.2],
            "groupwise_false_negative_rate": [0.1, 0.2],
            "bias_risk": [0.05, 0.10],
            "remaining_capacity_total": [1.0, 2.0],
            "remaining_capacity_by_batch": ['{"b1": 1}', '{"b1": 2}'],
        }
    )


def _historical_frame(with_sensitive: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3", "c4"],
            "split": ["train", "train", "val", "val"],
            "y_true": [1, 0, 1, 0],
            "expert_a": [1, 0, 1, 1],
            "expert_b": [1, 1, 0, 0],
            "oracle_expert": [1, 0, 1, 0],
        }
    )
    if with_sensitive:
        frame["customer_age"] = ["young", "young", "old", "old"]
    return frame


def test_capacity_is_never_exceeded_when_capacity_exists() -> None:
    decisions = route_cases(
        _core(),
        ranked_experts=["expert_a", "expert_b"],
        expert_metrics=_expert_metrics(),
        capacity_state=CapacityState(
            remaining={("b1", "expert_a"): 1, ("b1", "expert_b"): 2},
            batch_column="batch",
        ),
        fp_cost=0.057,
        fn_cost=1.0,
        fairness_penalty_weight=0.2,
        capacity_penalty_weight=0.2,
        human_review_cost=0.05,
        escalation_cost=0.10,
        top_k=2,
    )
    usage_a = int((decisions["selected_expert"] == "expert_a").sum()) + int(
        decisions["selected_expert"].astype(str).str.contains("expert_a\\||\\|expert_a|^expert_a$", regex=True).sum()
    ) - int((decisions["selected_expert"] == "expert_a").sum())
    assert usage_a <= 1


def test_selected_expert_is_available() -> None:
    core = _core()
    decisions = route_cases(
        core,
        ranked_experts=["expert_a", "expert_b"],
        expert_metrics=_expert_metrics(),
        capacity_state=CapacityState(remaining=None, batch_column=None),
        fp_cost=0.057,
        fn_cost=1.0,
        fairness_penalty_weight=0.2,
        capacity_penalty_weight=0.2,
        human_review_cost=0.05,
        escalation_cost=0.10,
        top_k=2,
    )
    merged = decisions.merge(core[["case_id", "expert_a", "expert_b"]], on="case_id", how="left")
    human_rows = merged[merged["selected_route"] == "Human Expert"]
    for _, row in human_rows.iterrows():
        assert row["selected_expert"] in {"expert_a", "expert_b"}
        assert pd.notna(row[row["selected_expert"]])


def test_oracle_is_not_used_in_deployable_routing() -> None:
    decisions = route_cases(
        _core(),
        ranked_experts=["expert_a", "expert_b"],
        expert_metrics=_expert_metrics(),
        capacity_state=CapacityState(remaining=None, batch_column=None),
        fp_cost=0.057,
        fn_cost=1.0,
        fairness_penalty_weight=0.2,
        capacity_penalty_weight=0.2,
        human_review_cost=0.05,
        escalation_cost=0.10,
        top_k=2,
    )
    assert not decisions["selected_route"].astype(str).str.contains("oracle", case=False).any()
    assert not decisions["decision_reason"].astype(str).str.contains("oracle", case=False).any()
    assert not decisions["selected_expert"].astype(str).str.contains("oracle", case=False).any()


def test_experts_have_reliability_rows() -> None:
    reliability = estimate_expert_reliability(
        _historical_frame(),
        expert_columns=["expert_a", "expert_b"],
        label_column="y_true",
        sensitive_attributes=["customer_age"],
        fp_cost=0.057,
        fn_cost=1.0,
        capacity_by_expert={"expert_a": {"b1": 1}},
        capacity_totals={"expert_a": 1, "expert_b": 0},
        fairness_penalty_weight=0.2,
        capacity_penalty_weight=0.2,
        capacity_enabled=True,
    )
    assert reliability["expert"].tolist() == ["expert_a", "expert_b"]
    assert {"accuracy", "false_positive_rate", "false_negative_rate", "cost_sensitive_loss", "expected_cost"}.issubset(
        reliability.columns
    )


def test_missing_capacity_does_not_crash() -> None:
    reliability = estimate_expert_reliability(
        _historical_frame(),
        expert_columns=["expert_a", "expert_b"],
        label_column="y_true",
        sensitive_attributes=["customer_age"],
        fp_cost=0.057,
        fn_cost=1.0,
        fairness_penalty_weight=0.2,
        capacity_penalty_weight=0.2,
        capacity_enabled=False,
    )
    assert (reliability["remaining_capacity_total"] == 0.0).all()
    assert not reliability["capacity_enabled"].any()


def test_fairness_metrics_only_if_sensitive_attribute_exists() -> None:
    no_sensitive = estimate_expert_reliability(
        _historical_frame(with_sensitive=False),
        expert_columns=["expert_a", "expert_b"],
        label_column="y_true",
        sensitive_attributes=["customer_age"],
        fp_cost=0.057,
        fn_cost=1.0,
    )
    assert not no_sensitive["fairness_available"].any()
    assert (no_sensitive["bias_risk"] == 0.0).all()


def test_oracle_is_not_used_as_expert_in_reliability() -> None:
    reliability = estimate_expert_reliability(
        _historical_frame(),
        expert_columns=["expert_a", "expert_b", "oracle_expert"],
        label_column="y_true",
        sensitive_attributes=["customer_age"],
        fp_cost=0.057,
        fn_cost=1.0,
    )
    assert "oracle_expert" not in reliability["expert"].tolist()
