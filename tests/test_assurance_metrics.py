from __future__ import annotations

import math

import pandas as pd

from src.evaluation.assurance_metrics import (
    assurance_effectiveness,
    audit_coverage_rate,
    compute_assurance_metrics,
    correct_rejection_rate,
    evidence_completeness_rate,
    overreliance_rate,
    underreliance_rate,
    wrong_confident_avoidance_rate,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3", "c4"],
            "selected_route": ["AI", "Human Expert", "Human Expert", "AI"],
            "final_prediction": [1, 0, 1, 1],
            "y_true": [1, 1, 0, 0],
            "ai_pred": [1, 0, 1, 1],
            "ai_correct": [1, 0, 0, 0],
            "wrong_confident_label_offline": [0, 1, 0, 1],
            "risk_category": ["low", "high", "high", "medium"],
            "numerical_confidence": [0.9, 0.8, 0.7, 0.95],
            "distance_uncertainty": [0.1, 0.4, 0.3, 0.2],
            "calibration_risk": [0.05, 0.2, 0.3, 0.1],
            "neighbor_error_rate": [0.1, 0.5, 0.4, 0.2],
            "wrong_confident_risk": [0.2, 0.8, 0.4, 0.9],
            "magd_assurance_risk": [0.2, 0.8, 0.75, 0.5],
            "decision_reason": ["a", "b", "c", "d"],
        }
    )


def _ai_only_reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3", "c4"],
            "final_prediction": [1, 0, 1, 1],
        }
    )


def test_wrong_confident_avoidance_rate() -> None:
    assert math.isclose(wrong_confident_avoidance_rate(_frame()), 0.5)


def test_correct_rejection_rate() -> None:
    assert math.isclose(correct_rejection_rate(_frame()), 2.0 / 3.0)


def test_overreliance_rate() -> None:
    assert math.isclose(overreliance_rate(_frame()), 1.0 / 3.0)


def test_underreliance_rate() -> None:
    assert math.isclose(underreliance_rate(_frame()), 0.0)


def test_assurance_effectiveness() -> None:
    frame = _frame().copy()
    frame.loc[frame["case_id"] == "c2", "final_prediction"] = 1
    value = assurance_effectiveness(frame, _ai_only_reference(), fp_cost=1.0, fn_cost=5.0)
    assert math.isclose(value, 5.0)


def test_audit_coverage_rate() -> None:
    frame = _frame().copy()
    frame.loc[0, "magd_assurance_risk"] = None
    value = audit_coverage_rate(frame, ["numerical_confidence", "distance_uncertainty", "magd_assurance_risk"])
    assert math.isclose(value, 0.75)


def test_evidence_completeness_rate() -> None:
    frame = _frame().copy()
    frame.loc[0, "magd_assurance_risk"] = None
    value = evidence_completeness_rate(frame, ["numerical_confidence", "distance_uncertainty", "magd_assurance_risk"])
    assert math.isclose(value, (2 / 3 + 1 + 1 + 1) / 4)


def test_compute_assurance_metrics() -> None:
    metrics = compute_assurance_metrics(
        _frame(),
        ai_only_reference=_ai_only_reference(),
        fp_cost=1.0,
        fn_cost=5.0,
        required_evidence_columns=[
            "numerical_confidence",
            "distance_uncertainty",
            "calibration_risk",
            "neighbor_error_rate",
            "wrong_confident_risk",
            "magd_assurance_risk",
            "selected_route",
            "final_prediction",
            "decision_reason",
        ],
    )
    assert "wrong_confident_avoidance_rate" in metrics
    assert "assurance_effectiveness" in metrics
    assert "audit_coverage" in metrics
    assert "evidence_completeness" in metrics


def test_correct_rejection_and_overreliance_share_denominator() -> None:
    frame = _frame()
    assert math.isclose(correct_rejection_rate(frame) + overreliance_rate(frame), 1.0)


def test_wrong_confident_avoidance_zero_denominator_safe() -> None:
    frame = _frame().copy()
    frame["wrong_confident_label_offline"] = 0
    assert math.isclose(wrong_confident_avoidance_rate(frame), 0.0)


def test_underreliance_zero_denominator_safe() -> None:
    frame = _frame().copy()
    frame["ai_correct"] = 0
    assert math.isclose(underreliance_rate(frame), 0.0)
