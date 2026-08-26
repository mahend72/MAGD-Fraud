from __future__ import annotations

import pandas as pd

from src.deferral.adaptive_threshold import compute_adaptive_thresholds


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3"],
            "magd_assurance_risk": [0.2, 0.6, 0.9],
            "business_risk": [0.1, 0.3, 0.9],
            "fairness_risk": [0.0, 0.2, 0.5],
            "capacity_pressure": [0.0, 0.4, 1.0],
        }
    )


def test_adaptive_thresholds_are_numeric() -> None:
    thresholds = compute_adaptive_thresholds(
        _frame(),
        base_threshold=0.5,
        value_weight=0.1,
        fairness_weight=0.1,
        capacity_weight=0.1,
    )
    assert pd.api.types.is_numeric_dtype(thresholds["adaptive_threshold"])
    assert pd.api.types.is_bool_dtype(thresholds["passes_threshold"])


def test_adaptive_thresholds_are_clipped_to_zero_one() -> None:
    thresholds = compute_adaptive_thresholds(
        _frame(),
        base_threshold=0.95,
        value_weight=0.5,
        fairness_weight=0.5,
        capacity_weight=0.5,
    )
    assert thresholds["adaptive_threshold"].between(0.0, 1.0).all()


def test_higher_contextual_risk_cannot_ease_ai_retention() -> None:
    low_risk_context = pd.DataFrame(
        {
            "case_id": ["c1"],
            "magd_assurance_risk": [0.4],
            "business_risk": [0.0],
            "fairness_risk": [0.0],
            "capacity_pressure": [0.0],
        }
    )
    high_risk_context = pd.DataFrame(
        {
            "case_id": ["c1"],
            "magd_assurance_risk": [0.4],
            "business_risk": [0.8],
            "fairness_risk": [0.6],
            "capacity_pressure": [0.9],
        }
    )
    kwargs = dict(base_threshold=0.5, value_weight=0.1, fairness_weight=0.1, capacity_weight=0.1)
    low_context_result = compute_adaptive_thresholds(low_risk_context, **kwargs)
    high_context_result = compute_adaptive_thresholds(high_risk_context, **kwargs)

    low_threshold = float(low_context_result["adaptive_threshold"].iloc[0])
    high_threshold = float(high_context_result["adaptive_threshold"].iloc[0])
    assert high_threshold <= low_threshold

    low_pass = bool(low_context_result["passes_threshold"].iloc[0])
    high_pass = bool(high_context_result["passes_threshold"].iloc[0])
    # Higher contextual risk must never make AI retention MORE likely (monotonicity).
    assert high_pass <= low_pass


def test_missing_optional_signals_do_not_crash() -> None:
    thresholds = compute_adaptive_thresholds(
        pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "magd_assurance_risk": [0.1, 0.8],
            }
        ),
        base_threshold=0.5,
        value_weight=0.1,
        fairness_weight=0.1,
        capacity_weight=0.1,
    )
    assert len(thresholds) == 2
    assert (thresholds["business_risk"] == 0.0).all()
    assert (thresholds["fairness_risk"] == 0.0).all()
    assert (thresholds["capacity_pressure"] == 0.0).all()
