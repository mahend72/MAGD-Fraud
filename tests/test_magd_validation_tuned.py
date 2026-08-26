from __future__ import annotations

import inspect

import pandas as pd

from src.assurance.magd_risk import compute_magd_risk
from src.deferral.magd_validation_tuned import apply_budgeted_policy, tune_validation_policy


def _core() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3", "c4"],
            "y_true": [0, 1, 1, 0],
            "ai_score": [0.1, 0.2, 0.8, 0.9],
            "ai_pred": [0, 0, 1, 1],
            "numerical_confidence": [0.9, 0.8, 0.8, 0.9],
            "distance_confidence": [0.9, 0.2, 0.7, 0.3],
            "distance_uncertainty": [0.1, 0.8, 0.3, 0.7],
            "calibration_risk": [0.0, 0.2, 0.1, 0.1],
            "neighbor_error_rate": [0.0, 0.7, 0.2, 0.3],
            "neighbor_fraud_rate": [0.0, 0.8, 0.3, 0.2],
            "neighbor_ai_agreement": [1.0, 0.2, 0.8, 0.7],
            "mean_neighbor_distance": [0.1, 0.8, 0.2, 0.6],
            "top_k": [5, 5, 5, 5],
            "wrong_confident_risk": [0.1, 0.9, 0.2, 0.5],
            "expert_a": [0, 1, 1, 0],
        }
    )


def test_magd_risk_increases_for_high_uncertainty_cases() -> None:
    weights = {
        "distance_uncertainty": 0.6,
        "calibration_risk": 0.1,
        "neighbor_error_rate": 0.2,
        "wrong_confident_risk": 0.1,
        "drift_risk": 0.0,
        "business_risk": 0.0,
    }
    scored = compute_magd_risk(_core(), weights, use_drift_risk=False, use_business_risk=False)
    assert scored.loc[scored["case_id"].eq("c2"), "magd_assurance_risk"].iloc[0] > scored.loc[
        scored["case_id"].eq("c1"), "magd_assurance_risk"
    ].iloc[0]


def test_budgeted_policy_defers_highest_magd_risk_cases() -> None:
    weights = {
        "distance_uncertainty": 1.0,
        "calibration_risk": 0.0,
        "neighbor_error_rate": 0.0,
        "wrong_confident_risk": 0.0,
        "drift_risk": 0.0,
        "business_risk": 0.0,
    }
    decisions, threshold = apply_budgeted_policy(
        _core(),
        weights=weights,
        budget=0.25,
        threshold=None,
        ranked_experts=["expert_a"],
        method_name="MAGD-Fraud-ValidationTuned",
    )
    assert threshold >= 0.0
    deferred = decisions.loc[decisions["selected_route"].eq("Human Expert")]
    assert len(deferred) >= 1
    assert "c2" in set(deferred["case_id"])
    assert decisions["tuned_on_split"].eq("validation").all()


def test_validation_tuning_entrypoint_does_not_load_test_predictions() -> None:
    source = inspect.getsource(tune_validation_policy)
    assert "test_predictions.csv" not in source
    assert "validation_core" in source
