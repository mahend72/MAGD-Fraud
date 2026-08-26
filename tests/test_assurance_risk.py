from __future__ import annotations

import pandas as pd

from src.assurance.assurance_risk import (
    add_risk_categories_and_actions,
    compute_assurance_risk,
    map_risk_category,
    recommended_action_for_category,
)


def _weights() -> dict[str, float]:
    return {
        "calibration_risk": 0.2,
        "distance_uncertainty": 0.2,
        "neighbor_error_rate": 0.2,
        "wrong_confident_risk": 0.3,
        "business_risk": 0.1,
    }


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3"],
            "calibration_risk": [0.1, 0.4, 0.8],
            "distance_uncertainty": [0.1, 0.5, 0.8],
            "neighbor_error_rate": [0.0, 0.4, 0.9],
            "wrong_confident_risk": [0.1, 0.6, 0.95],
            "business_risk": [0.0, 0.2, 1.0],
        }
    )


def test_risk_score_between_zero_and_one() -> None:
    scored = compute_assurance_risk(_frame(), _weights())
    assert scored["assurance_risk"].between(0.0, 1.0).all()


def test_category_mapping_works() -> None:
    assert map_risk_category(0.2, 0.33, 0.66) == "low"
    assert map_risk_category(0.5, 0.33, 0.66) == "medium"
    assert map_risk_category(0.9, 0.33, 0.66) == "high"


def test_recommended_action_works() -> None:
    assert recommended_action_for_category("low") == "AI"
    assert recommended_action_for_category("medium") == "Human Expert"
    assert recommended_action_for_category("high") == "Escalate"


def test_enrichment_adds_category_and_action() -> None:
    scored = compute_assurance_risk(_frame(), _weights())
    enriched = add_risk_categories_and_actions(scored, 0.33, 0.66)
    assert "risk_category" in enriched.columns
    assert "recommended_action" in enriched.columns
