from __future__ import annotations

import pandas as pd

from src.assurance.claim_evidence_matrix import build_claim_evidence_matrix


def test_claim_evidence_matrix_has_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "case_id": ["c1", "c2"],
            "ai_pred": [0, 1],
            "numerical_confidence": [0.9, 0.8],
            "calibration_risk": [0.1, 0.2],
            "distance_uncertainty": [0.2, 0.4],
            "neighbor_error_rate": [0.1, 0.3],
            "wrong_confident_risk": [0.2, 0.5],
            "risk_category": ["low", "high"],
            "selected_route": ["AI", "Escalate"],
            "selected_expert": ["", "expert_a|expert_b"],
            "final_prediction": [0, 1],
            "decision_reason": ["low_risk_ai_allowed", "high_magd_risk_escalate"],
            "fairness_risk": [0.0, 0.1],
            "capacity_status": [None, "assigned"],
        }
    )
    matrix = build_claim_evidence_matrix(frame)
    assert set(["claim_id", "claim_text", "evidence_fields", "coverage_score", "missing_evidence_count", "example_case_ids"]).issubset(matrix.columns)
    assert len(matrix) == 7
    assert matrix["example_case_ids"].astype(str).str.len().gt(0).all()


def test_claim_evidence_coverage_is_between_zero_and_one() -> None:
    frame = pd.DataFrame({"case_id": ["c1"], "ai_pred": [0], "selected_route": ["AI"], "decision_reason": ["ok"], "final_prediction": [0]})
    matrix = build_claim_evidence_matrix(frame)
    assert matrix["coverage_score"].between(0.0, 1.0).all()
