from __future__ import annotations

import math

import pandas as pd

from src.evaluation.audit_metrics import compute_audit_metrics


def test_audit_coverage_tracks_missing_evidence_and_rationale() -> None:
    frame = pd.DataFrame(
        {
            "case_id": ["c1", "c2"],
            "y_true": [0, 1],
            "final_prediction": [0, 1],
            "selected_route": ["AI", "Human Expert"],
            "decision_reason": ["ok", ""],
            "numerical_confidence": [0.9, 0.8],
            "distance_uncertainty": [0.1, None],
        }
    )
    metrics = compute_audit_metrics(frame, required_evidence_columns=["numerical_confidence", "distance_uncertainty"])

    assert math.isclose(metrics["audit_coverage"], 0.5)
    assert math.isclose(metrics["evidence_completeness"], 0.75)
    assert math.isclose(metrics["complete_decision_logs_rate"], 1.0)
    assert math.isclose(metrics["missing_evidence_rate"], 0.5)
    assert math.isclose(metrics["missing_rationale_rate"], 0.5)
