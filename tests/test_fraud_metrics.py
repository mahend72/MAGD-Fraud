from __future__ import annotations

import math

import pandas as pd

from src.evaluation.fraud_metrics import compute_fraud_metrics


def test_cost_sensitive_loss_matches_false_positive_and_false_negative_costs() -> None:
    frame = pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1],
            "final_prediction": [1, 0, 0, 1],
            "ai_score": [0.9, 0.1, 0.2, 0.8],
        }
    )
    metrics = compute_fraud_metrics(frame, score_column="ai_score", fp_cost=2.0, fn_cost=7.0)

    assert math.isclose(metrics["false_positives"], 1.0)
    assert math.isclose(metrics["false_negatives"], 1.0)
    assert math.isclose(metrics["cost_sensitive_loss"], 9.0)

