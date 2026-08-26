from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assurance.wrong_confident_detector import (
    add_offline_wrong_confident_labels,
    compute_deployable_wrong_confident_risk,
    evaluate_wrong_confident_detection,
    load_assurance_inputs,
)


def _signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3"],
            "split": ["val", "test", "test"],
            "ai_score": [0.95, 0.60, 0.20],
            "ai_pred": [1, 1, 0],
            "numerical_confidence": [0.95, 0.60, 0.80],
            "distance_confidence": [0.70, 0.55, 0.75],
            "distance_uncertainty": [0.30, 0.45, 0.25],
            "calibration_risk": [0.20, 0.10, 0.05],
            "neighbor_error_rate": [0.40, 0.20, 0.10],
            "y_true": [0, 1, 0],
        }
    )


def _weights() -> dict[str, float]:
    return {
        "numerical_confidence": 0.25,
        "distance_uncertainty": 0.2,
        "calibration_risk": 0.2,
        "neighbor_error_rate": 0.2,
        "confidence_disagreement": 0.15,
    }


def test_risk_score_between_zero_and_one() -> None:
    frame = compute_deployable_wrong_confident_risk(_signal_frame().drop(columns=["y_true"]), _weights())
    assert frame["wrong_confident_risk"].between(0.0, 1.0).all()


def test_deployable_score_does_not_use_y_true() -> None:
    base = _signal_frame()
    variant = base.copy()
    variant["y_true"] = [1, 0, 1]
    scored_a = compute_deployable_wrong_confident_risk(base, _weights())
    scored_b = compute_deployable_wrong_confident_risk(variant, _weights())
    columns = ["case_id", "split", "confidence_disagreement", "wrong_confident_risk"]
    pd.testing.assert_frame_equal(scored_a[columns], scored_b[columns])


def test_offline_label_exists_only_after_evaluation_labeling() -> None:
    base = compute_deployable_wrong_confident_risk(_signal_frame().drop(columns=["y_true"]), _weights())
    assert "wrong_confident_label_offline" not in base.columns
    labeled = add_offline_wrong_confident_labels(base.assign(y_true=_signal_frame()["y_true"]), 0.9)
    assert "wrong_confident_label_offline" in labeled.columns
    metrics = evaluate_wrong_confident_detection(labeled, risk_alert_threshold=0.5, top_k_fraction=0.5)
    assert {"precision", "recall", "f1", "top_k_capture_rate"}.issubset(metrics)


def test_zero_sum_weights_fail_safely() -> None:
    zero_weights = {
        "numerical_confidence": 0.0,
        "distance_uncertainty": 0.0,
        "calibration_risk": 0.0,
        "neighbor_error_rate": 0.0,
        "confidence_disagreement": 0.0,
    }
    with pytest.raises(ValueError, match="positive"):
        compute_deployable_wrong_confident_risk(_signal_frame().drop(columns=["y_true"]), zero_weights)


def test_missing_inputs_raise_clear_errors(tmp_path: Path) -> None:
    repo_dir = tmp_path
    assurance_dir = repo_dir / "data" / "outputs" / "assurance"
    assurance_dir.mkdir(parents=True, exist_ok=True)

    config_path = repo_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  output_dir: data/outputs",
                "data:",
                "  dataset_name: fifar",
                "  train_path: data/processed/X_train.csv",
                "  val_path: data/processed/X_val.csv",
                "  test_path: data/processed/X_test.csv",
                "  y_train_path: data/processed/y_train.csv",
                "  y_val_path: data/processed/y_val.csv",
                "  y_test_path: data/processed/y_test.csv",
                "  sensitive_attribute: customer_age",
            ]
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "case_id": ["c1"],
            "split": ["test"],
            "ai_score": [0.9],
            "ai_pred": [1],
            "numerical_confidence": [0.9],
        }
    ).to_csv(assurance_dir / "numerical_confidence.csv", index=False)

    with pytest.raises(FileNotFoundError, match="calibration risk outputs"):
        load_assurance_inputs(config_path)
