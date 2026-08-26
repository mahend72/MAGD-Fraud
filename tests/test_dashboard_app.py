from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dashboard.app import (
    PROTOTYPE_NOTICE,
    build_dashboard_case_table,
    dashboard_missing_messages,
    load_dashboard_data,
)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _setup_dashboard_repo(tmp_path: Path) -> Path:
    repo = tmp_path
    config_path = repo / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  output_dir: data/outputs",
                "paths:",
                "  outputs_dir: data/outputs",
                "  processed_data_dir: data/processed",
            ]
        ),
        encoding="utf-8",
    )

    _write_csv(
        repo / "data" / "outputs" / "audit_pack" / "decision_audit_log.csv",
        pd.DataFrame(
            {
                "case_id": ["c1"],
                "ai_score": [0.8],
                "ai_pred": [1],
                "numerical_confidence": [0.8],
                "calibration_risk": [0.1],
                "distance_uncertainty": [0.2],
                "neighbor_error_rate": [0.3],
                "wrong_confident_risk": [0.4],
                "magd_assurance_risk": [0.6],
                "risk_category": ["medium"],
                "selected_route": ["Human Expert"],
                "selected_expert": ["expert_a"],
                "final_prediction": [0],
                "decision_reason": ["medium_risk_to_expert"],
                "method": ["magd_constrained_calibrated"],
                "y_true": [0],
                "is_correct": [1],
                "ai_correct": [0],
            }
        ),
    )
    _write_csv(
        repo / "data" / "outputs" / "assurance" / "wrong_confident_risk.csv",
        pd.DataFrame(
            {
                "case_id": ["c1"],
                "split": ["test"],
                "numerical_confidence": [0.8],
                "distance_confidence": [0.7],
                "distance_uncertainty": [0.3],
                "calibration_risk": [0.1],
                "neighbor_error_rate": [0.2],
                "confidence_disagreement": [0.1],
                "wrong_confident_risk": [0.4],
                "wrong_confident_label_offline": [1],
            }
        ),
    )
    _write_csv(
        repo / "data" / "outputs" / "assurance" / "local_reliability.csv",
        pd.DataFrame(
            {
                "case_id": ["c1"],
                "split": ["test"],
                "neighbor_error_rate": [0.2],
                "neighbor_fraud_rate": [0.5],
                "neighbor_ai_agreement": [0.7],
                "mean_neighbor_distance": [1.2],
                "knn_k": [25],
            }
        ),
    )
    _write_csv(
        repo / "data" / "outputs" / "assurance_deferral" / "expert_reliability.csv",
        pd.DataFrame(
            {
                "expert": ["expert_a"],
                "accuracy": [0.9],
                "false_positive_rate": [0.1],
                "false_negative_rate": [0.2],
                "cost_sensitive_loss": [0.3],
                "bias_risk": [0.0],
                "capacity_enabled": [False],
                "expected_cost": [0.2],
            }
        ),
    )
    _write_csv(
        repo / "data" / "outputs" / "final_metrics" / "all_method_metrics.csv",
        pd.DataFrame({"method": ["MAGD-Constrained"], "precision": [0.5]}),
    )
    _write_csv(
        repo / "data" / "outputs" / "final_metrics" / "audit_metrics.csv",
        pd.DataFrame({"method": ["MAGD-Constrained"], "audit_coverage": [1.0]}),
    )
    _write_csv(
        repo / "data" / "outputs" / "audit_pack" / "claim_evidence_matrix.csv",
        pd.DataFrame({"claim_id": ["C1"], "coverage_score": [1.0], "example_case_ids": ["c1"]}),
    )
    _write_csv(repo / "data" / "processed" / "test_metadata.csv", pd.DataFrame({"case_id": ["c1"]}))
    return config_path


def test_dashboard_imports_and_loads_expected_files(tmp_path: Path) -> None:
    config_path = _setup_dashboard_repo(tmp_path)
    data = load_dashboard_data(config_path)
    assert PROTOTYPE_NOTICE.startswith("This dashboard is a research prototype")
    assert not data["frames"]["decision_audit_log"].empty
    assert not data["frames"]["wrong_confident_risk"].empty
    assert not data["frames"]["local_reliability"].empty


def test_dashboard_missing_files_show_friendly_messages(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  output_dir: data/outputs",
                "paths:",
                "  outputs_dir: data/outputs",
                "  processed_data_dir: data/processed",
            ]
        ),
        encoding="utf-8",
    )
    data = load_dashboard_data(config_path)
    messages = dashboard_missing_messages(data)
    assert messages
    assert any("not available yet" in message.lower() for message in messages)


def test_dashboard_hides_test_labels_unless_evaluation_mode_enabled(tmp_path: Path) -> None:
    config_path = _setup_dashboard_repo(tmp_path)
    data = load_dashboard_data(config_path)
    default_frame = build_dashboard_case_table(data, evaluation_mode=False)
    evaluation_frame = build_dashboard_case_table(data, evaluation_mode=True)
    assert "y_true" not in default_frame.columns
    assert "is_correct" not in default_frame.columns
    assert "y_true" in evaluation_frame.columns
    assert "is_correct" in evaluation_frame.columns
