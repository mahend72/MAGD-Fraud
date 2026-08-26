from __future__ import annotations

from pathlib import Path

import math
import pandas as pd

from scripts.evaluate_all_methods import run_evaluation


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _setup_evaluation_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path
    outputs_root = repo / "data" / "outputs"
    baselines_dir = outputs_root / "baselines"
    assurance_dir = outputs_root / "assurance"
    model_dir = outputs_root / "model"
    assurance_deferral_dir = outputs_root / "assurance_deferral"
    processed_dir = repo / "data" / "processed"
    raw_dir = repo / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    config_path = repo / "config.yaml"
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
                "paths:",
                "  raw_data_dir: data/raw",
                "  processed_data_dir: data/processed",
                "  outputs_dir: data/outputs",
                "dataset:",
                "  main_file: data/raw/main.csv",
                "  train_file: data/raw/train.csv",
                "  test_file: data/raw/test.csv",
                "columns:",
                "  case_id: case_id",
                "  batch_id: batch",
                "  time: month",
                "  label: fraud_label",
                "  train_label: fraud_label",
                "  test_label: fraud_label",
                "  model_score: model_score",
                "  sensitive_attributes:",
                "    - customer_age",
                "costs:",
                "  false_positive: 2.0",
                "  false_negative: 7.0",
            ]
        ),
        encoding="utf-8",
    )

    _write_csv(raw_dir / "main.csv", pd.DataFrame({"case_id": [1], "fraud_label": [0], "batch": ["b1"], "month": [1], "customer_age": ["young"], "model_score": [0.5]}))
    _write_csv(raw_dir / "train.csv", pd.DataFrame({"case_id": ["t1"], "fraud_label": [0], "batch": ["b1"], "month": [1], "customer_age": ["young"], "model_score": [0.5]}))
    _write_csv(raw_dir / "test.csv", pd.DataFrame({"case_id": ["c1", "c2"], "fraud_label": [0, 1], "batch": ["b1", "b1"], "month": [2, 2], "customer_age": ["young", "old"], "model_score": [0.8, 0.2]}))

    _write_csv(processed_dir / "test_metadata.csv", pd.DataFrame({"case_id": ["c1", "c2"], "batch": ["b1", "b1"], "month": [2, 2], "customer_age": ["young", "old"]}))

    _write_csv(
        model_dir / "test_predictions.csv",
        pd.DataFrame({"case_id": ["c1", "c2"], "y_true": [0, 1], "ai_score": [0.8, 0.2], "ai_pred": [1, 0], "split": ["test", "test"]}),
    )
    _write_csv(
        assurance_dir / "wrong_confident_risk.csv",
        pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "ai_score": [0.8, 0.2],
                "ai_pred": [1, 0],
                "numerical_confidence": [0.8, 0.8],
                "distance_confidence": [0.2, 0.4],
                "distance_uncertainty": [0.8, 0.6],
                "calibration_risk": [0.3, 0.4],
                "neighbor_error_rate": [0.7, 0.2],
                "wrong_confident_risk": [0.9, 0.1],
                "wrong_confident_label_offline": [1, 0],
            }
        ),
    )
    _write_csv(
        assurance_dir / "magd_risk.csv",
        pd.DataFrame({"case_id": ["c1", "c2"], "magd_assurance_risk": [0.85, 0.25], "risk_category": ["high", "low"]}),
    )

    base_logs = {
        "ai_only_decisions.csv": pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "y_true": [0, 1],
                "ai_score": [0.8, 0.2],
                "ai_pred": [1, 0],
                "selected_route": ["AI", "AI"],
                "selected_expert": ["", ""],
                "final_prediction": [1, 0],
                "decision_reason": ["ai_only", "ai_only"],
            }
        ),
        "best_expert_decisions.csv": pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "y_true": [0, 1],
                "ai_score": [0.8, 0.2],
                "ai_pred": [1, 0],
                "selected_route": ["Human Expert", "Human Expert"],
                "selected_expert": ["expert_a", "expert_a"],
                "final_prediction": [0, 1],
                "decision_reason": ["expert", "expert"],
            }
        ),
        "random_expert_decisions.csv": pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "y_true": [0, 1],
                "ai_score": [0.8, 0.2],
                "ai_pred": [1, 0],
                "selected_route": ["Human Expert", "Human Expert"],
                "selected_expert": ["expert_b", "expert_b"],
                "final_prediction": [1, 1],
                "decision_reason": ["random", "random"],
            }
        ),
        "numerical_threshold_decisions.csv": pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "y_true": [0, 1],
                "ai_score": [0.8, 0.2],
                "ai_pred": [1, 0],
                "selected_route": ["Human Expert", "AI"],
                "selected_expert": ["expert_a", ""],
                "final_prediction": [0, 0],
                "decision_reason": ["threshold", "threshold"],
            }
        ),
        "distance_threshold_decisions.csv": pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "y_true": [0, 1],
                "ai_score": [0.8, 0.2],
                "ai_pred": [1, 0],
                "selected_route": ["Escalate", "AI"],
                "selected_expert": ["expert_a|expert_b", ""],
                "final_prediction": [0, 0],
                "decision_reason": ["distance", "distance"],
            }
        ),
        "oracle_upper_bound_decisions.csv": pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "y_true": [0, 1],
                "ai_score": [0.8, 0.2],
                "ai_pred": [1, 0],
                "selected_route": ["Human Expert", "Human Expert"],
                "selected_expert": ["oracle", "oracle"],
                "final_prediction": [0, 1],
                "decision_reason": ["oracle", "oracle"],
            }
        ),
    }
    for name, frame in base_logs.items():
        _write_csv(baselines_dir / name, frame)

    _write_csv(
        assurance_deferral_dir / "assurance_guided_decisions.csv",
        pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "y_true": [0, 1],
                "ai_score": [0.8, 0.2],
                "ai_pred": [1, 0],
                "selected_route": ["Escalate", "AI"],
                "selected_expert": ["expert_a|expert_b", ""],
                "final_prediction": [0, 0],
                "decision_reason": ["assurance", "assurance"],
                "capacity_status": ["assigned", "not_applicable"],
            }
        ),
    )
    _write_csv(
        assurance_deferral_dir / "magd_constrained_decisions.csv",
        pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "y_true": [0, 1],
                "ai_score": [0.8, 0.2],
                "ai_pred": [1, 0],
                "selected_route": ["Escalate", "Human Expert"],
                "selected_expert": ["expert_a|expert_b", "expert_a"],
                "final_prediction": [0, 1],
                "decision_reason": ["magd", "magd"],
                "capacity_status": ["assigned", "assigned"],
            }
        ),
    )
    return config_path


def test_run_evaluation_outputs_one_row_per_method_and_uses_config_costs(tmp_path: Path) -> None:
    config_path = _setup_evaluation_fixture_repo(tmp_path)
    final_dir, method_count = run_evaluation(config_path)
    all_metrics = pd.read_csv(final_dir / "all_method_metrics.csv")
    assert len(all_metrics) == method_count
    assert all_metrics["method"].is_unique
    ai_only = all_metrics.loc[all_metrics["method"] == "AI-only"].iloc[0]
    assert math.isclose(float(ai_only["cost_sensitive_loss"]), 9.0)


def test_run_evaluation_writes_requested_paper_tables(tmp_path: Path) -> None:
    config_path = _setup_evaluation_fixture_repo(tmp_path)
    run_evaluation(config_path)
    paper_dir = config_path.parent / "data" / "outputs" / "paper_tables"
    for name in ["human_ai_metrics.csv", "baseline_comparison.csv", "intervention_calibrated_results.csv"]:
        assert (paper_dir / name).exists()
    intervention = pd.read_csv(paper_dir / "intervention_calibrated_results.csv")
    assert len(intervention) == 1
    assert intervention.loc[0, "method"] == "MAGD-Constrained"
    expected_diagnostics = {
        "audit_coverage_satisfied",
        "feasible",
        "overreliance_satisfied",
        "wrong_confident_avoidance_satisfied",
    }
    assert expected_diagnostics.issubset(intervention.columns)


def test_run_evaluation_preserves_intervention_calibrated_constraint_diagnostics(tmp_path: Path) -> None:
    config_path = _setup_evaluation_fixture_repo(tmp_path)
    paper_dir = config_path.parent / "data" / "outputs" / "paper_tables"
    _write_csv(
        paper_dir / "intervention_calibrated_results.csv",
        pd.DataFrame(
            [
                {
                    "stage": "calibrated_test",
                    "feasible": True,
                    "audit_coverage_satisfied": True,
                    "overreliance_satisfied": False,
                    "wrong_confident_avoidance_satisfied": True,
                }
            ]
        ),
    )

    run_evaluation(config_path)

    intervention = pd.read_csv(paper_dir / "intervention_calibrated_results.csv")
    assert bool(intervention.loc[0, "feasible"])
    assert bool(intervention.loc[0, "audit_coverage_satisfied"])
    assert not bool(intervention.loc[0, "overreliance_satisfied"])
    assert bool(intervention.loc[0, "wrong_confident_avoidance_satisfied"])


def test_audit_coverage_computed_from_required_fields(tmp_path: Path) -> None:
    config_path = _setup_evaluation_fixture_repo(tmp_path)
    run_evaluation(config_path)
    all_metrics = pd.read_csv(config_path.parent / "data" / "outputs" / "final_metrics" / "all_method_metrics.csv")
    magd = all_metrics.loc[all_metrics["method"] == "MAGD-Constrained"].iloc[0]
    assert 0.0 <= float(magd["audit_coverage"]) <= 1.0
