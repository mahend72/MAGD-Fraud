from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.deferral.magd_constrained import REQUIRED_AUDIT_COLUMNS, run_magd_constrained


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _setup_constrained_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path
    output_root = repo / "data" / "outputs"
    model_dir = output_root / "model"
    assurance_dir = output_root / "assurance"
    policy_dir = output_root / "magd_policy"
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
                "  expert_predictions_file: data/raw/expert_predictions.csv",
                "  historical_expert_predictions_file: data/raw/historical_expert_predictions.csv",
                "columns:",
                "  case_id: case_id",
                "  batch_id: batch",
                "  time: month",
                "  label: fraud_label",
                "  train_label: fraud_label",
                "  test_label: fraud_label",
                "  model_score: model_score",
                "  expert_case_id: case_id",
                "  expert_prediction: expert_a",
                "  sensitive_attributes:",
                "    - customer_age",
                "magd:",
                "  mode: constrained",
            ]
        ),
        encoding="utf-8",
    )

    _write_csv(raw_dir / "main.csv", pd.DataFrame({"case_id": [1], "fraud_label": [0], "batch": ["b1"], "month": [1], "customer_age": ["young"], "model_score": [0.5]}))
    _write_csv(
        raw_dir / "train.csv",
        pd.DataFrame(
            {
                "case_id": ["t1", "t2", "t3", "t4", "v1", "v2", "v3", "v4"],
                "fraud_label": [0, 1, 0, 1, 0, 1, 0, 1],
                "batch": ["b1"] * 8,
                "month": [1, 1, 1, 1, 2, 2, 2, 2],
                "customer_age": ["young", "old", "young", "old", "young", "old", "young", "old"],
                "model_score": [0.15, 0.85, 0.25, 0.75, 0.9, 0.35, 0.8, 0.3],
            }
        ),
    )
    _write_csv(
        raw_dir / "test.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2", "s3"],
                "fraud_label": [0, 1, 1],
                "batch": ["b1", "b1", "b1"],
                "month": [3, 3, 3],
                "customer_age": ["young", "old", "young"],
                "model_score": [0.9, 0.25, 0.55],
            }
        ),
    )
    _write_csv(
        raw_dir / "historical_expert_predictions.csv",
        pd.DataFrame(
            {
                "case_id": ["t1", "t2", "t3", "t4", "v1", "v2", "v3", "v4"],
                "expert_a": [0, 1, 0, 1, 0, 1, 0, 1],
                "expert_b": [1, 1, 0, 0, 1, 0, 1, 0],
            }
        ),
    )
    _write_csv(
        raw_dir / "expert_predictions.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2", "s3"],
                "expert_a": [0, 1, 1],
                "expert_b": [1, 0, 1],
            }
        ),
    )

    split_rows = {
        "train": [("t1", 0), ("t2", 1), ("t3", 0), ("t4", 1)],
        "val": [("v1", 0), ("v2", 1), ("v3", 0), ("v4", 1)],
        "test": [("s1", 0), ("s2", 1), ("s3", 1)],
    }
    features = {
        "train": pd.DataFrame({"f1": [0.0, 1.0, 0.2, 0.8], "f2": [0.1, 0.9, 0.3, 0.7]}),
        "val": pd.DataFrame({"f1": [0.05, 0.95, 0.25, 0.75], "f2": [0.15, 0.85, 0.35, 0.65]}),
        "test": pd.DataFrame({"f1": [0.1, 0.9, 0.55], "f2": [0.2, 0.8, 0.45]}),
    }
    scores = {
        "train": [0.15, 0.85, 0.25, 0.75],
        "val": [0.9, 0.35, 0.8, 0.3],
        "test": [0.9, 0.25, 0.55],
    }
    preds = {
        "train": [0, 1, 0, 1],
        "val": [1, 0, 1, 0],
        "test": [1, 0, 1],
    }
    for split_name, rows in split_rows.items():
        ids = [row[0] for row in rows]
        labels = [row[1] for row in rows]
        _write_csv(processed_dir / f"X_{split_name}.csv", features[split_name])
        _write_csv(processed_dir / f"y_{split_name}.csv", pd.DataFrame({"fraud_label": labels}))
        metadata_name = f"{split_name}_metadata.csv" if split_name != "test" else "test_metadata.csv"
        ages = ["young", "old", "young", "old"] if len(ids) == 4 else ["young", "old", "young"]
        _write_csv(
            processed_dir / metadata_name,
            pd.DataFrame({"case_id": ids, "batch": ["b1"] * len(ids), "month": [1] * len(ids), "customer_age": ages}),
        )
        _write_csv(
            model_dir / f"{split_name}_predictions.csv",
            pd.DataFrame(
                {
                    "case_id": ids,
                    "y_true": labels,
                    "ai_score": scores[split_name],
                    "ai_pred": preds[split_name],
                    "split": [split_name] * len(ids),
                }
            ),
        )

    _write_csv(
        assurance_dir / "numerical_confidence.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2", "s3"],
                "split": ["test", "test", "test"],
                "ai_score": [0.9, 0.25, 0.55],
                "ai_pred": [1, 0, 1],
                "numerical_confidence": [0.9, 0.75, 0.55],
            }
        ),
    )
    _write_csv(
        assurance_dir / "calibration_risk.csv",
        pd.DataFrame({"case_id": ["s1", "s2", "s3"], "split": ["test", "test", "test"], "calibration_risk": [0.25, 0.55, 0.45]}),
    )
    _write_csv(
        assurance_dir / "distance_uncertainty.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2", "s3"],
                "split": ["test", "test", "test"],
                "distance_confidence": [0.2, 0.45, 0.35],
                "distance_uncertainty": [0.8, 0.55, 0.65],
            }
        ),
    )
    _write_csv(
        assurance_dir / "local_reliability.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2", "s3"],
                "split": ["test", "test", "test"],
                "neighbor_error_rate": [0.75, 0.35, 0.7],
                "neighbor_fraud_rate": [0.6, 0.5, 0.7],
                "neighbor_ai_agreement": [0.25, 0.55, 0.35],
                "mean_neighbor_distance": [0.7, 0.4, 0.6],
                "top_k": [4, 4, 4],
            }
        ),
    )
    _write_csv(
        assurance_dir / "wrong_confident_risk.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2", "s3"],
                "split": ["test", "test", "test"],
                "wrong_confident_risk": [0.95, 0.45, 0.85],
                "wrong_confident_label_offline": [1, 0, 1],
            }
        ),
    )
    _write_csv(
        assurance_dir / "magd_risk.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2", "s3"],
                "split": ["test", "test", "test"],
                "distance_uncertainty": [0.8, 0.55, 0.65],
                "calibration_risk": [0.25, 0.55, 0.45],
                "neighbor_error_rate": [0.75, 0.35, 0.7],
                "wrong_confident_risk": [0.95, 0.45, 0.85],
                "drift_risk": [0.0, 0.0, 0.0],
                "business_risk": [0.1, 0.1, 0.2],
                "drift_available": [False, False, False],
                "business_available": [False, False, False],
                "magd_assurance_risk": [0.82, 0.48, 0.78],
                "risk_category": ["high", "medium", "high"],
            }
        ),
    )
    _write_csv(
        policy_dir / "learned_weights.csv",
        pd.DataFrame(
            [
                {"variant": "heuristic", "objective": 0.3, "distance_uncertainty": 0.25, "calibration_risk": 0.2, "neighbor_error_rate": 0.2, "wrong_confident_risk": 0.25, "drift_risk": 0.05, "business_risk": 0.05},
                {"variant": "learned", "objective": 0.25, "distance_uncertainty": 0.3, "calibration_risk": 0.15, "neighbor_error_rate": 0.15, "wrong_confident_risk": 0.3, "drift_risk": 0.05, "business_risk": 0.05},
                {"variant": "constrained", "objective": 0.2, "distance_uncertainty": 0.28, "calibration_risk": 0.17, "neighbor_error_rate": 0.2, "wrong_confident_risk": 0.25, "drift_risk": 0.05, "business_risk": 0.05},
            ]
        ),
    )
    (policy_dir / "optimization_diagnostics.json").write_text(
        json.dumps({"constrained": {"optimizer": "grid_search", "used_validation_only": True}}, indent=2),
        encoding="utf-8",
    )
    return config_path


def test_constraints_evaluated_on_validation_during_tuning(tmp_path: Path) -> None:
    config_path = _setup_constrained_fixture_repo(tmp_path)
    artifacts = run_magd_constrained(config_path)
    assert artifacts.diagnostics["used_validation_only"] is True
    assert artifacts.diagnostics["selected_validation_metrics"]["phase"] == "validation"
    assert artifacts.diagnostics["validation_rows"] > 0


def test_no_test_labels_used_for_tuning(tmp_path: Path) -> None:
    config_path = _setup_constrained_fixture_repo(tmp_path)
    first = run_magd_constrained(config_path)
    first_routes = first.calibrated_decisions[["case_id", "selected_route", "selected_expert", "final_prediction"]].copy()
    first_params = json.loads((first.policy_dir / "constrained_policy_config.json").read_text(encoding="utf-8"))["selected_params"]

    model_dir = config_path.parent / "data" / "outputs" / "model"
    test_predictions = pd.read_csv(model_dir / "test_predictions.csv")
    test_predictions["y_true"] = [1, 0, 0]
    test_predictions.to_csv(model_dir / "test_predictions.csv", index=False)

    second = run_magd_constrained(config_path)
    second_routes = second.calibrated_decisions[["case_id", "selected_route", "selected_expert", "final_prediction"]].copy()
    second_params = json.loads((second.policy_dir / "constrained_policy_config.json").read_text(encoding="utf-8"))["selected_params"]
    pd.testing.assert_frame_equal(first_routes, second_routes)
    assert first_params == second_params


def test_calibrated_policy_has_non_trivial_deferral_if_feasible(tmp_path: Path) -> None:
    config_path = _setup_constrained_fixture_repo(tmp_path)
    artifacts = run_magd_constrained(config_path)
    if artifacts.diagnostics["selected_constraint_status"]["feasible"]:
        deferred = artifacts.calibrated_decisions["selected_route"].astype(str).isin(["Human Expert", "Escalate"])
        assert float(deferred.mean()) > 0.0


def test_decisions_have_complete_audit_fields(tmp_path: Path) -> None:
    config_path = _setup_constrained_fixture_repo(tmp_path)
    artifacts = run_magd_constrained(config_path)
    required = {
        "case_id",
        "y_true",
        "ai_score",
        "ai_pred",
        "numerical_confidence",
        "calibration_risk",
        "distance_uncertainty",
        "neighbor_error_rate",
        "wrong_confident_risk",
        "magd_assurance_risk",
        "risk_category",
        "selected_route",
        "selected_expert",
        "final_prediction",
        "is_correct",
        "decision_reason",
        "method",
    }
    assert required.issubset(artifacts.calibrated_decisions.columns)
    assert artifacts.calibrated_decisions[REQUIRED_AUDIT_COLUMNS].notna().all().all()
    assert artifacts.calibrated_decisions["decision_reason"].astype(str).str.strip().ne("").all()


def test_constrained_diagnostics_explain_constraint_status(tmp_path: Path) -> None:
    config_path = _setup_constrained_fixture_repo(tmp_path)
    artifacts = run_magd_constrained(config_path)
    diagnostics_path = artifacts.policy_dir / "constrained_policy_diagnostics.json"
    config_json_path = artifacts.policy_dir / "constrained_policy_config.json"
    results_path = config_path.parent / "data" / "outputs" / "paper_tables" / "intervention_calibrated_results.csv"
    assert diagnostics_path.exists()
    assert config_json_path.exists()
    assert results_path.exists()

    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    status = diagnostics["selected_constraint_status"]
    expected_keys = {
        "feasible",
        "overreliance_satisfied",
        "wrong_confident_avoidance_satisfied",
        "min_deferral_satisfied",
        "max_deferral_satisfied",
        "audit_coverage_satisfied",
    }
    assert expected_keys.issubset(status)
    results = pd.read_csv(results_path)
    assert expected_keys.issubset(results.columns)
    assert (config_path.parent / "data" / "outputs" / "assurance_deferral" / "magd_constrained_initial_decisions.csv").exists()
    assert (config_path.parent / "data" / "outputs" / "assurance_deferral" / "magd_constrained_calibrated_decisions.csv").exists()
