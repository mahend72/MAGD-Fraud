from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.deferral.capacity_assignment import CapacityState
from src.deferral.magd_deferral import route_magd_cases, run_magd_deferral


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _setup_deferral_fixture_repo(tmp_path: Path) -> Path:
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
                "  mode: heuristic",
            ]
        ),
        encoding="utf-8",
    )

    _write_csv(raw_dir / "main.csv", pd.DataFrame({"case_id": [1], "fraud_label": [0], "batch": ["b1"], "month": [1], "customer_age": ["young"], "model_score": [0.5]}))
    _write_csv(
        raw_dir / "train.csv",
        pd.DataFrame(
            {
                "case_id": ["t1", "t2", "v1", "v2"],
                "fraud_label": [0, 1, 0, 1],
                "batch": ["b1", "b1", "b1", "b1"],
                "month": [1, 1, 2, 2],
                "customer_age": ["young", "old", "young", "old"],
                "model_score": [0.2, 0.8, 0.9, 0.4],
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
                "model_score": [0.8, 0.3, 0.55],
            }
        ),
    )
    _write_csv(
        raw_dir / "historical_expert_predictions.csv",
        pd.DataFrame({"case_id": ["t1", "t2", "v1", "v2"], "expert_a": [0, 1, 0, 1], "expert_b": [1, 1, 0, 0]}),
    )
    _write_csv(
        raw_dir / "expert_predictions.csv",
        pd.DataFrame({"case_id": ["s1", "s2", "s3"], "expert_a": [0, 1, 1], "expert_b": [1, 0, 1]}),
    )

    split_rows = {
        "train": [("t1", 0), ("t2", 1)],
        "val": [("v1", 0), ("v2", 1)],
        "test": [("s1", 0), ("s2", 1), ("s3", 1)],
    }
    for split_name, rows in split_rows.items():
        ids = [row[0] for row in rows]
        labels = [row[1] for row in rows]
        _write_csv(processed_dir / f"X_{split_name}.csv", pd.DataFrame({"f1": [0.0, 1.0, 0.5][: len(rows)], "f2": [0.1, 0.9, 0.5][: len(rows)]}))
        _write_csv(processed_dir / f"y_{split_name}.csv", pd.DataFrame({"fraud_label": labels}))
        metadata_name = f"{split_name}_metadata.csv" if split_name != "test" else "test_metadata.csv"
        _write_csv(
            processed_dir / metadata_name,
            pd.DataFrame({"case_id": ids, "batch": ["b1"] * len(rows), "month": [1] * len(rows), "customer_age": ["young", "old", "young"][: len(rows)]}),
        )
        _write_csv(
            model_dir / f"{split_name}_predictions.csv",
            pd.DataFrame(
                {
                    "case_id": ids,
                    "y_true": labels,
                    "ai_score": [0.9, 0.4, 0.6][: len(rows)],
                    "ai_pred": [0, 0, 1][: len(rows)],
                    "split": [split_name] * len(rows),
                }
            ),
        )

    _write_csv(
        assurance_dir / "numerical_confidence.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2", "s3"],
                "split": ["test", "test", "test"],
                "ai_score": [0.9, 0.4, 0.6],
                "ai_pred": [0, 0, 1],
                "numerical_confidence": [0.9, 0.6, 0.6],
            }
        ),
    )
    _write_csv(
        assurance_dir / "calibration_risk.csv",
        pd.DataFrame({"case_id": ["s1", "s2", "s3"], "split": ["test", "test", "test"], "calibration_risk": [0.1, 0.3, 0.5]}),
    )
    _write_csv(
        assurance_dir / "distance_uncertainty.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2", "s3"],
                "split": ["test", "test", "test"],
                "distance_confidence": [0.8, 0.5, 0.3],
                "distance_uncertainty": [0.2, 0.5, 0.7],
            }
        ),
    )
    _write_csv(
        assurance_dir / "local_reliability.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2", "s3"],
                "split": ["test", "test", "test"],
                "neighbor_error_rate": [0.1, 0.4, 0.6],
                "neighbor_fraud_rate": [0.2, 0.5, 0.7],
                "neighbor_ai_agreement": [0.8, 0.5, 0.3],
                "mean_neighbor_distance": [0.1, 0.3, 0.5],
                "top_k": [25, 25, 25],
            }
        ),
    )
    _write_csv(
        assurance_dir / "wrong_confident_risk.csv",
        pd.DataFrame({"case_id": ["s1", "s2", "s3"], "split": ["test", "test", "test"], "wrong_confident_risk": [0.2, 0.4, 0.9]}),
    )
    _write_csv(
        assurance_dir / "magd_risk.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2", "s3"],
                "split": ["test", "test", "test"],
                "distance_uncertainty": [0.2, 0.5, 0.7],
                "calibration_risk": [0.1, 0.3, 0.5],
                "neighbor_error_rate": [0.1, 0.4, 0.6],
                "wrong_confident_risk": [0.2, 0.4, 0.9],
                "drift_risk": [0.0, 0.0, 0.0],
                "business_risk": [0.1, 0.1, 0.2],
                "drift_available": [False, False, False],
                "business_available": [False, False, False],
                "magd_assurance_risk": [0.2, 0.5, 0.8],
                "risk_category": ["low", "medium", "high"],
            }
        ),
    )
    _write_csv(
        policy_dir / "learned_weights.csv",
        pd.DataFrame(
            [
                {"variant": "heuristic", "objective": 0.3, "distance_uncertainty": 0.25, "calibration_risk": 0.20, "neighbor_error_rate": 0.20, "wrong_confident_risk": 0.25, "drift_risk": 0.05, "business_risk": 0.05},
                {"variant": "learned", "objective": 0.2, "distance_uncertainty": 0.30, "calibration_risk": 0.15, "neighbor_error_rate": 0.15, "wrong_confident_risk": 0.30, "drift_risk": 0.05, "business_risk": 0.05},
            ]
        ),
    )
    (policy_dir / "optimization_diagnostics.json").write_text(
        '{"heuristic": {"optimizer": "config", "used_validation_only": true}, "learned": {"optimizer": "grid_search", "used_validation_only": true}}',
        encoding="utf-8",
    )
    return config_path


def _core(y_true: list[int] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3"],
            "ai_score": [0.9, 0.4, 0.6],
            "ai_pred": [1, 0, 1],
            "numerical_confidence": [0.9, 0.6, 0.6],
            "distance_confidence": [0.8, 0.5, 0.3],
            "distance_uncertainty": [0.2, 0.5, 0.7],
            "calibration_risk": [0.1, 0.3, 0.5],
            "neighbor_error_rate": [0.1, 0.4, 0.6],
            "neighbor_fraud_rate": [0.2, 0.5, 0.7],
            "neighbor_ai_agreement": [0.8, 0.5, 0.3],
            "mean_neighbor_distance": [0.1, 0.3, 0.5],
            "wrong_confident_risk": [0.2, 0.4, 0.9],
            "magd_assurance_risk": [0.2, 0.5, 0.8],
            "risk_category": ["low", "medium", "high"],
            "base_threshold": [0.5, 0.5, 0.5],
            "business_risk": [0.1, 0.1, 0.2],
            "fairness_risk": [0.0, 0.2, 0.3],
            "capacity_pressure": [0.0, 0.1, 0.2],
            "adaptive_threshold": [0.55, 0.53, 0.57],
            "passes_threshold": [True, False, False],
            "batch": ["b1", "b1", "b1"],
            "expert_a": [1, 1, 1],
            "expert_b": [1, 0, 0],
        }
    )
    if y_true is not None:
        frame["y_true"] = y_true
    return frame


def _expert_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expert": ["expert_a", "expert_b"],
            "accuracy": [0.9, 0.8],
            "false_positive_rate": [0.1, 0.2],
            "false_negative_rate": [0.1, 0.2],
            "cost_sensitive_loss": [1.0, 2.0],
            "groupwise_false_positive_rate": [0.1, 0.2],
            "groupwise_false_negative_rate": [0.1, 0.2],
            "bias_risk": [0.05, 0.1],
            "remaining_capacity_total": [3.0, 3.0],
            "remaining_capacity_by_batch": ['{"b1": 3}', '{"b1": 3}'],
        }
    )


def test_no_y_true_used_for_routing() -> None:
    decisions_a = route_magd_cases(
        _core(y_true=[0, 1, 0]),
        ranked_experts=["expert_a", "expert_b"],
        expert_metrics=_expert_metrics(),
        capacity_state=CapacityState(remaining=None, batch_column=None),
        fp_cost=0.057,
        fn_cost=1.0,
        fairness_penalty_weight=0.2,
        capacity_penalty_weight=0.2,
        human_review_cost=0.05,
        escalation_cost=0.1,
        top_k=2,
        wrong_confident_threshold=0.7,
    )
    decisions_b = route_magd_cases(
        _core(y_true=[1, 0, 1]),
        ranked_experts=["expert_a", "expert_b"],
        expert_metrics=_expert_metrics(),
        capacity_state=CapacityState(remaining=None, batch_column=None),
        fp_cost=0.057,
        fn_cost=1.0,
        fairness_penalty_weight=0.2,
        capacity_penalty_weight=0.2,
        human_review_cost=0.05,
        escalation_cost=0.1,
        top_k=2,
        wrong_confident_threshold=0.7,
    )
    compare_cols = ["case_id", "selected_route", "selected_expert", "final_prediction", "decision_reason"]
    assert decisions_a[compare_cols].equals(decisions_b[compare_cols])


def test_all_decisions_have_route_and_reason() -> None:
    decisions = route_magd_cases(
        _core(),
        ranked_experts=["expert_a", "expert_b"],
        expert_metrics=_expert_metrics(),
        capacity_state=CapacityState(remaining=None, batch_column=None),
        fp_cost=0.057,
        fn_cost=1.0,
        fairness_penalty_weight=0.2,
        capacity_penalty_weight=0.2,
        human_review_cost=0.05,
        escalation_cost=0.1,
        top_k=2,
        wrong_confident_threshold=0.7,
    )
    assert decisions["selected_route"].astype(str).str.strip().ne("").all()
    assert decisions["decision_reason"].astype(str).str.strip().ne("").all()


def test_all_decisions_have_audit_evidence() -> None:
    decisions = route_magd_cases(
        _core(),
        ranked_experts=["expert_a", "expert_b"],
        expert_metrics=_expert_metrics(),
        capacity_state=CapacityState(remaining=None, batch_column=None),
        fp_cost=0.057,
        fn_cost=1.0,
        fairness_penalty_weight=0.2,
        capacity_penalty_weight=0.2,
        human_review_cost=0.05,
        escalation_cost=0.1,
        top_k=2,
        wrong_confident_threshold=0.7,
    )
    evidence_columns = [
        "numerical_confidence",
        "distance_uncertainty",
        "calibration_risk",
        "neighbor_error_rate",
        "wrong_confident_risk",
        "magd_assurance_risk",
        "adaptive_threshold",
        "ai_expected_cost",
    ]
    assert not decisions[evidence_columns].isna().any().any()


def test_escalation_route_works() -> None:
    decisions = route_magd_cases(
        _core(),
        ranked_experts=["expert_a", "expert_b"],
        expert_metrics=_expert_metrics(),
        capacity_state=CapacityState(remaining=None, batch_column=None),
        fp_cost=0.057,
        fn_cost=1.0,
        fairness_penalty_weight=0.2,
        capacity_penalty_weight=0.2,
        human_review_cost=0.05,
        escalation_cost=0.1,
        top_k=2,
        wrong_confident_threshold=0.7,
    )
    assert decisions.loc[decisions["case_id"] == "c3", "selected_route"].iloc[0] == "Escalate"


def test_escalation_tie_is_resolved_deterministically() -> None:
    # c3 escalates (risk_category="high") with top_k=2 experts split 1-0 (expert_a=1,
    # expert_b=0) - an exact vote tie. The outcome must be identical every time the same
    # inputs are routed, with no dependence on dict/set iteration order or randomness.
    kwargs = dict(
        ranked_experts=["expert_a", "expert_b"],
        expert_metrics=_expert_metrics(),
        capacity_state=CapacityState(remaining=None, batch_column=None),
        fp_cost=0.057,
        fn_cost=1.0,
        fairness_penalty_weight=0.2,
        capacity_penalty_weight=0.2,
        human_review_cost=0.05,
        escalation_cost=0.1,
        top_k=2,
        wrong_confident_threshold=0.7,
    )
    first = route_magd_cases(_core(), **kwargs)
    second = route_magd_cases(_core(), **kwargs)
    tie_row_first = first.loc[first["case_id"] == "c3"].iloc[0]
    tie_row_second = second.loc[second["case_id"] == "c3"].iloc[0]
    assert tie_row_first["selected_route"] == "Escalate"
    assert tie_row_first["final_prediction"] == tie_row_second["final_prediction"]
    assert tie_row_first["selected_expert"] == tie_row_second["selected_expert"]


def test_no_available_expert_falls_back_to_ai() -> None:
    decisions = route_magd_cases(
        _core(),
        ranked_experts=[],
        expert_metrics=_expert_metrics(),
        capacity_state=CapacityState(remaining=None, batch_column=None),
        fp_cost=0.057,
        fn_cost=1.0,
        fairness_penalty_weight=0.2,
        capacity_penalty_weight=0.2,
        human_review_cost=0.05,
        escalation_cost=0.1,
        top_k=2,
        wrong_confident_threshold=0.7,
    )
    high_risk_row = decisions.loc[decisions["case_id"] == "c3"].iloc[0]
    assert high_risk_row["selected_route"] == "AI"
    assert high_risk_row["decision_reason"] == "no_expert_available_ai_fallback"
    assert int(high_risk_row["final_prediction"]) == int(_core().loc[_core()["case_id"] == "c3", "ai_pred"].iloc[0])


def test_run_magd_deferral_does_not_use_test_labels_for_routing(tmp_path: Path) -> None:
    config_path = _setup_deferral_fixture_repo(tmp_path)
    first = run_magd_deferral(config_path, mode="learned")
    first_routes = first.decisions[["case_id", "selected_route", "selected_expert", "final_prediction"]].copy()

    model_dir = config_path.parent / "data" / "outputs" / "model"
    test_predictions = pd.read_csv(model_dir / "test_predictions.csv")
    test_predictions["y_true"] = [1, 0, 0]
    test_predictions.to_csv(model_dir / "test_predictions.csv", index=False)

    second = run_magd_deferral(config_path, mode="learned")
    second_routes = second.decisions[["case_id", "selected_route", "selected_expert", "final_prediction"]].copy()
    pd.testing.assert_frame_equal(first_routes, second_routes)


def test_run_magd_deferral_writes_mode_outputs_and_metrics(tmp_path: Path) -> None:
    config_path = _setup_deferral_fixture_repo(tmp_path)
    heuristic = run_magd_deferral(config_path, mode="heuristic")
    learned = run_magd_deferral(config_path, mode="learned")
    assert {"selected_route", "decision_reason", "method"}.issubset(heuristic.decisions.columns)
    assert {"selected_route", "decision_reason", "method"}.issubset(learned.decisions.columns)
    assert (heuristic.output_dir / "magd_heuristic_decisions.csv").exists()
    assert (heuristic.output_dir / "magd_learned_decisions.csv").exists()
    metrics = pd.read_csv(heuristic.output_dir / "magd_metrics.csv")
    assert {"magd_heuristic", "magd_learned"}.issubset(set(metrics["method"].astype(str)))
