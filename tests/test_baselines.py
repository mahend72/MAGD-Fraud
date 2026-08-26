from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.deferral.baselines import run_baselines


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _setup_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path
    model_dir = repo / "data" / "outputs" / "model"
    assurance_dir = repo / "data" / "outputs" / "assurance"
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
                "  label: fraud_label",
                "  train_label: fraud_label",
                "  test_label: fraud_label",
                "  expert_case_id: case_id",
                "  expert_prediction: expert_a",
                "  sensitive_attributes:",
                "    - customer_age",
            ]
        ),
        encoding="utf-8",
    )

    _write_csv(raw_dir / "main.csv", pd.DataFrame({"case_id": [1], "fraud_label": [0], "batch": ["b1"], "month": [1], "customer_age": ["young"], "model_score": [0.5]}))
    _write_csv(
        raw_dir / "train.csv",
        pd.DataFrame({"case_id": ["t1", "t2", "v1", "v2"], "fraud_label": [0, 1, 0, 1], "batch": ["b1", "b1", "b1", "b1"], "month": [1, 1, 2, 2], "customer_age": ["young", "old", "young", "old"], "model_score": [0.2, 0.8, 0.9, 0.4]}),
    )
    _write_csv(raw_dir / "test.csv", pd.DataFrame({"case_id": ["s1", "s2"], "fraud_label": [0, 1], "batch": ["b1", "b1"], "month": [3, 3], "customer_age": ["young", "old"], "model_score": [0.8, 0.3]}))
    _write_csv(
        raw_dir / "historical_expert_predictions.csv",
        pd.DataFrame(
            {
                "case_id": ["t1", "t2", "v1", "v2"],
                "expert_a": [0, 1, 0, 1],
                "expert_b": [1, 1, 0, 0],
                "oracle_helper": [0, 1, 0, 1],
            }
        ),
    )
    _write_csv(
        raw_dir / "expert_predictions.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2"],
                "expert_a": [0, 1],
                "expert_b": [1, 0],
                "oracle_helper": [0, 1],
            }
        ),
    )

    for split_name, rows in {
        "train": [("t1", 0), ("t2", 1)],
        "val": [("v1", 0), ("v2", 1)],
        "test": [("s1", 0), ("s2", 1)],
    }.items():
        ids = [row[0] for row in rows]
        labels = [row[1] for row in rows]
        _write_csv(processed_dir / f"X_{split_name}.csv", pd.DataFrame({"f1": [0.0, 1.0][: len(rows)]}))
        _write_csv(processed_dir / f"y_{split_name}.csv", pd.DataFrame({"fraud_label": labels}))
        metadata_name = f"{split_name}_metadata.csv" if split_name != "test" else "test_metadata.csv"
        _write_csv(processed_dir / metadata_name, pd.DataFrame({"case_id": ids, "customer_age": ["young", "old"][: len(rows)]}))

    _write_csv(
        model_dir / "val_predictions.csv",
        pd.DataFrame(
            {
                "case_id": ["v1", "v2"],
                "y_true": [0, 1],
                "ai_score": [0.9, 0.4],
                "ai_pred": [0, 0],
                "split": ["val", "val"],
            }
        ),
    )
    _write_csv(
        model_dir / "test_predictions.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2"],
                "y_true": [0, 1],
                "ai_score": [0.8, 0.3],
                "ai_pred": [0, 0],
                "split": ["test", "test"],
            }
        ),
    )
    _write_csv(
        assurance_dir / "numerical_confidence.csv",
        pd.DataFrame(
            {
                "case_id": ["v1", "v2", "s1", "s2"],
                "split": ["val", "val", "test", "test"],
                "ai_score": [0.9, 0.4, 0.8, 0.3],
                "ai_pred": [0, 0, 0, 0],
                "numerical_confidence": [0.9, 0.6, 0.8, 0.7],
            }
        ),
    )
    _write_csv(
        assurance_dir / "distance_uncertainty.csv",
        pd.DataFrame(
            {
                "case_id": ["v1", "v2", "s1", "s2"],
                "split": ["val", "val", "test", "test"],
                "y_true": [0, 1, 0, 1],
                "ai_score": [0.9, 0.4, 0.8, 0.3],
                "ai_pred": [0, 0, 0, 0],
                "distance_confidence": [0.8, 0.2, 0.75, 0.25],
            }
        ),
    )
    return config_path


def test_every_method_creates_decisions_and_metrics(tmp_path: Path) -> None:
    config_path = _setup_fixture_repo(tmp_path)
    artifacts = run_baselines(config_path)

    expected = [
        "ai_only_decisions.csv",
        "best_expert_decisions.csv",
        "random_expert_decisions.csv",
        "confidence_threshold_decisions.csv",
        "distance_threshold_decisions.csv",
        "oracle_upper_bound_decisions.csv",
    ]
    for name in expected:
        frame = pd.read_csv(artifacts.output_dir / name)
        assert {"case_id", "y_true", "ai_score", "ai_pred", "selected_route", "selected_expert", "final_prediction", "decision_reason", "method"}.issubset(frame.columns)
        assert len(frame) == 2

    metrics = pd.read_csv(artifacts.output_dir / "baseline_metrics.csv")
    assert set(metrics["method"]) == {"ai_only", "best_expert", "random_expert", "confidence_threshold", "distance_threshold", "oracle_upper_bound"}


def test_every_case_has_exactly_one_route(tmp_path: Path) -> None:
    config_path = _setup_fixture_repo(tmp_path)
    artifacts = run_baselines(config_path)
    for path in artifacts.output_dir.glob("*_decisions.csv"):
        frame = pd.read_csv(path)
        assert frame["case_id"].is_unique
        assert frame["selected_route"].notna().all()


def test_oracle_marked_non_deployable(tmp_path: Path) -> None:
    config_path = _setup_fixture_repo(tmp_path)
    artifacts = run_baselines(config_path)
    metrics = pd.read_csv(artifacts.output_dir / "baseline_metrics.csv")
    oracle_row = metrics.loc[metrics["method"] == "oracle_upper_bound"].iloc[0]
    assert bool(oracle_row["deployable"]) is False
    oracle_decisions = pd.read_csv(artifacts.output_dir / "oracle_upper_bound_decisions.csv")
    assert oracle_decisions["decision_reason"].astype(str).str.contains("oracle_non_deployable|oracle_upper_bound").all()


def test_deployable_baselines_do_not_use_test_y_true_for_routing(tmp_path: Path) -> None:
    config_path = _setup_fixture_repo(tmp_path)
    first = run_baselines(config_path)
    first_conf = pd.read_csv(first.output_dir / "confidence_threshold_decisions.csv")
    first_dist = pd.read_csv(first.output_dir / "distance_threshold_decisions.csv")

    model_dir = config_path.parent / "data" / "outputs" / "model"
    test_predictions = pd.read_csv(model_dir / "test_predictions.csv")
    test_predictions["y_true"] = [1, 0]
    test_predictions.to_csv(model_dir / "test_predictions.csv", index=False)

    second = run_baselines(config_path)
    second_conf = pd.read_csv(second.output_dir / "confidence_threshold_decisions.csv")
    second_dist = pd.read_csv(second.output_dir / "distance_threshold_decisions.csv")

    pd.testing.assert_frame_equal(
        first_conf[["case_id", "selected_route", "selected_expert", "final_prediction"]],
        second_conf[["case_id", "selected_route", "selected_expert", "final_prediction"]],
    )
    pd.testing.assert_frame_equal(
        first_dist[["case_id", "selected_route", "selected_expert", "final_prediction"]],
        second_dist[["case_id", "selected_route", "selected_expert", "final_prediction"]],
    )
