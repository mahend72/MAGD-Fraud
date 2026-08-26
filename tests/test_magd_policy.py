from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd

from src.deferral.magd_policy import (
    _normalize_weights,
    _POLICY_RUN_CACHE,
    load_policy_weights,
    load_validation_policy_frame,
    run_magd_policy,
)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _setup_policy_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path
    model_dir = repo / "data" / "outputs" / "model"
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
                "case_id": ["t1", "t2", "t3", "t4", "v1", "v2", "v3", "v4"],
                "fraud_label": [0, 1, 0, 1, 0, 1, 0, 1],
                "batch": ["b1"] * 8,
                "month": [1, 1, 1, 1, 2, 2, 2, 2],
                "customer_age": ["young", "old", "young", "old", "young", "old", "young", "old"],
                "model_score": [0.2, 0.8, 0.3, 0.7, 0.85, 0.35, 0.75, 0.25],
            }
        ),
    )
    _write_csv(
        raw_dir / "test.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2"],
                "fraud_label": [0, 1],
                "batch": ["b1", "b1"],
                "month": [3, 3],
                "customer_age": ["young", "old"],
                "model_score": [0.8, 0.3],
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
        pd.DataFrame({"case_id": ["s1", "s2"], "expert_a": [0, 1], "expert_b": [1, 0]}),
    )

    split_rows = {
        "train": [("t1", 0), ("t2", 1), ("t3", 0), ("t4", 1)],
        "val": [("v1", 0), ("v2", 1), ("v3", 0), ("v4", 1)],
        "test": [("s1", 0), ("s2", 1)],
    }
    features = {
        "train": pd.DataFrame({"f1": [0.0, 1.0, 0.2, 0.8], "f2": [0.1, 0.9, 0.3, 0.7]}),
        "val": pd.DataFrame({"f1": [0.05, 0.95, 0.25, 0.75], "f2": [0.15, 0.85, 0.35, 0.65]}),
        "test": pd.DataFrame({"f1": [0.1, 0.9], "f2": [0.2, 0.8]}),
    }
    scores = {
        "train": [0.2, 0.8, 0.3, 0.7],
        "val": [0.85, 0.35, 0.75, 0.25],
        "test": [0.8, 0.3],
    }
    preds = {
        "train": [0, 1, 0, 1],
        "val": [1, 0, 1, 0],
        "test": [1, 0],
    }
    for split_name, rows in split_rows.items():
        ids = [row[0] for row in rows]
        labels = [row[1] for row in rows]
        _write_csv(processed_dir / f"X_{split_name}.csv", features[split_name])
        _write_csv(processed_dir / f"y_{split_name}.csv", pd.DataFrame({"fraud_label": labels}))
        metadata_name = f"{split_name}_metadata.csv" if split_name != "test" else "test_metadata.csv"
        _write_csv(
            processed_dir / metadata_name,
            pd.DataFrame({"case_id": ids, "batch": ["b1"] * len(ids), "month": [1] * len(ids), "customer_age": ["young", "old"] * (len(ids) // 2)}),
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
    return config_path


def test_normalized_weights_sum_to_one_for_active_signals() -> None:
    weights = _normalize_weights(
        {
            "distance_uncertainty": 2.0,
            "calibration_risk": 1.0,
            "neighbor_error_rate": 1.0,
        },
        ["distance_uncertainty", "calibration_risk", "neighbor_error_rate"],
    )
    total = (
        weights["distance_uncertainty"]
        + weights["calibration_risk"]
        + weights["neighbor_error_rate"]
    )
    assert abs(total - 1.0) < 1e-9


def test_policy_learning_uses_validation_predictions_source() -> None:
    source = inspect.getsource(load_validation_policy_frame)
    assert "val_predictions.csv" in source
    assert "test_predictions.csv" not in source


def test_learned_weights_valid_and_normalized(tmp_path: Path) -> None:
    config_path = _setup_policy_fixture_repo(tmp_path)
    artifacts = run_magd_policy(config_path)
    for variant in ["heuristic", "learned"]:
        row = artifacts.learned_weights.loc[artifacts.learned_weights["variant"] == variant].iloc[0]
        weights = [
            float(row["distance_uncertainty"]),
            float(row["calibration_risk"]),
            float(row["neighbor_error_rate"]),
            float(row["wrong_confident_risk"]),
        ]
        assert all(weight >= 0.0 for weight in weights)
        assert abs(sum(weights) - 1.0) < 1e-6


def test_stale_learned_weights_are_not_silently_reused(tmp_path: Path) -> None:
    config_path = _setup_policy_fixture_repo(tmp_path)
    policy_dir = config_path.parent / "data" / "outputs" / "magd_policy"
    policy_dir.mkdir(parents=True, exist_ok=True)

    # Simulate a stale learned_weights.csv left over from an earlier, incompatible run
    # (e.g. a different dataset/config) - an obviously-wrong, unnormalized row that no
    # genuine optimization run for this dataset would ever produce.
    stale = pd.DataFrame(
        [
            {
                "variant": "heuristic",
                "objective": -999.0,
                "distance_uncertainty": 9.0,
                "calibration_risk": 0.0,
                "neighbor_error_rate": 0.0,
                "wrong_confident_risk": 0.0,
                "drift_risk": 0.0,
                "business_risk": 0.0,
            }
        ]
    )
    stale.to_csv(policy_dir / "learned_weights.csv", index=False)
    _POLICY_RUN_CACHE.clear()

    weights = load_policy_weights(config_path, variant="heuristic")
    # A freshly-recomputed heuristic weight vector must be normalized (sums to 1) and must
    # not carry over the stale file's raw, unnormalized 9.0 value untouched.
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["distance_uncertainty"] != 9.0

    on_disk = pd.read_csv(policy_dir / "learned_weights.csv")
    heuristic_row = on_disk.loc[on_disk["variant"] == "heuristic"].iloc[0]
    assert float(heuristic_row["objective"]) != -999.0


def test_policy_learning_not_affected_by_test_labels(tmp_path: Path) -> None:
    config_path = _setup_policy_fixture_repo(tmp_path)
    first = run_magd_policy(config_path).learned_weights.copy()

    model_dir = config_path.parent / "data" / "outputs" / "model"
    test_predictions = pd.read_csv(model_dir / "test_predictions.csv")
    test_predictions["y_true"] = [1, 0]
    test_predictions.to_csv(model_dir / "test_predictions.csv", index=False)

    second = run_magd_policy(config_path).learned_weights.copy()
    compare_columns = ["variant", "distance_uncertainty", "calibration_risk", "neighbor_error_rate", "wrong_confident_risk"]
    pd.testing.assert_frame_equal(first[compare_columns], second[compare_columns])
