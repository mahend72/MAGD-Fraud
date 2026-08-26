from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.models.train_model import run_model_predictions


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_config(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _base_config(tmp_path: Path, *, use_existing_scores: bool = True) -> Path:
    config = {
        "experiment": {
            "output_dir": "data/outputs",
            "use_existing_scores": use_existing_scores,
            "seed": 42,
        },
        "paths": {
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "outputs_dir": "data/outputs",
        },
        "columns": {
            "case_id": "case_id",
            "label": "label",
            "model_score": "model_score",
            "sensitive_attributes": [],
        },
        "model": {
            "model_type": "random_forest",
            "fallback_model": "random_forest",
            "threshold": 0.5,
            "random_forest": {
                "n_estimators": 50,
                "random_state": 42,
            },
        },
        "costs": {
            "false_positive": 0.057,
            "false_negative": 1.0,
            "human_review": 0.05,
            "escalation": 0.10,
        },
    }
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)
    return config_path


def _create_processed_inputs(tmp_path: Path, *, use_existing_scores: bool = True, y_test: list[int] | None = None) -> Path:
    processed_dir = tmp_path / "data/processed"
    _write_csv(processed_dir / "X_train.csv", pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [0, 0, 1, 1]}))
    _write_csv(processed_dir / "X_val.csv", pd.DataFrame({"f1": [0, 1], "f2": [1, 0]}))
    _write_csv(processed_dir / "X_test.csv", pd.DataFrame({"f1": [0, 1], "f2": [0, 1]}))

    _write_csv(processed_dir / "y_train.csv", pd.DataFrame({"label": [0, 1, 0, 1]}))
    _write_csv(processed_dir / "y_val.csv", pd.DataFrame({"label": [0, 1]}))
    _write_csv(processed_dir / "y_test.csv", pd.DataFrame({"label": y_test or [0, 1]}))

    train_metadata = pd.DataFrame({"case_id": ["c1", "c2", "c3", "c4"]})
    val_metadata = pd.DataFrame({"case_id": ["c5", "c6"]})
    test_metadata = pd.DataFrame({"case_id": ["c7", "c8"]})
    if use_existing_scores:
        train_metadata["model_score"] = [0.1, 0.9, 0.2, 0.8]
        val_metadata["model_score"] = [0.3, 0.7]
        test_metadata["model_score"] = [0.4, 0.6]

    _write_csv(processed_dir / "train_metadata.csv", train_metadata)
    _write_csv(processed_dir / "val_metadata.csv", val_metadata)
    _write_csv(processed_dir / "test_metadata.csv", test_metadata)
    _write_csv(processed_dir / "expert_predictions.csv", pd.DataFrame(columns=["case_id", "expert_prediction"]))
    _write_csv(processed_dir / "capacity.csv", pd.DataFrame(columns=["case_id", "capacity"]))

    return _base_config(tmp_path, use_existing_scores=use_existing_scores)


def test_predictions_have_required_columns(tmp_path: Path) -> None:
    config_path = _create_processed_inputs(tmp_path, use_existing_scores=True)
    artifacts = run_model_predictions(config_path)

    required = {"case_id", "y_true", "ai_score", "ai_pred", "split"}
    assert required.issubset(artifacts.train_predictions.columns)
    assert required.issubset(artifacts.val_predictions.columns)
    assert required.issubset(artifacts.test_predictions.columns)


def test_ai_score_between_zero_and_one(tmp_path: Path) -> None:
    config_path = _create_processed_inputs(tmp_path, use_existing_scores=True)
    artifacts = run_model_predictions(config_path)

    assert artifacts.train_predictions["ai_score"].between(0.0, 1.0).all()
    assert artifacts.val_predictions["ai_score"].between(0.0, 1.0).all()
    assert artifacts.test_predictions["ai_score"].between(0.0, 1.0).all()


def test_ai_pred_binary(tmp_path: Path) -> None:
    config_path = _create_processed_inputs(tmp_path, use_existing_scores=True)
    artifacts = run_model_predictions(config_path)

    assert set(artifacts.train_predictions["ai_pred"].unique().tolist()).issubset({0, 1})
    assert set(artifacts.val_predictions["ai_pred"].unique().tolist()).issubset({0, 1})
    assert set(artifacts.test_predictions["ai_pred"].unique().tolist()).issubset({0, 1})


def test_no_test_labels_are_used_for_threshold_tuning(tmp_path: Path) -> None:
    config_path_a = _create_processed_inputs(tmp_path / "a", use_existing_scores=False, y_test=[0, 1])
    config_path_b = _create_processed_inputs(tmp_path / "b", use_existing_scores=False, y_test=[1, 0])

    artifacts_a = run_model_predictions(config_path_a)
    artifacts_b = run_model_predictions(config_path_b)

    compare_columns = ["case_id", "ai_score", "ai_pred", "split"]
    pd.testing.assert_frame_equal(
        artifacts_a.test_predictions[compare_columns].reset_index(drop=True),
        artifacts_b.test_predictions[compare_columns].reset_index(drop=True),
    )


def test_model_metrics_csv_is_created(tmp_path: Path) -> None:
    config_path = _create_processed_inputs(tmp_path, use_existing_scores=True)
    artifacts = run_model_predictions(config_path)
    model_metrics_path = artifacts.model_dir / "model_metrics.csv"

    assert model_metrics_path.exists()
    metrics = pd.read_csv(model_metrics_path)
    required_metrics = {
        "split",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "false_positive_rate",
        "false_negative_rate",
        "cost_sensitive_loss",
    }
    assert required_metrics.issubset(metrics.columns)
