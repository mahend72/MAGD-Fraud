from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.data.preprocess import prepare_data


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_config(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _base_config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "outputs_dir": "data/outputs",
        },
        "dataset": {
            "main_file": "data/raw/main.csv",
            "expert_predictions_file": "data/raw/expert.csv",
            "capacity_file": "data/raw/capacity.csv",
        },
        "columns": {
            "case_id": "case_id",
            "application_id": "application_id",
            "batch_id": "batch_id",
            "time": None,
            "label": "label",
            "prediction": "model_prediction",
            "model_score": "model_score",
            "expert_case_id": "case_id",
            "expert_prediction": "expert_prediction",
            "capacity_case_id": "case_id",
            "capacity": "capacity_limit",
            "sensitive_attributes": ["age_band", "gender"],
        },
        "split": {
            "train_size": 0.5,
            "val_size": 0.25,
            "test_size": 0.25,
            "random_state": 7,
            "stratify": True,
        },
    }


def _create_input_files(tmp_path: Path) -> Path:
    main = pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(1, 9)],
            "application_id": [f"a{i}" for i in range(1, 9)],
            "batch_id": ["b1", "b1", "b2", "b2", "b3", "b3", "b4", "b4"],
            "amount_band": ["low", "high", "low", "high", "low", "high", "low", "high"],
            "channel": ["web", "branch", "web", "branch", "web", "branch", "web", "branch"],
            "model_prediction": [0, 1, 0, 1, 0, 1, 0, 1],
            "model_score": [0.1, 0.8, 0.2, 0.9, 0.3, 0.75, 0.4, 0.7],
            "age_band": ["18-24", "25-34", "35-44", "18-24", "25-34", "35-44", "18-24", "25-34"],
            "gender": ["F", "M", "F", "M", "F", "M", "F", "M"],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    expert = pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(1, 9)],
            "expert_prediction": [0, 1, 0, 1, 0, 1, 1, 1],
        }
    )
    capacity = pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(1, 9)],
            "capacity_limit": [5, 5, 6, 6, 7, 7, 8, 8],
        }
    )

    _write_csv(tmp_path / "data/raw/main.csv", main)
    _write_csv(tmp_path / "data/raw/expert.csv", expert)
    _write_csv(tmp_path / "data/raw/capacity.csv", capacity)

    config = _base_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)
    return config_path


def test_no_label_leakage(tmp_path: Path) -> None:
    config_path = _create_input_files(tmp_path)
    artifacts = prepare_data(config_path)

    assert "label" not in artifacts.X_train.columns
    assert "label" not in artifacts.X_val.columns
    assert "label" not in artifacts.X_test.columns


def test_same_number_of_rows_in_x_and_y(tmp_path: Path) -> None:
    config_path = _create_input_files(tmp_path)
    artifacts = prepare_data(config_path)

    assert len(artifacts.X_train) == len(artifacts.y_train)
    assert len(artifacts.X_val) == len(artifacts.y_val)
    assert len(artifacts.X_test) == len(artifacts.y_test)


def test_expert_prediction_table_aligns_with_case_ids(tmp_path: Path) -> None:
    config_path = _create_input_files(tmp_path)
    artifacts = prepare_data(config_path)
    main_case_ids = {f"c{i}" for i in range(1, 9)}

    assert set(artifacts.expert_predictions["case_id"].astype(str)).issubset(main_case_ids)


def test_processed_files_are_created(tmp_path: Path) -> None:
    config_path = _create_input_files(tmp_path)
    prepare_data(config_path)

    expected_files = [
        "X_train.csv",
        "X_val.csv",
        "X_test.csv",
        "y_train.csv",
        "y_val.csv",
        "y_test.csv",
        "train_metadata.csv",
        "val_metadata.csv",
        "test_metadata.csv",
        "expert_predictions.csv",
        "capacity.csv",
    ]
    for file_name in expected_files:
        assert (tmp_path / "data/processed" / file_name).exists(), file_name
