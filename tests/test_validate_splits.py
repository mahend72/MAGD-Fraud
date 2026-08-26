from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.validate_splits import validate_data_splits


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_config(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _base_config(tmp_path: Path) -> dict:
    return {
        "experiment": {
            "output_dir": "data/outputs",
        },
        "paths": {
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "outputs_dir": "data/outputs",
        },
        "columns": {
            "case_id": "case_id",
            "label": "label",
            "sensitive_attributes": ["customer_age"],
        },
    }


def _create_processed_split_artifacts(tmp_path: Path, *, include_sensitive: bool = True, include_capacity: bool = True) -> Path:
    processed_dir = tmp_path / "data/processed"

    _write_csv(processed_dir / "X_train.csv", pd.DataFrame({"f1": [1, 2], "f2": [0, 1]}))
    _write_csv(processed_dir / "X_val.csv", pd.DataFrame({"f1": [3], "f2": [1]}))
    _write_csv(processed_dir / "X_test.csv", pd.DataFrame({"f1": [4], "f2": [0]}))

    _write_csv(processed_dir / "y_train.csv", pd.DataFrame({"label": [0, 1]}))
    _write_csv(processed_dir / "y_val.csv", pd.DataFrame({"label": [1]}))
    _write_csv(processed_dir / "y_test.csv", pd.DataFrame({"label": [0]}))

    train_metadata = pd.DataFrame({"case_id": ["c1", "c2"]})
    val_metadata = pd.DataFrame({"case_id": ["c3"]})
    test_metadata = pd.DataFrame({"case_id": ["c4"]})
    if include_sensitive:
        train_metadata["customer_age"] = ["18-24", "25-34"]
        val_metadata["customer_age"] = ["25-34"]
        test_metadata["customer_age"] = ["35-44"]

    _write_csv(processed_dir / "train_metadata.csv", train_metadata)
    _write_csv(processed_dir / "val_metadata.csv", val_metadata)
    _write_csv(processed_dir / "test_metadata.csv", test_metadata)

    _write_csv(
        processed_dir / "expert_predictions.csv",
        pd.DataFrame(
            {
                "case_id": ["c1", "c2", "c3", "c4"],
                "expert_a": [0, 1, 1, 0],
                "expert_b": [0, 1, 0, 0],
            }
        ),
    )
    capacity = pd.DataFrame(columns=["case_id", "capacity"])
    if include_capacity:
        capacity = pd.DataFrame({"case_id": ["c1", "c2"], "capacity": [1, 1]})
    _write_csv(processed_dir / "capacity.csv", capacity)

    config = _base_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)
    return config_path


def test_data_loader_returns_train_val_test(tmp_path: Path) -> None:
    config_path = _create_processed_split_artifacts(tmp_path)
    artifacts = validate_data_splits(config_path)

    assert len(artifacts.split_data.X_train) == 2
    assert len(artifacts.split_data.X_val) == 1
    assert len(artifacts.split_data.X_test) == 1


def test_split_ids_do_not_overlap(tmp_path: Path) -> None:
    config_path = _create_processed_split_artifacts(tmp_path)
    artifacts = validate_data_splits(config_path)
    case_id_column = artifacts.split_data.case_id_column

    train_ids = set(artifacts.split_data.train_metadata[case_id_column].astype(str))
    val_ids = set(artifacts.split_data.val_metadata[case_id_column].astype(str))
    test_ids = set(artifacts.split_data.test_metadata[case_id_column].astype(str))
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)


def test_labels_are_binary(tmp_path: Path) -> None:
    config_path = _create_processed_split_artifacts(tmp_path)
    artifacts = validate_data_splits(config_path)

    assert set(artifacts.split_data.y_train["label"].unique().tolist()).issubset({0, 1})
    assert set(artifacts.split_data.y_val["label"].unique().tolist()).issubset({0, 1})
    assert set(artifacts.split_data.y_test["label"].unique().tolist()).issubset({0, 1})


def test_fraud_prevalence_is_computed(tmp_path: Path) -> None:
    config_path = _create_processed_split_artifacts(tmp_path)
    artifacts = validate_data_splits(config_path)

    assert artifacts.statistics.train_fraud_prevalence == pytest.approx(0.5)
    assert artifacts.statistics.validation_fraud_prevalence == pytest.approx(1.0)
    assert artifacts.statistics.test_fraud_prevalence == pytest.approx(0.0)


def test_missing_optional_sensitive_and_capacity_fields_are_logged_but_do_not_crash(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_path = _create_processed_split_artifacts(tmp_path, include_sensitive=False, include_capacity=False)
    caplog.set_level(logging.INFO)

    artifacts = validate_data_splits(config_path)

    assert artifacts.statistics.capacity_configured is False
    assert any("Sensitive attribute" in record.message for record in caplog.records)
    assert any("Capacity is not configured" in record.message for record in caplog.records)


def test_dataset_summary_files_are_written(tmp_path: Path) -> None:
    config_path = _create_processed_split_artifacts(tmp_path)
    artifacts = validate_data_splits(config_path)

    assert (artifacts.paper_tables_dir / "dataset_summary.csv").exists()
    assert (artifacts.paper_tables_dir / "dataset_summary.md").exists()
