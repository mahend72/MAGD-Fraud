from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import load_yaml
from src.utils.logging_utils import get_logger

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".json"}
COMMON_ID_COLUMNS = ["case_id", "application_id", "alert_id", "id"]
LOGGER = get_logger(__name__)


@dataclass
class LoadedFiFARData:
    config: dict[str, Any]
    config_path: Path
    raw_data_dir: Path
    processed_data_dir: Path
    main_df: pd.DataFrame
    expert_df: pd.DataFrame | None
    capacity_df: pd.DataFrame | None
    case_id_column: str
    application_id_column: str | None
    batch_id_column: str | None
    time_column: str | None
    label_column: str
    model_prediction_column: str | None
    model_score_column: str | None
    expert_case_id_column: str | None
    expert_prediction_column: str | None
    capacity_case_id_column: str | None
    capacity_column: str | None
    sensitive_attributes: list[str]
    model_scores_available: bool
    has_predefined_test_split: bool


@dataclass
class LoadedSplitData:
    config: dict[str, Any]
    config_path: Path
    processed_data_dir: Path
    outputs_dir: Path
    case_id_column: str
    label_column: str
    sensitive_attributes: list[str]
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_val: pd.DataFrame
    y_test: pd.DataFrame
    train_metadata: pd.DataFrame
    val_metadata: pd.DataFrame
    test_metadata: pd.DataFrame
    expert_predictions: pd.DataFrame
    capacity: pd.DataFrame
    n_experts: int
    capacity_configured: bool


@dataclass
class SplitStatistics:
    total_cases: int
    train_cases: int
    validation_cases: int
    test_cases: int
    train_fraud_prevalence: float
    validation_fraud_prevalence: float
    test_fraud_prevalence: float
    n_features: int
    n_experts: int
    sensitive_attribute_used: str
    capacity_configured: bool


def read_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = Path(config_path).resolve()
    config = load_yaml(resolved)
    return config, resolved


def resolve_outputs_dir(config_path: Path, config: dict[str, Any]) -> Path:
    outputs_dir = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_dir.is_absolute():
        outputs_dir = (config_path.parent / outputs_dir).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def resolve_config_path(config_path: Path, configured_path: str | None, *, required: bool, field_name: str) -> Path | None:
    if not configured_path:
        if required:
            raise ValueError(f"Missing `{field_name}` in config.")
        return None
    path = Path(configured_path)
    if not path.is_absolute():
        path = (config_path.parent / configured_path).resolve()
    if required and not path.exists():
        raise FileNotFoundError(f"Configured file for `{field_name}` does not exist: {path}")
    if path and path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type for `{field_name}`: {path}")
    return path


def ensure_directory(config_path: Path, configured_path: str | None, *, field_name: str) -> Path:
    if not configured_path:
        raise ValueError(f"Missing `{field_name}` in config.")
    path = Path(configured_path)
    if not path.is_absolute():
        path = (config_path.parent / configured_path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_existing_directory(config_path: Path, configured_path: str | None, *, field_name: str) -> Path:
    if not configured_path:
        raise ValueError(f"Missing `{field_name}` in config.")
    path = Path(configured_path)
    if not path.is_absolute():
        path = (config_path.parent / configured_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Configured directory for `{field_name}` does not exist: {path}. "
            "Place the FiFAR files there or update config.yaml."
        )
    if not path.is_dir():
        raise NotADirectoryError(f"Configured path for `{field_name}` is not a directory: {path}")
    return path


def read_table(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix == ".json":
        return pd.read_json(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")


def _first_present_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _resolve_column(frame: pd.DataFrame, configured_name: str | None, *, field_name: str, required: bool, fallbacks: list[str] | None = None) -> str | None:
    if configured_name:
        if configured_name not in frame.columns:
            raise KeyError(f"Configured column `{field_name}={configured_name}` not found in dataframe columns.")
        return configured_name
    fallback_name = _first_present_column(frame, fallbacks or [])
    if fallback_name:
        return fallback_name
    if required:
        raise ValueError(
            f"Missing required column mapping for `{field_name}`. "
            f"Add it to config.yaml. Available columns: {list(frame.columns)}"
        )
    return None


def load_fifar_data(config_path: str | Path) -> LoadedFiFARData:
    config, resolved_config_path = read_config(config_path)

    raw_data_dir = resolve_existing_directory(
        resolved_config_path,
        config.get("paths", {}).get("raw_data_dir"),
        field_name="paths.raw_data_dir",
    )
    processed_data_dir = ensure_directory(
        resolved_config_path,
        config.get("paths", {}).get("processed_data_dir"),
        field_name="paths.processed_data_dir",
    )

    dataset_cfg = config.get("dataset", {})
    columns_cfg = config.get("columns", {})

    main_file = resolve_config_path(
        resolved_config_path,
        dataset_cfg.get("main_file"),
        required=False,
        field_name="dataset.main_file",
    )
    train_file = resolve_config_path(
        resolved_config_path,
        dataset_cfg.get("train_file"),
        required=False,
        field_name="dataset.train_file",
    )
    test_file = resolve_config_path(
        resolved_config_path,
        dataset_cfg.get("test_file"),
        required=False,
        field_name="dataset.test_file",
    )
    expert_file = resolve_config_path(
        resolved_config_path,
        dataset_cfg.get("expert_predictions_file"),
        required=False,
        field_name="dataset.expert_predictions_file",
    )
    capacity_file = resolve_config_path(
        resolved_config_path,
        dataset_cfg.get("capacity_file"),
        required=False,
        field_name="dataset.capacity_file",
    )

    has_predefined_test_split = train_file is not None and test_file is not None
    if has_predefined_test_split:
        train_df = read_table(train_file)
        test_df = read_table(test_file)
        if train_df.empty:
            raise ValueError(f"Train dataset is empty: {train_file}")
        if test_df.empty:
            raise ValueError(f"Test dataset is empty: {test_file}")

        train_label_name = columns_cfg.get("train_label") or columns_cfg.get("label")
        test_label_name = columns_cfg.get("test_label") or columns_cfg.get("label")
        if not train_label_name or train_label_name not in train_df.columns:
            raise ValueError(
                "Configured `columns.train_label` (or `columns.label`) is missing from the train file."
            )
        if not test_label_name or test_label_name not in test_df.columns:
            raise ValueError(
                "Configured `columns.test_label` (or `columns.label`) is missing from the test file."
            )

        canonical_label = columns_cfg.get("label") or "label"
        train_df = train_df.copy()
        test_df = test_df.copy()
        if train_label_name != canonical_label:
            train_df[canonical_label] = train_df[train_label_name]
        if test_label_name != canonical_label:
            test_df[canonical_label] = test_df[test_label_name]
        train_df["__split__"] = "train"
        test_df["__split__"] = "test"
        main_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
        columns_cfg = {**columns_cfg, "label": canonical_label}
    elif main_file is not None:
        main_df = read_table(main_file)
        if main_df.empty:
            raise ValueError(f"Main dataset is empty: {main_file}")
    else:
        raise ValueError(
            "Provide either `dataset.main_file` or both `dataset.train_file` and `dataset.test_file` in config."
        )

    case_id_column = _resolve_column(
        main_df,
        columns_cfg.get("case_id"),
        field_name="columns.case_id",
        required=False,
        fallbacks=COMMON_ID_COLUMNS,
    )
    application_id_column = _resolve_column(
        main_df,
        columns_cfg.get("application_id"),
        field_name="columns.application_id",
        required=False,
        fallbacks=["application_id", "app_id"],
    )
    if not case_id_column and not application_id_column:
        raise ValueError(
            "Neither case_id nor application_id could be resolved. "
            "Set `columns.case_id` or `columns.application_id` in config.yaml."
        )
    if not case_id_column and application_id_column:
        case_id_column = application_id_column

    batch_id_column = _resolve_column(
        main_df,
        columns_cfg.get("batch_id"),
        field_name="columns.batch_id",
        required=False,
        fallbacks=["batch_id", "batch", "queue_batch"],
    )
    time_column = _resolve_column(
        main_df,
        columns_cfg.get("time"),
        field_name="columns.time",
        required=False,
        fallbacks=["event_time", "timestamp", "created_at", "date", "time"],
    )
    label_column = _resolve_column(
        main_df,
        columns_cfg.get("label"),
        field_name="columns.label",
        required=True,
        fallbacks=[],
    )
    model_prediction_column = _resolve_column(
        main_df,
        columns_cfg.get("prediction"),
        field_name="columns.prediction",
        required=False,
        fallbacks=[],
    )
    model_score_column = _resolve_column(
        main_df,
        columns_cfg.get("model_score"),
        field_name="columns.model_score",
        required=False,
        fallbacks=[],
    )

    sensitive_attributes = []
    for column in columns_cfg.get("sensitive_attributes", []):
        if column not in main_df.columns:
            raise KeyError(f"Configured sensitive attribute column not found: {column}")
        sensitive_attributes.append(column)

    expert_df: pd.DataFrame | None = None
    expert_case_id_column: str | None = None
    expert_prediction_column: str | None = None
    if expert_file is not None:
        expert_df = read_table(expert_file)
        if expert_df.empty:
            raise ValueError(f"Expert predictions dataset is empty: {expert_file}")
        expert_case_id_column = _resolve_column(
            expert_df,
            columns_cfg.get("expert_case_id") or columns_cfg.get("case_id"),
            field_name="columns.expert_case_id",
            required=False,
            fallbacks=[case_id_column, application_id_column] + COMMON_ID_COLUMNS,
        )
        if expert_case_id_column is None:
            raise ValueError("Could not resolve case id column for expert predictions table.")
        expert_prediction_column = _resolve_column(
            expert_df,
            columns_cfg.get("expert_prediction"),
            field_name="columns.expert_prediction",
            required=True,
            fallbacks=[],
        )

    capacity_df: pd.DataFrame | None = None
    capacity_case_id_column: str | None = None
    capacity_column: str | None = None
    if capacity_file is not None:
        capacity_df = read_table(capacity_file)
        if capacity_df.empty:
            raise ValueError(f"Capacity dataset is empty: {capacity_file}")
        capacity_case_id_column = _resolve_column(
            capacity_df,
            columns_cfg.get("capacity_case_id") or columns_cfg.get("case_id"),
            field_name="columns.capacity_case_id",
            required=False,
            fallbacks=[case_id_column, application_id_column] + COMMON_ID_COLUMNS,
        )
        capacity_column = _resolve_column(
            capacity_df,
            columns_cfg.get("capacity"),
            field_name="columns.capacity",
            required=True,
            fallbacks=[],
        )

    return LoadedFiFARData(
        config=config,
        config_path=resolved_config_path,
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        main_df=main_df,
        expert_df=expert_df,
        capacity_df=capacity_df,
        case_id_column=case_id_column,
        application_id_column=application_id_column,
        batch_id_column=batch_id_column,
        time_column=time_column,
        label_column=label_column,
        model_prediction_column=model_prediction_column,
        model_score_column=model_score_column,
        expert_case_id_column=expert_case_id_column,
        expert_prediction_column=expert_prediction_column,
        capacity_case_id_column=capacity_case_id_column,
        capacity_column=capacity_column,
        sensitive_attributes=sensitive_attributes,
        model_scores_available=model_score_column is not None,
        has_predefined_test_split=has_predefined_test_split,
    )


def _read_processed_csv(processed_dir: Path, file_name: str) -> pd.DataFrame:
    path = processed_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split artifact: {path}")
    return pd.read_csv(path)


def _resolve_n_experts(expert_predictions: pd.DataFrame, case_id_column: str) -> int:
    if expert_predictions.empty:
        return 0
    expert_columns = [column for column in expert_predictions.columns if column != case_id_column]
    return len(expert_columns)


def load_processed_splits(config_path: str | Path) -> LoadedSplitData:
    config, resolved_config_path = read_config(config_path)
    processed_data_dir = ensure_directory(
        resolved_config_path,
        config.get("paths", {}).get("processed_data_dir"),
        field_name="paths.processed_data_dir",
    )
    outputs_dir = resolve_outputs_dir(resolved_config_path, config)

    columns_cfg = config.get("columns", {})
    case_id_column = str(columns_cfg.get("case_id") or "case_id")
    label_column = str(columns_cfg.get("label") or "label")
    sensitive_attributes = list(columns_cfg.get("sensitive_attributes", []))

    train_metadata = _read_processed_csv(processed_data_dir, "train_metadata.csv")
    val_metadata = _read_processed_csv(processed_data_dir, "val_metadata.csv")
    test_metadata = _read_processed_csv(processed_data_dir, "test_metadata.csv")
    for split_name, metadata in [("train", train_metadata), ("validation", val_metadata), ("test", test_metadata)]:
        if case_id_column not in metadata.columns:
            raise KeyError(f"Missing `{case_id_column}` in {split_name} metadata.")
        metadata[case_id_column] = metadata[case_id_column].astype(str)

    for sensitive in sensitive_attributes:
        missing_splits = [
            split_name
            for split_name, metadata in [("train", train_metadata), ("validation", val_metadata), ("test", test_metadata)]
            if sensitive not in metadata.columns
        ]
        if missing_splits:
            LOGGER.warning(
                "Sensitive attribute `%s` is missing from metadata for splits: %s. Validation will continue.",
                sensitive,
                ", ".join(missing_splits),
            )

    capacity = _read_processed_csv(processed_data_dir, "capacity.csv")
    if capacity.empty:
        LOGGER.info("Capacity artifact is empty; capacity is not configured.")
    expert_predictions = _read_processed_csv(processed_data_dir, "expert_predictions.csv")
    if expert_predictions.empty:
        LOGGER.info("Expert predictions artifact is empty; no synthetic experts were loaded.")
    elif case_id_column in expert_predictions.columns:
        expert_predictions[case_id_column] = expert_predictions[case_id_column].astype(str)

    return LoadedSplitData(
        config=config,
        config_path=resolved_config_path,
        processed_data_dir=processed_data_dir,
        outputs_dir=outputs_dir,
        case_id_column=case_id_column,
        label_column=label_column,
        sensitive_attributes=sensitive_attributes,
        X_train=_read_processed_csv(processed_data_dir, "X_train.csv"),
        X_val=_read_processed_csv(processed_data_dir, "X_val.csv"),
        X_test=_read_processed_csv(processed_data_dir, "X_test.csv"),
        y_train=_read_processed_csv(processed_data_dir, "y_train.csv"),
        y_val=_read_processed_csv(processed_data_dir, "y_val.csv"),
        y_test=_read_processed_csv(processed_data_dir, "y_test.csv"),
        train_metadata=train_metadata,
        val_metadata=val_metadata,
        test_metadata=test_metadata,
        expert_predictions=expert_predictions,
        capacity=capacity,
        n_experts=_resolve_n_experts(expert_predictions, case_id_column),
        capacity_configured=not capacity.empty,
    )


def compute_split_statistics(split_data: LoadedSplitData) -> SplitStatistics:
    def prevalence(label_frame: pd.DataFrame) -> float:
        if split_data.label_column not in label_frame.columns:
            candidate_column = label_frame.columns[0]
        else:
            candidate_column = split_data.label_column
        return float(pd.to_numeric(label_frame[candidate_column], errors="coerce").mean())

    sensitive_attribute_used = ", ".join(split_data.sensitive_attributes) if split_data.sensitive_attributes else "none"
    return SplitStatistics(
        total_cases=len(split_data.y_train) + len(split_data.y_val) + len(split_data.y_test),
        train_cases=len(split_data.y_train),
        validation_cases=len(split_data.y_val),
        test_cases=len(split_data.y_test),
        train_fraud_prevalence=prevalence(split_data.y_train),
        validation_fraud_prevalence=prevalence(split_data.y_val),
        test_fraud_prevalence=prevalence(split_data.y_test),
        n_features=split_data.X_train.shape[1],
        n_experts=split_data.n_experts,
        sensitive_attribute_used=sensitive_attribute_used,
        capacity_configured=split_data.capacity_configured,
    )
