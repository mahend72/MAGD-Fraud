from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.io import load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".json"}

LABEL_KEYWORDS = {
    "label",
    "target",
    "class",
    "outcome",
    "fraud",
    "is_fraud",
    "fraud_flag",
    "ground_truth",
    "truth",
    "y",
}
PREDICTION_KEYWORDS = {
    "pred",
    "prediction",
    "predicted",
    "model_output",
    "decision",
    "class",
    "ai_decision",
}
SCORE_KEYWORDS = {
    "score",
    "prob",
    "probability",
    "confidence",
    "risk",
    "logit",
    "uncertainty",
    "calibration",
    "likelihood",
}
EXPERT_KEYWORDS = {
    "expert",
    "analyst",
    "reviewer",
    "human",
    "investigator",
    "adjudicated",
    "adjudication",
    "manual",
}
CAPACITY_KEYWORDS = {
    "capacity",
    "queue",
    "sla",
    "workload",
    "bandwidth",
    "limit",
    "availability",
    "staffing",
}
SENSITIVE_KEYWORDS = {
    "age",
    "gender",
    "sex",
    "race",
    "ethnicity",
    "demographic",
    "nationality",
    "country",
    "citizenship",
    "marital",
    "religion",
    "disability",
    "zip",
    "postcode",
}


@dataclass
class DatasetSummary:
    file_path: Path
    rows: int
    columns: int
    column_names: list[str]
    missing_summary: pd.DataFrame
    possible_label_columns: list[str]
    possible_model_score_columns: list[str]
    possible_model_prediction_columns: list[str]
    possible_expert_prediction_columns: list[str]
    possible_capacity_columns: list[str]
    possible_sensitive_columns: list[str]


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name)


def _matches_keywords(column: str, keywords: set[str]) -> bool:
    normalized = _normalize_name(column)
    parts = {part for part in normalized.split("_") if part}
    return any(keyword in normalized or keyword in parts for keyword in keywords)


def _detect_columns(columns: Iterable[str], keywords: set[str]) -> list[str]:
    return [column for column in columns if _matches_keywords(column, keywords)]


def _read_table(file_path: Path) -> pd.DataFrame:
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


def _missing_value_summary(frame: pd.DataFrame) -> pd.DataFrame:
    missing_count = frame.isna().sum()
    missing_pct = (missing_count / len(frame) * 100.0) if len(frame) else 0.0
    summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "dtype": frame.dtypes.astype(str),
        }
    )
    return summary.sort_values(["missing_count", "missing_pct"], ascending=False)


def inspect_file(file_path: Path) -> DatasetSummary:
    frame = _read_table(file_path)
    column_names = [str(column) for column in frame.columns]
    return DatasetSummary(
        file_path=file_path,
        rows=len(frame),
        columns=len(frame.columns),
        column_names=column_names,
        missing_summary=_missing_value_summary(frame),
        possible_label_columns=_detect_columns(column_names, LABEL_KEYWORDS),
        possible_model_score_columns=_detect_columns(column_names, SCORE_KEYWORDS),
        possible_model_prediction_columns=_detect_columns(column_names, PREDICTION_KEYWORDS),
        possible_expert_prediction_columns=[
            column
            for column in column_names
            if _matches_keywords(column, EXPERT_KEYWORDS)
            and (
                _matches_keywords(column, PREDICTION_KEYWORDS)
                or _matches_keywords(column, LABEL_KEYWORDS)
            )
        ],
        possible_capacity_columns=_detect_columns(column_names, CAPACITY_KEYWORDS),
        possible_sensitive_columns=_detect_columns(column_names, SENSITIVE_KEYWORDS),
    )


def _resolve_raw_dir(config_path: str | Path) -> tuple[Path, dict]:
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    raw_data_dir = config.get("paths", {}).get("raw_data_dir")
    if not raw_data_dir:
        raise ValueError("Missing `paths.raw_data_dir` in config.")
    raw_dir = Path(raw_data_dir)
    if not raw_dir.is_absolute():
        raw_dir = (config_path.parent / raw_dir).resolve()
    return raw_dir, config


def find_supported_files(raw_dir: Path, file_patterns: list[str] | None = None) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Configured raw data directory does not exist: {raw_dir}. "
            "Place FiFAR dataset files inside data/raw/ or update config.yaml."
        )
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Configured raw data path is not a directory: {raw_dir}")

    patterns = file_patterns or ["*"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in raw_dir.glob(pattern) if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)

    unique_files = sorted(set(files))
    if not unique_files:
        raise FileNotFoundError(
            f"No supported dataset files found in {raw_dir}. "
            "Expected one of: .csv, .tsv, .parquet, .xlsx, .xls, .json"
        )
    return unique_files


def format_summary(summary: DatasetSummary, max_candidates: int = 10) -> str:
    def fmt(values: list[str]) -> str:
        if not values:
            return "None detected"
        clipped = values[:max_candidates]
        suffix = "" if len(values) <= max_candidates else f" ... (+{len(values) - max_candidates} more)"
        return ", ".join(clipped) + suffix

    missing_display = summary.missing_summary.to_string()
    return "\n".join(
        [
            f"File: {summary.file_path.name}",
            f"Shape: {summary.rows} rows x {summary.columns} columns",
            f"Columns: {', '.join(summary.column_names) if summary.column_names else 'No columns'}",
            "Missing value summary:",
            missing_display,
            f"Possible label columns: {fmt(summary.possible_label_columns)}",
            f"Possible model score columns: {fmt(summary.possible_model_score_columns)}",
            f"Possible model prediction columns: {fmt(summary.possible_model_prediction_columns)}",
            f"Possible expert prediction columns: {fmt(summary.possible_expert_prediction_columns)}",
            f"Possible capacity columns: {fmt(summary.possible_capacity_columns)}",
            f"Possible sensitive columns: {fmt(summary.possible_sensitive_columns)}",
        ]
    )


def run_inspection(config_path: str | Path) -> list[DatasetSummary]:
    raw_dir, config = _resolve_raw_dir(config_path)
    patterns = config.get("dataset", {}).get("file_patterns", [])
    max_candidates = config.get("inspection", {}).get("max_candidate_columns_per_group", 10)
    files = find_supported_files(raw_dir, patterns)

    print(f"Configured raw data directory: {raw_dir}")
    print("Files discovered:")
    for file_path in files:
        print(f" - {file_path.name}")

    summaries: list[DatasetSummary] = []
    for file_path in files:
        LOGGER.info("Inspecting %s", file_path)
        summary = inspect_file(file_path)
        summaries.append(summary)
        print()
        print(format_summary(summary, max_candidates=max_candidates))
    return summaries
