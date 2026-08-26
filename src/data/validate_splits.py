from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.load_fifar import LoadedSplitData, SplitStatistics, compute_split_statistics, load_processed_splits
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class SplitValidationArtifacts:
    split_data: LoadedSplitData
    statistics: SplitStatistics
    summary_table: pd.DataFrame
    paper_tables_dir: Path


def _label_column(frame: pd.DataFrame, configured_label: str) -> str:
    return configured_label if configured_label in frame.columns else frame.columns[0]


def _validate_binary_labels(split_data: LoadedSplitData) -> None:
    for split_name, labels in [
        ("train", split_data.y_train),
        ("validation", split_data.y_val),
        ("test", split_data.y_test),
    ]:
        label_column = _label_column(labels, split_data.label_column)
        values = pd.to_numeric(labels[label_column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Labels for {split_name} split contain non-numeric values.")
        unique_values = set(values.astype(int).unique().tolist())
        if not unique_values.issubset({0, 1}):
            raise ValueError(f"Labels for {split_name} split must be binary 0/1, got {sorted(unique_values)}.")


def _validate_matching_lengths(split_data: LoadedSplitData) -> None:
    for split_name, X_frame, y_frame, metadata_frame in [
        ("train", split_data.X_train, split_data.y_train, split_data.train_metadata),
        ("validation", split_data.X_val, split_data.y_val, split_data.val_metadata),
        ("test", split_data.X_test, split_data.y_test, split_data.test_metadata),
    ]:
        if len(X_frame) != len(y_frame) or len(X_frame) != len(metadata_frame):
            raise ValueError(
                f"Split `{split_name}` has inconsistent row counts: "
                f"X={len(X_frame)}, y={len(y_frame)}, metadata={len(metadata_frame)}."
            )


def _validate_case_id_overlap(split_data: LoadedSplitData) -> None:
    case_id_column = split_data.case_id_column
    train_ids = set(split_data.train_metadata[case_id_column].astype(str))
    val_ids = set(split_data.val_metadata[case_id_column].astype(str))
    test_ids = set(split_data.test_metadata[case_id_column].astype(str))

    overlaps = {
        "train/validation": train_ids & val_ids,
        "train/test": train_ids & test_ids,
        "validation/test": val_ids & test_ids,
    }
    duplicate_pairs = {name: values for name, values in overlaps.items() if values}
    if duplicate_pairs:
        preview = {
            name: sorted(list(values))[:5]
            for name, values in duplicate_pairs.items()
        }
        raise ValueError(f"Case IDs overlap across splits: {preview}")


def _build_summary_table(split_data: LoadedSplitData, statistics: SplitStatistics) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "total_cases": statistics.total_cases,
                "train_cases": statistics.train_cases,
                "validation_cases": statistics.validation_cases,
                "test_cases": statistics.test_cases,
                "train_fraud_prevalence": statistics.train_fraud_prevalence,
                "validation_fraud_prevalence": statistics.validation_fraud_prevalence,
                "test_fraud_prevalence": statistics.test_fraud_prevalence,
                "number_of_synthetic_experts": statistics.n_experts,
                "sensitive_attribute_used": statistics.sensitive_attribute_used,
                "capacity_configured": "yes" if statistics.capacity_configured else "no",
            }
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    row_lines = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *row_lines]) + "\n"


def _save_summary(summary_table: pd.DataFrame, paper_tables_dir: Path) -> None:
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(paper_tables_dir / "dataset_summary.csv", index=False)
    (paper_tables_dir / "dataset_summary.md").write_text(_markdown_table(summary_table), encoding="utf-8")


def validate_data_splits(config_path: str | Path) -> SplitValidationArtifacts:
    split_data = load_processed_splits(config_path)
    _validate_matching_lengths(split_data)
    _validate_binary_labels(split_data)
    _validate_case_id_overlap(split_data)

    statistics = compute_split_statistics(split_data)
    LOGGER.info(
        "Split sizes: train=%s, validation=%s, test=%s",
        statistics.train_cases,
        statistics.validation_cases,
        statistics.test_cases,
    )
    LOGGER.info(
        "Feature counts: train=%s, validation=%s, test=%s",
        split_data.X_train.shape[1],
        split_data.X_val.shape[1],
        split_data.X_test.shape[1],
    )
    LOGGER.info(
        "Fraud prevalence: train=%.4f, validation=%.4f, test=%.4f",
        statistics.train_fraud_prevalence,
        statistics.validation_fraud_prevalence,
        statistics.test_fraud_prevalence,
    )
    LOGGER.info("Synthetic experts loaded: %s", statistics.n_experts)
    if not split_data.capacity_configured:
        LOGGER.warning("Capacity is not configured; validation will continue.")

    paper_tables_dir = split_data.outputs_dir / "paper_tables"
    summary_table = _build_summary_table(split_data, statistics)
    _save_summary(summary_table, paper_tables_dir)

    return SplitValidationArtifacts(
        split_data=split_data,
        statistics=statistics,
        summary_table=summary_table,
        paper_tables_dir=paper_tables_dir,
    )
