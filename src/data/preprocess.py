from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.load_fifar import LoadedFiFARData, load_fifar_data
from src.data.validate_splits import validate_data_splits
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PreparedArtifacts:
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
    processed_dir: Path
    model_scores_available: bool


def _split_config(loaded: LoadedFiFARData) -> dict:
    split_cfg = loaded.config.get("split", {})
    train_size = float(split_cfg.get("train_size", 0.7))
    val_size = float(split_cfg.get("val_size", 0.15))
    test_size = float(split_cfg.get("test_size", 0.15))
    total = round(train_size + val_size + test_size, 10)
    if total != 1.0:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    return {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "random_state": int(split_cfg.get("random_state", 42)),
        "stratify": bool(split_cfg.get("stratify", True)),
    }


def _normalise_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    return series


def _sorted_unique_groups(series: pd.Series) -> list:
    non_null = series.dropna()
    if non_null.empty:
        return []
    if pd.api.types.is_datetime64_any_dtype(non_null):
        return sorted(non_null.unique())
    return list(pd.Index(non_null).sort_values().unique())


def _temporal_or_batch_split(frame: pd.DataFrame, group_column: str, split_cfg: dict) -> tuple[pd.Index, pd.Index, pd.Index]:
    ordered_groups = _sorted_unique_groups(frame[group_column])
    if len(ordered_groups) < 3:
        raise ValueError(f"Need at least 3 distinct groups in `{group_column}` for temporal/batch split.")

    group_to_order = {group: idx for idx, group in enumerate(ordered_groups)}
    ordered = frame.assign(_group_order=frame[group_column].map(group_to_order)).sort_values("_group_order")
    unique_count = len(ordered_groups)
    train_cut = max(1, int(unique_count * split_cfg["train_size"]))
    val_cut = max(train_cut + 1, int(unique_count * (split_cfg["train_size"] + split_cfg["val_size"])))
    if val_cut >= unique_count:
        val_cut = unique_count - 1
    train_groups = set(ordered_groups[:train_cut])
    val_groups = set(ordered_groups[train_cut:val_cut])
    test_groups = set(ordered_groups[val_cut:])

    if not train_groups or not val_groups or not test_groups:
        raise ValueError(f"Could not create non-empty temporal/batch split from `{group_column}`.")

    train_idx = ordered.index[ordered[group_column].isin(train_groups)]
    val_idx = ordered.index[ordered[group_column].isin(val_groups)]
    test_idx = ordered.index[ordered[group_column].isin(test_groups)]
    return train_idx, val_idx, test_idx


def _stratified_split(frame: pd.DataFrame, label_column: str, split_cfg: dict) -> tuple[pd.Index, pd.Index, pd.Index]:
    random_state = split_cfg["random_state"]
    train_fraction = split_cfg["train_size"]
    val_fraction = split_cfg["val_size"]

    if split_cfg["stratify"] and frame[label_column].nunique(dropna=False) > 1:
        train_parts: list[pd.Index] = []
        val_parts: list[pd.Index] = []
        test_parts: list[pd.Index] = []

        for _, group in frame.groupby(label_column, dropna=False):
            group = group.sample(frac=1.0, random_state=random_state)
            n = len(group)
            train_end = max(1, int(round(n * train_fraction)))
            val_end = max(train_end + 1, int(round(n * (train_fraction + val_fraction)))) if n >= 3 else min(n, train_end)
            if val_end >= n and n >= 3:
                val_end = n - 1

            train_parts.append(group.index[:train_end])
            val_parts.append(group.index[train_end:val_end])
            test_parts.append(group.index[val_end:])

        train_idx = pd.Index([idx for part in train_parts for idx in part])
        val_idx = pd.Index([idx for part in val_parts for idx in part])
        test_idx = pd.Index([idx for part in test_parts for idx in part])
    else:
        shuffled = frame.sample(frac=1.0, random_state=random_state)
        n = len(shuffled)
        train_end = max(1, int(round(n * train_fraction)))
        val_end = max(train_end + 1, int(round(n * (train_fraction + val_fraction)))) if n >= 3 else min(n, train_end)
        if val_end >= n and n >= 3:
            val_end = n - 1
        train_idx = shuffled.index[:train_end]
        val_idx = shuffled.index[train_end:val_end]
        test_idx = shuffled.index[val_end:]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Unable to create non-empty train/validation/test split. Check dataset size and split fractions.")
    return train_idx, val_idx, test_idx


def _select_split_indices(loaded: LoadedFiFARData) -> tuple[pd.Index, pd.Index, pd.Index]:
    frame = loaded.main_df.copy()
    split_cfg = _split_config(loaded)

    if loaded.has_predefined_test_split and "__split__" in frame.columns:
        train_pool = frame.loc[frame["__split__"] == "train"].copy()
        test_idx = frame.index[frame["__split__"] == "test"]
        if train_pool.empty or len(test_idx) == 0:
            raise ValueError("Predefined train/test split is empty.")
        if loaded.time_column is not None:
            train_pool[loaded.time_column] = pd.to_datetime(train_pool[loaded.time_column], errors="coerce")
            if train_pool[loaded.time_column].notna().sum() >= 3 and train_pool[loaded.time_column].nunique(dropna=True) >= 3:
                train_idx, val_idx, _ = _temporal_or_batch_split(train_pool, loaded.time_column, split_cfg)
                return train_idx, val_idx, test_idx
        if loaded.batch_id_column is not None and train_pool[loaded.batch_id_column].nunique(dropna=True) >= 3:
            train_idx, val_idx, _ = _temporal_or_batch_split(train_pool, loaded.batch_id_column, split_cfg)
            return train_idx, val_idx, test_idx

        train_idx, val_idx, _ = _stratified_split(train_pool, loaded.label_column, split_cfg)
        return train_idx, val_idx, test_idx

    if loaded.time_column is not None:
        frame[loaded.time_column] = pd.to_datetime(frame[loaded.time_column], errors="coerce")
        if frame[loaded.time_column].notna().sum() >= 3 and frame[loaded.time_column].nunique(dropna=True) >= 3:
            LOGGER.info("Using temporal split on `%s`", loaded.time_column)
            return _temporal_or_batch_split(frame, loaded.time_column, split_cfg)

    if loaded.batch_id_column is not None and frame[loaded.batch_id_column].nunique(dropna=True) >= 3:
        LOGGER.info("Using batch split on `%s`", loaded.batch_id_column)
        return _temporal_or_batch_split(frame, loaded.batch_id_column, split_cfg)

    LOGGER.info("Falling back to stratified split.")
    return _stratified_split(frame, loaded.label_column, split_cfg)


def _build_feature_matrix(loaded: LoadedFiFARData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = loaded.main_df.copy()
    protected_columns = {
        loaded.label_column,
        loaded.case_id_column,
        loaded.application_id_column,
        loaded.batch_id_column,
        loaded.time_column,
        "__split__",
    }
    protected_columns.update(loaded.sensitive_attributes)
    protected_columns = {column for column in protected_columns if column}

    feature_frame = frame.drop(columns=list(protected_columns), errors="ignore")
    y_frame = pd.DataFrame({_label_output_name(loaded.label_column): _normalise_labels(frame[loaded.label_column])})

    categorical_columns = feature_frame.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    feature_frame = pd.get_dummies(feature_frame, columns=categorical_columns, dummy_na=True)

    metadata_columns = [loaded.case_id_column]
    if loaded.application_id_column and loaded.application_id_column != loaded.case_id_column:
        metadata_columns.append(loaded.application_id_column)
    if loaded.batch_id_column:
        metadata_columns.append(loaded.batch_id_column)
    if loaded.time_column:
        metadata_columns.append(loaded.time_column)
    metadata_columns.extend([column for column in loaded.sensitive_attributes if column not in metadata_columns])
    if loaded.model_prediction_column and loaded.model_prediction_column not in metadata_columns:
        metadata_columns.append(loaded.model_prediction_column)
    if loaded.model_score_column and loaded.model_score_column not in metadata_columns:
        metadata_columns.append(loaded.model_score_column)
    metadata_frame = frame.loc[:, metadata_columns].copy()

    return feature_frame, y_frame, metadata_frame


def _label_output_name(label_column: str) -> str:
    return label_column


def _save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _prepare_optional_tables(loaded: LoadedFiFARData) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_values = set(loaded.main_df[loaded.case_id_column].astype(str))

    if loaded.expert_df is not None:
        expert = loaded.expert_df.copy()
        expert_case_column = loaded.expert_case_id_column
        assert expert_case_column is not None
        expert[expert_case_column] = expert[expert_case_column].astype(str)
        missing_ids = set(expert[expert_case_column]) - id_values
        if missing_ids:
            raise ValueError(
                f"Expert prediction table contains case ids not present in main table. "
                f"Sample missing ids: {sorted(list(missing_ids))[:5]}"
            )
        expert_output = expert
    else:
        expert_output = pd.DataFrame(columns=["case_id", "expert_prediction"])

    if loaded.capacity_df is not None:
        capacity = loaded.capacity_df.copy()
        if loaded.capacity_case_id_column is not None:
            capacity[loaded.capacity_case_id_column] = capacity[loaded.capacity_case_id_column].astype(str)
            missing_ids = set(capacity[loaded.capacity_case_id_column]) - id_values
            if missing_ids:
                raise ValueError(
                    f"Capacity table contains case ids not present in main table. "
                    f"Sample missing ids: {sorted(list(missing_ids))[:5]}"
                )
        capacity_output = capacity
    else:
        capacity_output = pd.DataFrame(columns=["case_id", "capacity"])

    return expert_output, capacity_output


def prepare_data(config_path: str | Path) -> PreparedArtifacts:
    loaded = load_fifar_data(config_path)

    if not loaded.model_scores_available:
        LOGGER.warning(
            "Model score column is not configured or not available in the main table. "
            "Later pipeline stages should train or derive a model score."
        )

    X, y, metadata = _build_feature_matrix(loaded)
    train_idx, val_idx, test_idx = _select_split_indices(loaded)

    X_train = X.loc[train_idx].reset_index(drop=True)
    X_val = X.loc[val_idx].reset_index(drop=True)
    X_test = X.loc[test_idx].reset_index(drop=True)
    y_train = y.loc[train_idx].reset_index(drop=True)
    y_val = y.loc[val_idx].reset_index(drop=True)
    y_test = y.loc[test_idx].reset_index(drop=True)
    train_metadata = metadata.loc[train_idx].reset_index(drop=True)
    val_metadata = metadata.loc[val_idx].reset_index(drop=True)
    test_metadata = metadata.loc[test_idx].reset_index(drop=True)

    expert_predictions, capacity = _prepare_optional_tables(loaded)
    processed_dir = loaded.processed_data_dir

    _save_frame(X_train, processed_dir / "X_train.csv")
    _save_frame(X_val, processed_dir / "X_val.csv")
    _save_frame(X_test, processed_dir / "X_test.csv")
    _save_frame(y_train, processed_dir / "y_train.csv")
    _save_frame(y_val, processed_dir / "y_val.csv")
    _save_frame(y_test, processed_dir / "y_test.csv")
    _save_frame(train_metadata, processed_dir / "train_metadata.csv")
    _save_frame(val_metadata, processed_dir / "val_metadata.csv")
    _save_frame(test_metadata, processed_dir / "test_metadata.csv")
    _save_frame(expert_predictions, processed_dir / "expert_predictions.csv")
    _save_frame(capacity, processed_dir / "capacity.csv")

    validate_data_splits(config_path)

    return PreparedArtifacts(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_metadata=train_metadata,
        val_metadata=val_metadata,
        test_metadata=test_metadata,
        expert_predictions=expert_predictions,
        capacity=capacity,
        processed_dir=processed_dir,
        model_scores_available=loaded.model_scores_available,
    )
