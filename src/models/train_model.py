from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any

import pandas as pd

from src.models.predict import (
    build_prediction_frame,
    compute_binary_metrics,
    decision_threshold,
    load_processed_split,
    predict_scores,
    threshold_source,
)
from src.utils.io import load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ModelRunArtifacts:
    model_source: str
    model_name: str
    model_dir: Path
    metrics: pd.DataFrame
    train_predictions: pd.DataFrame
    val_predictions: pd.DataFrame
    test_predictions: pd.DataFrame


def _output_directories(config_path: Path) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    model_dir = outputs_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    return outputs_root, model_dir


def _training_costs(config: dict) -> tuple[float, float]:
    costs_cfg = config.get("costs", {})
    if costs_cfg:
        return float(costs_cfg.get("false_positive", 1.0)), float(costs_cfg.get("false_negative", 5.0))
    model_cfg = config.get("model", {})
    return float(model_cfg.get("false_positive_cost", 1.0)), float(model_cfg.get("false_negative_cost", 5.0))


def _model_context(config_path: Path, config: dict):
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = (config_path.parent / processed_dir).resolve()
    columns_cfg = config.get("columns", {})
    case_id_column = str(columns_cfg.get("case_id") or "case_id")
    model_score_column = columns_cfg.get("model_score")
    train_metadata = pd.read_csv(processed_dir / "train_metadata.csv")
    model_scores_available = bool(model_score_column) and model_score_column in train_metadata.columns
    return SimpleNamespace(
        processed_data_dir=processed_dir,
        case_id_column=case_id_column,
        model_score_column=model_score_column,
        model_scores_available=model_scores_available,
    )


def _use_existing_scores(config: dict, loaded) -> bool:
    experiment_cfg = config.get("experiment", {})
    use_existing = bool(experiment_cfg.get("use_existing_scores", True))
    return use_existing and loaded.model_scores_available


def _initialise_xgboost(config: dict, y_train: pd.Series) -> tuple[Any, str]:
    from xgboost import XGBClassifier

    model_cfg = config.get("model", {})
    xgb_cfg = model_cfg.get("xgboost", {})
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = negatives / positives if positives > 0 and negatives > 0 else 1.0
    model = XGBClassifier(
        n_estimators=int(xgb_cfg.get("n_estimators", 300)),
        max_depth=int(xgb_cfg.get("max_depth", 6)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
        subsample=float(xgb_cfg.get("subsample", 0.9)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.9)),
        reg_lambda=float(xgb_cfg.get("reg_lambda", 1.0)),
        random_state=int(xgb_cfg.get("random_state", config.get("experiment", {}).get("seed", 42))),
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
    )
    return model, "xgboost"


def _initialise_random_forest(config: dict) -> tuple[Any, str]:
    from sklearn.ensemble import RandomForestClassifier

    model_cfg = config.get("model", {})
    rf_cfg = model_cfg.get("random_forest", {})
    model = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 300)),
        max_depth=rf_cfg.get("max_depth"),
        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 1)),
        random_state=int(rf_cfg.get("random_state", config.get("experiment", {}).get("seed", 42))),
        class_weight="balanced",
        n_jobs=-1,
    )
    return model, "random_forest"


def _initialise_model(config: dict, y_train: pd.Series) -> tuple[Any, str]:
    preferred = str(config.get("model", {}).get("model_type", "xgboost")).lower()
    fallback = str(config.get("model", {}).get("fallback_model", "random_forest")).lower()
    order = [preferred]
    if fallback != preferred:
        order.append(fallback)

    last_error: Exception | None = None
    for model_name in order:
        try:
            if model_name == "xgboost":
                return _initialise_xgboost(config, y_train)
            if model_name == "random_forest":
                return _initialise_random_forest(config)
            raise ValueError(f"Unsupported model type requested in config: {model_name}")
        except Exception as exc:
            LOGGER.warning("Model `%s` unavailable, trying next option: %s", model_name, exc)
            last_error = exc

    raise ImportError("Unable to initialize any configured model type.") from last_error


def _save_model_artifact(model_dir: Path, model: Any, model_name: str, feature_names: list[str], threshold: float, source: str) -> Path:
    artifact = {
        "model": model,
        "model_name": model_name,
        "feature_names": feature_names,
        "threshold": threshold,
        "threshold_source": threshold_source(),
        "source": source,
    }
    artifact_path = model_dir / "model.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    return artifact_path


def _metrics_row(
    *,
    split_name: str,
    model_name: str,
    y_true: pd.Series,
    y_score: pd.Series | Any,
    threshold: float,
    fp_cost: float,
    fn_cost: float,
) -> dict[str, float | str]:
    metrics = compute_binary_metrics(
        y_true=y_true,
        y_score=y_score,
        threshold=threshold,
        false_positive_cost=fp_cost,
        false_negative_cost=fn_cost,
    )
    return {
        "split": split_name,
        "model_name": model_name,
        "threshold_source": threshold_source(),
        **metrics,
    }


def _score_existing_ai(
    loaded,
    config: dict,
    threshold: float,
    model_dir: Path,
) -> ModelRunArtifacts:
    processed_dir = loaded.processed_data_dir
    case_id_column = loaded.case_id_column
    score_column = loaded.model_score_column
    assert score_column is not None

    fp_cost, fn_cost = _training_costs(config)
    prediction_frames: dict[str, pd.DataFrame] = {}
    metrics_rows: list[dict[str, float | str]] = []

    for split_name in ["train", "val", "test"]:
        _, y, metadata = load_processed_split(processed_dir, split_name)
        if score_column not in metadata.columns:
            raise ValueError(
                f"Expected model score column `{score_column}` in {split_name}_metadata.csv, "
                "but it was not found. Re-run preprocessing with the configured score column."
            )
        prediction_frame = build_prediction_frame(
            case_ids=metadata[case_id_column],
            y_true=y.iloc[:, 0],
            ai_scores=metadata[score_column],
            threshold=threshold,
            split_name=split_name,
        )
        prediction_frames[split_name] = prediction_frame
        metrics_rows.append(
            _metrics_row(
                split_name=split_name,
                model_name="existing_ai_scores",
                y_true=prediction_frame["y_true"],
                y_score=prediction_frame["ai_score"],
                threshold=threshold,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
            )
        )

    manifest = {
        "source": "existing_scores",
        "model_name": "existing_ai_scores",
        "threshold": threshold,
        "threshold_source": threshold_source(),
        "score_column": score_column,
    }
    with (model_dir / "model_manifest.pkl").open("wb") as handle:
        pickle.dump(manifest, handle)

    return ModelRunArtifacts(
        model_source="existing_scores",
        model_name="existing_ai_scores",
        model_dir=model_dir,
        metrics=pd.DataFrame(metrics_rows),
        train_predictions=prediction_frames["train"],
        val_predictions=prediction_frames["val"],
        test_predictions=prediction_frames["test"],
    )


def _train_new_model(
    loaded,
    config: dict,
    threshold: float,
    model_dir: Path,
) -> ModelRunArtifacts:
    processed_dir = loaded.processed_data_dir
    X_train, y_train, train_metadata = load_processed_split(processed_dir, "train")
    X_val, y_val, val_metadata = load_processed_split(processed_dir, "val")
    X_test, y_test, test_metadata = load_processed_split(processed_dir, "test")

    y_train_series = y_train.iloc[:, 0].astype(int)
    y_val_series = y_val.iloc[:, 0].astype(int)
    y_test_series = y_test.iloc[:, 0].astype(int)

    model, model_name = _initialise_model(config, y_train_series)
    model.fit(X_train, y_train_series)
    _save_model_artifact(model_dir, model, model_name, X_train.columns.tolist(), threshold, "trained_model")

    fp_cost, fn_cost = _training_costs(config)
    metrics_rows: list[dict[str, float | str]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    split_payloads = {
        "train": (X_train, y_train_series, train_metadata),
        "val": (X_val, y_val_series, val_metadata),
        "test": (X_test, y_test_series, test_metadata),
    }

    for split_name, (X_split, y_split, metadata_split) in split_payloads.items():
        scores = predict_scores(model, X_split)
        prediction_frame = build_prediction_frame(
            case_ids=metadata_split[loaded.case_id_column],
            y_true=y_split,
            ai_scores=scores,
            threshold=threshold,
            split_name=split_name,
        )
        prediction_frames[split_name] = prediction_frame
        metrics_rows.append(
            _metrics_row(
                split_name=split_name,
                model_name=model_name,
                y_true=prediction_frame["y_true"],
                y_score=prediction_frame["ai_score"],
                threshold=threshold,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
            )
        )

    return ModelRunArtifacts(
        model_source="trained_model",
        model_name=model_name,
        model_dir=model_dir,
        metrics=pd.DataFrame(metrics_rows),
        train_predictions=prediction_frames["train"],
        val_predictions=prediction_frames["val"],
        test_predictions=prediction_frames["test"],
    )


def _save_outputs(artifacts: ModelRunArtifacts) -> None:
    artifacts.train_predictions.to_csv(artifacts.model_dir / "train_predictions.csv", index=False)
    artifacts.val_predictions.to_csv(artifacts.model_dir / "val_predictions.csv", index=False)
    artifacts.test_predictions.to_csv(artifacts.model_dir / "test_predictions.csv", index=False)
    artifacts.metrics.to_csv(artifacts.model_dir / "model_metrics.csv", index=False)


def run_model_predictions(config_path: str | Path) -> ModelRunArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    loaded = _model_context(resolved_config_path, config)
    _, model_dir = _output_directories(resolved_config_path)
    threshold = decision_threshold(resolved_config_path)

    if _use_existing_scores(config, loaded):
        LOGGER.info("Using existing benchmark AI scores from the FiFAR dataset.")
        artifacts = _score_existing_ai(loaded, config, threshold, model_dir)
    else:
        LOGGER.info("Training a tabular fraud classifier. Threshold is fixed from config and not tuned on test labels.")
        artifacts = _train_new_model(loaded, config, threshold, model_dir)

    _save_outputs(artifacts)
    return artifacts


def train_or_load_model(config_path: str | Path) -> ModelRunArtifacts:
    return run_model_predictions(config_path)
