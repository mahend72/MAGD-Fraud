from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.predict import load_processed_split
from src.utils.io import load_yaml


@dataclass
class EmbeddingArtifacts:
    method: str
    n_components: int
    feature_names: list[str]
    train_embeddings: pd.DataFrame
    val_embeddings: pd.DataFrame
    test_embeddings: pd.DataFrame
    train_labels: pd.Series
    val_labels: pd.Series
    test_labels: pd.Series
    train_metadata: pd.DataFrame
    val_metadata: pd.DataFrame
    test_metadata: pd.DataFrame


def _impute_with_train_medians(
    train_X: pd.DataFrame,
    val_X: pd.DataFrame,
    test_X: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_numeric = train_X.apply(pd.to_numeric, errors="coerce")
    val_numeric = val_X.apply(pd.to_numeric, errors="coerce")
    test_numeric = test_X.apply(pd.to_numeric, errors="coerce")

    medians = train_numeric.median(axis=0)
    medians = medians.fillna(0.0)
    train_filled = train_numeric.fillna(medians).fillna(0.0)
    val_filled = val_numeric.fillna(medians).fillna(0.0)
    test_filled = test_numeric.fillna(medians).fillna(0.0)
    return train_filled, val_filled, test_filled


def _embedding_config(config: dict) -> tuple[str, int]:
    assurance_cfg = config.get("assurance", {})
    embedding_cfg = assurance_cfg.get("distance_embedding", {})
    method = str(assurance_cfg.get("embedding_method", embedding_cfg.get("method", "pca"))).lower()
    n_components = int(assurance_cfg.get("pca_components", embedding_cfg.get("n_components", 8)))
    if method != "pca":
        raise ValueError(f"Unsupported embedding method `{method}`. Only `pca` is implemented.")
    if n_components <= 0:
        raise ValueError(f"`assurance.distance_embedding.n_components` must be positive, got {n_components}")
    return method, n_components


def _fit_pca(train_X: pd.DataFrame, n_components: int) -> tuple[Any, int]:
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for PCA embeddings. Install dependencies from requirements.txt."
        ) from exc

    max_components = min(len(train_X), train_X.shape[1], n_components)
    if max_components <= 0:
        raise ValueError("Cannot compute PCA embeddings on an empty training feature matrix.")
    pca = PCA(n_components=max_components, random_state=42)
    pca.fit(train_X)
    return pca, max_components


def _transform_frame(model: Any, frame: pd.DataFrame, component_count: int) -> pd.DataFrame:
    transformed = model.transform(frame)
    columns = [f"embedding_{idx}" for idx in range(component_count)]
    return pd.DataFrame(transformed, columns=columns, index=frame.index)


def _resolve_assurance_output_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("experiment", {}).get("output_dir", config.get("paths", {}).get("outputs_dir", "data/outputs")))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    assurance_dir = outputs_root / "assurance"
    assurance_dir.mkdir(parents=True, exist_ok=True)
    return assurance_dir


def embedding_frame_for_split(
    embeddings: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    split_name: str,
) -> pd.DataFrame:
    if "case_id" not in metadata.columns:
        raise ValueError(f"Metadata for split `{split_name}` is missing required `case_id` column.")
    case_ids = metadata["case_id"].astype(str).reset_index(drop=True)
    split_series = pd.Series([split_name] * len(case_ids), name="split")
    return pd.concat(
        [
            case_ids.rename("case_id"),
            split_series,
            embeddings.reset_index(drop=True),
        ],
        axis=1,
    )


def write_embedding_outputs(
    config_path: str | Path,
    artifacts: EmbeddingArtifacts | None = None,
) -> dict[str, Path]:
    resolved_config_path = Path(config_path).resolve()
    embedding_artifacts = artifacts or load_embedding_artifacts(resolved_config_path)
    assurance_dir = _resolve_assurance_output_dir(resolved_config_path)

    split_frames = {
        "train": embedding_frame_for_split(
            embedding_artifacts.train_embeddings,
            embedding_artifacts.train_metadata,
            split_name="train",
        ),
        "val": embedding_frame_for_split(
            embedding_artifacts.val_embeddings,
            embedding_artifacts.val_metadata,
            split_name="val",
        ),
        "test": embedding_frame_for_split(
            embedding_artifacts.test_embeddings,
            embedding_artifacts.test_metadata,
            split_name="test",
        ),
    }

    output_paths: dict[str, Path] = {}
    for split_name, frame in split_frames.items():
        path = assurance_dir / f"embeddings_{split_name}.csv"
        frame.to_csv(path, index=False)
        output_paths[split_name] = path
    return output_paths


def load_embedding_artifacts(config_path: str | Path) -> EmbeddingArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    processed_dir_cfg = config.get("paths", {}).get("processed_data_dir", "data/processed")
    processed_dir = Path(processed_dir_cfg)
    if not processed_dir.is_absolute():
        processed_dir = (resolved_config_path.parent / processed_dir).resolve()

    method, n_components = _embedding_config(config)
    X_train, y_train, train_metadata = load_processed_split(processed_dir, "train")
    X_val, y_val, val_metadata = load_processed_split(processed_dir, "val")
    X_test, y_test, test_metadata = load_processed_split(processed_dir, "test")
    X_train, X_val, X_test = _impute_with_train_medians(X_train, X_val, X_test)

    pca, fitted_components = _fit_pca(X_train, n_components)
    train_embeddings = _transform_frame(pca, X_train, fitted_components)
    val_embeddings = _transform_frame(pca, X_val, fitted_components)
    test_embeddings = _transform_frame(pca, X_test, fitted_components)

    return EmbeddingArtifacts(
        method=method,
        n_components=fitted_components,
        feature_names=X_train.columns.tolist(),
        train_embeddings=train_embeddings,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        train_labels=y_train.iloc[:, 0].astype(int),
        val_labels=y_val.iloc[:, 0].astype(int),
        test_labels=y_test.iloc[:, 0].astype(int),
        train_metadata=train_metadata,
        val_metadata=val_metadata,
        test_metadata=test_metadata,
    )
