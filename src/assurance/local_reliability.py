from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import load_yaml


@dataclass
class LocalReliabilityArtifacts:
    local_reliability: pd.DataFrame
    assurance_dir: Path


def _resolve_output_dirs(config_path: Path) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    model_dir = outputs_root / "model"
    assurance_dir = outputs_root / "assurance"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Missing model outputs directory at {model_dir}. "
            "Run `python scripts/run_model_predictions.py --config config.yaml` first."
        )
    assurance_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, assurance_dir


def _neighbor_top_k(config: dict, train_size: int) -> int:
    assurance_cfg = config.get("assurance", {})
    top_k = int(assurance_cfg.get("knn_k", assurance_cfg.get("neighbor_top_k", 5)))
    if top_k <= 0:
        raise ValueError(f"`assurance.knn_k` must be positive, got {top_k}")
    if train_size <= 0:
        raise ValueError("Training embeddings are empty; cannot compute local reliability.")
    return min(top_k, train_size)


def _load_prediction_file(model_dir: Path, split_name: str) -> pd.DataFrame:
    path = model_dir / f"{split_name}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {split_name} predictions at {path}. "
            "Run `python scripts/run_model_predictions.py --config config.yaml` first."
        )
    frame = pd.read_csv(path)
    required = {"case_id", "ai_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{split_name.title()} predictions are missing required columns: {sorted(missing)}")
    return frame


def _load_train_predictions(model_dir: Path) -> pd.DataFrame:
    frame = _load_prediction_file(model_dir, "train")
    required = {"y_true"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Train predictions are missing required columns: {sorted(missing)}")
    return frame


def _load_embedding_file(assurance_dir: Path, split_name: str) -> pd.DataFrame:
    path = assurance_dir / f"embeddings_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {split_name} embeddings at {path}. "
            "Run `python scripts/run_distance_uncertainty.py --config config.yaml` first."
        )
    frame = pd.read_csv(path)
    required = {"case_id", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{split_name.title()} embeddings are missing required columns: {sorted(missing)}")
    embedding_columns = [column for column in frame.columns if column.startswith("embedding_")]
    if not embedding_columns:
        raise ValueError(f"{split_name.title()} embeddings do not contain any `embedding_*` columns.")
    return frame


def _nearest_neighbor_query(
    query_vectors: np.ndarray,
    train_vectors: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for local reliability. Install dependencies from requirements.txt."
        ) from exc

    index = NearestNeighbors(n_neighbors=top_k, metric="euclidean", algorithm="ball_tree", n_jobs=-1)
    index.fit(train_vectors)
    chunk_size = 5000
    distance_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    for start in range(0, len(query_vectors), chunk_size):
        stop = min(start + chunk_size, len(query_vectors))
        distances, indices = index.kneighbors(query_vectors[start:stop], return_distance=True)
        distance_chunks.append(distances)
        index_chunks.append(indices)
    return np.vstack(distance_chunks), np.vstack(index_chunks)


def _prepare_train_reference(
    train_embeddings: pd.DataFrame,
    train_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    embedding_columns = [column for column in train_embeddings.columns if column.startswith("embedding_")]
    train_frame = train_embeddings.copy()
    train_frame["case_id"] = train_frame["case_id"].astype(str)

    predictions = train_predictions.copy()
    predictions["case_id"] = predictions["case_id"].astype(str)
    predictions = predictions.drop_duplicates(subset=["case_id"], keep="first")

    merged = train_frame.merge(
        predictions[["case_id", "y_true", "ai_pred"]],
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    if merged[["y_true", "ai_pred"]].isna().any().any():
        raise ValueError("Train prediction rows do not align with training embedding case ids.")

    vectors = merged[embedding_columns].astype(float)
    labels = merged["y_true"].astype(int)
    ai_pred = merged["ai_pred"].astype(int)
    ai_correct = (labels == ai_pred).astype(int)
    return vectors, labels, ai_pred, ai_correct


def _prepare_query_frame(
    query_embeddings: pd.DataFrame,
    query_predictions: pd.DataFrame,
    *,
    split_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    embedding_columns = [column for column in query_embeddings.columns if column.startswith("embedding_")]
    query_frame = query_embeddings.copy()
    query_frame["case_id"] = query_frame["case_id"].astype(str)
    if "split" in query_frame.columns:
        query_frame["split"] = query_frame["split"].astype(str)
        observed_splits = set(query_frame["split"].dropna().unique().tolist())
        if observed_splits != {split_name}:
            raise ValueError(f"Expected only split `{split_name}` in query embeddings, got {sorted(observed_splits)}")

    predictions = query_predictions.copy()
    predictions["case_id"] = predictions["case_id"].astype(str)
    predictions = predictions.drop_duplicates(subset=["case_id"], keep="first")

    merged = query_frame.merge(
        predictions[["case_id", "ai_pred"]],
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    if merged["ai_pred"].isna().any():
        raise ValueError(f"{split_name.title()} prediction rows do not align with query embedding case ids.")

    vectors = merged[embedding_columns].astype(float)
    ai_pred = merged["ai_pred"].astype(int)
    return vectors, ai_pred


def build_local_reliability_for_queries(
    *,
    train_embeddings: pd.DataFrame,
    train_predictions: pd.DataFrame,
    query_embeddings: pd.DataFrame,
    query_predictions: pd.DataFrame,
    split_name: str,
    top_k: int,
) -> pd.DataFrame:
    train_vectors, train_labels, train_ai_pred, train_ai_correct = _prepare_train_reference(
        train_embeddings,
        train_predictions,
    )
    query_vectors, query_ai_pred = _prepare_query_frame(
        query_embeddings,
        query_predictions,
        split_name=split_name,
    )

    neighbor_distances_matrix, neighbor_indices = _nearest_neighbor_query(
        query_vectors.to_numpy(dtype=float),
        train_vectors.to_numpy(dtype=float),
        top_k,
    )

    query_case_ids = query_embeddings["case_id"].astype(str).reset_index(drop=True)
    rows: list[dict[str, float | int | str]] = []
    for query_idx, case_id in enumerate(query_case_ids):
        chosen_indices = neighbor_indices[query_idx]
        neighbor_distances = neighbor_distances_matrix[query_idx]
        predicted_label = int(query_ai_pred.iloc[query_idx])

        neighbor_error_rate = float(np.mean(1 - train_ai_correct.iloc[chosen_indices])) if len(chosen_indices) else 0.0
        neighbor_fraud_rate = float(np.mean(train_labels.iloc[chosen_indices])) if len(chosen_indices) else 0.0
        neighbor_ai_agreement = (
            float(np.mean(train_ai_pred.iloc[chosen_indices] == predicted_label)) if len(chosen_indices) else 0.0
        )
        mean_neighbor_distance = float(np.mean(neighbor_distances)) if len(neighbor_distances) else 0.0

        rows.append(
            {
                "case_id": case_id,
                "split": split_name,
                "neighbor_error_rate": neighbor_error_rate,
                "neighbor_fraud_rate": neighbor_fraud_rate,
                "neighbor_ai_agreement": neighbor_ai_agreement,
                "mean_neighbor_distance": mean_neighbor_distance,
                "knn_k": int(top_k),
                "top_k": int(top_k),
            }
        )

    return pd.DataFrame(rows)


def run_local_reliability(config_path: str | Path) -> LocalReliabilityArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    model_dir, assurance_dir = _resolve_output_dirs(resolved_config_path)

    train_embeddings = _load_embedding_file(assurance_dir, "train")
    val_embeddings = _load_embedding_file(assurance_dir, "val")
    test_embeddings = _load_embedding_file(assurance_dir, "test")
    train_predictions = _load_train_predictions(model_dir)
    val_predictions = _load_prediction_file(model_dir, "val")
    test_predictions = _load_prediction_file(model_dir, "test")
    top_k = _neighbor_top_k(config, len(train_embeddings))

    val_reliability = build_local_reliability_for_queries(
        train_embeddings=train_embeddings,
        train_predictions=train_predictions,
        query_embeddings=val_embeddings,
        query_predictions=val_predictions,
        split_name="val",
        top_k=top_k,
    )
    test_reliability = build_local_reliability_for_queries(
        train_embeddings=train_embeddings,
        train_predictions=train_predictions,
        query_embeddings=test_embeddings,
        query_predictions=test_predictions,
        split_name="test",
        top_k=top_k,
    )
    local_reliability = pd.concat([val_reliability, test_reliability], ignore_index=True)
    local_reliability.to_csv(assurance_dir / "local_reliability.csv", index=False)

    return LocalReliabilityArtifacts(local_reliability=local_reliability, assurance_dir=assurance_dir)
