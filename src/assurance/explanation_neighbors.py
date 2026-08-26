from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.embeddings import EmbeddingArtifacts, load_embedding_artifacts
from src.utils.io import load_yaml


@dataclass
class NeighborArtifacts:
    neighbor_evidence_long: pd.DataFrame
    neighbor_summary: pd.DataFrame


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
            "Run `python scripts/train_or_load_model.py --config config.yaml` first."
        )
    assurance_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, assurance_dir


def _neighbor_top_k(config: dict, train_size: int) -> int:
    top_k = int(config.get("assurance", {}).get("neighbor_top_k", 5))
    if top_k <= 0:
        raise ValueError(f"`assurance.neighbor_top_k` must be positive, got {top_k}")
    if train_size <= 0:
        raise ValueError("Training embeddings are empty; cannot compute nearest neighbors.")
    return min(top_k, train_size)


def _load_train_predictions(model_dir: Path) -> pd.DataFrame:
    train_predictions_path = model_dir / "train_predictions.csv"
    if not train_predictions_path.exists():
        raise FileNotFoundError(
            f"Missing train predictions at {train_predictions_path}. "
            "Run `python scripts/train_or_load_model.py --config config.yaml` first."
        )
    frame = pd.read_csv(train_predictions_path)
    required = {"case_id", "y_true", "ai_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Train predictions are missing required columns: {sorted(missing)}")
    return frame


def _nearest_neighbor_query(
    test_vectors: np.ndarray,
    train_vectors: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for nearest-neighbor evidence. Install dependencies from requirements.txt."
        ) from exc

    index = NearestNeighbors(n_neighbors=top_k, metric="euclidean", algorithm="auto")
    index.fit(train_vectors)
    distances, indices = index.kneighbors(test_vectors, return_distance=True)
    return distances, indices


def build_neighbor_evidence(
    embeddings: EmbeddingArtifacts,
    train_predictions: pd.DataFrame,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_case_ids = embeddings.train_metadata["case_id"].astype(str).reset_index(drop=True)
    test_case_ids = embeddings.test_metadata["case_id"].astype(str).reset_index(drop=True)

    train_predictions = train_predictions.copy()
    train_predictions["case_id"] = train_predictions["case_id"].astype(str)
    train_predictions = train_predictions.drop_duplicates(subset=["case_id"], keep="first")
    train_predictions = train_case_ids.to_frame(name="case_id").merge(train_predictions, on="case_id", how="left")
    if train_predictions[["y_true", "ai_pred"]].isna().any().any():
        raise ValueError("Train prediction rows do not align with training case ids.")

    neighbor_correct = (train_predictions["y_true"].astype(int) == train_predictions["ai_pred"].astype(int)).astype(int)

    train_vectors = embeddings.train_embeddings.to_numpy(dtype=float)
    test_vectors = embeddings.test_embeddings.to_numpy(dtype=float)
    neighbor_distances_matrix, neighbor_indices = _nearest_neighbor_query(test_vectors, train_vectors, top_k)

    long_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    train_labels = embeddings.train_labels.astype(int).reset_index(drop=True)
    train_ai_pred = train_predictions["ai_pred"].astype(int).reset_index(drop=True)
    neighbor_correct = neighbor_correct.reset_index(drop=True)

    for test_idx, case_id in enumerate(test_case_ids):
        chosen_indices = neighbor_indices[test_idx]
        neighbor_distances = neighbor_distances_matrix[test_idx]
        wrong_flags: list[int] = []
        fraud_labels: list[int] = []

        for rank, (neighbor_idx, distance) in enumerate(zip(chosen_indices, neighbor_distances, strict=True), start=1):
            is_correct = int(neighbor_correct.iloc[neighbor_idx])
            long_rows.append(
                {
                    "case_id": case_id,
                    "neighbor_rank": rank,
                    "neighbor_case_id": train_case_ids.iloc[neighbor_idx],
                    "neighbor_label": int(train_labels.iloc[neighbor_idx]),
                    "neighbor_ai_pred": int(train_ai_pred.iloc[neighbor_idx]),
                    "neighbor_ai_correct": is_correct,
                    "distance": float(distance),
                }
            )
            wrong_flags.append(1 - is_correct)
            fraud_labels.append(int(train_labels.iloc[neighbor_idx]))

        summary_rows.append(
            {
                "case_id": case_id,
                "neighbor_error_rate": float(np.mean(wrong_flags)) if wrong_flags else 0.0,
                "neighbor_fraud_rate": float(np.mean(fraud_labels)) if fraud_labels else 0.0,
                "mean_neighbor_distance": float(np.mean(neighbor_distances)) if len(neighbor_distances) else 0.0,
                "top_k": int(top_k),
            }
        )

    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def run_neighbor_evidence(config_path: str | Path) -> NeighborArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    model_dir, assurance_dir = _resolve_output_dirs(resolved_config_path)
    embeddings = load_embedding_artifacts(resolved_config_path)
    train_predictions = _load_train_predictions(model_dir)
    top_k = _neighbor_top_k(config, len(embeddings.train_embeddings))

    neighbor_evidence_long, neighbor_summary = build_neighbor_evidence(
        embeddings=embeddings,
        train_predictions=train_predictions,
        top_k=top_k,
    )

    neighbor_evidence_long.to_csv(assurance_dir / "neighbor_evidence_long.csv", index=False)
    neighbor_summary.to_csv(assurance_dir / "neighbor_summary.csv", index=False)

    return NeighborArtifacts(
        neighbor_evidence_long=neighbor_evidence_long,
        neighbor_summary=neighbor_summary,
    )
