from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_fifar import load_fifar_data
from src.models.embeddings import EmbeddingArtifacts, load_embedding_artifacts, write_embedding_outputs
from src.utils.io import load_yaml


@dataclass
class DistanceArtifacts:
    distance_frame: pd.DataFrame
    threshold_metrics: pd.DataFrame


def _resolve_output_dirs(config_path: Path) -> tuple[Path, Path, Path]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    model_dir = outputs_root / "model"
    assurance_dir = outputs_root / "assurance"
    plots_dir = outputs_root / "plots"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Missing model outputs directory at {model_dir}. "
            "Run `python scripts/run_model_predictions.py --config config.yaml` first."
        )
    assurance_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, assurance_dir, plots_dir


def _distance_thresholds(config: dict) -> list[float]:
    thresholds = [float(value) for value in config.get("assurance", {}).get("distance_thresholds", [0.5, 0.7, 0.9])]
    if not thresholds:
        raise ValueError("`assurance.distance_thresholds` must contain at least one threshold.")
    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Distance threshold must be in [0, 1], got {threshold}")
    return thresholds


def _costs(config: dict) -> tuple[float, float]:
    model_cfg = config.get("model", {})
    return float(model_cfg.get("false_positive_cost", 1.0)), float(model_cfg.get("false_negative_cost", 5.0))


def _load_prediction_file(model_dir: Path, split_name: str) -> pd.DataFrame:
    prediction_path = model_dir / f"{split_name}_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Missing {split_name} predictions at {prediction_path}. "
            "Run `python scripts/run_model_predictions.py --config config.yaml` first."
        )
    frame = pd.read_csv(prediction_path)
    required = {"case_id", "y_true", "ai_score", "ai_pred", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{split_name.title()} predictions are missing required columns: {sorted(missing)}")
    return frame


def _class_centroids(embeddings: pd.DataFrame, labels: pd.Series) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for label in [0, 1]:
        class_rows = embeddings.loc[labels.astype(int) == label]
        if class_rows.empty:
            raise ValueError(f"Cannot compute centroid for class {label}: no training examples available.")
        centroids[label] = class_rows.mean(axis=0).to_numpy(dtype=float)
    return centroids


def _euclidean_distance(point: np.ndarray, centroid: np.ndarray) -> float:
    return float(np.linalg.norm(point - centroid))


def _distance_frame_for_split(
    *,
    embeddings: pd.DataFrame,
    metadata: pd.DataFrame,
    predictions: pd.DataFrame,
    split_name: str,
    centroids: dict[int, np.ndarray],
) -> pd.DataFrame:
    if "case_id" not in metadata.columns:
        raise ValueError(f"Metadata for split `{split_name}` is missing required `case_id` column.")

    metadata_case_ids = metadata["case_id"].astype(str).reset_index(drop=True)
    prediction_case_ids = predictions["case_id"].astype(str).reset_index(drop=True)
    if not metadata_case_ids.equals(prediction_case_ids):
        raise ValueError(f"Case IDs for split `{split_name}` do not align between embeddings and predictions.")

    vectors = embeddings.to_numpy(dtype=float)
    predicted_labels = predictions["ai_pred"].astype(int).to_numpy()
    distances_to_0 = np.array([_euclidean_distance(vector, centroids[0]) for vector in vectors], dtype=float)
    distances_to_1 = np.array([_euclidean_distance(vector, centroids[1]) for vector in vectors], dtype=float)
    predicted_centroid_distances = np.where(predicted_labels == 1, distances_to_1, distances_to_0)

    return pd.DataFrame(
        {
            "case_id": prediction_case_ids,
            "split": pd.Series([split_name] * len(predictions), dtype="object"),
            "y_true": predictions["y_true"].astype(int).reset_index(drop=True),
            "ai_score": predictions["ai_score"].astype(float).reset_index(drop=True),
            "ai_pred": predictions["ai_pred"].astype(int).reset_index(drop=True),
            "distance_to_centroid_0": distances_to_0,
            "distance_to_centroid_1": distances_to_1,
            "distance_to_predicted_centroid": predicted_centroid_distances,
        }
    )


def _normalize_distances(
    *,
    reference_distances: pd.Series | np.ndarray,
    target_distances: pd.Series | np.ndarray,
) -> np.ndarray:
    reference = np.asarray(reference_distances, dtype=float)
    target = np.asarray(target_distances, dtype=float)
    if reference.size == 0:
        raise ValueError("Distance normalization requires at least one train/validation reference distance.")

    min_distance = float(np.nanmin(reference))
    max_distance = float(np.nanmax(reference))
    if not np.isfinite(min_distance) or not np.isfinite(max_distance):
        raise ValueError("Distance normalization reference contains non-finite values.")
    if max_distance <= min_distance:
        return np.ones_like(target, dtype=float)

    scaled = (target - min_distance) / (max_distance - min_distance)
    confidence = 1.0 - scaled
    return np.clip(confidence, 0.0, 1.0)


def compute_distance_confidence_frame(
    embeddings: EmbeddingArtifacts,
    prediction_frames: dict[str, pd.DataFrame] | pd.DataFrame,
) -> pd.DataFrame:
    centroids = _class_centroids(embeddings.train_embeddings, embeddings.train_labels)

    if isinstance(prediction_frames, pd.DataFrame):
        split_frames = {
            "test": _distance_frame_for_split(
                embeddings=embeddings.test_embeddings,
                metadata=embeddings.test_metadata,
                predictions=prediction_frames,
                split_name="test",
                centroids=centroids,
            )
        }
    else:
        split_frames = {
            "train": _distance_frame_for_split(
                embeddings=embeddings.train_embeddings,
                metadata=embeddings.train_metadata,
                predictions=prediction_frames["train"],
                split_name="train",
                centroids=centroids,
            ),
            "val": _distance_frame_for_split(
                embeddings=embeddings.val_embeddings,
                metadata=embeddings.val_metadata,
                predictions=prediction_frames["val"],
                split_name="val",
                centroids=centroids,
            ),
            "test": _distance_frame_for_split(
                embeddings=embeddings.test_embeddings,
                metadata=embeddings.test_metadata,
                predictions=prediction_frames["test"],
                split_name="test",
                centroids=centroids,
            ),
        }

    reference_distances = pd.concat(
        [
            split_frames[split_name]["distance_to_predicted_centroid"]
            for split_name in ["train", "val"]
            if split_name in split_frames
        ],
        ignore_index=True,
    )
    if reference_distances.empty:
        reference_distances = split_frames["test"]["distance_to_predicted_centroid"]

    ordered_splits = [name for name in ["train", "val", "test"] if name in split_frames]
    enriched_frames: list[pd.DataFrame] = []
    for split_name in ordered_splits:
        frame = split_frames[split_name].copy()
        confidence = _normalize_distances(
            reference_distances=reference_distances,
            target_distances=frame["distance_to_predicted_centroid"],
        )
        frame["distance_confidence"] = confidence
        frame["distance_uncertainty"] = 1.0 - confidence
        enriched_frames.append(frame)

    return pd.concat(enriched_frames, ignore_index=True)


def _resolve_expert_predictions(config_path: Path, case_ids: pd.Series) -> pd.Series | None:
    loaded = load_fifar_data(config_path)
    if loaded.expert_df is None or loaded.expert_prediction_column is None or loaded.expert_case_id_column is None:
        return None

    expert_frame = loaded.expert_df[[loaded.expert_case_id_column, loaded.expert_prediction_column]].copy()
    expert_frame[loaded.expert_case_id_column] = expert_frame[loaded.expert_case_id_column].astype(str)
    expert_frame = expert_frame.drop_duplicates(subset=[loaded.expert_case_id_column], keep="first")
    lookup = expert_frame.set_index(loaded.expert_case_id_column)[loaded.expert_prediction_column]
    resolved = case_ids.astype(str).map(lookup)
    if resolved.notna().all():
        return resolved.astype(int)
    return None


def _hard_prediction_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, float]:
    y_true_array = y_true.astype(int).to_numpy()
    y_pred_array = y_pred.astype(int).to_numpy()
    tp = int(np.sum((y_true_array == 1) & (y_pred_array == 1)))
    tn = int(np.sum((y_true_array == 0) & (y_pred_array == 0)))
    fp = int(np.sum((y_true_array == 0) & (y_pred_array == 1)))
    fn = int(np.sum((y_true_array == 1) & (y_pred_array == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    loss = false_positive_cost * fp + false_negative_cost * fn

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cost_sensitive_loss": float(loss),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def explore_distance_thresholds(
    distance_frame: pd.DataFrame,
    thresholds: list[float],
    false_positive_cost: float,
    false_negative_cost: float,
    expert_predictions: pd.Series | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    total_cases = len(distance_frame)

    for threshold in thresholds:
        use_ai_mask = distance_frame["distance_confidence"] >= threshold
        final_pred = distance_frame["ai_pred"].copy()
        expert_used_mask = pd.Series(False, index=distance_frame.index)

        if expert_predictions is not None:
            defer_mask = ~use_ai_mask
            final_pred.loc[defer_mask] = expert_predictions.loc[defer_mask].astype(int)
            expert_used_mask = defer_mask

        metrics = _hard_prediction_metrics(
            y_true=distance_frame["y_true"],
            y_pred=final_pred,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        ai_coverage = float(use_ai_mask.mean()) if total_cases else 0.0
        deferral_rate = float(expert_used_mask.mean()) if total_cases else 0.0
        rows.append(
            {
                "threshold": threshold,
                "ai_coverage": ai_coverage,
                "deferral_rate": deferral_rate,
                "expert_available": float(expert_predictions is not None),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def _plot_threshold_curve(frame: pd.DataFrame, x_column: str, y_column: str, title: str, plot_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to generate distance uncertainty plots. "
            "Install dependencies from requirements.txt."
        ) from exc

    plt.figure(figsize=(7, 5))
    plt.plot(frame[x_column], frame[y_column], marker="o", linewidth=2)
    plt.xlabel("Distance confidence threshold")
    plt.ylabel(y_column.replace("_", " ").title())
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def run_distance_uncertainty(config_path: str | Path) -> DistanceArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    model_dir, assurance_dir, plots_dir = _resolve_output_dirs(resolved_config_path)
    embeddings = load_embedding_artifacts(resolved_config_path)
    write_embedding_outputs(resolved_config_path, embeddings)

    prediction_frames = {
        "train": _load_prediction_file(model_dir, "train"),
        "val": _load_prediction_file(model_dir, "val"),
        "test": _load_prediction_file(model_dir, "test"),
    }

    distance_frame = compute_distance_confidence_frame(embeddings, prediction_frames)
    test_distance_frame = distance_frame.loc[distance_frame["split"] == "test"].reset_index(drop=True)
    expert_predictions = _resolve_expert_predictions(resolved_config_path, test_distance_frame["case_id"])
    fp_cost, fn_cost = _costs(config)
    thresholds = _distance_thresholds(config)
    threshold_metrics = explore_distance_thresholds(
        distance_frame=test_distance_frame,
        thresholds=thresholds,
        false_positive_cost=fp_cost,
        false_negative_cost=fn_cost,
        expert_predictions=expert_predictions,
    )

    distance_frame.to_csv(assurance_dir / "distance_uncertainty.csv", index=False)
    threshold_metrics.to_csv(assurance_dir / "threshold_exploration_distance.csv", index=False)
    _plot_threshold_curve(
        threshold_metrics,
        x_column="threshold",
        y_column="cost_sensitive_loss",
        title="Distance Threshold vs Cost-Sensitive Loss",
        plot_path=plots_dir / "threshold_vs_loss_distance.png",
    )
    _plot_threshold_curve(
        threshold_metrics,
        x_column="threshold",
        y_column="f1",
        title="Distance Threshold vs F1",
        plot_path=plots_dir / "threshold_vs_f1_distance.png",
    )

    return DistanceArtifacts(distance_frame=distance_frame, threshold_metrics=threshold_metrics)
