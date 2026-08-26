from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import load_yaml


@dataclass
class PredictionBundle:
    split_name: str
    predictions: pd.DataFrame


def load_processed_split(processed_dir: Path, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = pd.read_csv(processed_dir / f"X_{split_name}.csv")
    y = pd.read_csv(processed_dir / f"y_{split_name}.csv")
    metadata_name = f"{split_name}_metadata.csv" if split_name != "test" else "test_metadata.csv"
    metadata = pd.read_csv(processed_dir / metadata_name)
    return X, y, metadata


def load_model_artifact(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def decision_threshold(config_path: str | Path) -> float:
    config = load_yaml(config_path)
    threshold = float(config.get("model", {}).get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Model threshold must be between 0 and 1, got {threshold}")
    return threshold


def threshold_source() -> str:
    return "config_fixed"


def scores_to_predictions(scores: pd.Series | np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=float) >= threshold).astype(int)


def predict_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise ValueError("Model predict_proba output is malformed.")
        return probabilities[:, 1]
    if hasattr(model, "predict"):
        outputs = np.asarray(model.predict(X), dtype=float)
        if outputs.ndim != 1:
            raise ValueError("Model predict output must be one-dimensional.")
        if np.any((outputs < 0.0) | (outputs > 1.0)):
            return 1.0 / (1.0 + np.exp(-outputs))
        return outputs
    raise TypeError("Model does not expose predict_proba or predict.")


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    if positives == 0 or negatives == 0:
        return math.nan

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)

    sorted_scores = y_score[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = np.mean(np.arange(start + 1, end + 1, dtype=float))
        ranks[order[start:end]] = avg_rank
        start = end

    positive_rank_sum = ranks[y_true == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def pr_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = np.sum(y_true == 1)
    if positives == 0:
        return math.nan

    order = np.argsort(-y_score)
    sorted_true = y_true[order]
    tp = 0
    fp = 0
    recalls = [0.0]
    precisions = [1.0]

    for label in sorted_true:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / positives
        precisions.append(precision)
        recalls.append(recall)

    auc = 0.0
    for idx in range(1, len(recalls)):
        auc += (recalls[idx] - recalls[idx - 1]) * precisions[idx]
    return float(auc)


def compute_binary_metrics(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true).astype(int)
    y_score_array = np.asarray(y_score, dtype=float)
    y_pred_array = scores_to_predictions(y_score_array, threshold)

    tp = int(np.sum((y_true_array == 1) & (y_pred_array == 1)))
    tn = int(np.sum((y_true_array == 0) & (y_pred_array == 0)))
    fp = int(np.sum((y_true_array == 0) & (y_pred_array == 1)))
    fn = int(np.sum((y_true_array == 1) & (y_pred_array == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    cost_sensitive_loss = false_positive_cost * fp + false_negative_cost * fn

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc_score_manual(y_true_array, y_score_array),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "cost_sensitive_loss": float(cost_sensitive_loss),
        "threshold": float(threshold),
    }


def build_prediction_frame(
    *,
    case_ids: pd.Series,
    y_true: pd.Series,
    ai_scores: pd.Series | np.ndarray,
    threshold: float,
    split_name: str,
) -> pd.DataFrame:
    ai_score_series = pd.Series(ai_scores, name="ai_score").astype(float).clip(0.0, 1.0)
    ai_pred_series = pd.Series(scores_to_predictions(ai_score_series.to_numpy(), threshold), name="ai_pred").astype(int)
    y_true_series = pd.Series(y_true, name="y_true").astype(int)
    case_id_series = pd.Series(case_ids, name="case_id").astype(str)
    split_series = pd.Series([split_name] * len(case_id_series), name="split")
    return pd.concat(
        [
            case_id_series.reset_index(drop=True),
            y_true_series.reset_index(drop=True),
            ai_score_series.reset_index(drop=True),
            ai_pred_series.reset_index(drop=True),
            split_series.reset_index(drop=True),
        ],
        axis=1,
    )
