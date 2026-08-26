from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import load_yaml


@dataclass
class WrongConfidentArtifacts:
    risk_frame: pd.DataFrame
    metrics: dict[str, float]


def _resolve_output_dirs(config_path: Path) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    assurance_dir = outputs_root / "assurance"
    paper_tables_dir = outputs_root / "paper_tables"
    if not assurance_dir.exists():
        raise FileNotFoundError(
            f"Missing assurance outputs directory at {assurance_dir}. "
            "Run calibration, distance uncertainty, and local reliability first."
        )
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    return assurance_dir, paper_tables_dir


def _wrong_confident_config(config: dict) -> dict:
    wc_cfg = config.get("assurance", {}).get("wrong_confident", {})
    weights = wc_cfg.get("weights", {})
    resolved = {
        "high_conf_threshold": float(
            wc_cfg.get(
                "high_conf_threshold",
                config.get("assurance", {}).get("high_confidence_threshold", 0.80),
            )
        ),
        "risk_alert_threshold": float(
            wc_cfg.get(
                "risk_alert_threshold",
                config.get("magd", {}).get("thresholds", {}).get("wrong_confident", 0.70),
            )
        ),
        "top_k_fraction": float(wc_cfg.get("top_k_fraction", 0.10)),
        "weights": {
            "numerical_confidence": float(weights.get("numerical_confidence", 0.25)),
            "distance_uncertainty": float(weights.get("distance_uncertainty", 0.20)),
            "calibration_risk": float(weights.get("calibration_risk", 0.20)),
            "neighbor_error_rate": float(weights.get("neighbor_error_rate", 0.20)),
            "confidence_disagreement": float(weights.get("confidence_disagreement", 0.15)),
        },
    }
    return resolved


def _load_required_csv(path: Path, *, required_columns: set[str], friendly_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {friendly_name} at {path}.")
    frame = pd.read_csv(path)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{friendly_name} is missing required columns: {sorted(missing)}")
    return frame


def load_assurance_inputs(config_path: str | Path) -> pd.DataFrame:
    resolved_config_path = Path(config_path).resolve()
    assurance_dir, _ = _resolve_output_dirs(resolved_config_path)

    numerical = _load_required_csv(
        assurance_dir / "numerical_confidence.csv",
        required_columns={"case_id", "split", "ai_score", "ai_pred", "numerical_confidence"},
        friendly_name="numerical confidence outputs",
    )
    calibration = _load_required_csv(
        assurance_dir / "calibration_risk.csv",
        required_columns={"case_id", "split", "calibration_risk"},
        friendly_name="calibration risk outputs",
    )
    distance = _load_required_csv(
        assurance_dir / "distance_uncertainty.csv",
        required_columns={"case_id", "split", "distance_confidence", "distance_uncertainty"},
        friendly_name="distance uncertainty outputs",
    )

    local_path = assurance_dir / "local_reliability.csv"
    local = _load_required_csv(
        local_path,
        required_columns={"case_id", "neighbor_error_rate"},
        friendly_name="local reliability outputs",
    )

    if "split" not in local.columns:
        local = local.copy()
        local["split"] = pd.NA

    y_true_sources: list[pd.DataFrame] = []
    for candidate_name in ["calibration_risk.csv", "distance_uncertainty.csv"]:
        candidate = pd.read_csv(assurance_dir / candidate_name)
        if {"case_id", "split", "y_true"}.issubset(candidate.columns):
            y_true_sources.append(candidate[["case_id", "split", "y_true"]])
    if not y_true_sources:
        raise ValueError("Missing `y_true` for offline wrong-confident evaluation in assurance outputs.")
    y_true = pd.concat(y_true_sources, ignore_index=True).drop_duplicates(subset=["case_id", "split"], keep="first")

    merged = (
        numerical.merge(
            calibration[["case_id", "split", "calibration_risk"]],
            on=["case_id", "split"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            distance[["case_id", "split", "distance_confidence", "distance_uncertainty"]],
            on=["case_id", "split"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            local[
                [
                    "case_id",
                    "split",
                    "neighbor_error_rate",
                    "neighbor_fraud_rate",
                    "neighbor_ai_agreement",
                    "mean_neighbor_distance",
                    "knn_k",
                ]
                if {"split", "neighbor_fraud_rate", "neighbor_ai_agreement", "mean_neighbor_distance", "knn_k"}.issubset(local.columns)
                else [column for column in local.columns if column in {
                    "case_id",
                    "split",
                    "neighbor_error_rate",
                    "neighbor_fraud_rate",
                    "neighbor_ai_agreement",
                    "mean_neighbor_distance",
                    "knn_k",
                    "top_k",
                }]
            ],
            on=["case_id", "split"] if local["split"].notna().any() else ["case_id"],
            how="inner",
        )
        .merge(
            y_true,
            on=["case_id", "split"],
            how="left",
            validate="one_to_one",
        )
    )

    if "knn_k" not in merged.columns and "top_k" in merged.columns:
        merged["knn_k"] = merged["top_k"]
    required = {
        "case_id",
        "split",
        "ai_score",
        "ai_pred",
        "numerical_confidence",
        "distance_confidence",
        "distance_uncertainty",
        "calibration_risk",
        "neighbor_error_rate",
        "y_true",
    }
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"Wrong-confident inputs are missing required columns after merge: {sorted(missing)}")
    return merged


def compute_deployable_wrong_confident_risk(
    signal_frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    required_columns = {
        "case_id",
        "split",
        "ai_pred",
        "ai_score",
        "numerical_confidence",
        "distance_confidence",
        "distance_uncertainty",
        "calibration_risk",
        "neighbor_error_rate",
    }
    missing = required_columns - set(signal_frame.columns)
    if missing:
        raise ValueError(f"Signal frame missing required columns: {sorted(missing)}")

    frame = signal_frame.copy()
    frame["confidence_disagreement"] = (
        frame["numerical_confidence"].astype(float) - frame["distance_confidence"].astype(float)
    ).abs()

    component_columns = [
        "numerical_confidence",
        "distance_uncertainty",
        "calibration_risk",
        "neighbor_error_rate",
        "confidence_disagreement",
    ]
    for column in component_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"Signal column `{column}` contains non-numeric values.")
        frame[column] = frame[column].clip(0.0, 1.0)

    total_weight = sum(max(float(weight), 0.0) for weight in weights.values())
    if total_weight <= 0.0:
        raise ValueError("Wrong-confident weights must sum to a positive value.")

    risk = (
        float(weights["numerical_confidence"]) * frame["numerical_confidence"]
        + float(weights["distance_uncertainty"]) * frame["distance_uncertainty"]
        + float(weights["calibration_risk"]) * frame["calibration_risk"]
        + float(weights["neighbor_error_rate"]) * frame["neighbor_error_rate"]
        + float(weights["confidence_disagreement"]) * frame["confidence_disagreement"]
    ) / total_weight
    frame["wrong_confident_risk"] = risk.clip(0.0, 1.0)
    return frame


def add_offline_wrong_confident_labels(
    frame: pd.DataFrame,
    high_conf_threshold: float,
) -> pd.DataFrame:
    required_columns = {"y_true", "ai_pred", "numerical_confidence"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Offline labeling requires columns: {sorted(missing)}")

    labeled = frame.copy()
    labeled["wrong_confident_label_offline"] = (
        (labeled["ai_pred"].astype(int) != labeled["y_true"].astype(int))
        & (labeled["numerical_confidence"].astype(float) >= high_conf_threshold)
    ).astype(int)
    return labeled


def evaluate_wrong_confident_detection(
    frame: pd.DataFrame,
    risk_alert_threshold: float,
    top_k_fraction: float,
) -> dict[str, float]:
    if not 0.0 <= risk_alert_threshold <= 1.0:
        raise ValueError(f"risk_alert_threshold must be in [0, 1], got {risk_alert_threshold}")
    if not 0.0 < top_k_fraction <= 1.0:
        raise ValueError(f"top_k_fraction must be in (0, 1], got {top_k_fraction}")

    required_columns = {"wrong_confident_label_offline", "wrong_confident_risk"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Evaluation requires columns: {sorted(missing)}")

    eval_frame = frame.copy()
    predicted_positive = (eval_frame["wrong_confident_risk"].astype(float) >= risk_alert_threshold).astype(int)
    actual_positive = eval_frame["wrong_confident_label_offline"].astype(int)

    tp = int(((predicted_positive == 1) & (actual_positive == 1)).sum())
    fp = int(((predicted_positive == 1) & (actual_positive == 0)).sum())
    fn = int(((predicted_positive == 0) & (actual_positive == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    total_cases = len(eval_frame)
    top_k = max(1, int(np.ceil(total_cases * top_k_fraction)))
    ranked = eval_frame.sort_values("wrong_confident_risk", ascending=False)
    top_k_capture = float(ranked.head(top_k)["wrong_confident_label_offline"].sum())
    total_wrong_confident = int(actual_positive.sum())
    top_k_capture_rate = top_k_capture / total_wrong_confident if total_wrong_confident else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "top_k_capture_rate": top_k_capture_rate,
        "top_k_count": float(top_k),
        "offline_wrong_confident_count": float(total_wrong_confident),
        "risk_alert_threshold": float(risk_alert_threshold),
        "high_conf_threshold": float(frame.attrs.get("high_conf_threshold", np.nan)),
        "wrong_confident_detection_precision": precision,
        "wrong_confident_detection_recall": recall,
        "wrong_confident_detection_f1": f1,
        "top_k_wrong_confident_capture_rate": top_k_capture_rate,
    }


def run_wrong_confident_detection(config_path: str | Path) -> WrongConfidentArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    wc_cfg = _wrong_confident_config(config)
    assurance_dir, paper_tables_dir = _resolve_output_dirs(resolved_config_path)

    signal_frame = load_assurance_inputs(resolved_config_path)
    deployable = compute_deployable_wrong_confident_risk(signal_frame, wc_cfg["weights"])
    labeled = add_offline_wrong_confident_labels(deployable, wc_cfg["high_conf_threshold"])
    labeled.attrs["high_conf_threshold"] = wc_cfg["high_conf_threshold"]
    metrics = evaluate_wrong_confident_detection(
        labeled,
        risk_alert_threshold=wc_cfg["risk_alert_threshold"],
        top_k_fraction=wc_cfg["top_k_fraction"],
    )

    output_columns = [
        "case_id",
        "split",
        "numerical_confidence",
        "distance_confidence",
        "distance_uncertainty",
        "calibration_risk",
        "neighbor_error_rate",
        "confidence_disagreement",
        "wrong_confident_risk",
        "wrong_confident_label_offline",
    ]
    missing = set(output_columns) - set(labeled.columns)
    if missing:
        raise ValueError(f"Wrong-confident output is missing required columns: {sorted(missing)}")
    labeled[output_columns].to_csv(assurance_dir / "wrong_confident_risk.csv", index=False)
    pd.DataFrame([metrics]).to_csv(paper_tables_dir / "wrong_confident_detection.csv", index=False)
    return WrongConfidentArtifacts(risk_frame=labeled[output_columns], metrics=metrics)
