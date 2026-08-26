from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NumericalConfidenceArtifacts:
    frame: pd.DataFrame


@dataclass
class CalibrationArtifacts:
    ece: float
    calibration_table: pd.DataFrame
    assurance_frame: pd.DataFrame


def numerical_confidence_from_score(ai_scores: pd.Series | np.ndarray) -> pd.Series:
    score_series = pd.Series(ai_scores, dtype=float, name="ai_score").clip(0.0, 1.0)
    confidence = np.maximum(score_series.to_numpy(), 1.0 - score_series.to_numpy())
    return pd.Series(confidence, index=score_series.index, name="numerical_confidence").astype(float)


def add_numerical_confidence(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"case_id", "split", "ai_score", "ai_pred"}
    missing = required_columns - set(prediction_frame.columns)
    if missing:
        raise ValueError(f"Prediction frame missing required columns: {sorted(missing)}")

    enriched = prediction_frame.copy()
    enriched["case_id"] = enriched["case_id"].astype(str)
    enriched["split"] = enriched["split"].astype(str)
    enriched["ai_score"] = pd.to_numeric(enriched["ai_score"], errors="raise").clip(0.0, 1.0)
    enriched["ai_pred"] = pd.to_numeric(enriched["ai_pred"], errors="raise").astype(int)
    enriched["numerical_confidence"] = numerical_confidence_from_score(enriched["ai_score"])
    return enriched


def build_calibration_table(
    y_true: pd.Series | np.ndarray,
    ai_scores: pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    working = pd.DataFrame(
        {
            "y_true": pd.Series(y_true, dtype=int),
            "ai_score": pd.Series(ai_scores, dtype=float).clip(0.0, 1.0),
        }
    )
    working["ai_pred"] = (working["ai_score"] >= 0.5).astype(int)
    working["numerical_confidence"] = numerical_confidence_from_score(working["ai_score"])
    working["is_correct"] = (working["ai_pred"] == working["y_true"]).astype(int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    working["calibration_bin"] = np.digitize(
        working["numerical_confidence"].to_numpy(dtype=float),
        bin_edges[1:-1],
        right=False,
    ).astype(int)

    rows: list[dict[str, float | int]] = []
    for bin_idx in range(n_bins):
        mask = working["calibration_bin"] == bin_idx
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "calibration_bin": bin_idx,
                    "bin_lower": float(bin_edges[bin_idx]),
                    "bin_upper": float(bin_edges[bin_idx + 1]),
                    "count": 0,
                    "mean_confidence": np.nan,
                    "empirical_accuracy": np.nan,
                    "absolute_calibration_gap": np.nan,
                    "bin_weight": 0.0,
                }
            )
            continue
        mean_confidence = float(working.loc[mask, "numerical_confidence"].mean())
        empirical_accuracy = float(working.loc[mask, "is_correct"].mean())
        gap = abs(mean_confidence - empirical_accuracy)
        rows.append(
            {
                "calibration_bin": bin_idx,
                "bin_lower": float(bin_edges[bin_idx]),
                "bin_upper": float(bin_edges[bin_idx + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "absolute_calibration_gap": float(gap),
                "bin_weight": float(count / len(working)),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(calibration_table: pd.DataFrame) -> float:
    populated = calibration_table.dropna(subset=["absolute_calibration_gap"]).copy()
    if populated.empty:
        return 0.0
    return float((populated["bin_weight"] * populated["absolute_calibration_gap"]).sum())


def build_assurance_frame(prediction_frame: pd.DataFrame, calibration_table: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"case_id", "y_true", "ai_score", "ai_pred"}
    missing = required_columns - set(prediction_frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing required columns: {sorted(missing)}")

    enriched = prediction_frame.copy()
    if "split" not in enriched.columns:
        enriched["split"] = "unknown"
    enriched = add_numerical_confidence(enriched)
    bin_edges = np.concatenate(
        [
            calibration_table["bin_lower"].to_numpy(dtype=float),
            np.array([float(calibration_table["bin_upper"].iloc[-1])]),
        ]
    )
    enriched["calibration_bin"] = np.digitize(
        enriched["numerical_confidence"].to_numpy(dtype=float),
        bin_edges[1:-1],
        right=False,
    ).astype(int)
    gap_lookup = calibration_table.set_index("calibration_bin")["absolute_calibration_gap"].to_dict()
    enriched["calibration_risk"] = enriched["calibration_bin"].map(
        lambda bin_id: float(gap_lookup.get(int(bin_id), 0.0) or 0.0)
    )
    return enriched


def compute_calibration_artifacts(
    prediction_frame: pd.DataFrame,
    *,
    n_bins: int = 10,
) -> CalibrationArtifacts:
    calibration_table = build_calibration_table(
        prediction_frame["y_true"],
        prediction_frame["ai_score"],
        n_bins=n_bins,
    )
    assurance_frame = build_assurance_frame(prediction_frame, calibration_table)
    return CalibrationArtifacts(
        ece=expected_calibration_error(calibration_table),
        calibration_table=calibration_table,
        assurance_frame=assurance_frame,
    )
