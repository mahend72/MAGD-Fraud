from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.assurance.numerical_confidence import add_numerical_confidence
from src.models.predict import compute_binary_metrics
from src.utils.io import load_yaml


@dataclass
class CalibrationAssuranceArtifacts:
    assurance_dir: Path
    paper_tables_dir: Path
    numerical_confidence: pd.DataFrame
    calibration_bins: pd.DataFrame
    calibration_risk: pd.DataFrame
    ai_assurance: pd.DataFrame
    validation_ece: float


def _resolve_output_dirs(config_path: Path) -> tuple[Path, Path, Path]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    model_dir = outputs_root / "model"
    assurance_dir = outputs_root / "assurance"
    paper_tables_dir = outputs_root / "paper_tables"
    assurance_dir.mkdir(parents=True, exist_ok=True)
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, assurance_dir, paper_tables_dir


def _n_bins(config: dict) -> int:
    return int(config.get("assurance", {}).get("n_bins_calibration", config.get("assurance", {}).get("calibration_bins", 10)))


def _load_prediction_file(model_dir: Path, split_name: str) -> pd.DataFrame:
    path = model_dir / f"{split_name}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing model prediction file: {path}")
    frame = pd.read_csv(path)
    required_columns = {"case_id", "y_true", "ai_score", "ai_pred", "split"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file `{path}` is missing required columns: {sorted(missing)}")
    return frame


def build_calibration_bins(validation_predictions: pd.DataFrame, *, n_bins: int) -> pd.DataFrame:
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    validation = add_numerical_confidence(validation_predictions)
    validation["y_true"] = pd.to_numeric(validation["y_true"], errors="raise").astype(int)
    validation["is_correct"] = (validation["ai_pred"].astype(int) == validation["y_true"]).astype(int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(validation["numerical_confidence"].to_numpy(dtype=float), bin_edges[1:-1], right=False)
    validation["calibration_bin"] = bin_ids.astype(int)

    rows: list[dict[str, float | int | str]] = []
    for bin_idx in range(n_bins):
        mask = validation["calibration_bin"] == bin_idx
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "split_used_to_learn_bins": "validation",
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

        mean_conf = float(validation.loc[mask, "numerical_confidence"].mean())
        empirical_accuracy = float(validation.loc[mask, "is_correct"].mean())
        gap = abs(mean_conf - empirical_accuracy)
        rows.append(
            {
                "split_used_to_learn_bins": "validation",
                "calibration_bin": bin_idx,
                "bin_lower": float(bin_edges[bin_idx]),
                "bin_upper": float(bin_edges[bin_idx + 1]),
                "count": count,
                "mean_confidence": mean_conf,
                "empirical_accuracy": empirical_accuracy,
                "absolute_calibration_gap": float(gap),
                "bin_weight": float(count / len(validation)),
            }
        )

    return pd.DataFrame(rows)


def expected_calibration_error(calibration_bins: pd.DataFrame) -> float:
    populated = calibration_bins.dropna(subset=["absolute_calibration_gap"]).copy()
    if populated.empty:
        return 0.0
    return float((populated["bin_weight"] * populated["absolute_calibration_gap"]).sum())


def _safe_gap(gap_lookup: dict, bin_id: object) -> float:
    """Look up a bin's calibration gap, treating a missing or empty (NaN) bin as zero risk.

    `NaN or 0.0` evaluates to `NaN` in Python (NaN is truthy), so a naive `gap_lookup.get(...) or
    0.0` silently leaves NaN in place for empty bins - which then trips the "no NaN allowed"
    guard in `compute_magd_risk`. This makes the empty-bin fallback explicit and NaN-safe.
    """
    gap = gap_lookup.get(int(bin_id), 0.0)
    if gap is None or (isinstance(gap, float) and np.isnan(gap)):
        return 0.0
    return float(gap)


def assign_calibration_risk(predictions: pd.DataFrame, calibration_bins: pd.DataFrame) -> pd.DataFrame:
    enriched = add_numerical_confidence(predictions)
    bin_edges = np.concatenate(
        [
            calibration_bins["bin_lower"].to_numpy(dtype=float),
            np.array([float(calibration_bins["bin_upper"].iloc[-1])]),
        ]
    )
    bin_ids = np.digitize(enriched["numerical_confidence"].to_numpy(dtype=float), bin_edges[1:-1], right=False)
    gap_lookup = calibration_bins.set_index("calibration_bin")["absolute_calibration_gap"].to_dict()
    enriched["calibration_bin"] = pd.Series(bin_ids, index=enriched.index).astype(int)
    enriched["calibration_risk"] = enriched["calibration_bin"].map(lambda bin_id: _safe_gap(gap_lookup, bin_id))
    return enriched[
        [
            "case_id",
            "split",
            "y_true",
            "ai_score",
            "ai_pred",
            "numerical_confidence",
            "calibration_bin",
            "calibration_risk",
        ]
    ].copy()


def split_ece(scored_predictions: pd.DataFrame, calibration_bins: pd.DataFrame) -> float:
    gap_lookup = calibration_bins.set_index("calibration_bin")["absolute_calibration_gap"].to_dict()
    weight_rows = (
        scored_predictions.groupby("calibration_bin", dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    total = int(weight_rows["count"].sum())
    if total == 0:
        return 0.0
    weight_rows["weight"] = weight_rows["count"] / total
    weight_rows["gap"] = weight_rows["calibration_bin"].map(lambda value: _safe_gap(gap_lookup, value))
    return float((weight_rows["weight"] * weight_rows["gap"]).sum())


def build_ai_assurance_table(
    scored_predictions: pd.DataFrame,
    calibration_bins: pd.DataFrame,
    *,
    config_path: Path,
) -> pd.DataFrame:
    config = load_yaml(config_path)
    costs_cfg = config.get("costs", {})
    fp_cost = float(costs_cfg.get("false_positive", 1.0))
    fn_cost = float(costs_cfg.get("false_negative", 5.0))
    threshold = float(config.get("model", {}).get("threshold", 0.5))

    rows: list[dict[str, float | str]] = []
    for split_name, frame in scored_predictions.groupby("split", dropna=False):
        metrics = compute_binary_metrics(
            y_true=frame["y_true"],
            y_score=frame["ai_score"],
            threshold=threshold,
            false_positive_cost=fp_cost,
            false_negative_cost=fn_cost,
        )
        rows.append(
            {
                "split": str(split_name),
                "pr_auc": float(metrics["pr_auc"]),
                "f1": float(metrics["f1"]),
                "expected_calibration_error": float(split_ece(frame, calibration_bins)),
                "mean_numerical_confidence": float(pd.to_numeric(frame["numerical_confidence"], errors="coerce").mean()),
                "mean_calibration_risk": float(pd.to_numeric(frame["calibration_risk"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def run_calibration_assurance(config_path: str | Path) -> CalibrationAssuranceArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    model_dir, assurance_dir, paper_tables_dir = _resolve_output_dirs(resolved_config_path)
    n_bins = _n_bins(config)

    train_predictions = _load_prediction_file(model_dir, "train")
    val_predictions = _load_prediction_file(model_dir, "val")
    test_predictions = _load_prediction_file(model_dir, "test")

    calibration_bins = build_calibration_bins(val_predictions, n_bins=n_bins)
    validation_ece = expected_calibration_error(calibration_bins)
    calibration_bins = calibration_bins.copy()
    calibration_bins["expected_calibration_error"] = validation_ece

    train_scored = assign_calibration_risk(train_predictions, calibration_bins)
    val_scored = assign_calibration_risk(val_predictions, calibration_bins)
    test_scored = assign_calibration_risk(test_predictions, calibration_bins)
    all_scored = pd.concat([train_scored, val_scored, test_scored], ignore_index=True)

    numerical_confidence = all_scored[
        ["case_id", "split", "ai_score", "ai_pred", "numerical_confidence"]
    ].copy()
    calibration_risk = all_scored[
        ["case_id", "split", "ai_score", "ai_pred", "numerical_confidence", "calibration_bin", "calibration_risk"]
    ].copy()
    ai_assurance = build_ai_assurance_table(all_scored, calibration_bins, config_path=resolved_config_path)

    numerical_confidence.to_csv(assurance_dir / "numerical_confidence.csv", index=False)
    calibration_bins.to_csv(assurance_dir / "calibration_bins.csv", index=False)
    calibration_risk.to_csv(assurance_dir / "calibration_risk.csv", index=False)
    ai_assurance.to_csv(paper_tables_dir / "ai_assurance.csv", index=False)

    # Backward-compatible artifacts used by downstream pipeline stages.
    calibration_report = calibration_bins.rename(
        columns={
            "mean_confidence": "average_confidence",
            "empirical_accuracy": "observed_accuracy",
            "absolute_calibration_gap": "absolute_gap",
            "bin_weight": "weight",
        }
    )
    calibration_report["ece"] = validation_ece
    calibration_report.to_csv(assurance_dir / "calibration_report.csv", index=False)
    test_assurance_base = test_scored[
        ["case_id", "y_true", "ai_score", "ai_pred", "numerical_confidence", "calibration_risk"]
    ].copy()
    test_assurance_base.to_csv(assurance_dir / "test_assurance_base.csv", index=False)

    return CalibrationAssuranceArtifacts(
        assurance_dir=assurance_dir,
        paper_tables_dir=paper_tables_dir,
        numerical_confidence=numerical_confidence,
        calibration_bins=calibration_bins,
        calibration_risk=calibration_risk,
        ai_assurance=ai_assurance,
        validation_ece=validation_ece,
    )
