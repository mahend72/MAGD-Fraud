from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.io import load_yaml


@dataclass
class AssuranceRiskArtifacts:
    assurance_frame: pd.DataFrame
    assurance_dir: Path
    plots_dir: Path


def _resolve_output_dirs(config_path: Path) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    assurance_dir = outputs_root / "assurance"
    plots_dir = outputs_root / "plots"
    if not assurance_dir.exists():
        raise FileNotFoundError(
            f"Missing assurance outputs directory at {assurance_dir}. "
            "Run the upstream assurance scripts first."
        )
    plots_dir.mkdir(parents=True, exist_ok=True)
    return assurance_dir, plots_dir


def _risk_config(config: dict) -> dict:
    risk_cfg = config.get("assurance", {}).get("assurance_risk", {})
    weights = risk_cfg.get("weights", {})
    resolved = {
        "low_threshold": float(risk_cfg.get("low_threshold", 0.33)),
        "high_threshold": float(risk_cfg.get("high_threshold", 0.66)),
        "weights": {
            "calibration_risk": float(weights.get("calibration_risk", 0.2)),
            "distance_uncertainty": float(weights.get("distance_uncertainty", 0.2)),
            "neighbor_error_rate": float(weights.get("neighbor_error_rate", 0.2)),
            "wrong_confident_risk": float(weights.get("wrong_confident_risk", 0.3)),
            "business_risk": float(weights.get("business_risk", 0.1)),
        },
    }
    if not 0.0 <= resolved["low_threshold"] <= 1.0:
        raise ValueError("`assurance.assurance_risk.low_threshold` must be in [0, 1].")
    if not 0.0 <= resolved["high_threshold"] <= 1.0:
        raise ValueError("`assurance.assurance_risk.high_threshold` must be in [0, 1].")
    if resolved["low_threshold"] > resolved["high_threshold"]:
        raise ValueError("`low_threshold` must be less than or equal to `high_threshold`.")
    return resolved


def load_assurance_risk_inputs(config_path: str | Path) -> pd.DataFrame:
    resolved_config_path = Path(config_path).resolve()
    assurance_dir, _ = _resolve_output_dirs(resolved_config_path)

    base = pd.read_csv(assurance_dir / "test_assurance_base.csv")
    distance = pd.read_csv(assurance_dir / "distance_uncertainty.csv")
    neighbors = pd.read_csv(assurance_dir / "neighbor_summary.csv")
    wrong_conf = pd.read_csv(assurance_dir / "wrong_confident_risk.csv")

    merged = base.merge(
        distance[["case_id", "distance_uncertainty"]],
        on="case_id",
        how="inner",
    ).merge(
        neighbors[["case_id", "neighbor_error_rate"]],
        on="case_id",
        how="inner",
    ).merge(
        wrong_conf[["case_id", "wrong_confident_risk"]],
        on="case_id",
        how="inner",
    )
    return merged


def compute_assurance_risk(
    frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    working = frame.copy()
    if "business_risk" not in working.columns:
        working["business_risk"] = 0.0

    required_columns = {
        "case_id",
        "calibration_risk",
        "distance_uncertainty",
        "neighbor_error_rate",
        "wrong_confident_risk",
        "business_risk",
    }
    missing = required_columns - set(working.columns)
    if missing:
        raise ValueError(f"Assurance risk input frame missing required columns: {sorted(missing)}")

    total_weight = sum(max(weight, 0.0) for weight in weights.values())
    if total_weight <= 0.0:
        raise ValueError("Assurance risk weights must sum to a positive value.")

    raw_risk = (
        weights["calibration_risk"] * working["calibration_risk"].astype(float)
        + weights["distance_uncertainty"] * working["distance_uncertainty"].astype(float)
        + weights["neighbor_error_rate"] * working["neighbor_error_rate"].astype(float)
        + weights["wrong_confident_risk"] * working["wrong_confident_risk"].astype(float)
        + weights["business_risk"] * working["business_risk"].astype(float)
    ) / total_weight
    working["assurance_risk"] = raw_risk.clip(0.0, 1.0)
    return working


def map_risk_category(assurance_risk: float, low_threshold: float, high_threshold: float) -> str:
    if assurance_risk < low_threshold:
        return "low"
    if assurance_risk < high_threshold:
        return "medium"
    return "high"


def recommended_action_for_category(category: str) -> str:
    mapping = {
        "low": "AI",
        "medium": "Human Expert",
        "high": "Escalate",
    }
    if category not in mapping:
        raise ValueError(f"Unknown risk category: {category}")
    return mapping[category]


def add_risk_categories_and_actions(
    frame: pd.DataFrame,
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["risk_category"] = [
        map_risk_category(value, low_threshold, high_threshold)
        for value in enriched["assurance_risk"].astype(float)
    ]
    enriched["recommended_action"] = [
        recommended_action_for_category(category)
        for category in enriched["risk_category"]
    ]
    return enriched


def _plot_assurance_risk_distribution(frame: pd.DataFrame, plot_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to generate the assurance risk distribution plot."
        ) from exc

    plt.figure(figsize=(7, 5))
    plt.hist(frame["assurance_risk"].astype(float), bins=30)
    plt.xlabel("Assurance Risk")
    plt.ylabel("Case Count")
    plt.title("Assurance Risk Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def run_assurance_risk(config_path: str | Path) -> AssuranceRiskArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    risk_cfg = _risk_config(config)
    assurance_dir, plots_dir = _resolve_output_dirs(resolved_config_path)

    inputs = load_assurance_risk_inputs(resolved_config_path)
    scored = compute_assurance_risk(inputs, risk_cfg["weights"])
    enriched = add_risk_categories_and_actions(
        scored,
        low_threshold=risk_cfg["low_threshold"],
        high_threshold=risk_cfg["high_threshold"],
    )

    output_columns = [
        "case_id",
        "calibration_risk",
        "distance_uncertainty",
        "neighbor_error_rate",
        "wrong_confident_risk",
        "business_risk",
        "assurance_risk",
        "risk_category",
        "recommended_action",
    ]
    enriched[output_columns].to_csv(assurance_dir / "assurance_risk.csv", index=False)
    _plot_assurance_risk_distribution(enriched, plots_dir / "assurance_risk_distribution.png")

    return AssuranceRiskArtifacts(
        assurance_frame=enriched[output_columns],
        assurance_dir=assurance_dir,
        plots_dir=plots_dir,
    )
