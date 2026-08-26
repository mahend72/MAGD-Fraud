from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.io import load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class AdaptiveThresholdArtifacts:
    thresholds: pd.DataFrame
    assurance_dir: Path


def _resolve_assurance_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    assurance_dir = outputs_root / "assurance"
    if not assurance_dir.exists():
        raise FileNotFoundError(
            f"Missing assurance outputs directory at {assurance_dir}. "
            "Run the upstream MAGD assurance scripts first."
        )
    return assurance_dir


def _adaptive_threshold_config(config: dict) -> dict:
    adaptive_cfg = config.get("magd", {}).get("adaptive_threshold", {})
    resolved = {
        "enabled": bool(adaptive_cfg.get("enabled", True)),
        "base_threshold": float(adaptive_cfg.get("base_threshold", 0.50)),
        "value_weight": float(adaptive_cfg.get("value_weight", 0.10)),
        "fairness_weight": float(adaptive_cfg.get("fairness_weight", 0.10)),
        "capacity_weight": float(adaptive_cfg.get("capacity_weight", 0.10)),
    }
    for key in ["base_threshold", "value_weight", "fairness_weight", "capacity_weight"]:
        value = float(resolved[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"`magd.adaptive_threshold.{key}` must be in [0, 1], got {value}.")
    return resolved


def load_adaptive_threshold_inputs(config_path: str | Path) -> pd.DataFrame:
    resolved_config_path = Path(config_path).resolve()
    assurance_dir = _resolve_assurance_dir(resolved_config_path)
    path = assurance_dir / "magd_risk.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing MAGD risk artifact at {path}. "
            "Run `python scripts/run_magd_risk.py --config config.yaml` first."
        )
    frame = pd.read_csv(path)
    if "case_id" in frame.columns:
        frame["case_id"] = frame["case_id"].astype(str)
    return frame


def _resolve_optional_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in frame.columns:
        LOGGER.warning("Adaptive threshold input `%s` is unavailable; defaulting to 0.0.", column_name)
        return pd.Series([0.0] * len(frame), index=frame.index, dtype=float)
    series = pd.to_numeric(frame[column_name], errors="coerce")
    if series.isna().all():
        LOGGER.warning("Adaptive threshold input `%s` is all missing; defaulting to 0.0.", column_name)
        return pd.Series([0.0] * len(frame), index=frame.index, dtype=float)
    return series.fillna(0.0).astype(float)


def compute_adaptive_thresholds(
    frame: pd.DataFrame,
    *,
    base_threshold: float,
    value_weight: float,
    fairness_weight: float,
    capacity_weight: float,
) -> pd.DataFrame:
    required_columns = {"case_id", "magd_assurance_risk"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Adaptive threshold input frame missing required columns: {sorted(missing)}")

    working = frame.copy()
    working["magd_assurance_risk"] = pd.to_numeric(working["magd_assurance_risk"], errors="raise").astype(float)
    working["business_risk"] = _resolve_optional_series(working, "business_risk")
    working["fairness_risk"] = _resolve_optional_series(working, "fairness_risk")
    working["capacity_pressure"] = _resolve_optional_series(working, "capacity_pressure")
    working["base_threshold"] = float(base_threshold)

    # `adaptive_threshold` is the maximum MAGD risk still eligible for AI retention
    # (`passes_threshold = magd_assurance_risk < adaptive_threshold`, i.e. it is a risk
    # ceiling, not a risk floor). Higher business/fairness/capacity risk must therefore
    # LOWER this ceiling so that AI retention becomes strictly harder, never easier, as
    # contextual risk increases - the opposite of adding the risk terms to the base
    # threshold, which would raise the ceiling and make AI retention easier instead.
    adaptive = (
        float(base_threshold)
        - float(value_weight) * working["business_risk"]
        - float(fairness_weight) * working["fairness_risk"]
        - float(capacity_weight) * working["capacity_pressure"]
    ).clip(0.0, 1.0)
    working["adaptive_threshold"] = adaptive
    working["passes_threshold"] = (working["magd_assurance_risk"] < working["adaptive_threshold"]).astype(bool)

    return working[
        [
            "case_id",
            "base_threshold",
            "business_risk",
            "fairness_risk",
            "capacity_pressure",
            "adaptive_threshold",
            "magd_assurance_risk",
            "passes_threshold",
        ]
    ].copy()


def run_adaptive_threshold(config_path: str | Path) -> AdaptiveThresholdArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    adaptive_cfg = _adaptive_threshold_config(config)
    assurance_dir = _resolve_assurance_dir(resolved_config_path)
    inputs = load_adaptive_threshold_inputs(resolved_config_path)

    thresholds = compute_adaptive_thresholds(
        inputs,
        base_threshold=adaptive_cfg["base_threshold"],
        value_weight=adaptive_cfg["value_weight"],
        fairness_weight=adaptive_cfg["fairness_weight"],
        capacity_weight=adaptive_cfg["capacity_weight"],
    )
    thresholds.to_csv(assurance_dir / "adaptive_thresholds.csv", index=False)
    return AdaptiveThresholdArtifacts(thresholds=thresholds, assurance_dir=assurance_dir)
