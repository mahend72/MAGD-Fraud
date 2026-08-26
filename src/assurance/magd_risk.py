from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.io import load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

THRESHOLD_METADATA_FILENAME = "magd_risk_thresholds.json"


@dataclass
class MagdRiskArtifacts:
    magd_frame: pd.DataFrame
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


def _magd_config(config: dict) -> dict:
    magd_cfg = config.get("magd", {})
    signals_cfg = magd_cfg.get("signals", {})
    thresholds = magd_cfg.get("thresholds", {})
    weights = magd_cfg.get("risk_weights", magd_cfg.get("weights", {}))
    use_drift_risk = bool(magd_cfg.get("use_drift_risk", signals_cfg.get("use_drift_risk", False)))
    use_business_risk = bool(magd_cfg.get("use_business_risk", signals_cfg.get("use_business_risk", False)))
    resolved = {
        "use_drift_risk": use_drift_risk,
        "use_business_risk": use_business_risk,
        "threshold_mode": str(thresholds.get("mode", "validation_quantile")).lower(),
        "low_quantile": float(thresholds.get("low_quantile", 0.60)),
        "high_quantile": float(thresholds.get("high_quantile", 0.90)),
        "low_risk": float(thresholds.get("low_risk", 0.35)),
        "high_risk": float(thresholds.get("high_risk", 0.70)),
        "weights": {
            "distance_uncertainty": float(weights.get("distance_uncertainty", 0.25)),
            "calibration_risk": float(weights.get("calibration_risk", 0.20)),
            "neighbor_error_rate": float(weights.get("neighbor_error_rate", 0.20)),
            "wrong_confident_risk": float(weights.get("wrong_confident_risk", 0.25)),
            # Disabled signals must not contribute to compute_magd_risk's total_weight
            # denominator, matching magd_policy.py::_normalize_weights, which excludes
            # inactive signals entirely rather than padding the sum with an unused weight.
            "drift_risk": float(weights.get("drift_risk", 0.05)) if use_drift_risk else 0.0,
            "business_risk": float(weights.get("business_risk", 0.05)) if use_business_risk else 0.0,
        },
    }
    if not 0.0 <= resolved["low_risk"] <= 1.0:
        raise ValueError("`magd.thresholds.low_risk` must be in [0, 1].")
    if not 0.0 <= resolved["high_risk"] <= 1.0:
        raise ValueError("`magd.thresholds.high_risk` must be in [0, 1].")
    if resolved["low_risk"] > resolved["high_risk"]:
        raise ValueError("`magd.thresholds.low_risk` must be less than or equal to `magd.thresholds.high_risk`.")
    return resolved


def _load_required_csv(path: Path, *, required_columns: set[str], friendly_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {friendly_name} at {path}.")
    frame = pd.read_csv(path)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{friendly_name} is missing required columns: {sorted(missing)}")
    return frame


def load_magd_inputs(config_path: str | Path) -> pd.DataFrame:
    resolved_config_path = Path(config_path).resolve()
    assurance_dir, _ = _resolve_output_dirs(resolved_config_path)

    distance = _load_required_csv(
        assurance_dir / "distance_uncertainty.csv",
        required_columns={"case_id", "split", "distance_uncertainty"},
        friendly_name="distance uncertainty outputs",
    )
    calibration = _load_required_csv(
        assurance_dir / "calibration_risk.csv",
        required_columns={"case_id", "split", "calibration_risk"},
        friendly_name="calibration risk outputs",
    )
    wrong_conf = _load_required_csv(
        assurance_dir / "wrong_confident_risk.csv",
        required_columns={"case_id", "split", "wrong_confident_risk"},
        friendly_name="wrong-confident risk outputs",
    )
    local = _load_required_csv(
        assurance_dir / "local_reliability.csv",
        required_columns={"case_id", "neighbor_error_rate"},
        friendly_name="local reliability outputs",
    )
    if "split" not in local.columns:
        local = local.copy()
        local["split"] = pd.NA

    merged = (
        distance[["case_id", "split", "distance_uncertainty"]]
        .merge(
            calibration[["case_id", "split", "calibration_risk"]],
            on=["case_id", "split"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            wrong_conf[["case_id", "split", "wrong_confident_risk"]],
            on=["case_id", "split"],
            how="inner",
            validate="one_to_one",
        )
    )

    if local["split"].notna().any():
        merged = merged.merge(
            local[["case_id", "split", "neighbor_error_rate"]],
            on=["case_id", "split"],
            how="inner",
        )
    else:
        merged = merged.merge(
            local[["case_id", "neighbor_error_rate"]],
            on="case_id",
            how="inner",
        )

    return merged


def _resolve_optional_signal(
    frame: pd.DataFrame,
    column_name: str,
    *,
    enabled: bool,
) -> tuple[pd.Series, bool]:
    if not enabled:
        return pd.Series([0.0] * len(frame), index=frame.index, dtype=float), False
    if column_name not in frame.columns:
        LOGGER.warning("MAGD optional signal `%s` is unavailable; defaulting to 0.0.", column_name)
        return pd.Series([0.0] * len(frame), index=frame.index, dtype=float), False
    series = pd.to_numeric(frame[column_name], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return series, True


def compute_magd_risk(
    frame: pd.DataFrame,
    weights: dict[str, float],
    *,
    use_drift_risk: bool,
    use_business_risk: bool,
) -> pd.DataFrame:
    working = frame.copy()
    required_columns = {
        "case_id",
        "distance_uncertainty",
        "calibration_risk",
        "neighbor_error_rate",
        "wrong_confident_risk",
    }
    missing = required_columns - set(working.columns)
    if missing:
        raise ValueError(f"MAGD risk input frame missing required columns: {sorted(missing)}")

    for column in ["distance_uncertainty", "calibration_risk", "neighbor_error_rate", "wrong_confident_risk"]:
        working[column] = pd.to_numeric(working[column], errors="coerce")
        if working[column].isna().any():
            raise ValueError(f"MAGD risk input column `{column}` contains non-numeric values.")
        working[column] = working[column].clip(0.0, 1.0)

    drift_risk, drift_available = _resolve_optional_signal(working, "drift_risk", enabled=use_drift_risk)
    business_risk, business_available = _resolve_optional_signal(working, "business_risk", enabled=use_business_risk)
    working["drift_risk"] = drift_risk
    working["business_risk"] = business_risk
    working["drift_available"] = drift_available
    working["business_available"] = business_available
    working["drift_risk_available"] = drift_available
    working["business_risk_available"] = business_available

    total_weight = sum(max(float(weight), 0.0) for weight in weights.values())
    if total_weight <= 0.0:
        raise ValueError("MAGD risk weights must sum to a positive value.")

    raw_risk = (
        float(weights["distance_uncertainty"]) * working["distance_uncertainty"]
        + float(weights["calibration_risk"]) * working["calibration_risk"]
        + float(weights["neighbor_error_rate"]) * working["neighbor_error_rate"]
        + float(weights["wrong_confident_risk"]) * working["wrong_confident_risk"]
        + float(weights["drift_risk"]) * working["drift_risk"]
        + float(weights["business_risk"]) * working["business_risk"]
    ) / total_weight
    working["magd_assurance_risk"] = raw_risk.clip(0.0, 1.0)
    return working


def derive_validation_risk_thresholds(
    validation_risk: pd.Series,
    *,
    low_quantile: float,
    high_quantile: float,
) -> tuple[float, float]:
    """Derive low/high MAGD-risk thresholds from the validation-split risk distribution only.

    This is the leakage-safe replacement for hard-coded absolute thresholds: it looks only at
    validation-split `magd_assurance_risk` values, never train or test labels/risk.
    """
    values = pd.to_numeric(validation_risk, errors="coerce").dropna()
    if values.empty:
        raise ValueError("Cannot derive MAGD risk thresholds from an empty validation risk series.")
    low = float(values.quantile(low_quantile))
    high = float(values.quantile(high_quantile))
    if high <= low:
        high = min(1.0, low + 1e-6)
    return low, high


def resolve_magd_risk_thresholds(
    scored_frame: pd.DataFrame,
    *,
    threshold_mode: str,
    low_quantile: float,
    high_quantile: float,
    fallback_low_risk: float,
    fallback_high_risk: float,
) -> tuple[float, float, dict[str, object]]:
    """Resolve the low/high MAGD-risk thresholds to freeze for this run.

    In `validation_quantile` mode the thresholds are derived from the validation split's
    `magd_assurance_risk` distribution only; in `fixed` mode the configured absolute values are
    used unchanged. Either way the result is a single frozen pair applied uniformly to all splits.
    """
    if threshold_mode == "fixed":
        metadata = {
            "mode": "fixed",
            "low_risk": float(fallback_low_risk),
            "high_risk": float(fallback_high_risk),
        }
        return float(fallback_low_risk), float(fallback_high_risk), metadata

    if "split" not in scored_frame.columns:
        raise ValueError("Cannot derive validation-quantile MAGD risk thresholds without a `split` column.")
    validation_risk = scored_frame.loc[scored_frame["split"].astype(str) == "val", "magd_assurance_risk"]
    low_risk, high_risk = derive_validation_risk_thresholds(
        validation_risk,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
    )
    metadata = {
        "mode": "validation_quantile",
        "low_quantile": float(low_quantile),
        "high_quantile": float(high_quantile),
        "low_risk": float(low_risk),
        "high_risk": float(high_risk),
        "validation_rows": int(len(validation_risk)),
        "derived_from_split": "val",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return low_risk, high_risk, metadata


def load_frozen_magd_risk_thresholds(
    assurance_dir: Path,
    *,
    fallback_low_risk: float,
    fallback_high_risk: float,
) -> tuple[float, float]:
    """Load the low/high thresholds frozen by `run_magd_risk` for this run, if available.

    Falls back to the provided (typically config-default) values when no frozen threshold
    metadata exists yet, e.g. because `run_magd_risk` has not been run in this context.
    """
    path = Path(assurance_dir) / THRESHOLD_METADATA_FILENAME
    if not path.exists():
        return float(fallback_low_risk), float(fallback_high_risk)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data["low_risk"]), float(data["high_risk"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        LOGGER.warning("Could not read frozen MAGD risk thresholds at %s; using fallback values.", path)
        return float(fallback_low_risk), float(fallback_high_risk)


def map_magd_risk_category(magd_risk: float, low_risk: float, high_risk: float) -> str:
    if magd_risk < low_risk:
        return "low"
    if magd_risk < high_risk:
        return "medium"
    return "high"


def recommended_action_for_category(category: str) -> str:
    mapping = {"low": "AI", "medium": "Human Expert", "high": "Escalate"}
    if category not in mapping:
        raise ValueError(f"Unknown MAGD risk category: {category}")
    return mapping[category]


def add_risk_categories_and_actions(
    frame: pd.DataFrame,
    *,
    low_risk: float,
    high_risk: float,
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["risk_category"] = [
        map_magd_risk_category(value, low_risk, high_risk)
        for value in enriched["magd_assurance_risk"].astype(float)
    ]
    enriched["recommended_action"] = [recommended_action_for_category(category) for category in enriched["risk_category"]]
    return enriched


def _plot_magd_risk_distribution(frame: pd.DataFrame, plot_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to generate the MAGD risk distribution plot.") from exc

    plt.figure(figsize=(7, 5))
    plt.hist(frame["magd_assurance_risk"].astype(float), bins=30)
    plt.xlabel("MAGD Assurance Risk")
    plt.ylabel("Case Count")
    plt.title("MAGD Risk Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def run_magd_risk(config_path: str | Path) -> MagdRiskArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    magd_cfg = _magd_config(config)
    assurance_dir, plots_dir = _resolve_output_dirs(resolved_config_path)

    inputs = load_magd_inputs(resolved_config_path)
    scored = compute_magd_risk(
        inputs,
        magd_cfg["weights"],
        use_drift_risk=magd_cfg["use_drift_risk"],
        use_business_risk=magd_cfg["use_business_risk"],
    )
    low_risk, high_risk, threshold_metadata = resolve_magd_risk_thresholds(
        scored,
        threshold_mode=magd_cfg["threshold_mode"],
        low_quantile=magd_cfg["low_quantile"],
        high_quantile=magd_cfg["high_quantile"],
        fallback_low_risk=magd_cfg["low_risk"],
        fallback_high_risk=magd_cfg["high_risk"],
    )
    enriched = add_risk_categories_and_actions(
        scored,
        low_risk=low_risk,
        high_risk=high_risk,
    )
    (assurance_dir / THRESHOLD_METADATA_FILENAME).write_text(json.dumps(threshold_metadata, indent=2), encoding="utf-8")

    output_columns = [
        "case_id",
        "split",
        "distance_uncertainty",
        "calibration_risk",
        "neighbor_error_rate",
        "wrong_confident_risk",
        "drift_risk",
        "business_risk",
        "drift_available",
        "business_available",
        "magd_assurance_risk",
        "risk_category",
    ]
    enriched[output_columns].to_csv(assurance_dir / "magd_risk.csv", index=False)
    _plot_magd_risk_distribution(enriched, plots_dir / "magd_risk_distribution.png")

    return MagdRiskArtifacts(
        magd_frame=enriched,
        assurance_dir=assurance_dir,
        plots_dir=plots_dir,
    )
