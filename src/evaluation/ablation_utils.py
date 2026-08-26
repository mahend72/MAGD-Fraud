from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.load_fifar import load_fifar_data
from src.evaluation.audit_metrics import compute_audit_metrics
from src.evaluation.deferral_metrics import compute_deferral_metrics
from src.evaluation.fairness_metrics import compute_fairness_metrics
from src.evaluation.fraud_metrics import compute_fraud_metrics
from src.evaluation.reliance_metrics import compute_reliance_metrics
from src.utils.io import load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

ABLATION_VARIANTS: list[str] = [
    "distance_only",
    "distance_plus_calibration",
    "distance_plus_neighbor_error",
    "distance_plus_wrong_confident",
    "full_magd_heuristic",
    "full_magd_learned",
    "full_magd_constrained_initial",
    "full_magd_constrained_intervention_calibrated",
    "full_magd_constrained_fairness_if_available",
    "full_magd_constrained_capacity_if_available",
]

AUXILIARY_SIGNALS = ["calibration_risk", "neighbor_error_rate", "wrong_confident_risk", "drift_risk", "business_risk"]
REQUIRED_AUDIT_COLUMNS = [
    "numerical_confidence",
    "distance_uncertainty",
    "calibration_risk",
    "neighbor_error_rate",
    "wrong_confident_risk",
    "magd_assurance_risk",
    "selected_route",
    "final_prediction",
    "decision_reason",
]


@dataclass
class VariantSpec:
    name: str
    mode: str
    selected_signals: list[str]
    constrained_source: str | None = None
    requires_fairness: bool = False
    requires_capacity: bool = False


def variant_specs() -> list[VariantSpec]:
    return [
        VariantSpec("distance_only", "custom", ["distance_uncertainty"]),
        VariantSpec("distance_plus_calibration", "custom", ["distance_uncertainty", "calibration_risk"]),
        VariantSpec("distance_plus_neighbor_error", "custom", ["distance_uncertainty", "neighbor_error_rate"]),
        VariantSpec("distance_plus_wrong_confident", "custom", ["distance_uncertainty", "wrong_confident_risk"]),
        VariantSpec(
            "full_magd_heuristic",
            "policy",
            ["distance_uncertainty", "calibration_risk", "neighbor_error_rate", "wrong_confident_risk", "drift_risk", "business_risk"],
        ),
        VariantSpec(
            "full_magd_learned",
            "policy",
            ["distance_uncertainty", "calibration_risk", "neighbor_error_rate", "wrong_confident_risk", "drift_risk", "business_risk"],
        ),
        VariantSpec(
            "full_magd_constrained_initial",
            "constrained",
            ["distance_uncertainty", "calibration_risk", "neighbor_error_rate", "wrong_confident_risk", "drift_risk", "business_risk"],
            constrained_source="initial",
        ),
        VariantSpec(
            "full_magd_constrained_intervention_calibrated",
            "constrained",
            ["distance_uncertainty", "calibration_risk", "neighbor_error_rate", "wrong_confident_risk", "drift_risk", "business_risk"],
            constrained_source="calibrated",
        ),
        VariantSpec(
            "full_magd_constrained_fairness_if_available",
            "constrained",
            ["distance_uncertainty", "calibration_risk", "neighbor_error_rate", "wrong_confident_risk", "drift_risk", "business_risk"],
            constrained_source="calibrated",
            requires_fairness=True,
        ),
        VariantSpec(
            "full_magd_constrained_capacity_if_available",
            "constrained",
            ["distance_uncertainty", "calibration_risk", "neighbor_error_rate", "wrong_confident_risk", "drift_risk", "business_risk"],
            constrained_source="calibrated",
            requires_capacity=True,
        ),
    ]


def outputs_root_for_config(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    return outputs_root


def ensure_variant_allowed(spec: VariantSpec, config_path: Path) -> tuple[bool, str]:
    loaded = load_fifar_data(config_path)
    if spec.requires_fairness and not loaded.sensitive_attributes:
        return False, "skipped_no_sensitive_attributes"
    if spec.requires_capacity and (loaded.capacity_df is None or loaded.capacity_df.empty):
        return False, "skipped_no_capacity_data"
    return True, "ready"


def zero_out_unselected_signals(frame: pd.DataFrame, selected_signals: list[str]) -> pd.DataFrame:
    working = frame.copy()
    for signal in AUXILIARY_SIGNALS:
        if signal in working.columns and signal not in selected_signals:
            working[signal] = 0.0
    return working


def summarize_fairness_disparity(frame: pd.DataFrame, config_path: Path, *, fp_cost: float, fn_cost: float) -> tuple[float, bool]:
    sensitive_attributes = load_fifar_data(config_path).sensitive_attributes
    fairness = compute_fairness_metrics(frame, sensitive_attributes, fp_cost=fp_cost, fn_cost=fn_cost)
    if fairness.empty:
        return 0.0, False
    disparity_columns = [column for column in ["false_positive_rate_disparity", "false_negative_rate_disparity", "cost_disparity"] if column in fairness.columns]
    if not disparity_columns:
        return 0.0, True
    disparity = float(fairness[disparity_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy().max())
    return disparity, True


def _with_offline_wrong_confident_label(frame: pd.DataFrame, config_path: Path) -> pd.DataFrame:
    """Merges in the frozen offline wrong-confident label (test split) so
    compute_reliance_metrics can compute a genuine WCA instead of silently
    defaulting to 0.0. Pure reporting-input attachment - does not affect which
    decisions were made, only how WCA gets computed from them afterward."""
    if "wrong_confident_label_offline" in frame.columns or "case_id" not in frame.columns:
        return frame
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    wc_path = outputs_root / "assurance" / "wrong_confident_risk.csv"
    if not wc_path.exists():
        return frame
    wc = pd.read_csv(wc_path)
    wc["case_id"] = wc["case_id"].astype(str)
    working = frame.copy()
    working["case_id"] = working["case_id"].astype(str)
    if "split" in wc.columns and "split" in working.columns:
        split_value = str(working["split"].iloc[0]) if len(working) else None
        if split_value is not None:
            wc = wc.loc[wc["split"].astype(str) == split_value]
    return working.merge(wc[["case_id", "wrong_confident_label_offline"]].drop_duplicates("case_id"), on="case_id", how="left")


def compute_ablation_metrics(frame: pd.DataFrame, config_path: Path) -> dict[str, float]:
    config = load_yaml(config_path)
    fp_cost = float(config["costs"]["false_positive"])
    fn_cost = float(config["costs"]["false_negative"])
    frame = _with_offline_wrong_confident_label(frame, config_path)
    fraud = compute_fraud_metrics(frame, score_column="ai_score" if "ai_score" in frame.columns else None, fp_cost=fp_cost, fn_cost=fn_cost)
    reliance = compute_reliance_metrics(frame)
    deferral = compute_deferral_metrics(frame, oracle_reference=None)
    audit = compute_audit_metrics(frame, required_evidence_columns=REQUIRED_AUDIT_COLUMNS)
    fairness_disparity, fairness_available = summarize_fairness_disparity(frame, config_path, fp_cost=fp_cost, fn_cost=fn_cost)
    return {
        "precision": float(fraud["precision"]),
        "recall": float(fraud["recall"]),
        "f1": float(fraud["f1"]),
        "pr_auc": float(fraud["pr_auc"]) if pd.notna(fraud["pr_auc"]) else float("nan"),
        "cost_sensitive_loss": float(fraud["cost_sensitive_loss"]),
        "overreliance": float(reliance["overreliance"]),
        "underreliance": float(reliance["underreliance"]),
        "correct_rejection": float(reliance["correct_rejection"]),
        "wrong_confident_avoidance": float(reliance["wrong_confident_avoidance_rate"]),
        "ai_coverage": float(deferral["ai_coverage"]),
        "expert_deferral_rate": float(deferral["human_deferral_rate"]),
        "escalation_rate": float(deferral["escalation_rate"]),
        "capacity_violation_rate": float(deferral["capacity_violation_rate"]),
        "fairness_disparity": float(fairness_disparity),
        "fairness_available": float(1.0 if fairness_available else 0.0),
        "audit_coverage": float(audit["audit_coverage"]),
        "evidence_completeness": float(1.0 - audit["missing_evidence_rate"]),
    }


def required_plot_path(config_path: Path) -> Path:
    return outputs_root_for_config(config_path) / "plots" / "ablation_cost_vs_overreliance.png"

