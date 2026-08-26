from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import load_yaml


PLOT_FILES = [
    "reliability_diagram.png",
    "threshold_vs_loss_distance.png",
    "threshold_vs_f1_distance.png",
    "assurance_risk_distribution.png",
    "deferral_route_counts.png",
]


def resolve_audit_pack_dirs(config_path: Path) -> tuple[dict[str, Any], Path, Path]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    audit_pack_dir = outputs_root / "audit_pack"
    plots_dir = audit_pack_dir / "plots"
    audit_pack_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return config, outputs_root, audit_pack_dir


def ensure_exists(path: Path, *, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required {description} not found: {path}")
    return path


def copy_artifact(src: Path, dst: Path) -> None:
    ensure_exists(src, description="artifact")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_assurance_summary(decision_logs: pd.DataFrame) -> pd.DataFrame:
    working = decision_logs.copy()
    if "assurance_risk" not in working.columns and "magd_assurance_risk" in working.columns:
        working["assurance_risk"] = working["magd_assurance_risk"]
    required = [
        "risk_category",
        "selected_route",
        "assurance_risk",
        "calibration_risk",
        "distance_uncertainty",
        "neighbor_error_rate",
        "wrong_confident_risk",
    ]
    missing = [column for column in required if column not in working.columns]
    if missing:
        raise ValueError(f"Decision logs are missing required assurance summary columns: {missing}")

    summary = (
        working.groupby(["risk_category", "selected_route"], dropna=False)
        .agg(
            total_cases=("case_id", "count"),
            mean_assurance_risk=("assurance_risk", "mean"),
            mean_calibration_risk=("calibration_risk", "mean"),
            mean_distance_uncertainty=("distance_uncertainty", "mean"),
            mean_neighbor_error_rate=("neighbor_error_rate", "mean"),
            mean_wrong_confident_risk=("wrong_confident_risk", "mean"),
            accuracy=("is_correct", "mean"),
        )
        .reset_index()
    )
    total = len(working)
    summary["case_share"] = summary["total_cases"] / total if total else 0.0
    return summary.sort_values(["risk_category", "selected_route"]).reset_index(drop=True)


def build_audit_coverage(
    *,
    decision_logs: pd.DataFrame,
    audit_metrics: pd.DataFrame,
    fairness_available: bool,
    capacity_available: bool,
) -> dict[str, Any]:
    target_row = pd.DataFrame()
    for method_name in [
        "MAGD-Constrained",
        "MAGD-Constrained intervention-calibrated",
        "magd_constrained_calibrated",
        "assurance-guided deferral",
    ]:
        target_row = audit_metrics.loc[audit_metrics["method"] == method_name]
        if not target_row.empty:
            break
    if target_row.empty and not audit_metrics.empty:
        target_row = audit_metrics.tail(1)
    if target_row.empty:
        raise ValueError("Audit metrics do not contain a MAGD or assurance-guided method row.")
    row = target_row.iloc[0]

    evidence_columns = [
        "numerical_confidence",
        "distance_confidence",
        "distance_uncertainty",
        "calibration_risk",
        "neighbor_error_rate",
        "wrong_confident_risk",
        "assurance_risk",
        "decision_reason",
        "selected_route",
    ]
    available_evidence = {column: bool(column in decision_logs.columns) for column in evidence_columns}

    return {
        "audit_coverage": float(row["audit_coverage"]),
        "evidence_completeness": float(row.get("evidence_completeness", 0.0)),
        "complete_decision_logs_rate": float(row["complete_decision_logs_rate"]),
        "missing_evidence_rate": float(row["missing_evidence_rate"]),
        "missing_rationale_rate": float(row["missing_rationale_rate"]),
        "total_decisions": int(len(decision_logs)),
        "fairness_summary_available": fairness_available,
        "capacity_constraints_available": capacity_available,
        "available_evidence_columns": available_evidence,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _render_method_table(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return "_No data available._"
    table = frame[available].copy()
    header = "| " + " | ".join(table.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = []
    for record in table.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in record) + " |")
    return "\n".join([header, separator, *rows])


def build_audit_report_markdown(
    *,
    config: dict[str, Any],
    final_metrics: pd.DataFrame,
    reliance_metrics: pd.DataFrame,
    deferral_metrics: pd.DataFrame,
    audit_coverage: dict[str, Any],
    fairness_available: bool,
) -> str:
    dataset_cfg = config.get("dataset", {})
    project_name = config.get("project", {}).get("name", "haaf_fifar")
    train_file = dataset_cfg.get("train_file") or dataset_cfg.get("main_file") or "unknown"
    test_file = dataset_cfg.get("test_file") or dataset_cfg.get("main_file") or "unknown"
    expert_file = dataset_cfg.get("expert_predictions_file") or "not configured"
    capacity_file = dataset_cfg.get("capacity_file") or "not configured"

    baseline_methods = [
        "AI-only",
        "best expert only",
        "random expert",
        "numerical threshold",
        "distance threshold",
        "oracle upper bound",
    ]
    key_metric_columns = [
        "method",
        "precision",
        "recall",
        "f1",
        "cost_sensitive_loss",
        "ai_coverage",
        "human_deferral_rate",
        "escalation_rate",
    ]
    reliance_columns = [
        "method",
        "correct_reliance",
        "correct_rejection",
        "overreliance",
        "underreliance",
        "wrong_confident_avoidance_rate",
    ]
    deferral_columns = [
        "method",
        "ai_coverage",
        "human_deferral_rate",
        "escalation_rate",
        "deferral_precision",
        "deferral_recall",
        "capacity_violation_rate",
        "oracle_gap",
    ]

    final_metrics = final_metrics.copy()
    key_metrics_table = _render_method_table(final_metrics, key_metric_columns)
    reliance_table = _render_method_table(reliance_metrics, reliance_columns)
    deferral_table = _render_method_table(deferral_metrics, deferral_columns)

    limitations: list[str] = []
    if capacity_file == "not configured":
        limitations.append("No capacity table was configured, so capacity-sensitive evaluation is unconstrained.")
    if not fairness_available:
        limitations.append("No fairness summary was generated because no sensitive attributes were available after preprocessing.")
    limitations.append("Oracle upper bound is reported for reference only and is not deployable.")

    lines = [
        f"# Audit Report: {project_name}",
        "",
        "## Project Aim",
        "Generate an assurance-style record showing how Level 3 model assurance signals informed Level 4 human oversight decisions for financial fraud alert review.",
        "",
        "## Dataset Used",
        f"- Train data: `{train_file}`",
        f"- Test data: `{test_file}`",
        f"- Expert predictions: `{expert_file}`",
        f"- Capacity table: `{capacity_file}`",
        "",
        "## Baselines",
        *[f"- {method}" for method in baseline_methods],
        "",
        "## Assurance-Guided Method",
        "The main method combines calibration risk, distance uncertainty, neighbor error evidence, wrong-confident risk, and optional business risk into an assurance risk score. The selected route is AI for low risk when AI expected cost is lower, Human Expert for medium risk, and Escalate for high risk.",
        "",
        "## Key Metrics",
        key_metrics_table,
        "",
        "## Reliance Metrics",
        reliance_table,
        "",
        "## Deferral Metrics",
        deferral_table,
        "",
        "## Audit Coverage",
        f"- Audit coverage: `{audit_coverage['audit_coverage']:.4f}`",
        f"- Complete decision logs rate: `{audit_coverage['complete_decision_logs_rate']:.4f}`",
        f"- Missing evidence rate: `{audit_coverage['missing_evidence_rate']:.4f}`",
        f"- Missing rationale rate: `{audit_coverage['missing_rationale_rate']:.4f}`",
        f"- Total decisions audited: `{audit_coverage['total_decisions']}`",
        "",
        "## Limitations",
        *[f"- {item}" for item in limitations],
        "",
    ]
    return "\n".join(lines)
