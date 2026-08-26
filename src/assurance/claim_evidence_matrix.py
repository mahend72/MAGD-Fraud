from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.evaluation.audit_metrics import compute_audit_metrics
from src.utils.io import load_yaml


REQUIRED_AUDIT_COLUMNS = [
    "case_id",
    "ai_score",
    "ai_pred",
    "numerical_confidence",
    "calibration_risk",
    "distance_uncertainty",
    "neighbor_error_rate",
    "wrong_confident_risk",
    "magd_assurance_risk",
    "risk_category",
    "selected_route",
    "selected_expert",
    "final_prediction",
    "decision_reason",
    "method",
]

REQUIRED_EVIDENCE_COLUMNS = [
    "numerical_confidence",
    "calibration_risk",
    "distance_uncertainty",
    "neighbor_error_rate",
    "wrong_confident_risk",
    "magd_assurance_risk",
    "selected_route",
    "final_prediction",
    "decision_reason",
]

CLAIMS: list[tuple[str, str, list[str]]] = [
    ("C1", "AI confidence is reliable enough for use.", ["numerical_confidence", "calibration_risk"]),
    ("C2", "Distance uncertainty was assessed.", ["distance_uncertainty"]),
    ("C3", "Similar historical cases support or weaken AI reliance.", ["neighbor_error_rate"]),
    ("C4", "Wrong-confident AI risk was assessed.", ["wrong_confident_risk"]),
    ("C5", "Expert review or escalation was justified.", ["selected_route", "selected_expert", "decision_reason"]),
    ("C6", "The final routing decision is reconstructable.", ["ai_pred", "selected_route", "final_prediction", "decision_reason"]),
    ("C7", "Optional fairness/capacity evidence was considered when available.", ["fairness_risk", "capacity_status"]),
]


@dataclass
class ClaimEvidenceArtifacts:
    matrix: pd.DataFrame
    output_dir: Path
    audit_log: pd.DataFrame
    auditability: pd.DataFrame


def _resolve_outputs_root(config_path: Path) -> Path:
    config = load_yaml(config_path)
    output_dir = (
        config.get("experiment", {}).get("output_dir")
        or config.get("paths", {}).get("outputs_dir")
        or "data/outputs"
    )
    outputs_root = Path(output_dir)
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    return outputs_root


def _resolve_dirs(config_path: Path) -> tuple[Path, Path, Path]:
    outputs_root = _resolve_outputs_root(config_path)
    audit_dir = outputs_root / "audit_pack"
    paper_dir = outputs_root / "paper_tables"
    audit_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    return outputs_root, audit_dir, paper_dir


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _decision_candidates(outputs_root: Path) -> list[Path]:
    return [
        outputs_root / "assurance_deferral" / "magd_constrained_calibrated_decisions.csv",
        outputs_root / "assurance_deferral" / "magd_constrained_decisions.csv",
        outputs_root / "assurance_deferral" / "magd_heuristic_decisions.csv",
        outputs_root / "assurance_deferral" / "magd_learned_decisions.csv",
        outputs_root / "assurance_deferral" / "assurance_guided_decisions.csv",
        outputs_root / "ablations" / "ablation_decisions_full_magd_constrained_intervention_calibrated.csv",
        outputs_root / "ablations" / "ablation_decisions_full_magd_constrained_initial.csv",
        outputs_root / "ablations" / "ablation_decisions_full_magd_heuristic.csv",
    ]


def _normalize_method_name(frame: pd.DataFrame, decision_path: Path) -> pd.Series:
    if "method" in frame.columns and frame["method"].notna().any():
        return frame["method"].astype(str)
    stem = decision_path.stem
    if "calibrated" in stem:
        method = "MAGD-Constrained intervention-calibrated"
    elif "initial" in stem:
        method = "MAGD-Constrained initial"
    elif "heuristic" in stem:
        method = "MAGD-Heuristic"
    elif "learned" in stem:
        method = "MAGD-Learned"
    elif "assurance" in stem:
        method = "assurance-guided deferral"
    else:
        method = stem
    return pd.Series([method] * len(frame), index=frame.index)


def _load_optional_table(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    frame = pd.read_csv(path)
    if "case_id" in frame.columns:
        frame["case_id"] = frame["case_id"].astype(str)
    if columns is None:
        return frame
    available = [column for column in columns if column in frame.columns]
    return frame[available].copy()


def build_decision_audit_log(config_path: str | Path) -> pd.DataFrame:
    resolved_config_path = Path(config_path).resolve()
    outputs_root = _resolve_outputs_root(resolved_config_path)
    decision_path = _first_existing(_decision_candidates(outputs_root))
    if decision_path is None:
        raise FileNotFoundError("No routed MAGD decision log found for audit-pack generation.")

    frame = pd.read_csv(decision_path).copy()
    if "case_id" not in frame.columns:
        raise ValueError(f"Decision log is missing `case_id`: {decision_path}")
    frame["case_id"] = frame["case_id"].astype(str)

    wrong_conf_path = outputs_root / "assurance" / "wrong_confident_risk.csv"
    wrong_conf = _load_optional_table(
        wrong_conf_path,
        columns=[
            "case_id",
            "ai_score",
            "ai_pred",
            "numerical_confidence",
            "distance_confidence",
            "distance_uncertainty",
            "calibration_risk",
            "neighbor_error_rate",
            "wrong_confident_risk",
        ],
    )
    if not wrong_conf.empty:
        frame = frame.merge(wrong_conf, on="case_id", how="left", suffixes=("", "_assurance"))
        for column in [
            "ai_score",
            "ai_pred",
            "numerical_confidence",
            "distance_confidence",
            "distance_uncertainty",
            "calibration_risk",
            "neighbor_error_rate",
            "wrong_confident_risk",
        ]:
            fallback = f"{column}_assurance"
            if fallback in frame.columns:
                if column not in frame.columns:
                    frame[column] = frame[fallback]
                else:
                    frame[column] = frame[column].where(frame[column].notna(), frame[fallback])
                frame = frame.drop(columns=[fallback])

    magd_path = outputs_root / "assurance" / "magd_risk.csv"
    magd = _load_optional_table(
        magd_path,
        columns=[
            "case_id",
            "magd_assurance_risk",
            "risk_category",
            "drift_risk",
            "business_risk",
            "drift_available",
            "business_available",
        ],
    )
    if not magd.empty:
        frame = frame.merge(magd, on="case_id", how="left", suffixes=("", "_magd"))
        for column in [
            "magd_assurance_risk",
            "risk_category",
            "drift_risk",
            "business_risk",
            "drift_available",
            "business_available",
        ]:
            fallback = f"{column}_magd"
            if fallback in frame.columns:
                if column not in frame.columns:
                    frame[column] = frame[fallback]
                else:
                    frame[column] = frame[column].where(frame[column].notna(), frame[fallback])
                frame = frame.drop(columns=[fallback])

    if "selected_route" not in frame.columns:
        frame["selected_route"] = "AI"
    if "selected_expert" not in frame.columns:
        frame["selected_expert"] = ""
    frame["selected_expert"] = frame["selected_expert"].fillna("")
    if "decision_reason" not in frame.columns:
        frame["decision_reason"] = ""
    if "risk_category" not in frame.columns:
        frame["risk_category"] = ""
    if "method" not in frame.columns:
        frame["method"] = _normalize_method_name(frame, decision_path)
    else:
        frame["method"] = frame["method"].where(frame["method"].notna(), _normalize_method_name(frame, decision_path))

    if "fairness_risk" not in frame.columns:
        frame["fairness_risk"] = pd.NA
    if "capacity_status" not in frame.columns:
        frame["capacity_status"] = pd.NA
    if "drift_risk" not in frame.columns:
        frame["drift_risk"] = 0.0
    if "business_risk" not in frame.columns:
        frame["business_risk"] = 0.0
    if "is_correct" not in frame.columns and {"y_true", "final_prediction"}.issubset(frame.columns):
        frame["is_correct"] = (
            pd.to_numeric(frame["final_prediction"], errors="coerce")
            == pd.to_numeric(frame["y_true"], errors="coerce")
        ).astype("Int64")
    if "ai_correct" not in frame.columns and {"y_true", "ai_pred"}.issubset(frame.columns):
        frame["ai_correct"] = (
            pd.to_numeric(frame["ai_pred"], errors="coerce")
            == pd.to_numeric(frame["y_true"], errors="coerce")
        ).astype("Int64")

    for column in REQUIRED_AUDIT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    ordered_columns = REQUIRED_AUDIT_COLUMNS + [
        column
        for column in [
            "y_true",
            "is_correct",
            "ai_correct",
            "distance_confidence",
            "neighbor_fraud_rate",
            "neighbor_ai_agreement",
            "mean_neighbor_distance",
            "drift_risk",
            "business_risk",
            "drift_available",
            "business_available",
            "fairness_risk",
            "fairness_penalty",
            "capacity_status",
            "capacity_pressure",
        ]
        if column in frame.columns
    ]
    available_columns = [column for column in ordered_columns if column in frame.columns]
    return frame[available_columns].copy()


def _example_case_ids(frame: pd.DataFrame, evidence_fields: list[str], limit: int = 3) -> str:
    if "case_id" not in frame.columns or frame.empty:
        return ""
    present_fields = [field for field in evidence_fields if field in frame.columns]
    if not present_fields:
        return ""
    if set(evidence_fields) == {"fairness_risk", "capacity_status"}:
        optional_mask = pd.Series(False, index=frame.index)
        if "fairness_risk" in frame.columns:
            optional_mask = optional_mask | frame["fairness_risk"].notna()
        if "capacity_status" in frame.columns:
            optional_mask = optional_mask | frame["capacity_status"].notna()
        examples = frame.loc[optional_mask, "case_id"].astype(str).head(limit).tolist()
        return ", ".join(examples)
    complete = ~frame[present_fields].isna().any(axis=1)
    examples = frame.loc[complete, "case_id"].astype(str).head(limit).tolist()
    return ", ".join(examples)


def build_claim_evidence_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for claim_id, claim_text, evidence_fields in CLAIMS:
        present_fields = [field for field in evidence_fields if field in frame.columns]
        if set(evidence_fields) == {"fairness_risk", "capacity_status"}:
            optional_available = pd.Series(False, index=frame.index)
            if "fairness_risk" in frame.columns:
                optional_available = optional_available | frame["fairness_risk"].notna()
            if "capacity_status" in frame.columns:
                optional_available = optional_available | frame["capacity_status"].notna()
            coverage_score = float(optional_available.mean()) if len(frame) else 0.0
            missing_count = int((~optional_available).sum()) if len(frame) else 0
        elif present_fields:
            complete = ~frame[present_fields].isna().any(axis=1)
            coverage_score = float(complete.mean())
            missing_count = int((~complete).sum())
        else:
            coverage_score = 0.0
            missing_count = len(frame)
        rows.append(
            {
                "claim_id": claim_id,
                "claim_text": claim_text,
                "evidence_fields": ", ".join(evidence_fields),
                "coverage_score": coverage_score,
                "missing_evidence_count": missing_count,
                "example_case_ids": _example_case_ids(frame, evidence_fields),
            }
        )
    return pd.DataFrame(rows)


def build_auditability_table(audit_log: pd.DataFrame, claim_matrix: pd.DataFrame) -> pd.DataFrame:
    method = str(audit_log["method"].iloc[0]) if "method" in audit_log.columns and not audit_log.empty else "unknown"
    metrics = compute_audit_metrics(audit_log, REQUIRED_EVIDENCE_COLUMNS)
    claim_coverage = pd.to_numeric(claim_matrix.get("coverage_score", pd.Series(dtype=float)), errors="coerce")
    return pd.DataFrame(
        [
            {
                "method": method,
                "cases": int(len(audit_log)),
                "audit_coverage": float(metrics["audit_coverage"]),
                "evidence_completeness": float(metrics["evidence_completeness"]),
                "complete_decision_logs_rate": float(metrics["complete_decision_logs_rate"]),
                "missing_evidence_rate": float(metrics["missing_evidence_rate"]),
                "missing_rationale_rate": float(metrics["missing_rationale_rate"]),
                "mean_claim_coverage": float(claim_coverage.mean()) if not claim_coverage.empty else 0.0,
                "claims_with_example_cases": int(claim_matrix["example_case_ids"].astype(str).str.len().gt(0).sum())
                if "example_case_ids" in claim_matrix.columns
                else 0,
            }
        ]
    )


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.fillna("")
    header = "| " + " | ".join(display.columns.astype(str)) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows]) + "\n"


def run_claim_evidence_matrix(config_path: str | Path) -> ClaimEvidenceArtifacts:
    resolved_config_path = Path(config_path).resolve()
    _, audit_dir, paper_dir = _resolve_dirs(resolved_config_path)
    audit_log = build_decision_audit_log(resolved_config_path)
    matrix = build_claim_evidence_matrix(audit_log)
    auditability = build_auditability_table(audit_log, matrix)
    matrix.to_csv(audit_dir / "claim_evidence_matrix.csv", index=False)
    (audit_dir / "claim_evidence_matrix.md").write_text(markdown_table(matrix.round({"coverage_score": 4})), encoding="utf-8")
    auditability.to_csv(paper_dir / "auditability.csv", index=False)
    return ClaimEvidenceArtifacts(matrix=matrix, output_dir=audit_dir, audit_log=audit_log, auditability=auditability)
