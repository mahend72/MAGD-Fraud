from __future__ import annotations

import pandas as pd


def compute_audit_metrics(frame: pd.DataFrame, required_evidence_columns: list[str]) -> dict[str, float]:
    total = len(frame)
    if total == 0:
        return {
            "audit_coverage": 0.0,
            "evidence_completeness": 0.0,
            "complete_decision_logs_rate": 0.0,
            "missing_evidence_rate": 1.0,
            "missing_rationale_rate": 1.0,
        }

    essential_columns = ["case_id", "y_true", "final_prediction", "selected_route"]
    present_essentials = [column for column in essential_columns if column in frame.columns]
    complete_logs = ~frame[present_essentials].isna().any(axis=1) if present_essentials else pd.Series([False] * total)

    evidence_present_cols = [column for column in required_evidence_columns if column in frame.columns]
    if evidence_present_cols:
        complete_evidence = ~frame[evidence_present_cols].isna().any(axis=1)
        evidence_completeness = 1.0 - frame[evidence_present_cols].isna().mean(axis=1)
    else:
        complete_evidence = pd.Series([False] * total)
        evidence_completeness = pd.Series([0.0] * total)

    rationale_col = "decision_reason" if "decision_reason" in frame.columns else None
    if rationale_col:
        rationale = frame[rationale_col].fillna("").astype(str).str.strip()
        missing_rationale = rationale.eq("")
    else:
        missing_rationale = pd.Series([True] * total)

    return {
        "audit_coverage": float(complete_evidence.mean()),
        "evidence_completeness": float(evidence_completeness.mean()),
        "complete_decision_logs_rate": float(complete_logs.mean()),
        "missing_evidence_rate": float((~complete_evidence).mean()),
        "missing_rationale_rate": float(missing_rationale.mean()),
    }
