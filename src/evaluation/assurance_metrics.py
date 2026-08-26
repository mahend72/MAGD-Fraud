from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_conditional_rate(numerator_mask: pd.Series, denominator_mask: pd.Series) -> float:
    denominator = int(denominator_mask.sum())
    if denominator == 0:
        return 0.0
    return float((numerator_mask & denominator_mask).sum() / denominator)


def wrong_confident_avoidance_rate(frame: pd.DataFrame) -> float:
    wrong_conf = frame.get("wrong_confident_label_offline", pd.Series([0] * len(frame), index=frame.index)).astype(int)
    uses_ai = frame["selected_route"].astype(str).eq("AI")
    target = wrong_conf == 1
    if not target.any():
        return 0.0
    return float((~uses_ai[target]).mean())


def correct_rejection_rate(frame: pd.DataFrame) -> float:
    ai_wrong = ~frame["ai_correct"].astype(bool)
    uses_ai = frame["selected_route"].astype(str).eq("AI")
    return _safe_conditional_rate(~uses_ai, ai_wrong)


def overreliance_rate(frame: pd.DataFrame) -> float:
    ai_wrong = ~frame["ai_correct"].astype(bool)
    uses_ai = frame["selected_route"].astype(str).eq("AI")
    return _safe_conditional_rate(uses_ai, ai_wrong)


def underreliance_rate(frame: pd.DataFrame) -> float:
    ai_correct = frame["ai_correct"].astype(bool)
    uses_ai = frame["selected_route"].astype(str).eq("AI")
    return _safe_conditional_rate(~uses_ai, ai_correct)


def assurance_effectiveness(
    frame: pd.DataFrame,
    ai_only_reference: pd.DataFrame,
    *,
    fp_cost: float,
    fn_cost: float,
) -> float:
    if "case_id" not in frame.columns or "case_id" not in ai_only_reference.columns:
        raise ValueError("Assurance effectiveness requires `case_id` in both frames.")

    if "risk_category" in frame.columns:
        target = frame["risk_category"].astype(str).eq("high")
    elif "magd_assurance_risk" in frame.columns:
        target = pd.to_numeric(frame["magd_assurance_risk"], errors="coerce").fillna(0.0) >= 0.7
    else:
        target = pd.Series([True] * len(frame), index=frame.index)

    if not target.any():
        return 0.0

    eval_frame = frame.loc[target, ["case_id", "y_true", "final_prediction"]].copy()
    ref = ai_only_reference[["case_id", "final_prediction"]].rename(columns={"final_prediction": "ai_only_prediction"})
    merged = eval_frame.merge(ref, on="case_id", how="inner")
    if merged.empty:
        return 0.0

    y_true = merged["y_true"].astype(int)
    model_pred = merged["final_prediction"].astype(int)
    ai_only_pred = merged["ai_only_prediction"].astype(int)

    model_fp = int(((model_pred == 1) & (y_true == 0)).sum())
    model_fn = int(((model_pred == 0) & (y_true == 1)).sum())
    ai_fp = int(((ai_only_pred == 1) & (y_true == 0)).sum())
    ai_fn = int(((ai_only_pred == 0) & (y_true == 1)).sum())

    model_loss = fp_cost * model_fp + fn_cost * model_fn
    ai_only_loss = fp_cost * ai_fp + fn_cost * ai_fn
    return float(ai_only_loss - model_loss)


def audit_coverage_rate(frame: pd.DataFrame, required_evidence_columns: list[str]) -> float:
    if len(frame) == 0:
        return 0.0
    evidence_cols = [column for column in required_evidence_columns if column in frame.columns]
    if not evidence_cols:
        return 0.0
    complete = ~frame[evidence_cols].isna().any(axis=1)
    return float(complete.mean())


def evidence_completeness_rate(frame: pd.DataFrame, required_evidence_columns: list[str]) -> float:
    if len(frame) == 0:
        return 0.0
    evidence_cols = [column for column in required_evidence_columns if column in frame.columns]
    if not evidence_cols:
        return 0.0
    completeness = 1.0 - frame[evidence_cols].isna().mean(axis=1)
    return float(completeness.mean())


def compute_assurance_metrics(
    frame: pd.DataFrame,
    *,
    ai_only_reference: pd.DataFrame,
    fp_cost: float,
    fn_cost: float,
    required_evidence_columns: list[str],
) -> dict[str, float]:
    return {
        "wrong_confident_avoidance_rate": wrong_confident_avoidance_rate(frame),
        "correct_rejection_rate": correct_rejection_rate(frame),
        "overreliance_rate": overreliance_rate(frame),
        "underreliance_rate": underreliance_rate(frame),
        "assurance_effectiveness": assurance_effectiveness(
            frame,
            ai_only_reference,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        ),
        "audit_coverage": audit_coverage_rate(frame, required_evidence_columns),
        "evidence_completeness": evidence_completeness_rate(frame, required_evidence_columns),
    }
