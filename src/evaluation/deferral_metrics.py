from __future__ import annotations

import pandas as pd


def compute_deferral_metrics(frame: pd.DataFrame, oracle_reference: pd.DataFrame | None = None) -> dict[str, float]:
    route = frame["selected_route"].astype(str)
    ai_wrong = ~frame["ai_correct"].astype(bool)
    deferred = route.isin(["Human Expert", "Escalate"])

    ai_coverage = float((route == "AI").mean())
    human_deferral_rate = float((route == "Human Expert").mean())
    escalation_rate = float((route == "Escalate").mean())

    deferral_precision = float(ai_wrong[deferred].mean()) if deferred.any() else 0.0
    deferral_recall = float(deferred[ai_wrong].mean()) if ai_wrong.any() else 0.0

    capacity_violation = 0.0
    if "capacity_status" in frame.columns:
        violation_mask = frame["capacity_status"].astype(str).isin(["capacity_exhausted"])
        capacity_violation = float(violation_mask.mean())

    oracle_gap = 0.0
    if oracle_reference is not None and "is_correct" in oracle_reference.columns and "is_correct" in frame.columns:
        oracle_gap = float(oracle_reference["is_correct"].mean() - frame["is_correct"].mean())

    return {
        "ai_coverage": ai_coverage,
        "human_deferral_rate": human_deferral_rate,
        "escalation_rate": escalation_rate,
        "deferral_precision": deferral_precision,
        "deferral_recall": deferral_recall,
        "capacity_violation_rate": capacity_violation,
        "oracle_gap": oracle_gap,
    }
