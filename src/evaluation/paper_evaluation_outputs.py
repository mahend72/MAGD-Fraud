from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.deferral_metrics import compute_deferral_metrics
from src.evaluation.fraud_metrics import compute_fraud_metrics
from src.evaluation.reliance_metrics import compute_reliance_metrics
from src.evaluation.statistical_tests import mcnemar_paired_correctness
from src.utils.io import load_yaml


BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
BUDGET_METHODS = ["MAGD-Fraud", "Learning-to-defer", "Distance threshold", "Confidence threshold"]
RISK_BINS = ["Low", "Medium", "High", "Very high"]
CONSTRAINT_SETTINGS = [
    {"setting": "Strict", "deferral_budget": 0.01, "wca_target": 0.50, "overreliance_bound": 0.50},
    {"setting": "Moderate", "deferral_budget": 0.05, "wca_target": 0.30, "overreliance_bound": 0.70},
    {"setting": "Relaxed", "deferral_budget": 0.10, "wca_target": 0.20, "overreliance_bound": 0.80},
    {"setting": "High-review", "deferral_budget": 0.20, "wca_target": 0.20, "overreliance_bound": 0.85},
]
STAT_COMPARISONS = [
    ("MAGD-Fraud-ValidationTuned vs AI-only", "magd_validation_tuned_pred", "ai_pred"),
    ("MAGD-Fraud-ValidationTuned vs Assurance-guided deferral", "magd_validation_tuned_pred", "assurance_guided_pred"),
    ("MAGD-Fraud-ValidationTuned vs Learning-to-defer", "magd_validation_tuned_pred", "learning_to_defer_pred"),
]


@dataclass
class PaperEvaluationArtifacts:
    paper_tables_dir: Path
    budget_matched_results: pd.DataFrame
    magd_risk_calibration: pd.DataFrame
    constraint_sensitivity: pd.DataFrame
    statistical_tests: pd.DataFrame
    per_case_predictions: pd.DataFrame


def _outputs_root(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    return outputs_root


def _processed_root(config_path: Path) -> Path:
    config = load_yaml(config_path)
    processed_root = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_root.is_absolute():
        processed_root = (config_path.parent / processed_root).resolve()
    return processed_root


def _costs(config: dict[str, Any]) -> tuple[float, float]:
    costs = config.get("costs", {})
    if costs:
        return float(costs.get("false_positive", 1.0)), float(costs.get("false_negative", 5.0))
    model = config.get("model", {})
    return float(model.get("false_positive_cost", 1.0)), float(model.get("false_negative_cost", 5.0))


def _load_required_csv(path: Path, required: set[str], description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{description} is missing required columns: {missing}")
    if "case_id" in frame.columns:
        frame["case_id"] = frame["case_id"].astype(str)
    return frame


def _split_rows(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if "split" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["split"].astype(str).eq(split_name)].copy()


def _merge_common_signals(config_path: Path, core: pd.DataFrame, split_name: str) -> pd.DataFrame:
    outputs_root = _outputs_root(config_path)
    assurance_dir = outputs_root / "assurance"
    magd = _load_required_csv(
        assurance_dir / "magd_risk.csv",
        {"case_id", "split", "magd_assurance_risk"},
        "MAGD risk outputs",
    )
    wrong = _load_required_csv(
        assurance_dir / "wrong_confident_risk.csv",
        {"case_id", "split", "wrong_confident_label_offline"},
        "wrong-confident outputs",
    )
    numerical = _load_required_csv(
        assurance_dir / "numerical_confidence.csv",
        {"case_id", "split", "numerical_confidence"},
        "numerical confidence outputs",
    )
    distance = _load_required_csv(
        assurance_dir / "distance_uncertainty.csv",
        {"case_id", "split", "distance_uncertainty"},
        "distance uncertainty outputs",
    )
    working = core.copy()
    working["case_id"] = working["case_id"].astype(str)
    for frame, columns in [
        (magd, ["case_id", "magd_assurance_risk"]),
        (wrong, ["case_id", "wrong_confident_label_offline"]),
        (numerical, ["case_id", "numerical_confidence"]),
        (distance, ["case_id", "distance_uncertainty"]),
    ]:
        working = working.merge(_split_rows(frame, split_name)[columns], on="case_id", how="left", suffixes=("", "_paper"))
    if "magd_assurance_risk_paper" in working.columns:
        working["magd_assurance_risk"] = working["magd_assurance_risk"].fillna(working["magd_assurance_risk_paper"])
        working = working.drop(columns=["magd_assurance_risk_paper"])
    for duplicate in ["numerical_confidence_paper", "distance_uncertainty_paper"]:
        if duplicate in working.columns:
            base = duplicate.replace("_paper", "")
            working[base] = working[base].fillna(working[duplicate]) if base in working.columns else working[duplicate]
            working = working.drop(columns=[duplicate])
    return working


def _candidate_expert_column(frame: pd.DataFrame, preferred: str | None) -> str | None:
    if preferred and preferred in frame.columns:
        return preferred
    excluded = {"case_id", "application_id", "alert_id", "id"}
    for column in frame.columns:
        if column in excluded or column.startswith("model#") or "oracle" in column.lower():
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any() and numeric.dropna().isin([0, 1]).all():
            return column
    return None


def _align_expert_column(config_path: Path, case_ids: pd.Series, split_name: str) -> pd.Series:
    config = load_yaml(config_path)
    dataset = config.get("dataset", {})
    columns = config.get("columns", {})
    preferred = columns.get("expert_prediction")
    path_key = "expert_predictions_file" if split_name == "test" else "historical_expert_predictions_file"
    raw_path = dataset.get(path_key)
    if raw_path is None:
        return pd.Series([np.nan] * len(case_ids), index=case_ids.index, dtype=float)
    source_path = Path(raw_path)
    if not source_path.is_absolute():
        source_path = (config_path.parent / source_path).resolve()
    if not source_path.exists():
        return pd.Series([np.nan] * len(case_ids), index=case_ids.index, dtype=float)
    source = pd.read_csv(source_path)
    expert_column = _candidate_expert_column(source, preferred)
    if expert_column is None:
        return pd.Series([np.nan] * len(case_ids), index=case_ids.index, dtype=float)
    case_column = columns.get("expert_case_id") or columns.get("case_id") or source.columns[0]
    if case_column in source.columns:
        source = source.copy()
        source[case_column] = source[case_column].astype(str)
        aligned = pd.DataFrame({"case_id": case_ids.astype(str)}).merge(
            source[[case_column, expert_column]].rename(columns={case_column: "case_id"}),
            on="case_id",
            how="left",
        )[expert_column]
        if aligned.notna().any():
            return pd.to_numeric(aligned, errors="coerce")
    n = min(len(case_ids), len(source))
    values = pd.Series([np.nan] * len(case_ids), index=case_ids.index, dtype=float)
    values.iloc[:n] = pd.to_numeric(source[expert_column].iloc[:n], errors="coerce").to_numpy()
    return values


def _load_paper_core(config_path: Path, split_name: str) -> pd.DataFrame:
    outputs_root = _outputs_root(config_path)
    model = _load_required_csv(
        outputs_root / "model" / f"{split_name}_predictions.csv",
        {"case_id", "y_true", "ai_score", "ai_pred"},
        f"{split_name} model predictions",
    )
    core = _merge_common_signals(config_path, _split_rows(model, split_name), split_name)
    core["paper_expert_pred"] = _align_expert_column(config_path, core["case_id"], split_name)
    return core


def _ranked_expert_columns(config_path: Path, frames: list[pd.DataFrame]) -> list[str]:
    _ = config_path
    available = {column for frame in frames for column in frame.columns}
    return ["paper_expert_pred"] if "paper_expert_pred" in available else []


def _with_best_expert_prediction(frame: pd.DataFrame, ranked_experts: list[str]) -> pd.DataFrame:
    working = frame.copy()
    expert_columns = [expert for expert in ranked_experts if expert in working.columns]
    if not expert_columns:
        working["_paper_best_expert_pred"] = np.nan
        return working
    working["_paper_best_expert_pred"] = working[expert_columns].bfill(axis=1).iloc[:, 0]
    return working


def _select_threshold_for_budget(validation_scores: pd.Series, budget: float) -> tuple[float, float]:
    scores = pd.to_numeric(validation_scores, errors="coerce").dropna().astype(float)
    if scores.empty:
        raise ValueError("No validation scores are available for threshold selection.")
    values, counts = np.unique(scores.to_numpy(dtype=float), return_counts=True)
    n = float(len(scores))
    rates = np.cumsum(counts[::-1])[::-1] / n
    gaps = np.abs(rates - float(budget))
    best_gap = gaps.min()
    best_positions = np.flatnonzero(np.isclose(gaps, best_gap))
    best_position = int(best_positions[-1])
    best_threshold = float(values[best_position])
    best_rate = float(rates[best_position])
    return best_threshold, best_rate


def _route_by_risk(
    core: pd.DataFrame,
    risk_scores: pd.Series,
    threshold: float,
    ranked_experts: list[str],
    method: str,
) -> pd.DataFrame:
    working = core.copy().reset_index(drop=True)
    scores = pd.to_numeric(risk_scores, errors="coerce").reset_index(drop=True)
    defer = scores >= float(threshold)
    expert_columns = [expert for expert in ranked_experts if expert in working.columns]
    selected_route = pd.Series(["AI"] * len(working), dtype="object")
    selected_expert = pd.Series([""] * len(working), dtype="object")
    final_prediction = working["ai_pred"].astype(int).reset_index(drop=True).copy()
    for expert in expert_columns:
        available = defer & selected_expert.eq("") & working[expert].notna().reset_index(drop=True)
        final_prediction.loc[available] = working.loc[available, expert].astype(int).to_numpy()
        selected_expert.loc[available] = expert
        selected_route.loc[available] = "Human Expert"

    decision_reason = pd.Series([f"{method}_ai"] * len(working), dtype="object")
    decision_reason.loc[selected_route.eq("Human Expert")] = f"{method}_budget_defer"
    decision_reason.loc[defer & selected_route.eq("AI")] = f"{method}_expert_unavailable_ai_fallback"
    routed = working.copy()
    routed["deferral_score"] = scores
    routed["selected_route"] = selected_route
    routed["selected_expert"] = selected_expert
    routed["final_prediction"] = final_prediction.astype(int)
    routed["decision_reason"] = decision_reason
    routed["used_ai"] = routed["selected_route"].eq("AI")
    routed["is_correct"] = (routed["final_prediction"].astype(int) == routed["y_true"].astype(int)).astype(int)
    routed["ai_correct"] = (routed["ai_pred"].astype(int) == routed["y_true"].astype(int)).astype(int)
    routed["method"] = method
    return routed


def _metric_row(method: str, budget: float, threshold: float, decisions: pd.DataFrame, fp_cost: float, fn_cost: float) -> dict[str, Any]:
    fraud = compute_fraud_metrics(decisions, score_column="ai_score", fp_cost=fp_cost, fn_cost=fn_cost)
    deferral = compute_deferral_metrics(decisions, oracle_reference=None)
    # _load_paper_core -> _merge_common_signals already requires wrong_confident_label_offline
    # (raises if missing), so this was already correctly populated; require_wrong_confident_label=True
    # makes that pre-existing guarantee explicit rather than implicit.
    reliance = compute_reliance_metrics(decisions, require_wrong_confident_label=True)
    human_deferral = float(deferral["human_deferral_rate"] + deferral["escalation_rate"])
    return {
        "method": method,
        "budget": float(budget),
        "threshold": float(threshold),
        "recall": float(fraud["recall"]),
        "precision": float(fraud["precision"]),
        "f1": float(fraud["f1"]),
        "cost_loss": float(fraud["cost_sensitive_loss"]),
        "ai_coverage": float(deferral["ai_coverage"]),
        "human_deferral": human_deferral,
        "wca": float(reliance["wrong_confident_avoidance_rate"]),
        "correct_rejection": float(reliance["correct_rejection"]),
        "overreliance": float(reliance["overreliance"]),
    }


def _diagnostic_budget_row(method: str, budget: float, reason: str) -> dict[str, Any]:
    row = {column: np.nan for column in ["threshold", "recall", "precision", "f1", "cost_loss", "ai_coverage", "human_deferral", "wca", "correct_rejection", "overreliance"]}
    row.update({"method": method, "budget": float(budget), "diagnostic": reason})
    return row


def _learning_to_defer_scores(config_path: Path, val_core: pd.DataFrame, test_core: pd.DataFrame, fp_cost: float, fn_cost: float) -> tuple[pd.Series, pd.Series, str | None]:
    _ = (fp_cost, fn_cost)
    score_path = _outputs_root(config_path) / "baselines" / "learning_to_defer_budget_scores.csv"
    if not score_path.exists():
        return pd.Series(dtype=float), pd.Series(dtype=float), "L2D budget matching requires learning_to_defer_budget_scores.csv, but it is missing."
    scores = pd.read_csv(score_path)
    required = {"case_id", "split", "l2d_deferral_score"}
    missing = sorted(required - set(scores.columns))
    if missing:
        return pd.Series(dtype=float), pd.Series(dtype=float), f"L2D budget score artifact is missing columns: {missing}"
    if "diagnostic" in scores.columns:
        diagnostics = scores["diagnostic"].dropna().astype(str)
        diagnostics = diagnostics[diagnostics.str.len() > 0]
        if not diagnostics.empty and scores["l2d_deferral_score"].isna().all():
            return pd.Series(dtype=float), pd.Series(dtype=float), diagnostics.iloc[0]
    scores["case_id"] = scores["case_id"].astype(str)
    val_lookup = scores.loc[scores["split"].astype(str).eq("val")].set_index("case_id")["l2d_deferral_score"]
    test_lookup = scores.loc[scores["split"].astype(str).eq("test")].set_index("case_id")["l2d_deferral_score"]
    val_scores = val_core["case_id"].astype(str).map(val_lookup)
    test_scores = test_core["case_id"].astype(str).map(test_lookup)
    if val_scores.isna().all() or test_scores.isna().all():
        return pd.Series(dtype=float), pd.Series(dtype=float), "L2D budget score artifact does not align with validation/test case IDs."
    return val_scores.astype(float), test_scores.astype(float), None


def build_budget_matched_results(config_path: str | Path) -> pd.DataFrame:
    resolved = Path(config_path).resolve()
    config = load_yaml(resolved)
    fp_cost, fn_cost = _costs(config)
    val_core = _load_paper_core(resolved, "val")
    test_core = _load_paper_core(resolved, "test")
    ranked = _ranked_expert_columns(resolved, [val_core, test_core])
    val_core = _with_best_expert_prediction(val_core, ranked)
    test_core = _with_best_expert_prediction(test_core, ranked)
    ranked = ["_paper_best_expert_pred"]
    l2d_val, l2d_test, l2d_error = _learning_to_defer_scores(resolved, val_core, test_core, fp_cost, fn_cost)
    score_specs = {
        "MAGD-Fraud": (val_core["magd_assurance_risk"], test_core["magd_assurance_risk"], None),
        "Learning-to-defer": (l2d_val, l2d_test, l2d_error),
        "Distance threshold": (val_core["distance_uncertainty"], test_core["distance_uncertainty"], None),
        "Confidence threshold": (1.0 - val_core["numerical_confidence"].astype(float), 1.0 - test_core["numerical_confidence"].astype(float), None),
    }
    rows: list[dict[str, Any]] = []
    for method in BUDGET_METHODS:
        val_scores, test_scores, error = score_specs[method]
        for budget in BUDGETS:
            if error:
                rows.append(_diagnostic_budget_row(method, budget, error))
                continue
            try:
                threshold, validation_deferral = _select_threshold_for_budget(val_scores, budget)
                decisions = _route_by_risk(test_core, test_scores, threshold, ranked, method)
                row = _metric_row(method, budget, threshold, decisions, fp_cost, fn_cost)
                actual = float(row["human_deferral"])
                row["actual_budget_gap"] = abs(actual - float(budget))
                row["validation_deferral"] = float(validation_deferral)
                row["validation_budget_gap"] = abs(float(validation_deferral) - float(budget))
                row["coverage_sum"] = float(row["ai_coverage"] + row["human_deferral"])
                rows.append(row)
            except Exception as exc:
                rows.append(_diagnostic_budget_row(method, budget, str(exc)))
    return pd.DataFrame(rows)


def build_magd_risk_calibration(config_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = Path(config_path).resolve()
    config = load_yaml(resolved)
    outputs_root = _outputs_root(resolved)
    magd = _load_required_csv(outputs_root / "assurance" / "magd_risk.csv", {"case_id", "split", "magd_assurance_risk"}, "MAGD risk outputs")
    model = _load_required_csv(outputs_root / "model" / "test_predictions.csv", {"case_id", "y_true", "ai_pred"}, "test predictions")
    numerical = _load_required_csv(outputs_root / "assurance" / "numerical_confidence.csv", {"case_id", "split", "numerical_confidence"}, "numerical confidence outputs")
    tuned_path = outputs_root / "assurance_deferral" / "magd_validation_tuned_decisions.csv"
    tuned = _load_required_csv(tuned_path, {"case_id", "selected_route"}, "MAGD-Fraud validation-tuned decisions") if tuned_path.exists() else pd.DataFrame(columns=["case_id", "selected_route"])
    high_conf = float(config.get("assurance", {}).get("high_confidence_threshold", config.get("magd", {}).get("thresholds", {}).get("high_confidence", 0.8)))
    working = (
        _split_rows(magd, "test")[["case_id", "magd_assurance_risk"]]
        .merge(model[["case_id", "y_true", "ai_pred"]], on="case_id", how="inner")
        .merge(_split_rows(numerical, "test")[["case_id", "numerical_confidence"]], on="case_id", how="left")
        .merge(tuned[["case_id", "selected_route"]], on="case_id", how="left")
    )
    working = working.sort_values("magd_assurance_risk", kind="mergesort").reset_index(drop=True)
    risk_rank = working["magd_assurance_risk"].rank(method="first")
    working["risk_bin"] = pd.qcut(risk_rank, q=4, labels=RISK_BINS)
    working["ai_error_flag"] = (working["ai_pred"].astype(int) != working["y_true"].astype(int)).astype(int)
    working["wc_error_flag"] = (working["ai_error_flag"].eq(1) & working["numerical_confidence"].astype(float).ge(high_conf)).astype(int)
    working["deferral_flag"] = working["selected_route"].astype(str).isin(["Human Expert", "Escalate"]).astype(int)
    rows = []
    thresholds: dict[str, Any] = {"binning": "quartiles_over_test_magd_assurance_risk", "wrong_confident_confidence_threshold": high_conf, "bins": []}
    for label in RISK_BINS:
        bucket = working.loc[working["risk_bin"].astype(str).eq(label)]
        rows.append(
            {
                "risk_bin": label,
                "cases": int(len(bucket)),
                "fraud_prevalence": float(bucket["y_true"].astype(int).mean()) if len(bucket) else 0.0,
                "ai_error": float(bucket["ai_error_flag"].mean()) if len(bucket) else 0.0,
                "wc_error": float(bucket["wc_error_flag"].mean()) if len(bucket) else 0.0,
                "deferral": float(bucket["deferral_flag"].mean()) if len(bucket) else 0.0,
            }
        )
        thresholds["bins"].append(
            {
                "risk_bin": label,
                "min_inclusive": float(bucket["magd_assurance_risk"].min()) if len(bucket) else math.nan,
                "max_inclusive": float(bucket["magd_assurance_risk"].max()) if len(bucket) else math.nan,
                "cases": int(len(bucket)),
            }
        )
    return pd.DataFrame(rows), thresholds


def build_constraint_sensitivity(config_path: str | Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    resolved = Path(config_path).resolve()
    config = load_yaml(resolved)
    fp_cost, fn_cost = _costs(config)
    val_core = _load_paper_core(resolved, "val")
    test_core = _load_paper_core(resolved, "test")
    ranked = _ranked_expert_columns(resolved, [val_core, test_core])
    val_core = _with_best_expert_prediction(val_core, ranked)
    test_core = _with_best_expert_prediction(test_core, ranked)
    ranked = ["_paper_best_expert_pred"]
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    validation_risks = pd.to_numeric(val_core["magd_assurance_risk"], errors="coerce").dropna().astype(float)
    for setting in CONSTRAINT_SETTINGS:
        candidates: list[tuple[bool, float, float, float, dict[str, Any], pd.DataFrame]] = []
        threshold_candidates: set[float] = set()
        for budget in sorted({float(setting["deferral_budget"]), 0.01, 0.02, 0.05, 0.10, 0.20}):
            threshold, _ = _select_threshold_for_budget(validation_risks, budget)
            threshold_candidates.add(float(threshold))
        for quantile in np.linspace(0.75, 0.995, 26):
            threshold_candidates.add(float(validation_risks.quantile(float(quantile))))
        for threshold in sorted(threshold_candidates):
            val_decisions = _route_by_risk(val_core, val_core["magd_assurance_risk"], float(threshold), ranked, "MAGD-Fraud")
            metrics = _metric_row("MAGD-Fraud", float(setting["deferral_budget"]), float(threshold), val_decisions, fp_cost, fn_cost)
            feasible = (
                float(metrics["human_deferral"]) <= float(setting["deferral_budget"]) + 0.005
                and float(metrics["wca"]) >= float(setting["wca_target"])
                and float(metrics["overreliance"]) <= float(setting["overreliance_bound"])
            )
            budget_gap = abs(float(metrics["human_deferral"]) - float(setting["deferral_budget"]))
            violation = (
                max(0.0, float(setting["wca_target"]) - float(metrics["wca"]))
                + max(0.0, float(metrics["overreliance"]) - float(setting["overreliance_bound"]))
                + max(0.0, float(metrics["human_deferral"]) - float(setting["deferral_budget"]))
            )
            candidates.append((feasible, violation, budget_gap, float(metrics["cost_loss"]), metrics, val_decisions))
        if not candidates:
            rows.append({**setting, "feasible": False, "recall": np.nan, "precision": np.nan, "f1": np.nan, "cost_loss": np.nan, "wca": np.nan, "correct_rejection": np.nan, "overreliance": np.nan, "human_deferral": np.nan, "ai_coverage": np.nan})
            diagnostics.append({**setting, "optimizer_status": "failed", "reason": "No validation MAGD risk thresholds available."})
            continue
        candidates.sort(key=lambda item: (not item[0], item[1], item[2], item[3]))
        feasible, _, _, _, val_metrics, _ = candidates[0]
        threshold = float(val_metrics["threshold"])
        test_decisions = _route_by_risk(test_core, test_core["magd_assurance_risk"], threshold, ranked, "MAGD-Fraud")
        test_metrics = _metric_row("MAGD-Fraud", float(setting["deferral_budget"]), threshold, test_decisions, fp_cost, fn_cost)
        rows.append(
            {
                **setting,
                "feasible": bool(feasible),
                "recall": test_metrics["recall"],
                "precision": test_metrics["precision"],
                "f1": test_metrics["f1"],
                "cost_loss": test_metrics["cost_loss"],
                "wca": test_metrics["wca"],
                "correct_rejection": test_metrics["correct_rejection"],
                "overreliance": test_metrics["overreliance"],
                "human_deferral": test_metrics["human_deferral"],
                "ai_coverage": test_metrics["ai_coverage"],
            }
        )
        diagnostics.append(
            {
                **setting,
                "optimizer_status": "feasible" if feasible else "infeasible_best_available",
                "reason": "" if feasible else "No validation threshold satisfied all sensitivity constraints.",
                "selected_threshold": threshold,
                "actual_validation_deferral": float(val_metrics["human_deferral"]),
                "actual_test_deferral": float(test_metrics["human_deferral"]),
                "constraint_satisfaction": {
                    "validation_budget_satisfied": float(val_metrics["human_deferral"]) <= float(setting["deferral_budget"]) + 0.005,
                    "validation_wca_satisfied": float(val_metrics["wca"]) >= float(setting["wca_target"]),
                    "validation_overreliance_satisfied": float(val_metrics["overreliance"]) <= float(setting["overreliance_bound"]),
                    "test_budget_satisfied": float(test_metrics["human_deferral"]) <= float(setting["deferral_budget"]) + 0.005,
                    "test_wca_satisfied": float(test_metrics["wca"]) >= float(setting["wca_target"]),
                    "test_overreliance_satisfied": float(test_metrics["overreliance"]) <= float(setting["overreliance_bound"]),
                },
            }
        )
    return pd.DataFrame(rows), diagnostics


def _normalize_existing_decision(path: Path, outputs_root: Path, method: str) -> pd.DataFrame:
    frame = _load_required_csv(path, {"case_id", "y_true", "ai_pred", "final_prediction"}, f"{method} decisions")
    wrong = _load_required_csv(outputs_root / "assurance" / "wrong_confident_risk.csv", {"case_id", "wrong_confident_label_offline"}, "wrong-confident outputs")
    magd = _load_required_csv(outputs_root / "assurance" / "magd_risk.csv", {"case_id", "magd_assurance_risk"}, "MAGD risk outputs")
    numerical = _load_required_csv(outputs_root / "assurance" / "numerical_confidence.csv", {"case_id", "numerical_confidence"}, "numerical confidence outputs")
    if "selected_route" not in frame.columns:
        frame["selected_route"] = "AI"
    frame = (
        frame.merge(wrong[["case_id", "wrong_confident_label_offline"]].drop_duplicates("case_id"), on="case_id", how="left")
        .merge(magd[["case_id", "magd_assurance_risk"]].drop_duplicates("case_id"), on="case_id", how="left")
        .merge(numerical[["case_id", "numerical_confidence"]].drop_duplicates("case_id"), on="case_id", how="left")
    )
    if "ai_score" not in frame.columns:
        model = pd.read_csv(outputs_root / "model" / "test_predictions.csv")
        model["case_id"] = model["case_id"].astype(str)
        frame = frame.merge(model[["case_id", "ai_score"]], on="case_id", how="left")
    frame["used_ai"] = frame["selected_route"].astype(str).eq("AI")
    frame["is_correct"] = (frame["final_prediction"].astype(int) == frame["y_true"].astype(int)).astype(int)
    frame["ai_correct"] = (frame["ai_pred"].astype(int) == frame["y_true"].astype(int)).astype(int)
    frame["method"] = method
    return frame.sort_values("case_id").reset_index(drop=True)


def build_per_case_predictions(config_path: str | Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    resolved = Path(config_path).resolve()
    outputs_root = _outputs_root(resolved)
    required_method_paths = {
        "ai": outputs_root / "baselines" / "ai_only_decisions.csv",
        "magd_validation_tuned": outputs_root / "assurance_deferral" / "magd_validation_tuned_decisions.csv",
        "learning_to_defer": outputs_root / "baselines" / "learning_to_defer_decisions.csv",
    }
    # Legacy pre-MAGD comparator: not produced by any stage of the current MAGD pipeline,
    # so it is only included in per-case predictions (and downstream statistical
    # comparisons) if a decision log genuinely exists on disk.
    optional_method_paths = {
        "assurance_guided": outputs_root / "assurance_deferral" / "assurance_guided_decisions.csv",
    }
    normalized = {name: _normalize_existing_decision(path, outputs_root, name) for name, path in required_method_paths.items()}
    for name, path in optional_method_paths.items():
        if path.exists():
            normalized[name] = _normalize_existing_decision(path, outputs_root, name)

    base = normalized["ai"][["case_id", "y_true", "ai_pred", "numerical_confidence", "magd_assurance_risk"]].rename(
        columns={"numerical_confidence": "ai_confidence", "magd_assurance_risk": "magd_risk"}
    )
    base = base.merge(normalized["magd_validation_tuned"][["case_id", "final_prediction", "selected_route"]].rename(columns={"final_prediction": "magd_validation_tuned_pred", "selected_route": "magd_route"}), on="case_id", how="inner")
    if "assurance_guided" in normalized:
        base = base.merge(normalized["assurance_guided"][["case_id", "final_prediction"]].rename(columns={"final_prediction": "assurance_guided_pred"}), on="case_id", how="inner")
    base = base.merge(normalized["learning_to_defer"][["case_id", "final_prediction", "selected_route"]].rename(columns={"final_prediction": "learning_to_defer_pred", "selected_route": "l2d_route"}), on="case_id", how="inner")
    return base.sort_values("case_id").reset_index(drop=True), normalized


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _cost(y_true: np.ndarray, y_pred: np.ndarray, fp_cost: float, fn_cost: float) -> float:
    return float(fp_cost * np.sum((y_true == 0) & (y_pred == 1)) + fn_cost * np.sum((y_true == 1) & (y_pred == 0)))


def build_statistical_tests(config_path: str | Path, per_case: pd.DataFrame) -> pd.DataFrame:
    resolved = Path(config_path).resolve()
    config = load_yaml(resolved)
    fp_cost, fn_cost = _costs(config)
    stats_cfg = config.get("statistics", {})
    n_bootstrap = int(stats_cfg.get("bootstrap_iterations", 1000))
    alpha = 1.0 - float(stats_cfg.get("confidence_level", 0.95))
    rng = np.random.default_rng(int(config.get("experiment", {}).get("seed", 42)))
    y_true = per_case["y_true"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for comparison, magd_col, baseline_col in STAT_COMPARISONS:
        if magd_col not in per_case.columns or baseline_col not in per_case.columns:
            # e.g. the legacy assurance-guided comparator's decision log does not exist
            # for this run; skip rather than fabricate or crash.
            continue
        magd_pred = per_case[magd_col].astype(int).to_numpy()
        baseline_pred = per_case[baseline_col].astype(int).to_numpy()
        delta_f1_samples = np.empty(n_bootstrap, dtype=float)
        delta_cost_samples = np.empty(n_bootstrap, dtype=float)
        n = len(per_case)
        for idx in range(n_bootstrap):
            sample = rng.integers(0, n, size=n)
            delta_f1_samples[idx] = _f1(y_true[sample], magd_pred[sample]) - _f1(y_true[sample], baseline_pred[sample])
            delta_cost_samples[idx] = _cost(y_true[sample], magd_pred[sample], fp_cost, fn_cost) - _cost(y_true[sample], baseline_pred[sample], fp_cost, fn_cost)
        mcnemar = mcnemar_paired_correctness((magd_pred == y_true).astype(int), (baseline_pred == y_true).astype(int))
        rows.append(
            {
                "comparison": comparison,
                "delta_f1": _f1(y_true, magd_pred) - _f1(y_true, baseline_pred),
                "delta_f1_ci_low": float(np.quantile(delta_f1_samples, alpha / 2.0)),
                "delta_f1_ci_high": float(np.quantile(delta_f1_samples, 1.0 - alpha / 2.0)),
                "delta_cost": _cost(y_true, magd_pred, fp_cost, fn_cost) - _cost(y_true, baseline_pred, fp_cost, fn_cost),
                "delta_cost_ci_low": float(np.quantile(delta_cost_samples, alpha / 2.0)),
                "delta_cost_ci_high": float(np.quantile(delta_cost_samples, 1.0 - alpha / 2.0)),
                "mcnemar_p": float(mcnemar["p_value"]),
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def _artifact_table_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"paper_table_or_figure": "budget-matched table/figure", "source_csv": "budget_matched_results.csv", "description": "Budget-matched deferral comparison across MAGD-Fraud and baseline routing scores."},
            {"paper_table_or_figure": "MAGD risk calibration table/figure", "source_csv": "magd_risk_calibration.csv", "description": "Held-out test MAGD risk quartiles with fraud prevalence, AI error, wrong-confident error, and MAGD deferral."},
            {"paper_table_or_figure": "constraint sensitivity table/figure", "source_csv": "constraint_sensitivity.csv", "description": "Validation-selected MAGD risk thresholds under strict, moderate, relaxed, and high-review constraints."},
            {"paper_table_or_figure": "statistical comparison table", "source_csv": "statistical_tests.csv", "description": "Paired bootstrap confidence intervals and McNemar tests over held-out test cases."},
        ]
    )


def run_paper_evaluation_outputs(config_path: str | Path) -> PaperEvaluationArtifacts:
    resolved = Path(config_path).resolve()
    outputs_root = _outputs_root(resolved)
    paper_dir = outputs_root / "paper_tables"
    final_dir = outputs_root / "final_metrics"
    policy_dir = outputs_root / "magd_policy"
    paper_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)

    budget = build_budget_matched_results(resolved)
    calibration, calibration_thresholds = build_magd_risk_calibration(resolved)
    sensitivity, sensitivity_diagnostics = build_constraint_sensitivity(resolved)
    per_case, _ = build_per_case_predictions(resolved)
    stats = build_statistical_tests(resolved, per_case)
    artifact_map = _artifact_table_map()

    budget.to_csv(paper_dir / "budget_matched_results.csv", index=False)
    calibration.to_csv(paper_dir / "magd_risk_calibration.csv", index=False)
    (paper_dir / "magd_risk_calibration_thresholds.json").write_text(json.dumps(calibration_thresholds, indent=2), encoding="utf-8")
    sensitivity.to_csv(paper_dir / "constraint_sensitivity.csv", index=False)
    (policy_dir / "constraint_sensitivity_diagnostics.json").write_text(json.dumps(sensitivity_diagnostics, indent=2), encoding="utf-8")
    stats.to_csv(paper_dir / "statistical_tests.csv", index=False)
    per_case.to_csv(final_dir / "per_case_predictions.csv", index=False)
    artifact_map.to_csv(paper_dir / "artifact_table_map.csv", index=False)

    return PaperEvaluationArtifacts(
        paper_tables_dir=paper_dir,
        budget_matched_results=budget,
        magd_risk_calibration=calibration,
        constraint_sensitivity=sensitivity,
        statistical_tests=stats,
        per_case_predictions=per_case,
    )
