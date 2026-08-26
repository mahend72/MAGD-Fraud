from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_fifar import load_fifar_data, read_table, resolve_config_path
from src.deferral.baselines import _build_capacity_state, _candidate_expert_columns, _historical_best_experts
from src.deferral.capacity_assignment import CapacityState
from src.deferral.expert_routing import (
    _ai_expected_cost,
    _assign_expert_candidate,
    _best_available_expert_candidate,
    _escalation_majority_vote,
    _load_expert_tables,
    _resolve_output_dir,
    _routing_config,
    estimate_expert_reliability,
    _remaining_capacity_summary,
)
from src.deferral.magd_policy import load_policy_diagnostics, load_policy_weights, resolve_magd_thresholds_for_config
from src.utils.io import load_yaml


@dataclass
class MagdDeferralArtifacts:
    decisions: pd.DataFrame
    metrics: pd.DataFrame
    output_dir: Path


def _filter_split(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if "split" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["split"].astype(str) == split_name].copy()


def _load_final_inputs(config_path: Path) -> pd.DataFrame:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = (config_path.parent / processed_dir).resolve()

    model_test = pd.read_csv(outputs_root / "model" / "test_predictions.csv")
    numerical_confidence = pd.read_csv(outputs_root / "assurance" / "numerical_confidence.csv")
    calibration_risk = pd.read_csv(outputs_root / "assurance" / "calibration_risk.csv")
    distance = pd.read_csv(outputs_root / "assurance" / "distance_uncertainty.csv")
    local_reliability = pd.read_csv(outputs_root / "assurance" / "local_reliability.csv")
    wrong_conf = pd.read_csv(outputs_root / "assurance" / "wrong_confident_risk.csv")
    magd = pd.read_csv(outputs_root / "assurance" / "magd_risk.csv")
    test_metadata = pd.read_csv(processed_dir / "test_metadata.csv")

    for frame in [model_test, numerical_confidence, calibration_risk, distance, local_reliability, wrong_conf, magd, test_metadata]:
        if "case_id" in frame.columns:
            frame["case_id"] = frame["case_id"].astype(str)

    model_test = _filter_split(model_test, "test")
    numerical_confidence = _filter_split(numerical_confidence, "test")
    calibration_risk = _filter_split(calibration_risk, "test")
    distance = _filter_split(distance, "test")
    local_reliability = _filter_split(local_reliability, "test")
    wrong_conf = _filter_split(wrong_conf, "test")
    magd = _filter_split(magd, "test")

    return (
        model_test.merge(
            numerical_confidence[["case_id", "numerical_confidence"]],
            on="case_id",
            how="inner",
        )
        .merge(
            calibration_risk[["case_id", "calibration_risk"]],
            on="case_id",
            how="inner",
        )
        .merge(
            distance[["case_id", "distance_confidence", "distance_uncertainty"]],
            on="case_id",
            how="inner",
        )
        .merge(
            local_reliability[
                [
                    "case_id",
                    "neighbor_error_rate",
                    "neighbor_fraud_rate",
                    "neighbor_ai_agreement",
                    "mean_neighbor_distance",
                    "top_k",
                ]
            ],
            on="case_id",
            how="inner",
        )
        .merge(
            wrong_conf[["case_id", "wrong_confident_risk"]],
            on="case_id",
            how="inner",
        )
        .merge(
            magd[
                [
                    "case_id",
                    "drift_risk",
                    "business_risk",
                    "magd_assurance_risk",
                    "risk_category",
                ]
            ],
            on="case_id",
            how="inner",
        )
        .merge(
            test_metadata,
            on="case_id",
            how="left",
        )
    )


def _build_expert_reliability(config_path: Path) -> tuple[pd.DataFrame, list[str]]:
    loaded = load_fifar_data(config_path)
    _, historical_df = _load_expert_tables(config_path)
    capacity_by_expert, capacity_totals = _remaining_capacity_summary(config_path)

    expert_metrics = pd.DataFrame()
    ranked_experts: list[str] = []
    if historical_df is None:
        return expert_metrics, ranked_experts

    train_label_column = loaded.config.get("columns", {}).get("train_label") or loaded.config.get("columns", {}).get("label")
    train_file = resolve_config_path(
        loaded.config_path,
        loaded.config.get("dataset", {}).get("train_file"),
        required=False,
        field_name="dataset.train_file",
    )
    if train_file is None:
        return expert_metrics, ranked_experts

    train_df = read_table(train_file)
    case_col = loaded.case_id_column
    label_frame = train_df[[case_col, train_label_column] + loaded.sensitive_attributes].copy()
    label_frame[case_col] = label_frame[case_col].astype(str)
    historical_case_col = case_col if case_col in historical_df.columns else historical_df.columns[0]
    hist = historical_df.copy()
    hist[historical_case_col] = hist[historical_case_col].astype(str)
    merged = label_frame.merge(hist, left_on=case_col, right_on=historical_case_col, how="inner")
    if len(merged) < max(100, int(0.01 * min(len(label_frame), len(hist)))):
        n_rows = min(len(label_frame), len(hist))
        merged = pd.concat(
            [
                label_frame.reset_index(drop=True).iloc[:n_rows],
                hist.drop(columns=[historical_case_col], errors="ignore").reset_index(drop=True).iloc[:n_rows],
            ],
            axis=1,
        )

    cfg = _routing_config(load_yaml(config_path))
    expert_metrics = estimate_expert_reliability(
        merged,
        expert_columns=[column for column in _candidate_expert_columns(hist) if column in merged.columns],
        label_column=train_label_column,
        sensitive_attributes=loaded.sensitive_attributes,
        fp_cost=cfg["fp_cost"],
        fn_cost=cfg["fn_cost"],
        capacity_by_expert=capacity_by_expert,
        capacity_totals=capacity_totals,
    )
    ranked_experts = _historical_best_experts(config_path, historical_df)
    return expert_metrics, ranked_experts


def route_magd_cases(
    core: pd.DataFrame,
    *,
    ranked_experts: list[str],
    expert_metrics: pd.DataFrame,
    capacity_state: CapacityState,
    fp_cost: float,
    fn_cost: float,
    fairness_penalty_weight: float,
    capacity_penalty_weight: float,
    human_review_cost: float,
    escalation_cost: float,
    top_k: int,
    wrong_confident_threshold: float,
    ai_cost_margin: float = 0.01,
    ai_cost_scale: float = 1.0,
) -> pd.DataFrame:
    expert_lookup = expert_metrics.set_index("expert").to_dict(orient="index") if not expert_metrics.empty else {}
    decisions: list[dict[str, object]] = []

    for _, row in core.iterrows():
        ai_expected_cost = _ai_expected_cost(row, fp_cost, fn_cost) * float(ai_cost_scale)
        passes_threshold = bool(row.get("passes_threshold", False))
        risk_category = str(row.get("risk_category", ""))
        wrong_conf_high = float(row["wrong_confident_risk"]) >= wrong_confident_threshold

        cand_name, cand_pred, cand_cost, cand_fairness = _best_available_expert_candidate(
            row,
            ranked_experts,
            expert_lookup,
            capacity_state,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            fairness_penalty_weight=fairness_penalty_weight,
            capacity_penalty_weight=capacity_penalty_weight,
            human_review_cost=human_review_cost,
        )

        selected_route = "AI"
        selected_expert = ""
        final_prediction = int(row["ai_pred"])
        best_expert_expected_cost = cand_cost
        escalation_expected_cost = np.nan
        capacity_status = "not_applicable"
        fairness_penalty = 0.0
        decision_reason = "low_risk_ai_allowed"

        if risk_category == "high" or wrong_conf_high:
            escalation_pred, escalation_experts, escalation_cost_value, escalation_fairness_penalty, escalation_status = _escalation_majority_vote(
                row,
                ranked_experts,
                expert_lookup,
                capacity_state,
                top_k=top_k,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
                fairness_penalty_weight=fairness_penalty_weight,
                capacity_penalty_weight=capacity_penalty_weight,
                escalation_cost=escalation_cost,
            )
            escalation_expected_cost = float(escalation_cost_value) if escalation_cost_value is not None else np.nan
            if escalation_pred is not None:
                selected_route = "Escalate"
                selected_expert = "|".join(escalation_experts)
                final_prediction = int(escalation_pred)
                capacity_status = escalation_status
                fairness_penalty = float(escalation_fairness_penalty)
                decision_reason = (
                    "high_wrong_confident_risk_escalate" if wrong_conf_high and risk_category != "high" else "high_assurance_risk_escalate"
                )
            else:
                assigned_name, assigned_pred, assigned_cost, assigned_fairness, assigned_status = _assign_expert_candidate(
                    row,
                    cand_name,
                    cand_pred,
                    cand_cost,
                    cand_fairness,
                    capacity_state,
                )
                if assigned_name is not None and assigned_pred is not None:
                    selected_route = "Human Expert"
                    selected_expert = assigned_name
                    final_prediction = int(assigned_pred)
                    best_expert_expected_cost = assigned_cost
                    capacity_status = assigned_status
                    fairness_penalty = float(assigned_fairness)
                    decision_reason = "capacity_limit_use_next_best_expert"
                else:
                    capacity_status = assigned_status
                    decision_reason = "no_expert_available_ai_fallback"
        elif risk_category == "low" and passes_threshold:
            selected_route = "AI"
            final_prediction = int(row["ai_pred"])
            capacity_status = "threshold_passed"
            decision_reason = "low_risk_ai_allowed"
        else:
            ai_clearly_lower = cand_cost is None or ai_expected_cost + float(ai_cost_margin) < float(cand_cost)
            if risk_category == "medium" and ai_clearly_lower:
                selected_route = "AI"
                final_prediction = int(row["ai_pred"])
                capacity_status = "ai_lower_expected_cost"
                decision_reason = "medium_risk_ai_lower_expected_cost"
                decisions.append(
                    {
                        "case_id": str(row["case_id"]),
                        "ai_score": float(row["ai_score"]),
                        "ai_pred": int(row["ai_pred"]),
                        "numerical_confidence": float(row["numerical_confidence"]),
                        "distance_confidence": float(row["distance_confidence"]),
                        "distance_uncertainty": float(row["distance_uncertainty"]),
                        "calibration_risk": float(row["calibration_risk"]),
                        "neighbor_error_rate": float(row["neighbor_error_rate"]),
                        "neighbor_fraud_rate": float(row["neighbor_fraud_rate"]),
                        "neighbor_ai_agreement": float(row["neighbor_ai_agreement"]),
                        "mean_neighbor_distance": float(row["mean_neighbor_distance"]),
                        "wrong_confident_risk": float(row["wrong_confident_risk"]),
                        "magd_assurance_risk": float(row["magd_assurance_risk"]),
                        "risk_category": risk_category,
                        "base_threshold": float(row["base_threshold"]),
                        "business_risk": float(row.get("business_risk", 0.0)),
                        "fairness_risk": float(row.get("fairness_risk", 0.0)),
                        "capacity_pressure": float(row.get("capacity_pressure", 0.0)),
                        "adaptive_threshold": float(row["adaptive_threshold"]),
                        "passes_threshold": bool(passes_threshold),
                        "selected_route": selected_route,
                        "selected_expert": selected_expert,
                        "final_prediction": int(final_prediction),
                        "ai_expected_cost": float(ai_expected_cost),
                        "best_expert_expected_cost": float(best_expert_expected_cost) if best_expert_expected_cost is not None else np.nan,
                        "escalation_expected_cost": float(escalation_expected_cost) if pd.notna(escalation_expected_cost) else np.nan,
                        "capacity_status": capacity_status,
                        "fairness_penalty": float(fairness_penalty),
                        "decision_reason": decision_reason,
                    }
                )
                continue
            assigned_name, assigned_pred, assigned_cost, assigned_fairness, assigned_status = _assign_expert_candidate(
                row,
                cand_name,
                cand_pred,
                cand_cost,
                cand_fairness,
                capacity_state,
            )
            if assigned_name is not None and assigned_pred is not None:
                selected_route = "Human Expert"
                selected_expert = assigned_name
                final_prediction = int(assigned_pred)
                best_expert_expected_cost = assigned_cost
                capacity_status = assigned_status
                fairness_penalty = float(assigned_fairness)
                if assigned_fairness > 0.0 and risk_category == "medium":
                    decision_reason = "fairness_risk_avoid_biased_expert"
                elif float(row.get("neighbor_error_rate", 0.0)) >= 0.5:
                    decision_reason = "similar_cases_often_misclassified"
                elif float(row.get("calibration_risk", 0.0)) >= 0.1:
                    decision_reason = "confidence_not_reliable"
                else:
                    decision_reason = "medium_risk_defer_to_best_expert" if risk_category == "medium" else "expert_lower_expected_cost"
            else:
                selected_route = "AI"
                final_prediction = int(row["ai_pred"])
                capacity_status = assigned_status
                decision_reason = "no_expert_available_ai_fallback"

        decisions.append(
            {
                "case_id": str(row["case_id"]),
                "ai_score": float(row["ai_score"]),
                "ai_pred": int(row["ai_pred"]),
                "numerical_confidence": float(row["numerical_confidence"]),
                "distance_confidence": float(row["distance_confidence"]),
                "distance_uncertainty": float(row["distance_uncertainty"]),
                "calibration_risk": float(row["calibration_risk"]),
                "neighbor_error_rate": float(row["neighbor_error_rate"]),
                "neighbor_fraud_rate": float(row["neighbor_fraud_rate"]),
                "neighbor_ai_agreement": float(row["neighbor_ai_agreement"]),
                "mean_neighbor_distance": float(row["mean_neighbor_distance"]),
                "wrong_confident_risk": float(row["wrong_confident_risk"]),
                "magd_assurance_risk": float(row["magd_assurance_risk"]),
                "risk_category": risk_category,
                "base_threshold": float(row["base_threshold"]),
                "business_risk": float(row.get("business_risk", 0.0)),
                "fairness_risk": float(row.get("fairness_risk", 0.0)),
                "capacity_pressure": float(row.get("capacity_pressure", 0.0)),
                "adaptive_threshold": float(row["adaptive_threshold"]),
                "passes_threshold": bool(passes_threshold),
                "selected_route": selected_route,
                "selected_expert": selected_expert,
                "final_prediction": int(final_prediction),
                "ai_expected_cost": float(ai_expected_cost),
                "best_expert_expected_cost": float(best_expert_expected_cost) if best_expert_expected_cost is not None else np.nan,
                "escalation_expected_cost": float(escalation_expected_cost) if pd.notna(escalation_expected_cost) else np.nan,
                "capacity_status": capacity_status,
                "fairness_penalty": float(fairness_penalty),
                "decision_reason": decision_reason,
            }
        )

    return pd.DataFrame(decisions)


def _metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    route_counts = decisions["selected_route"].value_counts(normalize=True)
    return pd.DataFrame(
        [
            {
                "total_cases": float(len(decisions)),
                "ai_route_rate": float(route_counts.get("AI", 0.0)),
                "expert_route_rate": float(route_counts.get("Human Expert", 0.0)),
                "escalate_route_rate": float(route_counts.get("Escalate", 0.0)),
                "mean_ai_expected_cost": float(decisions["ai_expected_cost"].mean()),
                "mean_best_expert_expected_cost": float(decisions["best_expert_expected_cost"].dropna().mean()) if decisions["best_expert_expected_cost"].notna().any() else np.nan,
                "mean_escalation_expected_cost": float(decisions["escalation_expected_cost"].dropna().mean()) if decisions["escalation_expected_cost"].notna().any() else np.nan,
                "threshold_pass_rate": float(decisions["passes_threshold"].mean()),
            }
        ]
    )


def _decision_output_name(mode: str) -> str:
    if mode == "heuristic":
        return "magd_heuristic_decisions.csv"
    if mode == "learned":
        return "magd_learned_decisions.csv"
    return f"magd_{mode}_decisions.csv"


def _write_metrics_snapshot(output_dir: Path, metrics: pd.DataFrame, mode: str) -> None:
    metrics_path = output_dir / "magd_metrics.csv"
    current = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    updated = pd.concat(
        [
            current.loc[current.get("method", pd.Series(dtype=str)).astype(str) != mode] if not current.empty and "method" in current.columns else current.iloc[0:0],
            metrics,
        ],
        ignore_index=True,
    )
    updated.to_csv(metrics_path, index=False)


def run_magd_deferral(
    config_path: str | Path,
    *,
    policy_variant: str | None = None,
    mode: str | None = None,
    output_stem: str | None = None,
) -> MagdDeferralArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    cfg = _routing_config(config)
    output_dir = _resolve_output_dir(resolved_config_path)

    core = _load_final_inputs(resolved_config_path)
    selected_variant = (mode or policy_variant or str(config.get("magd", {}).get("mode", "heuristic"))).lower()
    weights = load_policy_weights(resolved_config_path, variant=selected_variant)
    policy_diagnostics = load_policy_diagnostics(resolved_config_path).get(selected_variant, {})
    weight_columns = [
        "distance_uncertainty",
        "calibration_risk",
        "neighbor_error_rate",
        "wrong_confident_risk",
        "drift_risk",
        "business_risk",
    ]
    existing_weight_columns = [column for column in weight_columns if column in core.columns]
    resolved_low_risk, resolved_high_risk = resolve_magd_thresholds_for_config(resolved_config_path, config)
    if existing_weight_columns:
        from src.assurance.magd_risk import add_risk_categories_and_actions, compute_magd_risk
        from src.deferral.adaptive_threshold import compute_adaptive_thresholds

        rescored = compute_magd_risk(
            core,
            weights,
            use_drift_risk=bool(config["magd"]["signals"].get("use_drift_risk", False)),
            use_business_risk=bool(config["magd"]["signals"].get("use_business_risk", False)),
        )
        rescored = add_risk_categories_and_actions(
            rescored,
            low_risk=resolved_low_risk,
            high_risk=resolved_high_risk,
        )
        adaptive = compute_adaptive_thresholds(
            rescored,
            base_threshold=float(config["magd"]["adaptive_threshold"]["base_threshold"]),
            value_weight=float(config["magd"]["adaptive_threshold"]["value_weight"]),
            fairness_weight=float(config["magd"]["adaptive_threshold"]["fairness_weight"]),
            capacity_weight=float(config["magd"]["adaptive_threshold"]["capacity_weight"]),
        )
        adaptive_columns = [
            "base_threshold",
            "business_risk",
            "fairness_risk",
            "capacity_pressure",
            "adaptive_threshold",
            "passes_threshold",
        ]
        core = rescored.drop(columns=[column for column in adaptive_columns if column in rescored.columns], errors="ignore").merge(
            adaptive[["case_id", *adaptive_columns]],
            on="case_id",
            how="left",
        )
    expert_df, _ = _load_expert_tables(resolved_config_path)
    if expert_df is not None:
        expert_df = expert_df.rename(columns={expert_df.columns[0]: "case_id"})
        expert_df["case_id"] = expert_df["case_id"].astype(str)
        core = core.merge(expert_df, on="case_id", how="left")

    expert_metrics, ranked_experts = _build_expert_reliability(resolved_config_path)
    if expert_df is not None:
        available_columns = set(expert_df.columns)
        ranked_experts = [expert for expert in ranked_experts if expert in available_columns]

    capacity_remaining, batch_column = _build_capacity_state(core, resolved_config_path)
    capacity_state = CapacityState(remaining=capacity_remaining, batch_column=batch_column)

    wrong_confident_threshold = float(
        config.get("magd", {}).get("thresholds", {}).get(
            "wrong_confident",
            config.get("assurance", {}).get("wrong_confident", {}).get("risk_alert_threshold", 0.7),
        )
    )
    decisions = route_magd_cases(
        core,
        ranked_experts=ranked_experts,
        expert_metrics=expert_metrics,
        capacity_state=capacity_state,
        fp_cost=cfg["fp_cost"],
        fn_cost=cfg["fn_cost"],
        fairness_penalty_weight=cfg["fairness_penalty_weight"],
        capacity_penalty_weight=cfg["capacity_penalty_weight"],
        human_review_cost=cfg["human_review_cost"],
        escalation_cost=cfg["escalation_cost"],
        top_k=cfg["top_k"],
        wrong_confident_threshold=wrong_confident_threshold,
        ai_cost_margin=float(config.get("magd", {}).get("ai_cost_margin", 0.01)),
    )
    decisions = decisions.merge(core[["case_id", "y_true"]], on="case_id", how="left")
    decisions["is_correct"] = (decisions["final_prediction"].astype(int) == decisions["y_true"].astype(int)).astype(int)
    decisions["policy_variant"] = selected_variant
    decisions["method"] = f"magd_{selected_variant}"
    decisions["policy_version"] = f"magd_{selected_variant}:{policy_diagnostics.get('optimizer', 'config')}"
    decisions["low_risk_threshold"] = float(resolved_low_risk)
    decisions["high_risk_threshold"] = float(resolved_high_risk)
    decisions["policy_weights_json"] = json.dumps(weights, sort_keys=True)
    metrics = _metrics(decisions)
    metrics["policy_variant"] = selected_variant
    metrics["method"] = f"magd_{selected_variant}"
    metrics["optimizer"] = str(policy_diagnostics.get("optimizer", "config"))
    metrics["used_validation_only"] = bool(policy_diagnostics.get("used_validation_only", selected_variant != "heuristic"))

    decision_name = output_stem if output_stem is not None else _decision_output_name(selected_variant)
    decisions.to_csv(output_dir / decision_name, index=False)
    _write_metrics_snapshot(output_dir, metrics, f"magd_{selected_variant}")
    return MagdDeferralArtifacts(decisions=decisions, metrics=metrics, output_dir=output_dir)
