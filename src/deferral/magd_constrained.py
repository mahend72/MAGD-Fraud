from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.assurance.magd_risk import add_risk_categories_and_actions, compute_magd_risk
from src.deferral.magd_policy import resolve_magd_thresholds_for_config
from src.data.load_fifar import load_fifar_data
from src.deferral.adaptive_threshold import compute_adaptive_thresholds
from src.deferral.baselines import _build_capacity_state
from src.deferral.capacity_assignment import CapacityState
from src.deferral.expert_routing import _load_expert_tables, _resolve_output_dir, _routing_config
from src.deferral.magd_deferral import _build_expert_reliability, _load_final_inputs, route_magd_cases
from src.deferral.magd_policy import load_policy_weights, load_validation_policy_frame
from src.evaluation.assurance_metrics import audit_coverage_rate
from src.evaluation.deferral_metrics import compute_deferral_metrics
from src.evaluation.fairness_metrics import compute_fairness_metrics
from src.evaluation.fraud_metrics import compute_fraud_metrics
from src.evaluation.reliance_metrics import compute_reliance_metrics
from src.utils.io import load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

REQUIRED_AUDIT_COLUMNS = [
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

CONSTRAINT_DIAGNOSTIC_COLUMNS = [
    "audit_coverage_satisfied",
    "feasible",
    "overreliance_satisfied",
    "wrong_confident_avoidance_satisfied",
]

CONSTRAINT_FAILURE_DIAGNOSTICS = {column: False for column in CONSTRAINT_DIAGNOSTIC_COLUMNS}


@dataclass
class MagdConstrainedArtifacts:
    initial_decisions: pd.DataFrame
    calibrated_decisions: pd.DataFrame
    results: pd.DataFrame
    output_dir: Path
    policy_dir: Path
    diagnostics: dict[str, object]


def ensure_constraint_diagnostics(
    frame: pd.DataFrame,
    defaults: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Ensure constrained MAGD result tables expose required guardrail diagnostics."""
    enriched = frame.copy()
    resolved_defaults = {**CONSTRAINT_FAILURE_DIAGNOSTICS, **(defaults or {})}
    for column in CONSTRAINT_DIAGNOSTIC_COLUMNS:
        default_value = bool(resolved_defaults.get(column, False))
        if column not in enriched.columns:
            enriched[column] = default_value
        else:
            enriched[column] = enriched[column].fillna(default_value).astype(bool)
    return enriched


def _resolve_policy_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    policy_dir = outputs_root / "magd_policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    return policy_dir


def _resolve_paper_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    paper_dir = outputs_root / "paper_tables"
    paper_dir.mkdir(parents=True, exist_ok=True)
    return paper_dir


def _constraint_config(config: dict) -> dict[str, float]:
    constrained_cfg = config["magd"]["constrained_policy"]
    return {
        "max_overreliance": float(constrained_cfg["max_overreliance"]),
        "min_wrong_confident_avoidance": float(constrained_cfg["min_wrong_confident_avoidance"]),
        "min_deferral_rate": float(constrained_cfg["min_deferral_rate"]),
        "max_deferral_rate": float(constrained_cfg["max_deferral_rate"]),
        "min_audit_coverage": float(constrained_cfg["min_audit_coverage"]),
        "lambda_overreliance": float(constrained_cfg["lambda_overreliance"]),
        "lambda_capacity": float(constrained_cfg["lambda_capacity"]),
        "lambda_fairness": float(constrained_cfg["lambda_fairness"]),
        "lambda_audit": float(constrained_cfg["lambda_audit"]),
    }


def _initial_policy_params(config: dict, config_path: Path) -> dict[str, float]:
    constraints = _constraint_config(config)
    min_deferral = constraints["min_deferral_rate"]
    max_deferral = constraints["max_deferral_rate"]
    # Start the constrained-policy search from the validation-derived MAGD risk thresholds
    # (Fix A), not the old fixed 0.35/0.70 defaults which sat above the realized risk range
    # and made medium/high risk categories unreachable regardless of what the optimizer did.
    resolved_low_risk, resolved_high_risk = resolve_magd_thresholds_for_config(config_path, config)
    return {
        "low_risk": resolved_low_risk,
        "high_risk": resolved_high_risk,
        "magd_high_threshold": resolved_high_risk,
        "wrong_confident_threshold": float(config["magd"]["thresholds"]["wrong_confident"]),
        "base_threshold": float(config["magd"]["adaptive_threshold"]["base_threshold"]),
        "ai_cost_scale": 1.0,
        "ai_cost_margin": float(config.get("magd", {}).get("ai_cost_margin", 0.01)),
        "deferral_target": (min_deferral + max_deferral) / 2.0,
    }


def _fairness_penalty(frame: pd.DataFrame, sensitive_attributes: list[str], fp_cost: float, fn_cost: float) -> float:
    fairness = compute_fairness_metrics(frame, sensitive_attributes, fp_cost=fp_cost, fn_cost=fn_cost)
    if fairness.empty:
        return 0.0
    penalties: list[float] = []
    for _, group in fairness.groupby("sensitive_column", dropna=False):
        if "group_cost" in group.columns:
            penalties.append(float(group["group_cost"].max() - group["group_cost"].min()))
        if "false_positive_rate" in group.columns:
            penalties.append(float(group["false_positive_rate"].max() - group["false_positive_rate"].min()))
        if "false_negative_rate" in group.columns:
            penalties.append(float(group["false_negative_rate"].max() - group["false_negative_rate"].min()))
    return max(penalties) if penalties else 0.0


def _prepare_test_core(config_path: Path) -> pd.DataFrame:
    core = _load_final_inputs(config_path)
    expert_df, _ = _load_expert_tables(config_path)
    if expert_df is not None:
        expert_df = expert_df.rename(columns={expert_df.columns[0]: "case_id"})
        expert_df["case_id"] = expert_df["case_id"].astype(str)
        core = core.merge(expert_df, on="case_id", how="left")
    return core


def _prepare_validation_core(config_path: Path) -> pd.DataFrame:
    return load_validation_policy_frame(config_path, max_rows=250, max_train_rows=10000)


def _apply_policy_to_core(
    *,
    core: pd.DataFrame,
    config_path: Path,
    weights: dict[str, float],
    params: dict[str, float],
    ranked_experts: list[str],
    expert_metrics: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    config = load_yaml(config_path)
    cfg = _routing_config(config)
    rescored = compute_magd_risk(
        core,
        weights,
        use_drift_risk=bool(config["magd"]["signals"].get("use_drift_risk", False)),
        use_business_risk=bool(config["magd"]["signals"].get("use_business_risk", False)),
    )
    rescored = add_risk_categories_and_actions(
        rescored,
        low_risk=float(params["low_risk"]),
        high_risk=float(params["high_risk"]),
    )
    forced_high = pd.to_numeric(rescored["magd_assurance_risk"], errors="coerce").fillna(0.0) >= float(params["magd_high_threshold"])
    rescored.loc[forced_high, "risk_category"] = "high"
    if "fairness_risk" not in rescored.columns:
        rescored["fairness_risk"] = 0.0
    if "capacity_pressure" not in rescored.columns:
        rescored["capacity_pressure"] = 0.0

    adaptive = compute_adaptive_thresholds(
        rescored,
        base_threshold=float(params["base_threshold"]),
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
    routed_core = rescored.drop(columns=[column for column in adaptive_columns if column in rescored.columns], errors="ignore").merge(
        adaptive[["case_id", *adaptive_columns]],
        on="case_id",
        how="left",
    )

    if not bool(config.get("magd", {}).get("expert_routing", {}).get("capacity_enabled", False)):
        capacity_remaining, batch_column = None, None
    else:
        capacity_remaining, batch_column = _build_capacity_state(routed_core, config_path)
    decisions = route_magd_cases(
        routed_core,
        ranked_experts=ranked_experts,
        expert_metrics=expert_metrics,
        capacity_state=CapacityState(remaining=capacity_remaining, batch_column=batch_column),
        fp_cost=cfg["fp_cost"],
        fn_cost=cfg["fn_cost"],
        fairness_penalty_weight=cfg["fairness_penalty_weight"],
        capacity_penalty_weight=cfg["capacity_penalty_weight"],
        human_review_cost=cfg["human_review_cost"],
        escalation_cost=cfg["escalation_cost"],
        top_k=cfg["top_k"],
        wrong_confident_threshold=float(params["wrong_confident_threshold"]),
        ai_cost_margin=float(params["ai_cost_margin"]),
        ai_cost_scale=float(params["ai_cost_scale"]),
    )

    if "y_true" in core.columns:
        decisions = decisions.merge(core[["case_id", "y_true"]], on="case_id", how="left")
        decisions["is_correct"] = (
            pd.to_numeric(decisions["final_prediction"], errors="coerce") == pd.to_numeric(decisions["y_true"], errors="coerce")
        ).astype(int)
        decisions["ai_correct"] = (
            pd.to_numeric(decisions["ai_pred"], errors="coerce") == pd.to_numeric(decisions["y_true"], errors="coerce")
        ).astype(int)
    decisions["used_ai"] = decisions["selected_route"].astype(str).eq("AI")
    decisions["method"] = method
    decisions["policy_version"] = f"{method}:magd_constrained"
    decisions["low_risk_threshold"] = float(params["low_risk"])
    decisions["high_risk_threshold"] = float(params["high_risk"])
    decisions["policy_weights_json"] = json.dumps(weights, sort_keys=True)
    return decisions


def _evaluate_decisions(
    *,
    decisions: pd.DataFrame,
    config_path: Path,
    params: dict[str, float],
    phase: str,
) -> dict[str, object]:
    config = load_yaml(config_path)
    constraints = _constraint_config(config)
    loaded = load_fifar_data(config_path)
    fp_cost = float(config["costs"]["false_positive"])
    fn_cost = float(config["costs"]["false_negative"])

    fraud = compute_fraud_metrics(decisions, score_column="ai_score", fp_cost=fp_cost, fn_cost=fn_cost)
    reliance = compute_reliance_metrics(decisions)
    deferral = compute_deferral_metrics(decisions, oracle_reference=None)
    deferral_rate = float(deferral["human_deferral_rate"] + deferral["escalation_rate"])
    audit_coverage = audit_coverage_rate(decisions, REQUIRED_AUDIT_COLUMNS)
    fairness_penalty = _fairness_penalty(decisions, loaded.sensitive_attributes, fp_cost=fp_cost, fn_cost=fn_cost)
    overreliance = float(reliance["overreliance"])
    wrong_confident_avoidance = float(reliance["wrong_confident_avoidance_rate"])
    capacity_violation = float(deferral["capacity_violation_rate"])
    audit_gap = max(0.0, constraints["min_audit_coverage"] - audit_coverage)

    constraint_status = {
        "overreliance_satisfied": overreliance <= constraints["max_overreliance"],
        "wrong_confident_avoidance_satisfied": wrong_confident_avoidance >= constraints["min_wrong_confident_avoidance"],
        "min_deferral_satisfied": deferral_rate >= constraints["min_deferral_rate"],
        "max_deferral_satisfied": deferral_rate <= constraints["max_deferral_rate"],
        "audit_coverage_satisfied": audit_coverage >= constraints["min_audit_coverage"],
    }
    feasible = all(constraint_status.values())
    deferral_target = float(params["deferral_target"])
    objective = (
        float(fraud["cost_sensitive_loss"])
        + constraints["lambda_overreliance"] * overreliance
        + constraints["lambda_capacity"] * capacity_violation
        + constraints["lambda_fairness"] * fairness_penalty
        + constraints["lambda_audit"] * audit_gap
        + 0.1 * abs(deferral_rate - deferral_target)
    )

    return {
        "phase": phase,
        "fraud_loss": float(fraud["cost_sensitive_loss"]),
        "precision": float(fraud["precision"]),
        "recall": float(fraud["recall"]),
        "f1": float(fraud["f1"]),
        "pr_auc": float(fraud["pr_auc"]) if pd.notna(fraud["pr_auc"]) else np.nan,
        "overreliance": overreliance,
        "wrong_confident_avoidance": wrong_confident_avoidance,
        "deferral_rate": deferral_rate,
        "ai_coverage": float(deferral["ai_coverage"]),
        "expert_deferral_rate": float(deferral["human_deferral_rate"]),
        "escalation_rate": float(deferral["escalation_rate"]),
        "audit_coverage": float(audit_coverage),
        "capacity_violation": capacity_violation,
        "fairness_penalty": fairness_penalty,
        "audit_gap": audit_gap,
        "objective": float(objective),
        "feasible": bool(feasible),
        **constraint_status,
    }


def _params_from_array(values: np.ndarray) -> dict[str, float]:
    return {
        "low_risk": float(values[0]),
        "high_risk": float(values[1]),
        "magd_high_threshold": float(values[2]),
        "wrong_confident_threshold": float(values[3]),
        "base_threshold": float(values[4]),
        "ai_cost_scale": float(values[5]),
        "deferral_target": float(values[6]),
    }


def _array_from_params(params: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            float(params["low_risk"]),
            float(params["high_risk"]),
            float(params["magd_high_threshold"]),
            float(params["wrong_confident_threshold"]),
            float(params["base_threshold"]),
            float(params["ai_cost_scale"]),
            float(params["deferral_target"]),
        ],
        dtype=float,
    )


def _candidate_grids(
    config: dict,
    initial_params: dict[str, float],
    *,
    validation_risk: pd.Series | None = None,
) -> list[dict[str, float]]:
    constraints = _constraint_config(config)

    def bounded(values: list[float]) -> list[float]:
        return sorted({round(min(0.99, max(0.01, value)), 4) for value in values})

    # Prefer a candidate range that actually covers the observed validation MAGD-risk
    # distribution (Fix D) instead of a fixed +/-0.1 offset from the initial thresholds,
    # which previously left medium/high risk categories unreachable whenever the realized
    # risk range didn't happen to sit near the old hard-coded 0.35/0.70 defaults.
    non_null_validation_risk = validation_risk.dropna() if validation_risk is not None else None
    if non_null_validation_risk is not None and len(non_null_validation_risk) > 0:
        low_grid = bounded(
            [float(non_null_validation_risk.quantile(q)) for q in (0.30, 0.50, 0.70)]
            + [initial_params["low_risk"]]
        )
        high_grid = bounded(
            [float(non_null_validation_risk.quantile(q)) for q in (0.75, 0.90, 0.97)]
            + [initial_params["high_risk"]]
        )
    else:
        low_grid = bounded([initial_params["low_risk"], initial_params["low_risk"] + 0.1])
        high_grid = bounded([initial_params["high_risk"] - 0.1, initial_params["high_risk"]])
    magd_high_grid = bounded([max(initial_params["magd_high_threshold"], initial_params["high_risk"])])
    wrong_conf_grid = bounded([initial_params["wrong_confident_threshold"], initial_params["wrong_confident_threshold"] + 0.1])
    base_threshold_grid = bounded([initial_params["base_threshold"] - 0.1, initial_params["base_threshold"]])
    ai_cost_scale_grid = [1.0]
    mid_deferral = (constraints["min_deferral_rate"] + constraints["max_deferral_rate"]) / 2.0
    deferral_target_grid = bounded([mid_deferral])

    rows: list[dict[str, float]] = []
    for low_risk, high_risk, magd_high, wrong_conf, base_threshold, ai_cost_scale, def_target in itertools.product(
        low_grid,
        high_grid,
        magd_high_grid,
        wrong_conf_grid,
        base_threshold_grid,
        ai_cost_scale_grid,
        deferral_target_grid,
    ):
        if low_risk > high_risk:
            continue
        rows.append(
            {
                "low_risk": float(low_risk),
                "high_risk": float(high_risk),
                "magd_high_threshold": float(max(magd_high, high_risk)),
                "wrong_confident_threshold": float(wrong_conf),
                "base_threshold": float(base_threshold),
                "ai_cost_scale": float(ai_cost_scale),
                "deferral_target": float(def_target),
            }
        )
    return rows


def _optimize_policy(
    *,
    config_path: Path,
    validation_core: pd.DataFrame,
    weights: dict[str, float],
    base_params: dict[str, float],
    ranked_experts: list[str],
    expert_metrics: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, object]]:
    config = load_yaml(config_path)
    constraints = _constraint_config(config)
    cache: dict[tuple[float, ...], tuple[dict[str, float], pd.DataFrame, dict[str, object]]] = {}

    def evaluate(candidate_params: dict[str, float]) -> tuple[pd.DataFrame, dict[str, object]]:
        full_params = {**base_params, **candidate_params}
        key = tuple(round(float(full_params[name]), 6) for name in ["low_risk", "high_risk", "magd_high_threshold", "wrong_confident_threshold", "base_threshold", "ai_cost_scale", "deferral_target"])
        if key not in cache:
            decisions = _apply_policy_to_core(
                core=validation_core,
                config_path=config_path,
                weights=weights,
                params=full_params,
                ranked_experts=ranked_experts,
                expert_metrics=expert_metrics,
                method="magd_constrained_validation",
            )
            metrics = _evaluate_decisions(decisions=decisions, config_path=config_path, params=full_params, phase="validation")
            cache[key] = (full_params, decisions, metrics)
        _, decisions, metrics = cache[key]
        return decisions, metrics

    initial_decisions, initial_metrics = evaluate(base_params)
    best_params = base_params.copy()
    best_metrics = initial_metrics
    best_search = "initial"
    feasible_found = bool(initial_metrics["feasible"])

    try:
        from scipy.optimize import minimize

        initial_array = _array_from_params(base_params)

        def objective_fn(array: np.ndarray) -> float:
            params = _params_from_array(array)
            _, metrics = evaluate(params)
            return float(metrics["objective"])

        def constraint_fn(name: str):
            def _inner(array: np.ndarray) -> float:
                params = _params_from_array(array)
                _, metrics = evaluate(params)
                if name == "overreliance":
                    return constraints["max_overreliance"] - float(metrics["overreliance"])
                if name == "wrong_confident_avoidance":
                    return float(metrics["wrong_confident_avoidance"]) - constraints["min_wrong_confident_avoidance"]
                if name == "min_deferral":
                    return float(metrics["deferral_rate"]) - constraints["min_deferral_rate"]
                if name == "max_deferral":
                    return constraints["max_deferral_rate"] - float(metrics["deferral_rate"])
                if name == "audit_coverage":
                    return float(metrics["audit_coverage"]) - constraints["min_audit_coverage"]
                if name == "threshold_order":
                    return float(params["high_risk"] - params["low_risk"])
                if name == "escalation_order":
                    return float(params["magd_high_threshold"] - params["high_risk"])
                raise KeyError(name)

            return _inner

        result = minimize(
            objective_fn,
            initial_array,
            method="SLSQP",
            bounds=[
                (0.05, 0.7),
                (0.2, 0.95),
                (0.2, 0.99),
                (0.2, 0.99),
                (0.05, 0.95),
                (0.5, 1.5),
                (constraints["min_deferral_rate"], constraints["max_deferral_rate"]),
            ],
            constraints=[
                {"type": "ineq", "fun": constraint_fn("overreliance")},
                {"type": "ineq", "fun": constraint_fn("wrong_confident_avoidance")},
                {"type": "ineq", "fun": constraint_fn("min_deferral")},
                {"type": "ineq", "fun": constraint_fn("max_deferral")},
                {"type": "ineq", "fun": constraint_fn("audit_coverage")},
                {"type": "ineq", "fun": constraint_fn("threshold_order")},
                {"type": "ineq", "fun": constraint_fn("escalation_order")},
            ],
            options={"maxiter": 100, "disp": False},
        )
        if result.success:
            candidate_params = {**base_params, **_params_from_array(result.x)}
            _, candidate_metrics = evaluate(candidate_params)
            best_params = candidate_params
            best_metrics = candidate_metrics
            best_search = "scipy_slsqp"
            feasible_found = bool(candidate_metrics["feasible"])
        else:
            LOGGER.warning("Constrained MAGD SLSQP failed: %s", result.message)
    except Exception as exc:
        LOGGER.warning("Constrained MAGD SciPy optimization unavailable or failed: %s", exc)

    if not feasible_found:
        validation_risk = compute_magd_risk(
            validation_core,
            weights,
            use_drift_risk=bool(config["magd"]["signals"].get("use_drift_risk", False)),
            use_business_risk=bool(config["magd"]["signals"].get("use_business_risk", False)),
        )["magd_assurance_risk"]
        grid_candidates = _candidate_grids(config, base_params, validation_risk=validation_risk)
        for candidate in grid_candidates:
            _, metrics = evaluate(candidate)
            if metrics["feasible"]:
                if (not feasible_found) or float(metrics["objective"]) < float(best_metrics["objective"]):
                    best_params = {**base_params, **candidate}
                    best_metrics = metrics
                    best_search = "grid_search"
                    feasible_found = True
        if not feasible_found:
            for candidate in grid_candidates:
                _, metrics = evaluate(candidate)
                if float(metrics["objective"]) < float(best_metrics["objective"]):
                    best_params = {**base_params, **candidate}
                    best_metrics = metrics
                    best_search = "grid_search_infeasible_best"

    diagnostics = {
        "optimizer": best_search,
        "used_validation_only": True,
        "validation_rows": int(len(validation_core)),
        "feasible_found": bool(feasible_found),
        "initial_validation_metrics": initial_metrics,
        "selected_validation_metrics": best_metrics,
        "selected_params": best_params,
    }
    return best_params, diagnostics


def _results_rows(*, initial_test: dict[str, object], calibrated_validation: dict[str, object], calibrated_test: dict[str, object]) -> pd.DataFrame:
    rows = []
    for name, metrics in [
        ("initial_test", initial_test),
        ("calibrated_validation", calibrated_validation),
        ("calibrated_test", calibrated_test),
    ]:
        row = {"stage": name}
        row.update(metrics)
        row["feasible"] = bool(metrics.get("feasible", False))
        row["overreliance_satisfied"] = bool(metrics.get("overreliance_satisfied", False))
        row["wrong_confident_avoidance_satisfied"] = bool(metrics.get("wrong_confident_avoidance_satisfied", False))
        row["audit_coverage_satisfied"] = bool(metrics.get("audit_coverage_satisfied", False))
        row["min_deferral_satisfied"] = bool(metrics.get("min_deferral_satisfied", False))
        row["max_deferral_satisfied"] = bool(metrics.get("max_deferral_satisfied", False))
        rows.append(row)
    return ensure_constraint_diagnostics(pd.DataFrame(rows))


def run_magd_constrained(config_path: str | Path) -> MagdConstrainedArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    output_dir = _resolve_output_dir(resolved_config_path)
    policy_dir = _resolve_policy_dir(resolved_config_path)
    paper_dir = _resolve_paper_dir(resolved_config_path)

    weights = load_policy_weights(resolved_config_path, variant="constrained")
    base_params = _initial_policy_params(config, resolved_config_path)
    validation_core = _prepare_validation_core(resolved_config_path)
    test_core = _prepare_test_core(resolved_config_path)
    expert_metrics, ranked_experts = _build_expert_reliability(resolved_config_path)

    initial_decisions = _apply_policy_to_core(
        core=test_core,
        config_path=resolved_config_path,
        weights=weights,
        params=base_params,
        ranked_experts=ranked_experts,
        expert_metrics=expert_metrics,
        method="magd_constrained_initial",
    )
    initial_decisions.to_csv(output_dir / "magd_constrained_initial_decisions.csv", index=False)

    calibrated_params, diagnostics = _optimize_policy(
        config_path=resolved_config_path,
        validation_core=validation_core,
        weights=weights,
        base_params=base_params,
        ranked_experts=ranked_experts,
        expert_metrics=expert_metrics,
    )
    calibrated_validation_decisions = _apply_policy_to_core(
        core=validation_core,
        config_path=resolved_config_path,
        weights=weights,
        params=calibrated_params,
        ranked_experts=ranked_experts,
        expert_metrics=expert_metrics,
        method="magd_constrained_validation",
    )
    calibrated_decisions = _apply_policy_to_core(
        core=test_core,
        config_path=resolved_config_path,
        weights=weights,
        params=calibrated_params,
        ranked_experts=ranked_experts,
        expert_metrics=expert_metrics,
        method="magd_constrained_calibrated",
    )

    calibrated_decisions.to_csv(output_dir / "magd_constrained_calibrated_decisions.csv", index=False)
    calibrated_decisions.to_csv(output_dir / "magd_constrained_decisions.csv", index=False)

    initial_metrics = _evaluate_decisions(
        decisions=initial_decisions,
        config_path=resolved_config_path,
        params=base_params,
        phase="test",
    )
    calibrated_validation_metrics = _evaluate_decisions(
        decisions=calibrated_validation_decisions,
        config_path=resolved_config_path,
        params=calibrated_params,
        phase="validation",
    )
    calibrated_test_metrics = _evaluate_decisions(
        decisions=calibrated_decisions,
        config_path=resolved_config_path,
        params=calibrated_params,
        phase="test",
    )
    if not bool(diagnostics.get("feasible_found", False)):
        calibrated_validation_metrics.update(CONSTRAINT_FAILURE_DIAGNOSTICS)
        calibrated_test_metrics.update(CONSTRAINT_FAILURE_DIAGNOSTICS)

    policy_config = {
        "variant": "constrained",
        "weights": weights,
        "initial_params": base_params,
        "selected_params": calibrated_params,
        "tuned_on_split": "validation",
    }
    diagnostics.update(
        {
            "selected_constraint_status": {
                key: calibrated_validation_metrics[key]
                for key in [
                    "feasible",
                    "overreliance_satisfied",
                    "wrong_confident_avoidance_satisfied",
                    "min_deferral_satisfied",
                    "max_deferral_satisfied",
                    "audit_coverage_satisfied",
                ]
            },
            "test_metrics": calibrated_test_metrics,
        }
    )

    (policy_dir / "constrained_policy_config.json").write_text(json.dumps(policy_config, indent=2), encoding="utf-8")
    (policy_dir / "constrained_policy_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    results = _results_rows(
        initial_test=initial_metrics,
        calibrated_validation=calibrated_validation_metrics,
        calibrated_test=calibrated_test_metrics,
    )
    results.to_csv(paper_dir / "intervention_calibrated_results.csv", index=False)

    return MagdConstrainedArtifacts(
        initial_decisions=initial_decisions,
        calibrated_decisions=calibrated_decisions,
        results=results,
        output_dir=output_dir,
        policy_dir=policy_dir,
        diagnostics=diagnostics,
    )
