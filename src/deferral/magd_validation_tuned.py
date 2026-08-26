from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.assurance.magd_risk import compute_magd_risk
from src.deferral.baselines import _expert_predictions_for_split, _historical_best_experts
from src.deferral.expert_routing import _load_expert_tables, _resolve_output_dir, _routing_config
from src.deferral.magd_policy import _active_weight_keys, _normalize_weights, weight_grid_candidates
from src.evaluation.assurance_metrics import audit_coverage_rate
from src.evaluation.deferral_metrics import compute_deferral_metrics
from src.evaluation.fraud_metrics import compute_fraud_metrics
from src.evaluation.reliance_metrics import compute_reliance_metrics
from src.utils.io import load_yaml


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


@dataclass
class MagdValidationTunedArtifacts:
    validation_results: pd.DataFrame
    test_results: pd.DataFrame
    decisions: pd.DataFrame
    selected_policy: dict[str, object]
    output_dir: Path
    policy_dir: Path


def _outputs_root(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    return outputs_root


def _policy_dir(config_path: Path) -> Path:
    output_dir = _outputs_root(config_path) / "magd_policy"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _paper_dir(config_path: Path) -> Path:
    output_dir = _outputs_root(config_path) / "paper_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _validation_tuned_config(config: dict) -> dict[str, object]:
    cfg = config.get("magd", {}).get("validation_tuned", {})
    budgets = cfg.get("deferral_budgets", [0.01, 0.02, 0.05, 0.10, 0.20])
    return {
        "deferral_budgets": [float(value) for value in budgets],
        "lambda_deferral": float(cfg.get("lambda_deferral", 0.02)),
        "lambda_recall": float(cfg.get("lambda_recall", 0.05)),
        "lambda_f1": float(cfg.get("lambda_f1", 0.05)),
        "min_deferral_rate": float(cfg.get("min_deferral_rate", 0.0)),
        "max_validation_rows": cfg.get("max_validation_rows", None),
        "objective": str(cfg.get("objective", "cost_recall_f1")),
    }


def _load_split_core(config_path: Path, split_name: str) -> pd.DataFrame:
    config = load_yaml(config_path)
    outputs_root = _outputs_root(config_path)
    assurance_dir = outputs_root / "assurance"
    model = pd.read_csv(outputs_root / "model" / f"{split_name}_predictions.csv")
    numerical = pd.read_csv(assurance_dir / "numerical_confidence.csv")
    calibration = pd.read_csv(assurance_dir / "calibration_risk.csv")
    distance = pd.read_csv(assurance_dir / "distance_uncertainty.csv")
    local = pd.read_csv(assurance_dir / "local_reliability.csv")
    wrong = pd.read_csv(assurance_dir / "wrong_confident_risk.csv")

    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = (config_path.parent / processed_dir).resolve()
    metadata_name = f"{split_name}_metadata.csv" if split_name != "test" else "test_metadata.csv"
    metadata = pd.read_csv(processed_dir / metadata_name)

    for frame in [model, numerical, calibration, distance, local, wrong, metadata]:
        frame["case_id"] = frame["case_id"].astype(str)
        if "split" in frame.columns:
            frame = frame

    def by_split(frame: pd.DataFrame) -> pd.DataFrame:
        if "split" not in frame.columns:
            return frame.copy()
        return frame.loc[frame["split"].astype(str).eq(split_name)].copy()

    core = (
        by_split(model)
        .merge(by_split(numerical)[["case_id", "numerical_confidence"]], on="case_id", how="inner")
        .merge(by_split(calibration)[["case_id", "calibration_risk"]], on="case_id", how="inner")
        .merge(by_split(distance)[["case_id", "distance_confidence", "distance_uncertainty"]], on="case_id", how="inner")
        .merge(
            by_split(local)[
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
        .merge(by_split(wrong)[["case_id", "wrong_confident_risk"]], on="case_id", how="inner")
        .merge(metadata, on="case_id", how="left")
    )
    expert_predictions = _expert_predictions_for_split(config_path, split_name, core["case_id"])
    return core.merge(expert_predictions, on="case_id", how="left")


def _weight_candidates(config: dict) -> list[dict[str, float]]:
    """Genuine, leakage-safe multi-evidence weight search space for MAGD-ValidationTuned.

    Includes the config-default (heuristic) weight vector, the equal-weight vector, and a
    systematic multi-evidence grid over the active signals (shared with MAGD-Learned's grid
    search via `weight_grid_candidates`), so the search can genuinely select a multi-signal
    blend rather than only ever being able to pick a config default or a single pure signal.
    No candidate here encodes any specific historical/reported weight vector - candidates are
    generated purely from the active-signal grid, and the winner is whichever minimizes the
    validation objective for this run's actual data.
    """
    active_keys = _active_weight_keys(config)
    candidates: list[dict[str, float]] = [
        _normalize_weights(config["magd"]["risk_weights"], active_keys),
        _normalize_weights({key: 1.0 for key in active_keys}, active_keys),
    ]
    candidates.extend(weight_grid_candidates(active_keys, levels=(0.0, 0.5, 1.0)))
    unique: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for weights in candidates:
        key = tuple(sorted((name, round(float(value), 6)) for name, value in weights.items()))
        if key not in seen:
            unique.append(weights)
            seen.add(key)
    return unique


def _best_expert_ranking(config_path: Path, core: pd.DataFrame) -> list[str]:
    _, historical_df = _load_expert_tables(config_path)
    expert_columns = [
        column
        for column in core.columns
        if column
        not in {
            "case_id",
            "split",
            "y_true",
            "ai_score",
            "ai_pred",
            "numerical_confidence",
            "distance_confidence",
            "distance_uncertainty",
            "calibration_risk",
            "neighbor_error_rate",
            "neighbor_fraud_rate",
            "neighbor_ai_agreement",
            "mean_neighbor_distance",
            "top_k",
            "wrong_confident_risk",
        }
        and pd.to_numeric(core[column], errors="coerce").dropna().isin([0, 1]).all()
    ]
    if historical_df is not None:
        try:
            ranked = _historical_best_experts(config_path, historical_df)
            ranked = [expert for expert in ranked if expert in expert_columns]
            if ranked:
                return ranked
        except Exception:
            pass
    return expert_columns


def _policy_metrics(decisions: pd.DataFrame, *, fp_cost: float, fn_cost: float) -> dict[str, float]:
    fraud = compute_fraud_metrics(decisions, score_column="ai_score", fp_cost=fp_cost, fn_cost=fn_cost)
    deferral = compute_deferral_metrics(decisions, oracle_reference=None)
    reliance = compute_reliance_metrics(decisions)
    return {
        **{key: float(value) for key, value in fraud.items()},
        **{key: float(value) for key, value in deferral.items()},
        "overreliance": float(reliance["overreliance"]),
        "wrong_confident_avoidance_rate": float(reliance["wrong_confident_avoidance_rate"]),
        "audit_coverage": float(audit_coverage_rate(decisions, REQUIRED_AUDIT_COLUMNS)),
    }


def apply_budgeted_policy(
    core: pd.DataFrame,
    *,
    weights: dict[str, float],
    budget: float,
    threshold: float | None,
    ranked_experts: list[str],
    method_name: str,
    use_drift_risk: bool = False,
    use_business_risk: bool = False,
) -> tuple[pd.DataFrame, float]:
    scored = compute_magd_risk(core, weights, use_drift_risk=use_drift_risk, use_business_risk=use_business_risk)
    if threshold is None:
        quantile = max(0.0, min(1.0, 1.0 - float(budget)))
        threshold = float(scored["magd_assurance_risk"].quantile(quantile))
    defer = scored["magd_assurance_risk"].astype(float) >= float(threshold)

    expert_columns = [expert for expert in ranked_experts if expert in scored.columns]
    selected_expert = pd.Series([""] * len(scored), index=scored.index, dtype="object")
    final_prediction = scored["ai_pred"].astype(int).copy()
    selected_route = pd.Series(["AI"] * len(scored), index=scored.index, dtype="object")
    for expert in expert_columns:
        available = defer & selected_expert.eq("") & scored[expert].notna()
        final_prediction.loc[available] = scored.loc[available, expert].astype(int)
        selected_expert.loc[available] = expert
        selected_route.loc[available] = "Human Expert"

    decision_reason = pd.Series(["validation_tuned_ai"] * len(scored), index=scored.index, dtype="object")
    decision_reason.loc[selected_route.eq("Human Expert")] = "validation_tuned_budgeted_high_magd_risk"
    decision_reason.loc[defer & selected_route.eq("AI")] = "validation_tuned_no_expert_available_ai_fallback"

    decisions = scored.copy()
    decisions["risk_category"] = np.where(defer, "defer_budget", "ai_budget")
    decisions["base_threshold"] = float(threshold)
    def optional_numeric(column: str) -> pd.Series:
        if column not in decisions.columns:
            return pd.Series([0.0] * len(decisions), index=decisions.index, dtype=float)
        return pd.to_numeric(decisions[column], errors="coerce").fillna(0.0).astype(float)

    decisions["business_risk"] = optional_numeric("business_risk")
    decisions["fairness_risk"] = optional_numeric("fairness_risk")
    decisions["capacity_pressure"] = optional_numeric("capacity_pressure")
    decisions["adaptive_threshold"] = float(threshold)
    decisions["passes_threshold"] = ~defer
    decisions["selected_route"] = selected_route
    decisions["selected_expert"] = selected_expert
    decisions["final_prediction"] = final_prediction.astype(int)
    decisions["decision_reason"] = decision_reason
    decisions["used_ai"] = selected_route.eq("AI")
    decisions["is_correct"] = (decisions["final_prediction"].astype(int) == decisions["y_true"].astype(int)).astype(int)
    decisions["ai_correct"] = (decisions["ai_pred"].astype(int) == decisions["y_true"].astype(int)).astype(int)
    decisions["method"] = method_name
    decisions["policy_variant"] = "validation_tuned"
    decisions["tuned_on_split"] = "validation"
    decisions["selected_budget"] = float(budget)
    decisions["selected_threshold"] = float(threshold)
    decisions["policy_version"] = f"{method_name}:validation_tuned"
    decisions["policy_weights_json"] = json.dumps(weights, sort_keys=True)
    return decisions, float(threshold)


def tune_validation_policy(
    config_path: str | Path,
    validation_core: pd.DataFrame,
    *,
    ranked_experts: list[str],
) -> tuple[dict[str, object], pd.DataFrame]:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    cfg = _routing_config(config)
    tuned_cfg = _validation_tuned_config(config)
    rows: list[dict[str, object]] = []
    best_policy: dict[str, object] | None = None
    best_objective = float("inf")

    for weights in _weight_candidates(config):
        for budget in tuned_cfg["deferral_budgets"]:
            decisions, threshold = apply_budgeted_policy(
                validation_core,
                weights=weights,
                budget=float(budget),
                threshold=None,
                ranked_experts=ranked_experts,
                method_name="MAGD-Fraud-ValidationTuned",
                use_drift_risk=bool(config["magd"]["signals"].get("use_drift_risk", False)),
                use_business_risk=bool(config["magd"]["signals"].get("use_business_risk", False)),
            )
            metrics = _policy_metrics(decisions, fp_cost=cfg["fp_cost"], fn_cost=cfg["fn_cost"])
            deferral_rate = float(metrics["human_deferral_rate"] + metrics["escalation_rate"])
            objective = (
                float(metrics["cost_sensitive_loss"]) / max(1.0, float(len(decisions)))
                + float(tuned_cfg["lambda_deferral"]) * deferral_rate
                - float(tuned_cfg["lambda_recall"]) * float(metrics["recall"])
                - float(tuned_cfg["lambda_f1"]) * float(metrics["f1"])
            )
            if deferral_rate < float(tuned_cfg["min_deferral_rate"]):
                objective += 1.0
            row = {
                "stage": "validation",
                "budget": float(budget),
                "threshold": float(threshold),
                "objective": float(objective),
                **weights,
                **metrics,
            }
            rows.append(row)
            if objective < best_objective:
                best_objective = float(objective)
                best_policy = {
                    "method": "MAGD-Fraud-ValidationTuned",
                    "tuned_on_split": "validation",
                    "selection_metric": tuned_cfg["objective"],
                    "objective": float(objective),
                    "budget": float(budget),
                    "threshold": float(threshold),
                    "weights": weights,
                    "validation_metrics": metrics,
                    "uses_test_labels_for_tuning": False,
                }

    if best_policy is None:
        raise ValueError("No validation-tuned MAGD policy candidates were evaluated.")
    return best_policy, pd.DataFrame(rows)


def run_magd_validation_tuned(config_path: str | Path) -> MagdValidationTunedArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    tuned_cfg = _validation_tuned_config(config)
    output_dir = _resolve_output_dir(resolved_config_path)
    policy_dir = _policy_dir(resolved_config_path)
    paper_dir = _paper_dir(resolved_config_path)
    cfg = _routing_config(config)

    validation_core = _load_split_core(resolved_config_path, "val")
    max_validation_rows = tuned_cfg["max_validation_rows"]
    if max_validation_rows is not None and len(validation_core) > int(max_validation_rows):
        validation_core = validation_core.sample(n=int(max_validation_rows), random_state=int(config.get("experiment", {}).get("seed", 42)))
    ranked_experts = _best_expert_ranking(resolved_config_path, validation_core)
    selected_policy, validation_results = tune_validation_policy(
        resolved_config_path,
        validation_core,
        ranked_experts=ranked_experts,
    )

    test_core = _load_split_core(resolved_config_path, "test")
    test_ranked_experts = [expert for expert in ranked_experts if expert in test_core.columns]
    if not test_ranked_experts:
        test_ranked_experts = _best_expert_ranking(resolved_config_path, test_core)
    decisions, _ = apply_budgeted_policy(
        test_core,
        weights=selected_policy["weights"],
        budget=float(selected_policy["budget"]),
        threshold=float(selected_policy["threshold"]),
        ranked_experts=test_ranked_experts,
        method_name="MAGD-Fraud-ValidationTuned",
        use_drift_risk=bool(config["magd"]["signals"].get("use_drift_risk", False)),
        use_business_risk=bool(config["magd"]["signals"].get("use_business_risk", False)),
    )
    test_metrics = _policy_metrics(decisions, fp_cost=cfg["fp_cost"], fn_cost=cfg["fn_cost"])
    selected_policy["test_metrics"] = test_metrics
    selected_policy["test_evaluated_once_after_selection"] = True
    selected_policy["ranked_experts"] = test_ranked_experts

    decisions.to_csv(output_dir / "magd_validation_tuned_decisions.csv", index=False)
    (policy_dir / "validation_tuned_policy_config.json").write_text(json.dumps(selected_policy, indent=2), encoding="utf-8")
    validation_results.to_csv(policy_dir / "validation_tuned_budget_results.csv", index=False)

    test_results = pd.DataFrame(
        [
            {
                "stage": "test",
                "budget": float(selected_policy["budget"]),
                "threshold": float(selected_policy["threshold"]),
                **selected_policy["weights"],
                **test_metrics,
            }
        ]
    )
    combined = pd.concat([validation_results, test_results], ignore_index=True, sort=False)
    combined.to_csv(paper_dir / "magd_validation_tuned_budget_results.csv", index=False)
    return MagdValidationTunedArtifacts(
        validation_results=validation_results,
        test_results=test_results,
        decisions=decisions,
        selected_policy=selected_policy,
        output_dir=output_dir,
        policy_dir=policy_dir,
    )
