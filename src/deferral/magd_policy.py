from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.assurance.local_reliability import build_local_reliability_for_queries
from src.assurance.magd_risk import add_risk_categories_and_actions, compute_magd_risk, load_frozen_magd_risk_thresholds
from src.assurance.numerical_confidence import compute_calibration_artifacts
from src.assurance.wrong_confident_detector import compute_deployable_wrong_confident_risk
from src.data.load_fifar import load_fifar_data
from src.deferral.adaptive_threshold import compute_adaptive_thresholds
from src.deferral.baselines import _build_capacity_state
from src.deferral.expert_routing import _routing_config, _load_expert_tables
from src.evaluation.assurance_metrics import audit_coverage_rate, correct_rejection_rate
from src.evaluation.deferral_metrics import compute_deferral_metrics
from src.evaluation.fairness_metrics import compute_fairness_metrics
from src.evaluation.fraud_metrics import compute_fraud_metrics
from src.evaluation.reliance_metrics import compute_reliance_metrics
from src.models.embeddings import _fit_pca, _impute_with_train_medians, _transform_frame
from src.models.predict import load_processed_split
from src.utils.io import load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PolicyResult:
    variant: str
    weights: dict[str, float]
    objective: float
    diagnostics: dict[str, float | str | bool]


@dataclass
class MagdPolicyArtifacts:
    learned_weights: pd.DataFrame
    diagnostics: dict[str, object]
    output_dir: Path


def _resolve_output_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    output_dir = outputs_root / "magd_policy"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_assurance_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    return outputs_root / "assurance"


def resolve_magd_thresholds_for_config(config_path: Path, config: dict) -> tuple[float, float]:
    thresholds_cfg = config.get("magd", {}).get("thresholds", {})
    fallback_low = float(thresholds_cfg.get("low_risk", 0.35))
    fallback_high = float(thresholds_cfg.get("high_risk", 0.70))
    if str(thresholds_cfg.get("mode", "validation_quantile")).lower() == "fixed":
        return fallback_low, fallback_high
    return load_frozen_magd_risk_thresholds(
        _resolve_assurance_dir(config_path),
        fallback_low_risk=fallback_low,
        fallback_high_risk=fallback_high,
    )


def weight_grid_candidates(active_keys: list[str], levels: tuple[float, ...] = (0.0, 0.5, 1.0)) -> list[dict[str, float]]:
    """Systematic multi-evidence weight grid used by both weight-search and the
    validation-tuned candidate list, so both consumers explore a genuine, shared
    multi-signal search space rather than a handful of ad hoc single-signal vectors."""
    candidates: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for combo in itertools.product(levels, repeat=len(active_keys)):
        if sum(combo) <= 0.0:
            continue
        weights = _normalize_weights(dict(zip(active_keys, combo, strict=False)), active_keys)
        key = tuple(sorted((name, round(float(value), 6)) for name, value in weights.items()))
        if key not in seen:
            candidates.append(weights)
            seen.add(key)
    return candidates


def _active_weight_keys(config: dict) -> list[str]:
    signals = config.get("magd", {}).get("signals", {})
    keys = [
        ("distance_uncertainty", bool(signals.get("use_distance_uncertainty", True))),
        ("calibration_risk", bool(signals.get("use_calibration_risk", True))),
        ("neighbor_error_rate", bool(signals.get("use_neighbor_error_rate", True))),
        ("wrong_confident_risk", bool(signals.get("use_wrong_confident_risk", True))),
        ("drift_risk", bool(signals.get("use_drift_risk", False))),
        ("business_risk", bool(signals.get("use_business_risk", False))),
    ]
    active = [key for key, enabled in keys if enabled]
    return active or ["distance_uncertainty", "calibration_risk", "neighbor_error_rate", "wrong_confident_risk"]


def _normalize_weights(raw_weights: dict[str, float], active_keys: list[str]) -> dict[str, float]:
    weights = {key: max(float(raw_weights.get(key, 0.0)), 0.0) for key in active_keys}
    total = sum(weights.values())
    if total <= 0.0:
        weights = {key: 1.0 / len(active_keys) for key in active_keys}
    else:
        weights = {key: value / total for key, value in weights.items()}
    full = {
        "distance_uncertainty": 0.0,
        "calibration_risk": 0.0,
        "neighbor_error_rate": 0.0,
        "wrong_confident_risk": 0.0,
        "drift_risk": 0.0,
        "business_risk": 0.0,
    }
    full.update(weights)
    return full


def _distance_for_query_split(
    *,
    train_embeddings: pd.DataFrame,
    train_labels: pd.Series,
    query_embeddings: pd.DataFrame,
    prediction_frame: pd.DataFrame,
) -> pd.DataFrame:
    train_numeric = train_embeddings.select_dtypes(include=[np.number]).copy()
    query_numeric = query_embeddings.select_dtypes(include=[np.number]).copy()
    centroids: dict[int, np.ndarray] = {}
    for label in [0, 1]:
        class_rows = train_numeric.loc[train_labels.astype(int) == label]
        if class_rows.empty:
            raise ValueError(f"Cannot compute centroid for class {label}: no training examples available.")
        centroids[label] = class_rows.mean(axis=0).to_numpy(dtype=float)

    query_vectors = query_numeric.to_numpy(dtype=float)
    predicted_labels = prediction_frame["ai_pred"].astype(int).to_numpy()
    distances_to_0 = np.array([float(np.linalg.norm(vector - centroids[0])) for vector in query_vectors], dtype=float)
    distances_to_1 = np.array([float(np.linalg.norm(vector - centroids[1])) for vector in query_vectors], dtype=float)
    distance_to_predicted = np.array(
        [distances_to_1[idx] if pred == 1 else distances_to_0[idx] for idx, pred in enumerate(predicted_labels)],
        dtype=float,
    )
    max_distance = float(distance_to_predicted.max()) if len(distance_to_predicted) else 0.0
    distance_confidence = np.ones_like(distance_to_predicted) if max_distance <= 0.0 else 1.0 - (distance_to_predicted / max_distance)
    distance_confidence = np.clip(distance_confidence, 0.0, 1.0)
    return pd.DataFrame(
        {
            "case_id": prediction_frame["case_id"].astype(str),
            "distance_confidence": distance_confidence,
            "distance_uncertainty": 1.0 - distance_confidence,
        }
    )


def _sample_prediction_frame(predictions: pd.DataFrame, *, max_rows: int | None, random_state: int = 42) -> pd.DataFrame:
    if max_rows is None or len(predictions) <= max_rows:
        return predictions.reset_index(drop=True)
    if "y_true" not in predictions.columns:
        return predictions.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    sampled_parts: list[pd.DataFrame] = []
    for _, frame in predictions.groupby("y_true", dropna=False):
        group_n = max(1, int(round(max_rows * len(frame) / len(predictions))))
        sampled_parts.append(frame.sample(n=min(group_n, len(frame)), random_state=random_state))
    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    return sampled.reset_index(drop=True)


def _sample_index_by_label(labels: pd.Series, *, max_rows: int | None, random_state: int = 42) -> pd.Index:
    if max_rows is None or len(labels) <= max_rows:
        return labels.index

    sampled_indices: list[int] = []
    for _, group in labels.groupby(labels, dropna=False):
        group_n = max(1, int(round(max_rows * len(group) / len(labels))))
        sampled_indices.extend(group.sample(n=min(group_n, len(group)), random_state=random_state).index.tolist())
    if len(sampled_indices) > max_rows:
        sampled_indices = (
            pd.Index(sampled_indices)
            .to_series()
            .sample(n=max_rows, random_state=random_state)
            .tolist()
        )
    return pd.Index(sampled_indices)


def load_validation_policy_frame(
    config_path: str | Path,
    *,
    max_rows: int | None = 250,
    max_train_rows: int | None = 10000,
) -> pd.DataFrame:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (resolved_config_path.parent / outputs_root).resolve()
    model_dir = outputs_root / "model"
    processed_dir_cfg = config.get("paths", {}).get("processed_data_dir", "data/processed")
    processed_dir = Path(processed_dir_cfg)
    if not processed_dir.is_absolute():
        processed_dir = (resolved_config_path.parent / processed_dir).resolve()

    train_predictions = pd.read_csv(model_dir / "train_predictions.csv")
    val_predictions = pd.read_csv(model_dir / "val_predictions.csv")
    for frame in [train_predictions, val_predictions]:
        frame["case_id"] = frame["case_id"].astype(str)
    X_train, y_train, train_metadata = load_processed_split(processed_dir, "train")
    X_val, _, val_metadata = load_processed_split(processed_dir, "val")
    sampled_train_index = _sample_index_by_label(y_train.iloc[:, 0].astype(int).reset_index(drop=True), max_rows=max_train_rows)
    train_metadata = train_metadata.iloc[sampled_train_index].reset_index(drop=True)
    train_labels = y_train.iloc[sampled_train_index, 0].astype(int).reset_index(drop=True)
    train_metadata["case_id"] = train_metadata["case_id"].astype(str)
    train_predictions = train_metadata[["case_id"]].merge(train_predictions, on="case_id", how="left")
    val_predictions = _sample_prediction_frame(val_predictions, max_rows=max_rows)
    LOGGER.info("Building validation MAGD features on %s sampled validation rows.", len(val_predictions))
    sampled_case_ids = set(val_predictions["case_id"].astype(str))
    sampled_mask = val_metadata["case_id"].astype(str).isin(sampled_case_ids)
    val_metadata = val_metadata.loc[sampled_mask].copy().reset_index(drop=True)
    train_X_sample = X_train.iloc[sampled_train_index].reset_index(drop=True)
    val_X_sample = X_val.loc[sampled_mask].reset_index(drop=True)
    train_filled, val_filled, _ = _impute_with_train_medians(train_X_sample, val_X_sample, val_X_sample.copy())
    n_components = int(config.get("assurance", {}).get("distance_embedding", {}).get("n_components", 8))
    pca, component_count = _fit_pca(train_filled, n_components)
    train_embeddings = _transform_frame(pca, train_filled, component_count)
    val_embeddings = _transform_frame(pca, val_filled, component_count)
    train_embeddings.insert(0, "case_id", train_metadata["case_id"].astype(str).reset_index(drop=True))
    val_embeddings.insert(0, "case_id", val_metadata["case_id"].astype(str).reset_index(drop=True))
    LOGGER.info("Using %s sampled historical training rows for MAGD policy learning.", len(train_embeddings))

    calibration = compute_calibration_artifacts(
        val_predictions[["case_id", "y_true", "ai_score", "ai_pred"]],
        n_bins=int(config.get("assurance", {}).get("calibration_bins", 10)),
    )
    distance = _distance_for_query_split(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        query_embeddings=val_embeddings,
        prediction_frame=val_predictions,
    )
    local_top_k = min(int(config.get("assurance", {}).get("neighbor_top_k", 5)), max(1, len(train_embeddings)))
    local_reliability = build_local_reliability_for_queries(
        train_embeddings=train_embeddings,
        train_predictions=train_predictions,
        query_embeddings=val_embeddings,
        query_predictions=val_predictions[["case_id", "ai_pred"]],
        split_name="val",
        top_k=local_top_k,
    )
    wc_weights = {
        "numerical_confidence": float(config.get("assurance", {}).get("wrong_confident", {}).get("weights", {}).get("numerical_confidence", 0.25)),
        "distance_uncertainty": float(config.get("assurance", {}).get("wrong_confident", {}).get("weights", {}).get("distance_uncertainty", 0.2)),
        "calibration_risk": float(config.get("assurance", {}).get("wrong_confident", {}).get("weights", {}).get("calibration_risk", 0.2)),
        "neighbor_error_rate": float(config.get("assurance", {}).get("wrong_confident", {}).get("weights", {}).get("neighbor_error_rate", 0.2)),
        "confidence_disagreement": float(config.get("assurance", {}).get("wrong_confident", {}).get("weights", {}).get("confidence_disagreement", 0.15)),
    }
    wrong_confident = compute_deployable_wrong_confident_risk(
        calibration.assurance_frame.merge(distance, on="case_id", how="inner").merge(
            local_reliability[["case_id", "neighbor_error_rate"]],
            on="case_id",
            how="inner",
        ),
        wc_weights,
    )

    val_core = (
        val_predictions.merge(
            calibration.assurance_frame[["case_id", "numerical_confidence", "calibration_risk"]],
            on="case_id",
            how="inner",
        )
        .merge(distance, on="case_id", how="inner")
        .merge(local_reliability, on="case_id", how="inner")
        .merge(wrong_confident[["case_id", "wrong_confident_risk"]], on="case_id", how="inner")
        .merge(val_metadata.assign(case_id=val_metadata["case_id"].astype(str)), on="case_id", how="left")
    )

    expert_df, historical_df = _load_expert_tables(resolved_config_path)
    if historical_df is not None:
        historical_df = historical_df.rename(columns={historical_df.columns[0]: "case_id"})
        historical_df["case_id"] = historical_df["case_id"].astype(str)
        val_core = val_core.merge(historical_df, on="case_id", how="left")
    elif expert_df is not None:
        expert_df = expert_df.rename(columns={expert_df.columns[0]: "case_id"})
        expert_df["case_id"] = expert_df["case_id"].astype(str)
        val_core = val_core.merge(expert_df, on="case_id", how="left")

    return val_core


def _sample_validation_core(core: pd.DataFrame, *, max_rows: int = 5000) -> pd.DataFrame:
    if len(core) <= max_rows:
        return core
    if "y_true" in core.columns:
        sampled = (
            core.groupby("y_true", dropna=False, group_keys=False)
            .apply(lambda frame: frame.sample(n=max(1, int(round(max_rows * len(frame) / len(core)))), random_state=42))
            .reset_index(drop=True)
        )
        if len(sampled) > max_rows:
            sampled = sampled.sample(n=max_rows, random_state=42).reset_index(drop=True)
        return sampled
    return core.sample(n=max_rows, random_state=42).reset_index(drop=True)


def _fairness_penalty(frame: pd.DataFrame, sensitive_attributes: list[str], fp_cost: float, fn_cost: float) -> float:
    fairness = compute_fairness_metrics(frame, sensitive_attributes, fp_cost=fp_cost, fn_cost=fn_cost)
    if fairness.empty:
        return 0.0
    penalties = []
    for sensitive, group in fairness.groupby("sensitive_column", dropna=False):
        if "group_cost" in group.columns:
            penalties.append(float(group["group_cost"].max() - group["group_cost"].min()))
        if "false_positive_rate" in group.columns:
            penalties.append(float(group["false_positive_rate"].max() - group["false_positive_rate"].min()))
        if "false_negative_rate" in group.columns:
            penalties.append(float(group["false_negative_rate"].max() - group["false_negative_rate"].min()))
    return max(penalties) if penalties else 0.0


def evaluate_policy_weights(
    *,
    config_path: str | Path,
    core: pd.DataFrame,
    weights: dict[str, float],
    expert_metrics: pd.DataFrame | None = None,
    ranked_experts: list[str] | None = None,
    sensitive_attributes: list[str] | None = None,
) -> dict[str, float]:
    from src.deferral.magd_deferral import _build_expert_reliability, route_magd_cases

    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    loaded = load_fifar_data(resolved_config_path)
    cfg = _routing_config(config)

    weighted = compute_magd_risk(
        core,
        weights,
        use_drift_risk=bool(config["magd"]["signals"].get("use_drift_risk", False)),
        use_business_risk=bool(config["magd"]["signals"].get("use_business_risk", False)),
    )
    resolved_low_risk, resolved_high_risk = resolve_magd_thresholds_for_config(resolved_config_path, config)
    categorized = add_risk_categories_and_actions(
        weighted,
        low_risk=resolved_low_risk,
        high_risk=resolved_high_risk,
    )
    adaptive = compute_adaptive_thresholds(
        categorized,
        base_threshold=float(config["magd"]["adaptive_threshold"]["base_threshold"]),
        value_weight=float(config["magd"]["adaptive_threshold"]["value_weight"]),
        fairness_weight=float(config["magd"]["adaptive_threshold"]["fairness_weight"]),
        capacity_weight=float(config["magd"]["adaptive_threshold"]["capacity_weight"]),
    )
    routed_core = categorized.merge(
        adaptive[["case_id", "adaptive_threshold", "passes_threshold", "base_threshold", "business_risk", "fairness_risk", "capacity_pressure"]],
        on="case_id",
        how="left",
    )

    if expert_metrics is None or ranked_experts is None:
        expert_metrics, ranked_experts = _build_expert_reliability(resolved_config_path)
    capacity_remaining, batch_column = _build_capacity_state(routed_core, resolved_config_path)
    from src.deferral.capacity_assignment import CapacityState

    capacity_state = CapacityState(remaining=capacity_remaining, batch_column=batch_column)
    decisions = route_magd_cases(
        routed_core,
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
        wrong_confident_threshold=float(config.get("assurance", {}).get("wrong_confident", {}).get("risk_alert_threshold", 0.7)),
    )
    decisions = decisions.merge(core[["case_id", "y_true"]], on="case_id", how="left")
    decisions["used_ai"] = decisions["selected_route"].astype(str).eq("AI")
    decisions["is_correct"] = (decisions["final_prediction"].astype(int) == decisions["y_true"].astype(int)).astype(int)
    decisions["ai_correct"] = (decisions["ai_pred"].astype(int) == decisions["y_true"].astype(int)).astype(int)

    fraud = compute_fraud_metrics(decisions, score_column="ai_score", fp_cost=cfg["fp_cost"], fn_cost=cfg["fn_cost"])
    reliance = compute_reliance_metrics(decisions)
    deferral = compute_deferral_metrics(decisions, oracle_reference=None)
    fairness_penalty = _fairness_penalty(decisions, sensitive_attributes or loaded.sensitive_attributes, fp_cost=cfg["fp_cost"], fn_cost=cfg["fn_cost"])
    audit_cov = audit_coverage_rate(
        decisions,
        [
            "numerical_confidence",
            "distance_uncertainty",
            "calibration_risk",
            "neighbor_error_rate",
            "wrong_confident_risk",
            "magd_assurance_risk",
            "selected_route",
            "final_prediction",
            "decision_reason",
        ],
    )
    correct_rej = correct_rejection_rate(decisions)
    return {
        "cost_sensitive_loss": float(fraud["cost_sensitive_loss"]),
        "overreliance": float(reliance["overreliance"]),
        "correct_rejection": float(correct_rej),
        "capacity_violation_rate": float(deferral["capacity_violation_rate"]),
        "fairness_penalty": float(fairness_penalty),
        "audit_coverage": float(audit_cov),
    }


def _objective_for_variant(variant: str, metrics: dict[str, float], config: dict) -> float:
    constrained_cfg = config["magd"]["constrained_policy"]
    objective = float(metrics["cost_sensitive_loss"])
    if variant in {"learned", "constrained"}:
        objective += float(constrained_cfg["lambda_overreliance"]) * float(metrics["overreliance"])
    if variant == "constrained":
        objective += float(constrained_cfg["lambda_capacity"]) * float(metrics["capacity_violation_rate"])
        objective += float(constrained_cfg["lambda_fairness"]) * float(metrics["fairness_penalty"])
        objective += float(constrained_cfg["lambda_audit"]) * max(0.0, float(constrained_cfg["min_audit_coverage"]) - float(metrics["audit_coverage"]))
        if float(metrics["correct_rejection"]) < float(constrained_cfg["min_correct_rejection"]):
            objective += 1e6
    return float(objective)


def _grid_search_weights(active_keys: list[str], variant: str, evaluator, config: dict) -> tuple[dict[str, float], dict[str, float]]:
    best_weights: dict[str, float] | None = None
    best_metrics: dict[str, float] | None = None
    best_objective = float("inf")

    for weights in weight_grid_candidates(active_keys):
        metrics = evaluator(weights)
        objective = _objective_for_variant(variant, metrics, config)
        if objective < best_objective:
            best_objective = objective
            best_weights = weights
            best_metrics = metrics

    if best_weights is None or best_metrics is None:
        best_weights = _normalize_weights({key: 1.0 for key in active_keys}, active_keys)
        best_metrics = evaluator(best_weights)
    return best_weights, best_metrics


def _coarse_search_weights(active_keys: list[str], variant: str, evaluator, config: dict) -> tuple[dict[str, float], dict[str, float]]:
    candidate_sets: list[dict[str, float]] = []
    candidate_sets.append(_normalize_weights(config["magd"]["risk_weights"], active_keys))
    candidate_sets.append(_normalize_weights({key: 1.0 for key in active_keys}, active_keys))
    for key in active_keys:
        candidate_sets.append(_normalize_weights({candidate: 1.0 if candidate == key else 0.0 for candidate in active_keys}, active_keys))

    best_weights: dict[str, float] | None = None
    best_metrics: dict[str, float] | None = None
    best_objective = float("inf")
    for weights in candidate_sets:
        metrics = evaluator(weights)
        objective = _objective_for_variant(variant, metrics, config)
        if objective < best_objective:
            best_objective = objective
            best_weights = weights
            best_metrics = metrics

    if best_weights is None or best_metrics is None:
        best_weights = _normalize_weights({key: 1.0 for key in active_keys}, active_keys)
        best_metrics = evaluator(best_weights)
    return best_weights, best_metrics


def _scipy_optimize(active_keys: list[str], variant: str, evaluator, config: dict) -> tuple[dict[str, float], dict[str, float], bool]:
    try:
        from scipy.optimize import minimize
    except Exception:
        weights, metrics = _coarse_search_weights(active_keys, variant, evaluator, config)
        return weights, metrics, False

    initial = np.array([1.0 / len(active_keys)] * len(active_keys), dtype=float)

    def objective_fn(params: np.ndarray) -> float:
        normalized = np.maximum(params, 0.0)
        if float(normalized.sum()) <= 0.0:
            normalized = np.ones_like(normalized)
        weights = _normalize_weights(dict(zip(active_keys, normalized.tolist(), strict=False)), active_keys)
        metrics = evaluator(weights)
        return _objective_for_variant(variant, metrics, config)

    result = minimize(
        objective_fn,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(active_keys),
        options={"maxiter": 50, "disp": False},
    )
    if not result.success:
        weights, metrics = _coarse_search_weights(active_keys, variant, evaluator, config)
        return weights, metrics, False

    learned = _normalize_weights(dict(zip(active_keys, result.x.tolist(), strict=False)), active_keys)
    return learned, evaluator(learned), True


def learn_policy_variant(
    config_path: str | Path,
    variant: str,
    *,
    core: pd.DataFrame | None = None,
    expert_metrics: pd.DataFrame | None = None,
    ranked_experts: list[str] | None = None,
    sensitive_attributes: list[str] | None = None,
) -> PolicyResult:
    from src.deferral.magd_deferral import _build_expert_reliability

    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    active_keys = _active_weight_keys(config)
    LOGGER.info("Learning MAGD policy variant `%s`.", variant)
    policy_core = core if core is not None else load_validation_policy_frame(resolved_config_path, max_rows=250, max_train_rows=10000)
    loaded = load_fifar_data(resolved_config_path) if sensitive_attributes is None else None
    policy_sensitive_attributes = sensitive_attributes if sensitive_attributes is not None else loaded.sensitive_attributes
    if expert_metrics is None or ranked_experts is None:
        expert_metrics, ranked_experts = _build_expert_reliability(resolved_config_path)

    def evaluator(candidate_weights: dict[str, float]) -> dict[str, float]:
        return evaluate_policy_weights(
            config_path=resolved_config_path,
            core=policy_core,
            weights=candidate_weights,
            expert_metrics=expert_metrics,
            ranked_experts=ranked_experts,
            sensitive_attributes=policy_sensitive_attributes,
        )

    if variant == "heuristic":
        weights = _normalize_weights(config["magd"]["risk_weights"], active_keys)
        metrics = evaluator(weights)
        return PolicyResult(
            variant=variant,
            weights=weights,
            objective=_objective_for_variant(variant, metrics, config),
            diagnostics={"optimizer": "config", "used_validation_only": True, "validation_rows": float(len(policy_core)), **metrics},
        )

    weights, metrics, scipy_success = _scipy_optimize(active_keys, variant, evaluator, config)
    return PolicyResult(
        variant=variant,
        weights=weights,
        objective=_objective_for_variant(variant, metrics, config),
        diagnostics={
            "optimizer": "scipy_slsqp" if scipy_success else "grid_search",
            "used_validation_only": True,
            "validation_rows": float(len(policy_core)),
            **metrics,
        },
    )


# In-process cache keyed by resolved config path, so a single pipeline run does not
# redundantly re-learn policy weights for every stage that needs them (heuristic, learned,
# constrained, and the deferral runner all call load_policy_weights/load_policy_diagnostics).
# A NEW process (a fresh `python scripts/...` invocation, or a fresh test) starts with an
# empty cache, so stale on-disk artifacts from a different run/dataset are never trusted -
# see `load_policy_weights` below, which always resolves through this cache/run rather than
# reading `learned_weights.csv` as an input.
_POLICY_RUN_CACHE: dict[str, "MagdPolicyArtifacts"] = {}


def _dataset_fingerprint(resolved_config_path: Path, config: dict) -> dict[str, object]:
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = (resolved_config_path.parent / processed_dir).resolve()
    row_counts: dict[str, int] = {}
    for split_name in ["train", "val", "test"]:
        y_path = processed_dir / f"y_{split_name}.csv"
        if y_path.exists():
            try:
                row_counts[split_name] = int(sum(1 for _ in y_path.open("r", encoding="utf-8")) - 1)
            except OSError:
                row_counts[split_name] = -1
    return {
        "processed_dir": str(processed_dir),
        "row_counts": row_counts,
        "seed": int(config.get("experiment", {}).get("seed", 42)),
        "magd_mode": str(config.get("magd", {}).get("mode", "constrained")),
    }


def run_magd_policy(config_path: str | Path, *, force: bool = False) -> MagdPolicyArtifacts:
    resolved_config_path = Path(config_path).resolve()
    cache_key = str(resolved_config_path)
    if not force and cache_key in _POLICY_RUN_CACHE:
        return _POLICY_RUN_CACHE[cache_key]

    output_dir = _resolve_output_dir(resolved_config_path)
    config = load_yaml(resolved_config_path)
    loaded = load_fifar_data(resolved_config_path)
    core = load_validation_policy_frame(resolved_config_path, max_rows=250, max_train_rows=10000)
    from src.deferral.magd_deferral import _build_expert_reliability

    expert_metrics, ranked_experts = _build_expert_reliability(resolved_config_path)
    results = [
        learn_policy_variant(
            resolved_config_path,
            variant,
            core=core,
            expert_metrics=expert_metrics,
            ranked_experts=ranked_experts,
            sensitive_attributes=loaded.sensitive_attributes,
        )
        for variant in ["heuristic", "learned", "constrained"]
    ]

    rows = []
    diagnostics: dict[str, object] = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": _dataset_fingerprint(resolved_config_path, config),
    }
    for result in results:
        diagnostics[result.variant] = result.diagnostics
        row = {"variant": result.variant, "objective": result.objective, **result.weights}
        rows.append(row)
    learned_weights = pd.DataFrame(rows)
    learned_weights.to_csv(output_dir / "learned_weights.csv", index=False)
    (output_dir / "optimization_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    artifacts = MagdPolicyArtifacts(learned_weights=learned_weights, diagnostics=diagnostics, output_dir=output_dir)
    _POLICY_RUN_CACHE[cache_key] = artifacts
    return artifacts


def load_policy_weights(config_path: str | Path, variant: str | None = None) -> dict[str, float]:
    """Resolve policy weights for `variant`, always via a fresh-for-this-process
    `run_magd_policy` run (see `_POLICY_RUN_CACHE`) rather than trusting whatever
    `learned_weights.csv` happens to be sitting on disk. This is the fix for the confirmed
    bug where a stale, unrelated-dataset `learned_weights.csv` was silently reused across
    incompatible runs."""
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    requested_variant = variant or str(config["magd"].get("mode", "constrained")).lower()
    artifacts = run_magd_policy(resolved_config_path)
    frame = artifacts.learned_weights

    if requested_variant not in set(frame["variant"].astype(str)):
        raise ValueError(f"Requested MAGD policy variant `{requested_variant}` is not available.")
    row = frame.loc[frame["variant"].astype(str) == requested_variant].iloc[0].to_dict()
    return _normalize_weights({key: float(row.get(key, 0.0)) for key in _active_weight_keys(config)}, _active_weight_keys(config))


def load_policy_diagnostics(config_path: str | Path) -> dict[str, object]:
    resolved_config_path = Path(config_path).resolve()
    return run_magd_policy(resolved_config_path).diagnostics
