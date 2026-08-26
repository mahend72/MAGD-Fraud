from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


MAGD_DEFAULTS: dict[str, Any] = {
    "mode": "constrained",
    "signals": {
        "use_distance_uncertainty": True,
        "use_calibration_risk": True,
        "use_neighbor_error_rate": True,
        "use_wrong_confident_risk": True,
        "use_drift_risk": False,
        "use_business_risk": False,
        "use_capacity_penalty": False,
        "use_fairness_penalty": True,
    },
    "risk_weights": {
        "distance_uncertainty": 0.25,
        "calibration_risk": 0.20,
        "neighbor_error_rate": 0.20,
        "wrong_confident_risk": 0.25,
        "drift_risk": 0.05,
        "business_risk": 0.05,
    },
    "thresholds": {
        "mode": "validation_quantile",
        "low_quantile": 0.60,
        "high_quantile": 0.90,
        "low_risk": 0.35,
        "high_risk": 0.70,
        "high_confidence": 0.80,
        "wrong_confident": 0.70,
    },
    "adaptive_threshold": {
        "enabled": True,
        "base_threshold": 0.50,
        "business_risk_weight": 0.10,
        "fairness_risk_weight": 0.10,
        "capacity_pressure_weight": 0.10,
    },
    "constrained_policy": {
        "enabled": True,
        "lambda_overreliance": 1.0,
        "lambda_capacity": 0.5,
        "lambda_fairness": 0.5,
        "lambda_audit": 0.25,
        "min_correct_rejection": 0.60,
        "min_audit_coverage": 0.95,
    },
    "validation_tuned": {
        "enabled": True,
        "deferral_budgets": [0.01, 0.02, 0.05, 0.10, 0.20],
        "objective": "cost_recall_f1",
        "lambda_deferral": 0.02,
        "lambda_recall": 0.05,
        "lambda_f1": 0.05,
        "min_deferral_rate": 0.0,
        "max_validation_rows": None,
    },
    "expert_routing": {
        "top_k_for_escalation": 5,
        "capacity_enabled": False,
        "fairness_enabled": True,
        "capacity_penalty_weight": 0.20,
        "fairness_penalty_weight": 0.20,
        "use_majority_vote_for_escalation": True,
    },
    "costs": {
        "false_positive": 0.057,
        "false_negative": 1.0,
        "escalation": 0.10,
        "human_review": 0.05,
    },
}

CONFIG_DEFAULTS: dict[str, Any] = {
    "project": {"name": "haaf_fifar"},
    "experiment": {
        "seed": 42,
        "output_dir": "data/outputs",
        "use_existing_scores": True,
        "test_labels_allowed_for_routing": False,
    },
    "data": {
        "dataset_name": "fifar",
        "train_path": "data/processed/X_train.csv",
        "val_path": "data/processed/X_val.csv",
        "test_path": "data/processed/X_test.csv",
        "y_train_path": "data/processed/y_train.csv",
        "y_val_path": "data/processed/y_val.csv",
        "y_test_path": "data/processed/y_test.csv",
        "sensitive_attribute": "customer_age",
    },
    "model": {
        "model_type": "xgboost",
        "fallback_model": "random_forest",
        "threshold": 0.5,
        "false_positive_cost": 1.0,
        "false_negative_cost": 5.0,
    },
    "costs": {
        "false_positive": 0.057,
        "false_negative": 1.0,
        "human_review": 0.05,
        "escalation": 0.10,
    },
    "assurance": {
        "n_bins_calibration": 10,
        "embedding_method": "pca",
        "pca_components": 20,
        "knn_k": 25,
        "high_confidence_threshold": 0.80,
    },
    "intervention_constraints": {
        "enabled": True,
        "max_overreliance": 0.40,
        "min_wrong_confident_avoidance": 0.60,
        "min_deferral_rate": 0.05,
        "max_deferral_rate": 0.40,
        "min_audit_coverage": 0.95,
    },
    "expert_routing": {
        "top_k_for_escalation": 5,
        "use_majority_vote_for_escalation": True,
        "capacity_enabled": False,
        "fairness_enabled": True,
        "capacity_penalty_weight": 0.20,
        "fairness_penalty_weight": 0.20,
    },
    "statistics": {
        "bootstrap_iterations": 1000,
        "confidence_level": 0.95,
    },
    "paths": {
        "raw_data_dir": "data/raw",
        "processed_data_dir": "data/processed",
        "outputs_dir": "data/outputs",
    },
    "dataset": {
        "main_file": None,
        "train_file": None,
        "test_file": None,
        "expert_predictions_file": None,
        "historical_expert_predictions_file": None,
        "capacity_file": None,
        "file_patterns": ["*.csv", "*.parquet", "*.tsv", "*.xlsx", "*.xls", "*.json"],
    },
    "columns": {
        "case_id": "case_id",
        "application_id": None,
        "batch_id": "batch",
        "time": "month",
        "label": "fraud_label",
        "train_label": "fraud_bool",
        "test_label": "fraud_label",
        "prediction": None,
        "model_score": "model_score",
        "expert_case_id": "case_id",
        "expert_prediction": "sparse#7",
        "capacity_case_id": None,
        "capacity": None,
        "sensitive_attributes": ["customer_age"],
    },
    "split": {
        "train_size": 0.7,
        "val_size": 0.15,
        "test_size": 0.15,
        "random_state": 42,
        "stratify": True,
    },
    "baselines": {
        "numerical_conf_threshold": 0.9,
        "distance_conf_threshold": 0.7,
        "random_state": 42,
    },
    "assurance_deferral": {
        "escalation_top_k_experts": 5,
        "assurance_risk_penalty": 1.0,
        "bias_penalty_weight": 1.0,
        "capacity_penalty_weight": 1.0,
    },
    "inspection": {
        "max_rows_preview": 5,
        "max_candidate_columns_per_group": 10,
    },
}
CONFIG_DEFAULTS["magd"] = deepcopy(MAGD_DEFAULTS)

LEGACY_SIGNAL_KEYS = {
    "use_distance_uncertainty": "use_distance_uncertainty",
    "use_calibration_risk": "use_calibration_risk",
    "use_neighbor_error": "use_neighbor_error_rate",
    "use_neighbor_error_rate": "use_neighbor_error_rate",
    "use_wrong_confident_risk": "use_wrong_confident_risk",
    "use_drift_risk": "use_drift_risk",
    "use_business_risk": "use_business_risk",
    "use_capacity_penalty": "use_capacity_penalty",
    "use_fairness_penalty": "use_fairness_penalty",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{key}` config section must be a mapping.")
    return value


def _require_numeric(mapping: dict[str, Any], keys: list[str], *, context: str) -> None:
    for key in keys:
        value = mapping.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"`{context}.{key}` must be numeric, got {type(value).__name__}.")


def _require_probability(mapping: dict[str, Any], keys: list[str], *, context: str) -> None:
    _require_numeric(mapping, keys, context=context)
    for key in keys:
        value = float(mapping[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"`{context}.{key}` must be between 0 and 1, got {value}.")


def _resolve_output_dir(config_path: Path, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = (config_path.parent / output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _normalize_experiment(config: dict[str, Any]) -> None:
    experiment = _deep_merge(CONFIG_DEFAULTS["experiment"], _require_mapping(config, "experiment"))
    _require_numeric(experiment, ["seed"], context="experiment")
    if not isinstance(experiment["output_dir"], str) or not experiment["output_dir"].strip():
        raise ValueError("`experiment.output_dir` must be a non-empty string.")
    if not isinstance(experiment["use_existing_scores"], bool):
        raise ValueError("`experiment.use_existing_scores` must be a boolean.")
    if not isinstance(experiment["test_labels_allowed_for_routing"], bool):
        raise ValueError("`experiment.test_labels_allowed_for_routing` must be a boolean.")
    config["experiment"] = experiment


def _normalize_data(config: dict[str, Any]) -> None:
    data = _deep_merge(CONFIG_DEFAULTS["data"], _require_mapping(config, "data"))
    required_string_keys = [
        "dataset_name",
        "train_path",
        "val_path",
        "test_path",
        "y_train_path",
        "y_val_path",
        "y_test_path",
        "sensitive_attribute",
    ]
    for key in required_string_keys:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`data.{key}` must be a non-empty string.")
    config["data"] = data


def _normalize_model(config: dict[str, Any]) -> None:
    model = _deep_merge(CONFIG_DEFAULTS["model"], _require_mapping(config, "model"))
    if not isinstance(model["model_type"], str) or not model["model_type"].strip():
        raise ValueError("`model.model_type` must be a non-empty string.")
    if not isinstance(model["fallback_model"], str) or not model["fallback_model"].strip():
        raise ValueError("`model.fallback_model` must be a non-empty string.")
    _require_probability(model, ["threshold"], context="model")
    _require_numeric(model, ["false_positive_cost", "false_negative_cost"], context="model")
    config["model"] = model


def _normalize_costs(config: dict[str, Any]) -> None:
    costs = _deep_merge(CONFIG_DEFAULTS["costs"], _require_mapping(config, "costs"))
    _require_numeric(costs, ["false_positive", "false_negative", "human_review", "escalation"], context="costs")
    config["costs"] = costs


def _normalize_assurance(config: dict[str, Any]) -> None:
    assurance = _deep_merge(CONFIG_DEFAULTS["assurance"], _require_mapping(config, "assurance"))
    _require_numeric(assurance, ["n_bins_calibration", "pca_components", "knn_k"], context="assurance")
    _require_probability(assurance, ["high_confidence_threshold"], context="assurance")
    if int(assurance["n_bins_calibration"]) <= 0:
        raise ValueError("`assurance.n_bins_calibration` must be positive.")
    if int(assurance["pca_components"]) <= 0:
        raise ValueError("`assurance.pca_components` must be positive.")
    if int(assurance["knn_k"]) <= 0:
        raise ValueError("`assurance.knn_k` must be positive.")
    if not isinstance(assurance["embedding_method"], str) or not assurance["embedding_method"].strip():
        raise ValueError("`assurance.embedding_method` must be a non-empty string.")
    config["assurance"] = assurance


def _normalize_intervention_constraints(config: dict[str, Any]) -> None:
    constraints = _deep_merge(
        CONFIG_DEFAULTS["intervention_constraints"],
        _require_mapping(config, "intervention_constraints"),
    )
    if not isinstance(constraints["enabled"], bool):
        raise ValueError("`intervention_constraints.enabled` must be a boolean.")
    probability_keys = [
        "max_overreliance",
        "min_wrong_confident_avoidance",
        "min_deferral_rate",
        "max_deferral_rate",
        "min_audit_coverage",
    ]
    _require_probability(constraints, probability_keys, context="intervention_constraints")
    if float(constraints["min_deferral_rate"]) > float(constraints["max_deferral_rate"]):
        raise ValueError(
            "`intervention_constraints.min_deferral_rate` must be less than or equal to "
            "`intervention_constraints.max_deferral_rate`."
        )
    config["intervention_constraints"] = constraints


def _normalize_expert_routing(config: dict[str, Any]) -> None:
    routing = _deep_merge(CONFIG_DEFAULTS["expert_routing"], _require_mapping(config, "expert_routing"))
    if not isinstance(routing["use_majority_vote_for_escalation"], bool):
        raise ValueError("`expert_routing.use_majority_vote_for_escalation` must be a boolean.")
    if not isinstance(routing["capacity_enabled"], bool):
        raise ValueError("`expert_routing.capacity_enabled` must be a boolean.")
    if not isinstance(routing["fairness_enabled"], bool):
        raise ValueError("`expert_routing.fairness_enabled` must be a boolean.")
    _require_numeric(
        routing,
        ["top_k_for_escalation", "capacity_penalty_weight", "fairness_penalty_weight"],
        context="expert_routing",
    )
    if int(routing["top_k_for_escalation"]) <= 0:
        raise ValueError("`expert_routing.top_k_for_escalation` must be positive.")
    config["expert_routing"] = routing


def _normalize_statistics(config: dict[str, Any]) -> None:
    statistics = _deep_merge(CONFIG_DEFAULTS["statistics"], _require_mapping(config, "statistics"))
    _require_numeric(statistics, ["bootstrap_iterations"], context="statistics")
    _require_probability(statistics, ["confidence_level"], context="statistics")
    if int(statistics["bootstrap_iterations"]) <= 0:
        raise ValueError("`statistics.bootstrap_iterations` must be positive.")
    config["statistics"] = statistics


def _normalize_magd(config: dict[str, Any]) -> None:
    raw_magd = _require_mapping(config, "magd")
    preprocessed = deepcopy(raw_magd)

    if "weights" in preprocessed:
        preprocessed["risk_weights"] = _deep_merge(
            preprocessed.get("risk_weights", {}),
            preprocessed["weights"],
        )

    signals = preprocessed.get("signals", {})
    if signals is None:
        signals = {}
    if not isinstance(signals, dict):
        raise ValueError("`magd.signals` must be a mapping.")
    for legacy_key, normalized_key in LEGACY_SIGNAL_KEYS.items():
        if legacy_key in preprocessed and normalized_key not in signals:
            signals[normalized_key] = preprocessed[legacy_key]
    preprocessed["signals"] = signals

    adaptive = preprocessed.get("adaptive_threshold", {})
    if adaptive is None:
        adaptive = {}
    if not isinstance(adaptive, dict):
        raise ValueError("`magd.adaptive_threshold` must be a mapping.")
    if "value_weight" in adaptive and "business_risk_weight" not in adaptive:
        adaptive["business_risk_weight"] = adaptive["value_weight"]
    if "fairness_weight" in adaptive and "fairness_risk_weight" not in adaptive:
        adaptive["fairness_risk_weight"] = adaptive["fairness_weight"]
    if "capacity_weight" in adaptive and "capacity_pressure_weight" not in adaptive:
        adaptive["capacity_pressure_weight"] = adaptive["capacity_weight"]
    preprocessed["adaptive_threshold"] = adaptive

    magd = _deep_merge(MAGD_DEFAULTS, preprocessed)

    weight_keys = [
        "distance_uncertainty",
        "calibration_risk",
        "neighbor_error_rate",
        "wrong_confident_risk",
        "drift_risk",
        "business_risk",
    ]
    threshold_keys = ["low_risk", "high_risk", "high_confidence", "wrong_confident"]
    threshold_quantile_keys = ["low_quantile", "high_quantile"]
    adaptive_keys = ["base_threshold", "business_risk_weight", "fairness_risk_weight", "capacity_pressure_weight"]
    constrained_keys = ["lambda_overreliance", "lambda_capacity", "lambda_fairness", "lambda_audit"]
    constrained_probability_keys = ["min_correct_rejection", "min_audit_coverage"]
    validation_tuned_keys = ["lambda_deferral", "lambda_recall", "lambda_f1"]
    validation_tuned_probability_keys = ["min_deferral_rate"]
    routing_numeric_keys = ["top_k_for_escalation", "capacity_penalty_weight", "fairness_penalty_weight"]

    _require_numeric(magd["risk_weights"], weight_keys, context="magd.risk_weights")
    _require_numeric(magd["signals"], list(set(LEGACY_SIGNAL_KEYS.values())), context="magd.signals")
    _require_probability(magd["thresholds"], threshold_keys, context="magd.thresholds")
    _require_probability(magd["thresholds"], threshold_quantile_keys, context="magd.thresholds")
    if str(magd["thresholds"].get("mode", "validation_quantile")).lower() not in {"fixed", "validation_quantile"}:
        raise ValueError("`magd.thresholds.mode` must be one of: fixed, validation_quantile.")
    magd["thresholds"]["mode"] = str(magd["thresholds"].get("mode", "validation_quantile")).lower()
    if float(magd["thresholds"]["low_quantile"]) >= float(magd["thresholds"]["high_quantile"]):
        raise ValueError("`magd.thresholds.low_quantile` must be less than `magd.thresholds.high_quantile`.")
    _require_probability(magd["adaptive_threshold"], adaptive_keys, context="magd.adaptive_threshold")
    _require_numeric(magd["constrained_policy"], constrained_keys, context="magd.constrained_policy")
    _require_probability(magd["constrained_policy"], constrained_probability_keys, context="magd.constrained_policy")
    _require_numeric(magd["validation_tuned"], validation_tuned_keys, context="magd.validation_tuned")
    _require_probability(magd["validation_tuned"], validation_tuned_probability_keys, context="magd.validation_tuned")
    if not isinstance(magd["validation_tuned"].get("deferral_budgets"), list) or not magd["validation_tuned"]["deferral_budgets"]:
        raise ValueError("`magd.validation_tuned.deferral_budgets` must be a non-empty list.")
    _require_probability(
        {"budget": max(float(value) for value in magd["validation_tuned"]["deferral_budgets"])},
        ["budget"],
        context="magd.validation_tuned.deferral_budgets",
    )
    _require_probability(
        {"budget": min(float(value) for value in magd["validation_tuned"]["deferral_budgets"])},
        ["budget"],
        context="magd.validation_tuned.deferral_budgets",
    )
    _require_numeric(magd["expert_routing"], routing_numeric_keys, context="magd.expert_routing")

    if int(magd["expert_routing"]["top_k_for_escalation"]) <= 0:
        raise ValueError("`magd.expert_routing.top_k_for_escalation` must be positive.")
    if float(magd["thresholds"]["low_risk"]) > float(magd["thresholds"]["high_risk"]):
        raise ValueError("`magd.thresholds.low_risk` must be less than or equal to `magd.thresholds.high_risk`.")
    if str(magd["mode"]).lower() not in {"heuristic", "learned", "constrained"}:
        raise ValueError("`magd.mode` must be one of: heuristic, learned, constrained.")

    for signal_key, signal_value in magd["signals"].items():
        magd[signal_key] = bool(signal_value)
    magd["use_neighbor_error"] = bool(magd["signals"]["use_neighbor_error_rate"])
    magd["adaptive_threshold"]["value_weight"] = float(magd["adaptive_threshold"]["business_risk_weight"])
    magd["adaptive_threshold"]["fairness_weight"] = float(magd["adaptive_threshold"]["fairness_risk_weight"])
    magd["adaptive_threshold"]["capacity_weight"] = float(magd["adaptive_threshold"]["capacity_pressure_weight"])
    magd["use_expert_reliability"] = True
    magd["weights"] = deepcopy(magd["risk_weights"])

    config["magd"] = magd


def _apply_legacy_compatibility(config: dict[str, Any]) -> None:
    experiment = config["experiment"]
    data = config["data"]
    model = config["model"]
    costs = config["costs"]
    assurance = config["assurance"]
    magd = config["magd"]
    intervention_constraints = config["intervention_constraints"]
    expert_routing = config["expert_routing"]

    config["paths"] = _deep_merge(
        CONFIG_DEFAULTS["paths"],
        _require_mapping(config, "paths"),
    )
    config["paths"]["outputs_dir"] = experiment["output_dir"]

    processed_dir = Path(data["train_path"]).parent.as_posix()
    config["paths"]["processed_data_dir"] = processed_dir

    config["columns"] = _deep_merge(CONFIG_DEFAULTS["columns"], _require_mapping(config, "columns"))
    sensitive_attributes = config["columns"].get("sensitive_attributes")
    if not isinstance(sensitive_attributes, list) or not sensitive_attributes:
        config["columns"]["sensitive_attributes"] = [data["sensitive_attribute"]]

    config["split"] = _deep_merge(CONFIG_DEFAULTS["split"], _require_mapping(config, "split"))
    config["split"]["random_state"] = int(experiment["seed"])

    dataset = _deep_merge(CONFIG_DEFAULTS["dataset"], _require_mapping(config, "dataset"))
    config["dataset"] = dataset

    config["model"]["threshold"] = float(model["threshold"])
    config["model"]["false_positive_cost"] = float(model["false_positive_cost"])
    config["model"]["false_negative_cost"] = float(model["false_negative_cost"])

    config["assurance"] = _deep_merge(
        {
            "calibration_bins": int(assurance["n_bins_calibration"]),
            "neighbor_top_k": int(assurance["knn_k"]),
            "distance_embedding": {
                "method": assurance["embedding_method"],
                "n_components": int(assurance["pca_components"]),
            },
            "wrong_confident": {
                "high_conf_threshold": float(assurance["high_confidence_threshold"]),
                "risk_alert_threshold": float(magd["thresholds"]["wrong_confident"]),
                "top_k_fraction": 0.1,
                "weights": {
                    "numerical_confidence": 0.25,
                    "distance_uncertainty": 0.2,
                    "calibration_risk": 0.2,
                    "neighbor_error_rate": 0.2,
                    "confidence_disagreement": 0.15,
                },
            },
            "distance_thresholds": [0.5, 0.6, 0.7, 0.8, 0.9],
            "assurance_risk": {
                "low_threshold": 0.33,
                "high_threshold": 0.66,
                "weights": {
                    "calibration_risk": 0.2,
                    "distance_uncertainty": 0.2,
                    "neighbor_error_rate": 0.2,
                    "wrong_confident_risk": 0.3,
                    "business_risk": 0.1,
                },
            },
        },
        assurance,
    )

    config["costs"] = costs
    config["magd"]["costs"] = _deep_merge(config["magd"]["costs"], costs)

    config["magd"]["expert_routing"] = _deep_merge(config["magd"]["expert_routing"], expert_routing)
    config["magd"]["signals"]["use_capacity_penalty"] = bool(expert_routing["capacity_enabled"])
    config["magd"]["signals"]["use_fairness_penalty"] = bool(expert_routing["fairness_enabled"])
    config["magd"]["use_capacity_penalty"] = bool(expert_routing["capacity_enabled"])
    config["magd"]["use_fairness_penalty"] = bool(expert_routing["fairness_enabled"])

    config["magd"]["constrained_policy"] = _deep_merge(
        config["magd"]["constrained_policy"],
        {
            "enabled": bool(intervention_constraints["enabled"]),
            "min_audit_coverage": float(intervention_constraints["min_audit_coverage"]),
        },
    )
    config["magd"]["constrained_policy"]["max_overreliance"] = float(intervention_constraints["max_overreliance"])
    config["magd"]["constrained_policy"]["min_wrong_confident_avoidance"] = float(
        intervention_constraints["min_wrong_confident_avoidance"]
    )
    config["magd"]["constrained_policy"]["min_deferral_rate"] = float(intervention_constraints["min_deferral_rate"])
    config["magd"]["constrained_policy"]["max_deferral_rate"] = float(intervention_constraints["max_deferral_rate"])

    config["assurance_deferral"] = _deep_merge(
        CONFIG_DEFAULTS["assurance_deferral"],
        _require_mapping(config, "assurance_deferral"),
    )
    config["assurance_deferral"]["escalation_top_k_experts"] = int(expert_routing["top_k_for_escalation"])
    config["assurance_deferral"]["capacity_penalty_weight"] = float(expert_routing["capacity_penalty_weight"])
    config["baselines"] = _deep_merge(CONFIG_DEFAULTS["baselines"], _require_mapping(config, "baselines"))
    config["baselines"]["random_state"] = int(experiment["seed"])


def normalize_config(config: dict[str, Any], *, config_path: Path) -> dict[str, Any]:
    normalized = _deep_merge(CONFIG_DEFAULTS, config)
    _normalize_experiment(normalized)
    _normalize_data(normalized)
    _normalize_model(normalized)
    _normalize_costs(normalized)
    _normalize_assurance(normalized)
    _normalize_intervention_constraints(normalized)
    _normalize_expert_routing(normalized)
    _normalize_statistics(normalized)
    _normalize_magd(normalized)
    _apply_legacy_compatibility(normalized)
    _resolve_output_dir(config_path, normalized["experiment"]["output_dir"])
    return normalized


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {config_path}")
    return normalize_config(data, config_path=config_path.resolve())
