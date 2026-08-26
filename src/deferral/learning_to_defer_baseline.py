from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.deferral.baselines import _expert_predictions_for_split, _historical_best_experts, _load_expert_tables
from src.evaluation.fraud_metrics import compute_fraud_metrics
from src.utils.io import load_yaml


@dataclass
class LearningToDeferArtifacts:
    decisions: pd.DataFrame
    metrics: pd.DataFrame
    output_dir: Path


STANDARD_FEATURE_COLUMNS = [
    "ai_score",
    "numerical_confidence",
    "distance_uncertainty",
    "calibration_risk",
    "neighbor_error_rate",
    "wrong_confident_risk",
]
"""Independently-deployable baseline features: the base model score plus the four
deployable assurance signals (numerical confidence, distance uncertainty, calibration
risk, neighbourhood reliability, wrong-confident risk), none of which is computed by
MAGD_AR itself - each has its own standalone module (numerical_confidence.py,
distance_uncertainty.py, calibration.py, local_reliability.py,
wrong_confident_detector.py) and is available whether or not MAGD_AR ever runs."""

AUGMENTED_FEATURE_COLUMNS = STANDARD_FEATURE_COLUMNS + ["magd_assurance_risk"]
"""Standard features plus magd_assurance_risk - the literal output of
compute_magd_risk(). Using this feature set makes Learning-to-Defer dependent on the
MAGD method it is meant to be an independent comparator against, so it must be
reported as an explicitly-labeled augmented variant, never as the plain baseline."""

# Backward-compatible alias: existing callers/tests that import FEATURE_COLUMNS
# directly continue to get the augmented (current, pre-existing) feature set unchanged.
FEATURE_COLUMNS = AUGMENTED_FEATURE_COLUMNS

FEATURE_SETS = {
    "standard": STANDARD_FEATURE_COLUMNS,
    "augmented": AUGMENTED_FEATURE_COLUMNS,
}


def _resolve_output_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    output_dir = outputs_root / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _model_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    return outputs_root / "model"


def _assurance_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    return outputs_root / "assurance"


def _baseline_config(config: dict) -> dict:
    baselines = config.get("baselines", {})
    return {
        "model_type": str(baselines.get("learning_to_defer_model", "logistic_regression")).lower(),
    }


def _cost_config(config: dict) -> tuple[float, float]:
    costs = config.get("costs", {})
    if costs:
        return float(costs.get("false_positive", 0.057)), float(costs.get("false_negative", 1.0))
    model = config.get("model", {})
    return float(model.get("false_positive_cost", 1.0)), float(model.get("false_negative_cost", 5.0))


def _load_required_csv(path: Path, *, required_columns: set[str], friendly_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {friendly_name} at {path}.")
    frame = pd.read_csv(path)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{friendly_name} is missing required columns: {sorted(missing)}")
    return frame


def _load_split_feature_frame(config_path: Path, split_name: str) -> pd.DataFrame:
    model_dir = _model_dir(config_path)
    assurance_dir = _assurance_dir(config_path)

    predictions = _load_required_csv(
        model_dir / f"{split_name}_predictions.csv",
        required_columns={"case_id", "y_true", "ai_score", "ai_pred", "split"},
        friendly_name=f"{split_name} predictions",
    )
    numerical = _load_required_csv(
        assurance_dir / "numerical_confidence.csv",
        required_columns={"case_id", "split", "numerical_confidence"},
        friendly_name="numerical confidence outputs",
    )
    distance = _load_required_csv(
        assurance_dir / "distance_uncertainty.csv",
        required_columns={"case_id", "split", "distance_uncertainty"},
        friendly_name="distance uncertainty outputs",
    )
    calibration = _load_required_csv(
        assurance_dir / "calibration_risk.csv",
        required_columns={"case_id", "split", "calibration_risk"},
        friendly_name="calibration risk outputs",
    )
    local = _load_required_csv(
        assurance_dir / "local_reliability.csv",
        required_columns={"case_id", "neighbor_error_rate"},
        friendly_name="local reliability outputs",
    )
    wrong_conf = _load_required_csv(
        assurance_dir / "wrong_confident_risk.csv",
        required_columns={"case_id", "split", "wrong_confident_risk"},
        friendly_name="wrong-confident risk outputs",
    )
    magd = _load_required_csv(
        assurance_dir / "magd_risk.csv",
        required_columns={"case_id", "split", "magd_assurance_risk"},
        friendly_name="MAGD risk outputs",
    )

    for frame in [predictions, numerical, distance, calibration, local, wrong_conf, magd]:
        frame["case_id"] = frame["case_id"].astype(str)
    working = predictions.loc[predictions["split"].astype(str) == split_name].copy()
    working = (
        working.merge(
            numerical.loc[numerical["split"].astype(str) == split_name, ["case_id", "numerical_confidence"]],
            on="case_id",
            how="inner",
        )
        .merge(
            distance.loc[distance["split"].astype(str) == split_name, ["case_id", "distance_uncertainty"]],
            on="case_id",
            how="inner",
        )
        .merge(
            calibration.loc[calibration["split"].astype(str) == split_name, ["case_id", "calibration_risk"]],
            on="case_id",
            how="inner",
        )
        .merge(
            wrong_conf.loc[wrong_conf["split"].astype(str) == split_name, ["case_id", "wrong_confident_risk"]],
            on="case_id",
            how="inner",
        )
        .merge(
            magd.loc[magd["split"].astype(str) == split_name, ["case_id", "magd_assurance_risk"]],
            on="case_id",
            how="inner",
        )
    )
    if "split" in local.columns and local["split"].notna().any():
        working = working.merge(
            local.loc[local["split"].astype(str) == split_name, ["case_id", "neighbor_error_rate"]],
            on="case_id",
            how="inner",
        )
    else:
        working = working.merge(local[["case_id", "neighbor_error_rate"]], on="case_id", how="inner")
    return working


def _best_available_expert(row: pd.Series, ranked_experts: list[str]) -> tuple[str | None, int | None]:
    for expert in ranked_experts:
        if expert in row.index and pd.notna(row[expert]):
            return expert, int(row[expert])
    return None, None


def _route_label(row: pd.Series, ranked_experts: list[str], fp_cost: float, fn_cost: float) -> int:
    ai_pred = int(row["ai_pred"])
    y_true = int(row["y_true"])
    ai_cost = fp_cost * int(ai_pred == 1 and y_true == 0) + fn_cost * int(ai_pred == 0 and y_true == 1)
    expert_name, expert_pred = _best_available_expert(row, ranked_experts)
    if expert_name is None or expert_pred is None:
        return 0
    expert_cost = fp_cost * int(expert_pred == 1 and y_true == 0) + fn_cost * int(expert_pred == 0 and y_true == 1)
    return int(expert_cost < ai_cost)


def _route_labels(frame: pd.DataFrame, ranked_experts: list[str], fp_cost: float, fn_cost: float) -> pd.Series:
    y_true = frame["y_true"].astype(int)
    ai_pred = frame["ai_pred"].astype(int)
    ai_cost = fp_cost * ((ai_pred == 1) & (y_true == 0)).astype(float) + fn_cost * ((ai_pred == 0) & (y_true == 1)).astype(float)
    expert_columns = [expert for expert in ranked_experts if expert in frame.columns]
    if not expert_columns:
        return pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)
    expert_pred = frame[expert_columns].bfill(axis=1).iloc[:, 0]
    has_expert = expert_pred.notna()
    expert_pred = pd.to_numeric(expert_pred, errors="coerce").fillna(ai_pred).astype(int)
    expert_cost = fp_cost * ((expert_pred == 1) & (y_true == 0)).astype(float) + fn_cost * ((expert_pred == 0) & (y_true == 1)).astype(float)
    return ((expert_cost < ai_cost) & has_expert).astype(int)


def _fit_rejector(X: pd.DataFrame, y: pd.Series, model_type: str):
    positive_rate = float(y.mean()) if len(y) else 0.0
    imbalance = positive_rate < 0.2 or positive_rate > 0.8

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced" if imbalance else None,
        )
        model.fit(X, y)
        return model, "random_forest"

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced" if imbalance else None,
    )
    model.fit(X, y)
    return model, "logistic_regression"


def run_learning_to_defer_baseline(config_path: str | Path, *, feature_set: str = "standard") -> LearningToDeferArtifacts:
    """Train and evaluate the Learning-to-Defer baseline.

    `feature_set="standard"` (L2D-Standard, THE DEFAULT and the official independent
    baseline used by final paper scripts) uses only independently-deployable baseline
    features (STANDARD_FEATURE_COLUMNS) - no MAGD-derived quantity. It owns the
    canonical, unsuffixed output filenames (learning_to_defer_decisions.csv /
    learning_to_defer_metrics.csv).

    `feature_set="augmented"` (L2D+MAGD) additionally includes magd_assurance_risk and
    must be reported as an explicitly-labeled OPTIONAL augmented experiment, never as
    an independent comparator against MAGD - it is written to
    learning_to_defer_augmented_decisions.csv / _metrics.csv, never the canonical name.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown Learning-to-Defer feature_set `{feature_set}`; expected one of {sorted(FEATURE_SETS)}.")
    feature_columns = FEATURE_SETS[feature_set]
    method_name = "learning_to_defer" if feature_set == "standard" else "learning_to_defer_augmented"
    file_stem = "learning_to_defer" if feature_set == "standard" else "learning_to_defer_augmented"

    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    baseline_cfg = _baseline_config(config)
    output_dir = _resolve_output_dir(resolved_config_path)
    fp_cost, fn_cost = _cost_config(config)

    train_core = _load_split_feature_frame(resolved_config_path, "train")
    val_core = _load_split_feature_frame(resolved_config_path, "val")
    test_core = _load_split_feature_frame(resolved_config_path, "test")

    _, historical_df = _load_expert_tables(resolved_config_path)
    ranked_experts = _historical_best_experts(resolved_config_path, historical_df) if historical_df is not None else []

    for split_name, frame in [("train", train_core), ("val", val_core), ("test", test_core)]:
        if frame.empty:
            continue
        expert_predictions = _expert_predictions_for_split(resolved_config_path, split_name, frame["case_id"])
        frame["case_id"] = frame["case_id"].astype(str)
        expert_predictions["case_id"] = expert_predictions["case_id"].astype(str)
        merged = frame.merge(expert_predictions, on="case_id", how="left")
        if split_name == "train":
            train_core = merged
        elif split_name == "val":
            val_core = merged
        else:
            test_core = merged

    available_experts = [expert for expert in ranked_experts if expert in train_core.columns or expert in val_core.columns or expert in test_core.columns]
    budget_score_diagnostic = ""
    budget_val_scores = pd.Series(dtype=float)
    budget_test_scores = pd.Series(dtype=float)
    history = pd.concat([train_core, val_core], ignore_index=True)
    history = history.dropna(subset=feature_columns + ["y_true", "ai_pred"]).copy()
    if history.empty:
        raise ValueError("No non-test rows with complete L2D features are available.")
    history["route_label"] = _route_labels(history, available_experts, fp_cost, fn_cost)

    rejector, model_name = _fit_rejector(
        history[feature_columns].astype(float),
        history["route_label"].astype(int),
        baseline_cfg["model_type"],
    )
    try:
        if hasattr(rejector, "predict_proba"):
            budget_val_scores = pd.Series(
                rejector.predict_proba(val_core[feature_columns].astype(float))[:, 1],
                index=val_core.index,
            )
            budget_test_scores = pd.Series(
                rejector.predict_proba(test_core[feature_columns].astype(float))[:, 1],
                index=test_core.index,
            )
        elif hasattr(rejector, "decision_function"):
            val_raw = np.asarray(rejector.decision_function(val_core[feature_columns].astype(float)), dtype=float)
            test_raw = np.asarray(rejector.decision_function(test_core[feature_columns].astype(float)), dtype=float)
            budget_val_scores = pd.Series(1.0 / (1.0 + np.exp(-val_raw)), index=val_core.index)
            budget_test_scores = pd.Series(1.0 / (1.0 + np.exp(-test_raw)), index=test_core.index)
        else:
            budget_score_diagnostic = "Learning-to-defer rejector has no score output for budget matching."
    except Exception as exc:
        budget_score_diagnostic = f"Learning-to-defer budget scores unavailable: {exc}"

    predicted_human = pd.Series(rejector.predict(test_core[feature_columns].astype(float)), index=test_core.index).astype(int)
    rows: list[dict[str, object]] = []
    for position, (_, row) in enumerate(test_core.iterrows()):
        use_human = bool(predicted_human.iloc[position] == 1)
        expert_name, expert_pred = _best_available_expert(row, available_experts)
        if use_human and expert_name is not None and expert_pred is not None:
            selected_route = "Human Expert"
            selected_expert = expert_name
            final_prediction = int(expert_pred)
            decision_reason = "learning_to_defer_rejector_prefers_human"
        else:
            selected_route = "AI"
            selected_expert = ""
            final_prediction = int(row["ai_pred"])
            decision_reason = (
                "learning_to_defer_rejector_prefers_ai"
                if not use_human
                else "learning_to_defer_expert_unavailable_ai_fallback"
            )
        rows.append(
            {
                "case_id": str(row["case_id"]),
                "y_true": int(row["y_true"]),
                "ai_score": float(row["ai_score"]),
                "ai_pred": int(row["ai_pred"]),
                "selected_route": selected_route,
                "selected_expert": selected_expert,
                "final_prediction": int(final_prediction),
                "decision_reason": decision_reason,
                "method": method_name,
                "rejector_model": model_name,
                "feature_set": feature_set,
            }
        )

    decisions = pd.DataFrame(rows)
    if len(budget_test_scores) == len(decisions):
        decisions["deferral_score"] = budget_test_scores.reset_index(drop=True).astype(float)
    elif budget_score_diagnostic:
        decisions["deferral_score_diagnostic"] = budget_score_diagnostic

    budget_score_rows = []
    if len(budget_val_scores) == len(val_core):
        budget_score_rows.append(
            pd.DataFrame(
                {
                    "case_id": val_core["case_id"].astype(str).to_numpy(),
                    "split": "val",
                    "l2d_deferral_score": budget_val_scores.reset_index(drop=True).astype(float).to_numpy(),
                    "diagnostic": "",
                }
            )
        )
    if len(budget_test_scores) == len(test_core):
        budget_score_rows.append(
            pd.DataFrame(
                {
                    "case_id": test_core["case_id"].astype(str).to_numpy(),
                    "split": "test",
                    "l2d_deferral_score": budget_test_scores.reset_index(drop=True).astype(float).to_numpy(),
                    "diagnostic": "",
                }
            )
        )
    if budget_score_rows:
        pd.concat(budget_score_rows, ignore_index=True).to_csv(output_dir / f"{file_stem}_budget_scores.csv", index=False)
    else:
        pd.DataFrame(
            [{"case_id": "", "split": "", "l2d_deferral_score": np.nan, "diagnostic": budget_score_diagnostic or "Learning-to-defer score unavailable."}]
        ).to_csv(output_dir / f"{file_stem}_budget_scores.csv", index=False)

    fraud = compute_fraud_metrics(decisions, score_column="ai_score", fp_cost=fp_cost, fn_cost=fn_cost)
    route_counts = decisions["selected_route"].value_counts(normalize=True)
    metrics = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "feature_set": feature_set,
                "feature_columns": ",".join(feature_columns),
                **fraud,
                "ai_coverage": float(route_counts.get("AI", 0.0)),
                "human_deferral_rate": float(route_counts.get("Human Expert", 0.0)),
            }
        ]
    )

    decisions.to_csv(output_dir / f"{file_stem}_decisions.csv", index=False)
    metrics.to_csv(output_dir / f"{file_stem}_metrics.csv", index=False)
    return LearningToDeferArtifacts(decisions=decisions, metrics=metrics, output_dir=output_dir)
