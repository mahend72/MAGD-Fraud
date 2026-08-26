from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_fifar import load_fifar_data, read_table, resolve_config_path
from src.utils.io import load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class BaselineArtifacts:
    baseline_metrics: pd.DataFrame
    output_dir: Path


METHOD_SPECS = {
    "ai_only": {"deployable": True, "file_name": "ai_only_decisions.csv"},
    "best_expert": {"deployable": True, "file_name": "best_expert_decisions.csv"},
    "random_expert": {"deployable": True, "file_name": "random_expert_decisions.csv"},
    "confidence_threshold": {"deployable": True, "file_name": "confidence_threshold_decisions.csv", "aliases": ["numerical_threshold_decisions.csv"]},
    "distance_threshold": {"deployable": True, "file_name": "distance_threshold_decisions.csv"},
    "oracle_upper_bound": {"deployable": False, "file_name": "oracle_upper_bound_decisions.csv"},
}


def _resolve_output_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    output_dir = outputs_root / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_paper_tables_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    output_dir = outputs_root / "paper_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _assurance_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    return outputs_root / "assurance"


def _model_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    return outputs_root / "model"


def _baseline_config(config: dict) -> dict:
    baseline_cfg = config.get("baselines", {})
    return {
        "numerical_conf_threshold": float(baseline_cfg.get("numerical_conf_threshold", 0.9)),
        "distance_conf_threshold": float(baseline_cfg.get("distance_conf_threshold", 0.7)),
        "random_state": int(baseline_cfg.get("random_state", 42)),
    }


def _costs(config: dict) -> tuple[float, float]:
    model_cfg = config.get("model", {})
    return float(model_cfg.get("false_positive_cost", 1.0)), float(model_cfg.get("false_negative_cost", 5.0))


def _load_required_csv(path: Path, *, required_columns: set[str], friendly_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {friendly_name} at {path}.")
    frame = pd.read_csv(path)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{friendly_name} is missing required columns: {sorted(missing)}")
    return frame


def _candidate_expert_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"case_id", "application_id", "alert_id", "id"}
    columns = [column for column in frame.columns if column not in excluded]
    filtered: list[str] = []
    for column in columns:
        lower = column.lower()
        if column.startswith("model#") or "oracle" in lower:
            continue
        filtered.append(column)
    return filtered


def _historical_best_experts(
    config_path: Path,
    historical_df: pd.DataFrame | None,
) -> list[str]:
    if historical_df is None:
        raise ValueError("Historical expert prediction file is missing; cannot rank experts.")

    loaded = load_fifar_data(config_path)
    train_label_column = loaded.config.get("columns", {}).get("train_label") or loaded.config.get("columns", {}).get("label")
    train_file = resolve_config_path(
        loaded.config_path,
        loaded.config.get("dataset", {}).get("train_file"),
        required=False,
        field_name="dataset.train_file",
    )
    if train_file is None:
        raise ValueError("Train file is required to compute best historical expert.")
    train_df = read_table(train_file)
    if train_label_column not in train_df.columns:
        raise ValueError(f"Train label column `{train_label_column}` not found in {train_file}")

    case_col = loaded.case_id_column
    label_frame = train_df[[case_col, train_label_column]].copy()
    label_frame[case_col] = label_frame[case_col].astype(str)
    historical_case_col = case_col if case_col in historical_df.columns else historical_df.columns[0]
    working_history = historical_df.copy()
    working_history[historical_case_col] = working_history[historical_case_col].astype(str)
    merged = label_frame.merge(working_history, left_on=case_col, right_on=historical_case_col, how="inner")
    overlap_floor = max(100, int(0.01 * min(len(train_df), len(historical_df))))
    if len(merged) < overlap_floor:
        LOGGER.info(
            "Historical expert predictions do not align well on case_id; falling back to row-order alignment for expert ranking."
        )
        n_rows = min(len(label_frame), len(working_history))
        merged = pd.concat(
            [
                label_frame.reset_index(drop=True).iloc[:n_rows],
                working_history.drop(columns=[historical_case_col], errors="ignore").reset_index(drop=True).iloc[:n_rows],
            ],
            axis=1,
        )
    expert_columns = [column for column in _candidate_expert_columns(historical_df) if column in merged.columns]
    scores: list[tuple[str, float]] = []
    for column in expert_columns:
        series = merged[column]
        valid = series.notna()
        if valid.sum() == 0:
            continue
        acc = (series[valid].astype(int) == merged.loc[valid, train_label_column].astype(int)).mean()
        scores.append((column, float(acc)))
    if not scores:
        raise ValueError("No expert columns available for historical ranking.")
    scores.sort(key=lambda item: item[1], reverse=True)
    return [column for column, _ in scores]


def _build_capacity_state(test_metadata: pd.DataFrame, config_path: Path) -> tuple[dict[tuple[str, str], int] | None, str | None]:
    loaded = load_fifar_data(config_path)
    if loaded.capacity_df is None or loaded.capacity_df.empty:
        LOGGER.info("Capacity data is missing; baselines will run without capacity constraints.")
        return None, None
    batch_column = loaded.batch_id_column if loaded.batch_id_column in test_metadata.columns else None
    if batch_column is None:
        LOGGER.info("Capacity data exists but test metadata lacks batch identifiers; ignoring capacity constraints.")
        return None, None

    capacity = loaded.capacity_df.copy()
    if "batch_id" in capacity.columns:
        capacity_batch_col = "batch_id"
    elif loaded.capacity_case_id_column and loaded.capacity_case_id_column in capacity.columns:
        capacity_batch_col = loaded.capacity_case_id_column
    else:
        LOGGER.info("Capacity table does not expose a recognizable batch column; ignoring capacity constraints.")
        return None, None

    expert_columns = _candidate_expert_columns(capacity)
    remaining: dict[tuple[str, str], int] = {}
    for _, row in capacity.iterrows():
        batch_value = str(row[capacity_batch_col])
        for expert in expert_columns:
            value = row.get(expert)
            if pd.notna(value):
                remaining[(batch_value, expert)] = int(value)
    return remaining, batch_column


def _take_available_expert(
    case_row: pd.Series,
    preferred_experts: list[str],
    available_predictions: dict[str, int],
    capacity_state: dict[tuple[str, str], int] | None,
    batch_column: str | None,
) -> tuple[str | None, int | None]:
    for expert in preferred_experts:
        if expert not in available_predictions:
            continue
        if capacity_state is not None and batch_column is not None:
            batch_value = str(case_row[batch_column])
            remaining = capacity_state.get((batch_value, expert), 0)
            if remaining <= 0:
                continue
            capacity_state[(batch_value, expert)] = remaining - 1
        return expert, int(available_predictions[expert])
    return None, None


def _load_expert_tables(config_path: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    loaded = load_fifar_data(config_path)
    expert_df = loaded.expert_df.copy() if loaded.expert_df is not None else None
    if expert_df is not None and loaded.expert_case_id_column is not None:
        expert_df[loaded.expert_case_id_column] = expert_df[loaded.expert_case_id_column].astype(str)

    config = loaded.config
    dataset_cfg = config.get("dataset", {})
    historical_path = resolve_config_path(
        loaded.config_path,
        dataset_cfg.get("historical_expert_predictions_file"),
        required=False,
        field_name="dataset.historical_expert_predictions_file",
    )
    historical_df = read_table(historical_path) if historical_path is not None else None
    if historical_df is not None:
        case_col = loaded.case_id_column if loaded.case_id_column in historical_df.columns else historical_df.columns[0]
        historical_df = historical_df.copy()
        historical_df[case_col] = historical_df[case_col].astype(str)
    return expert_df, historical_df


def _normalize_expert_source(frame: pd.DataFrame, *, case_id_column: str | None) -> pd.DataFrame:
    case_column = case_id_column if case_id_column and case_id_column in frame.columns else frame.columns[0]
    normalized = frame.copy()
    normalized[case_column] = normalized[case_column].astype(str)
    normalized = normalized.rename(columns={case_column: "case_id"})
    expert_columns = _candidate_expert_columns(normalized)
    return normalized[["case_id"] + expert_columns].drop_duplicates(subset=["case_id"], keep="first")


def _row_order_align(case_ids: pd.Series, source: pd.DataFrame) -> pd.DataFrame:
    n_rows = min(len(case_ids), len(source))
    aligned = pd.concat(
        [
            pd.DataFrame({"case_id": case_ids.astype(str).reset_index(drop=True).iloc[:n_rows]}),
            source.drop(columns=["case_id"], errors="ignore").reset_index(drop=True).iloc[:n_rows],
        ],
        axis=1,
    )
    if n_rows < len(case_ids):
        aligned = pd.concat(
            [aligned, pd.DataFrame({"case_id": case_ids.astype(str).reset_index(drop=True).iloc[n_rows:]})],
            ignore_index=True,
        )
    return aligned


def _expert_predictions_for_split(config_path: Path, split_name: str, case_ids: pd.Series) -> pd.DataFrame:
    loaded = load_fifar_data(config_path)
    expert_df, historical_df = _load_expert_tables(config_path)
    case_ids = case_ids.astype(str).reset_index(drop=True)
    empty = pd.DataFrame({"case_id": case_ids})

    if split_name == "test" and expert_df is not None and not expert_df.empty:
        normalized = _normalize_expert_source(expert_df, case_id_column=loaded.expert_case_id_column)
        merged = empty.merge(normalized, on="case_id", how="left")
        if merged.drop(columns=["case_id"]).notna().sum().sum() > 0:
            return merged
        LOGGER.info("Test expert predictions do not align on case_id; falling back to row-order alignment.")
        return _row_order_align(case_ids, normalized)

    if historical_df is not None and not historical_df.empty:
        normalized = _normalize_expert_source(historical_df, case_id_column=loaded.case_id_column)
        merged = empty.merge(normalized, on="case_id", how="left")
        if merged.drop(columns=["case_id"]).notna().sum().sum() > 0:
            return merged
        LOGGER.info("%s expert predictions do not align on case_id; falling back to row-order alignment.", split_name.title())
        return _row_order_align(case_ids, normalized)

    return empty


def _load_split_core(config_path: Path, split_name: str) -> pd.DataFrame:
    model_dir = _model_dir(config_path)
    assurance_dir = _assurance_dir(config_path)
    config = load_yaml(config_path)
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = (config_path.parent / processed_dir).resolve()
    metadata_name = f"{split_name}_metadata.csv" if split_name != "test" else "test_metadata.csv"

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
        required_columns={"case_id", "split", "distance_confidence"},
        friendly_name="distance uncertainty outputs",
    )
    metadata = _load_required_csv(
        processed_dir / metadata_name,
        required_columns={"case_id"},
        friendly_name=f"{split_name} metadata",
    )
    for frame in [predictions, numerical, distance, metadata]:
        frame["case_id"] = frame["case_id"].astype(str)

    core = (
        predictions.loc[predictions["split"].astype(str) == split_name]
        .merge(
            numerical.loc[numerical["split"].astype(str) == split_name, ["case_id", "numerical_confidence"]],
            on="case_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            distance.loc[distance["split"].astype(str) == split_name, ["case_id", "distance_confidence"]],
            on="case_id",
            how="left",
            validate="one_to_one",
        )
        .merge(metadata, on="case_id", how="left")
    )
    expert_predictions = _expert_predictions_for_split(config_path, split_name, core["case_id"])
    core = core.merge(expert_predictions, on="case_id", how="left")
    return core


def _hard_metrics(y_true: pd.Series, y_pred: pd.Series, fp_cost: float, fn_cost: float) -> dict[str, float]:
    y_true_array = y_true.astype(int).to_numpy()
    y_pred_array = y_pred.astype(int).to_numpy()
    tp = int(np.sum((y_true_array == 1) & (y_pred_array == 1)))
    tn = int(np.sum((y_true_array == 0) & (y_pred_array == 0)))
    fp = int(np.sum((y_true_array == 0) & (y_pred_array == 1)))
    fn = int(np.sum((y_true_array == 1) & (y_pred_array == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cost_sensitive_loss": float(fp_cost * fp + fn_cost * fn),
    }


def _decision_frame(
    *,
    core: pd.DataFrame,
    final_prediction: pd.Series,
    selected_route: pd.Series,
    selected_expert: pd.Series,
    decision_reason: pd.Series,
    method: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "case_id": core["case_id"].astype(str),
            "y_true": core["y_true"].astype(int),
            "ai_score": core["ai_score"].astype(float),
            "ai_pred": core["ai_pred"].astype(int),
            "selected_route": selected_route.astype(str),
            "selected_expert": selected_expert.astype(str),
            "final_prediction": final_prediction.astype(int),
            "decision_reason": decision_reason.astype(str),
            "method": pd.Series([method] * len(core), dtype="object"),
        }
    )
    return frame


def _best_expert_name(config_path: Path, expert_columns: list[str], historical_df: pd.DataFrame | None) -> str | None:
    if not expert_columns:
        return None
    if historical_df is None:
        return expert_columns[0]
    ranked = _historical_best_experts(config_path, historical_df)
    ranked = [expert for expert in ranked if expert in expert_columns]
    return ranked[0] if ranked else (expert_columns[0] if expert_columns else None)


def _available_experts_for_row(row: pd.Series, expert_columns: list[str]) -> dict[str, int]:
    return {expert: int(row[expert]) for expert in expert_columns if expert in row.index and pd.notna(row[expert])}


def _expert_columns_from_core(core: pd.DataFrame) -> list[str]:
    excluded = {"case_id", "y_true", "ai_score", "ai_pred", "split", "numerical_confidence", "distance_confidence"}
    columns: list[str] = []
    for column in core.columns:
        if column in excluded or "oracle" in column.lower():
            continue
        numeric = pd.to_numeric(core[column], errors="coerce")
        if numeric.notna().sum() > 0 and numeric.dropna().isin([0, 1]).all():
            columns.append(column)
    return columns


def _run_ai_only(core: pd.DataFrame) -> pd.DataFrame:
    return _decision_frame(
        core=core,
        final_prediction=core["ai_pred"],
        selected_route=pd.Series(["AI"] * len(core)),
        selected_expert=pd.Series([""] * len(core)),
        decision_reason=pd.Series(["ai_only"] * len(core)),
        method="ai_only",
    )


def _run_best_expert(core: pd.DataFrame, ranked_experts: list[str]) -> pd.DataFrame:
    best_expert = ranked_experts[0] if ranked_experts else None
    if best_expert is None or best_expert not in core.columns:
        return _decision_frame(
            core=core,
            final_prediction=core["ai_pred"],
            selected_route=pd.Series(["AI"] * len(core)),
            selected_expert=pd.Series([""] * len(core)),
            decision_reason=pd.Series(["best_expert_unavailable_ai_fallback"] * len(core)),
            method="best_expert",
        )
    available = core[best_expert].notna()
    final_prediction = pd.Series(np.where(available, core[best_expert].fillna(core["ai_pred"]), core["ai_pred"]).astype(int))
    selected_route = pd.Series(np.where(available, "Human Expert", "AI"))
    selected_expert = pd.Series(np.where(available, best_expert, ""))
    decision_reason = pd.Series(np.where(available, "best_expert", "best_expert_unavailable_ai_fallback"))
    return _decision_frame(
        core=core,
        final_prediction=final_prediction,
        selected_route=selected_route,
        selected_expert=selected_expert,
        decision_reason=decision_reason,
        method="best_expert",
    )


def _run_random_expert(core: pd.DataFrame, expert_columns: list[str], random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for _, row in core.iterrows():
        available = list(_available_experts_for_row(row, expert_columns).items())
        if available:
            expert_name, pred = available[int(rng.integers(0, len(available)))]
            final_prediction = int(pred)
            selected_route = "Human Expert"
            reason = "random_expert"
            selected_expert = expert_name
        else:
            final_prediction = int(row["ai_pred"])
            selected_route = "AI"
            reason = "random_expert_unavailable_ai_fallback"
            selected_expert = ""
        rows.append(
            {
                "case_id": str(row["case_id"]),
                "y_true": int(row["y_true"]),
                "ai_score": float(row["ai_score"]),
                "ai_pred": int(row["ai_pred"]),
                "selected_route": selected_route,
                "selected_expert": selected_expert,
                "final_prediction": final_prediction,
                "decision_reason": reason,
                "method": "random_expert",
            }
        )
    return pd.DataFrame(rows)


def _threshold_decisions(core: pd.DataFrame, signal_column: str, threshold: float, ranked_experts: list[str], method: str) -> pd.DataFrame:
    best_expert = ranked_experts[0] if ranked_experts else None
    use_ai = core[signal_column].astype(float) >= threshold
    if best_expert is None or best_expert not in core.columns:
        return _decision_frame(
            core=core,
            final_prediction=core["ai_pred"],
            selected_route=pd.Series(["AI"] * len(core)),
            selected_expert=pd.Series([""] * len(core)),
            decision_reason=pd.Series([f"{method}_ai"] * len(core)),
            method=method,
        )
    expert_available = core[best_expert].notna()
    defer_to_expert = (~use_ai) & expert_available
    final_prediction = pd.Series(np.where(defer_to_expert, core[best_expert].fillna(core["ai_pred"]), core["ai_pred"]).astype(int))
    selected_route = pd.Series(np.where(defer_to_expert, "Human Expert", "AI"))
    selected_expert = pd.Series(np.where(defer_to_expert, best_expert, ""))
    decision_reason = pd.Series(
        np.where(
            defer_to_expert,
            f"{method}_defer",
            np.where(use_ai, f"{method}_ai", f"{method}_expert_unavailable_ai_fallback"),
        )
    )
    return _decision_frame(
        core=core,
        final_prediction=final_prediction,
        selected_route=selected_route,
        selected_expert=selected_expert,
        decision_reason=decision_reason,
        method=method,
    )


def _select_threshold(
    val_core: pd.DataFrame,
    *,
    signal_column: str,
    ranked_experts: list[str],
    fp_cost: float,
    fn_cost: float,
    default_threshold: float,
    method: str,
) -> float:
    if signal_column not in val_core.columns or val_core[signal_column].dropna().empty:
        return default_threshold
    candidate_thresholds = sorted({round(float(v), 4) for v in np.linspace(0.0, 1.0, 21).tolist() + [float(default_threshold)]})
    best_threshold = default_threshold
    best_loss: float | None = None
    for threshold in candidate_thresholds:
        decisions = _threshold_decisions(val_core, signal_column, threshold, ranked_experts, method)
        loss = _hard_metrics(decisions["y_true"], decisions["final_prediction"], fp_cost, fn_cost)["cost_sensitive_loss"]
        if best_loss is None or loss < best_loss:
            best_loss = float(loss)
            best_threshold = float(threshold)
    return best_threshold


def _run_oracle(core: pd.DataFrame, ranked_experts: list[str]) -> pd.DataFrame:
    rows = []
    for _, row in core.iterrows():
        if int(row["ai_pred"]) == int(row["y_true"]):
            final_prediction = int(row["ai_pred"])
            selected_route = "AI"
            selected_expert = ""
            reason = "oracle_upper_bound_ai_correct"
        else:
            available = _available_experts_for_row(row, ranked_experts)
            correct = [(expert, pred) for expert, pred in available.items() if int(pred) == int(row["y_true"])]
            if correct:
                selected_expert, pred = correct[0]
                final_prediction = int(pred)
                selected_route = "Human Expert"
                reason = "oracle_non_deployable_upper_bound"
            else:
                final_prediction = int(row["ai_pred"])
                selected_route = "AI"
                selected_expert = ""
                reason = "oracle_non_deployable_ai_fallback"
        rows.append(
            {
                "case_id": str(row["case_id"]),
                "y_true": int(row["y_true"]),
                "ai_score": float(row["ai_score"]),
                "ai_pred": int(row["ai_pred"]),
                "selected_route": selected_route,
                "selected_expert": selected_expert,
                "final_prediction": final_prediction,
                "decision_reason": reason,
                "method": "oracle_upper_bound",
            }
        )
    return pd.DataFrame(rows)


def _save_decisions(output_dir: Path, method: str, frame: pd.DataFrame) -> None:
    spec = METHOD_SPECS[method]
    frame.to_csv(output_dir / spec["file_name"], index=False)
    for alias in spec.get("aliases", []):
        frame.to_csv(output_dir / alias, index=False)


def _baseline_metrics(method: str, frame: pd.DataFrame, fp_cost: float, fn_cost: float, *, threshold: float | None = None) -> dict[str, float | str | bool]:
    metrics = _hard_metrics(frame["y_true"], frame["final_prediction"], fp_cost, fn_cost)
    metrics.update(
        {
            "method": method,
            "deployable": bool(METHOD_SPECS[method]["deployable"]),
            "ai_coverage": float((frame["selected_route"] == "AI").mean()),
            "expert_deferral_rate": float((frame["selected_route"] == "Human Expert").mean()),
            "escalation_rate": float((frame["selected_route"] == "Escalate").mean()) if "Escalate" in set(frame["selected_route"]) else 0.0,
            "threshold": float(threshold) if threshold is not None else np.nan,
        }
    )
    return metrics


def run_baselines(config_path: str | Path) -> BaselineArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    baseline_cfg = _baseline_config(config)
    fp_cost, fn_cost = _costs(config)
    output_dir = _resolve_output_dir(resolved_config_path)
    paper_tables_dir = _resolve_paper_tables_dir(resolved_config_path)

    val_core = _load_split_core(resolved_config_path, "val")
    test_core = _load_split_core(resolved_config_path, "test")
    _, historical_df = _load_expert_tables(resolved_config_path)
    expert_columns = _expert_columns_from_core(test_core)
    ranked_experts = _historical_best_experts(resolved_config_path, historical_df) if historical_df is not None else expert_columns
    ranked_experts = [expert for expert in ranked_experts if expert in expert_columns]
    if not ranked_experts:
        ranked_experts = expert_columns

    confidence_threshold = _select_threshold(
        val_core,
        signal_column="numerical_confidence",
        ranked_experts=ranked_experts,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        default_threshold=baseline_cfg["numerical_conf_threshold"],
        method="confidence_threshold",
    )
    distance_threshold = _select_threshold(
        val_core,
        signal_column="distance_confidence",
        ranked_experts=ranked_experts,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        default_threshold=baseline_cfg["distance_conf_threshold"],
        method="distance_threshold",
    )

    outputs = {
        "ai_only": _run_ai_only(test_core),
        "best_expert": _run_best_expert(test_core, ranked_experts),
        "random_expert": _run_random_expert(test_core, expert_columns, baseline_cfg["random_state"]),
        "confidence_threshold": _threshold_decisions(test_core, "numerical_confidence", confidence_threshold, ranked_experts, "confidence_threshold"),
        "distance_threshold": _threshold_decisions(test_core, "distance_confidence", distance_threshold, ranked_experts, "distance_threshold"),
        "oracle_upper_bound": _run_oracle(test_core, ranked_experts),
    }

    metrics_rows: list[dict[str, float | str | bool]] = []
    for method, frame in outputs.items():
        _save_decisions(output_dir, method, frame)
        threshold = confidence_threshold if method == "confidence_threshold" else distance_threshold if method == "distance_threshold" else None
        metrics_rows.append(_baseline_metrics(method, frame, fp_cost, fn_cost, threshold=threshold))

    baseline_metrics = pd.DataFrame(metrics_rows)
    baseline_metrics.to_csv(output_dir / "baseline_metrics.csv", index=False)
    baseline_metrics.to_csv(paper_tables_dir / "baseline_comparison.csv", index=False)
    return BaselineArtifacts(baseline_metrics=baseline_metrics, output_dir=output_dir)
