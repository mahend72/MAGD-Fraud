from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_fifar import load_fifar_data, read_table, resolve_config_path
from src.deferral.baselines import _build_capacity_state, _candidate_expert_columns, _historical_best_experts
from src.deferral.capacity_assignment import CapacityState, allocate_expert, capacity_penalty_for_expert
from src.evaluation.fairness_metrics import compute_group_error_rates, summarize_disparity
from src.models.predict import load_processed_split
from src.utils.io import load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ExpertRoutingArtifacts:
    expert_reliability: pd.DataFrame
    routing_decisions: pd.DataFrame
    output_dir: Path


def _resolve_output_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    output_dir = outputs_root / "assurance_deferral"
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


def _routing_config(config: dict) -> dict:
    magd_cfg = config.get("magd", {})
    routing_cfg = magd_cfg.get("expert_routing", {})
    costs_cfg = magd_cfg.get("costs", {})
    return {
        "fp_cost": float(costs_cfg.get("false_positive", 0.057)),
        "fn_cost": float(costs_cfg.get("false_negative", 1.0)),
        "escalation_cost": float(costs_cfg.get("escalation", 0.10)),
        "human_review_cost": float(costs_cfg.get("human_review", 0.05)),
        "top_k": int(routing_cfg.get("top_k_for_escalation", 5)),
        "capacity_penalty_weight": float(routing_cfg.get("capacity_penalty_weight", 0.20)),
        "fairness_penalty_weight": float(routing_cfg.get("fairness_penalty_weight", 0.20)),
        "use_majority_vote_for_escalation": bool(routing_cfg.get("use_majority_vote_for_escalation", True)),
    }


def _load_expert_tables(config_path: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    loaded = load_fifar_data(config_path)
    expert_df = loaded.expert_df.copy() if loaded.expert_df is not None else None
    if expert_df is not None and loaded.expert_case_id_column is not None:
        expert_df[loaded.expert_case_id_column] = expert_df[loaded.expert_case_id_column].astype(str)

    dataset_cfg = loaded.config.get("dataset", {})
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


def _non_oracle_expert_columns(frame: pd.DataFrame) -> list[str]:
    columns = _candidate_expert_columns(frame)
    filtered: list[str] = []
    for column in columns:
        lower = column.lower()
        if "oracle" in lower:
            continue
        filtered.append(column)
    return filtered


def _remaining_capacity_summary(config_path: Path) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    loaded = load_fifar_data(config_path)
    if loaded.capacity_df is None or loaded.capacity_df.empty:
        return {}, {}

    capacity = loaded.capacity_df.copy()
    if "batch_id" in capacity.columns:
        batch_col = "batch_id"
    elif loaded.capacity_case_id_column and loaded.capacity_case_id_column in capacity.columns:
        batch_col = loaded.capacity_case_id_column
    else:
        return {}, {}

    per_expert_batch: dict[str, dict[str, int]] = {}
    per_expert_total: dict[str, int] = {}
    for expert in _candidate_expert_columns(capacity):
        batch_map: dict[str, int] = {}
        total = 0
        for _, row in capacity.iterrows():
            value = row.get(expert)
            if pd.isna(value):
                continue
            batch_value = str(row[batch_col])
            batch_capacity = int(value)
            batch_map[batch_value] = batch_capacity
            total += batch_capacity
        if batch_map:
            per_expert_batch[expert] = batch_map
            per_expert_total[expert] = total
    return per_expert_batch, per_expert_total


def _group_rates(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float]:
    y = y_true.astype(int)
    p = y_pred.astype(int)
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tp = int(((p == 1) & (y == 1)).sum())
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return float(fpr), float(fnr)


def _processed_dir(config_path: Path) -> Path:
    config = load_yaml(config_path)
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = (config_path.parent / processed_dir).resolve()
    return processed_dir


def _load_train_validation_reference(config_path: Path) -> pd.DataFrame:
    processed_dir = _processed_dir(config_path)
    loaded = load_fifar_data(config_path)

    X_train, y_train, train_metadata = load_processed_split(processed_dir, "train")
    X_val, y_val, val_metadata = load_processed_split(processed_dir, "val")
    _ = X_train, X_val

    train = train_metadata.copy()
    train["split"] = "train"
    train["y_true"] = y_train.iloc[:, 0].astype(int).reset_index(drop=True)

    val = val_metadata.copy()
    val["split"] = "val"
    val["y_true"] = y_val.iloc[:, 0].astype(int).reset_index(drop=True)

    reference = pd.concat([train, val], ignore_index=True)
    reference["case_id"] = reference["case_id"].astype(str)
    keep_columns = ["case_id", "split", "y_true"]
    for sensitive in loaded.sensitive_attributes:
        if sensitive in reference.columns:
            keep_columns.append(sensitive)
    return reference[keep_columns].copy()


def _normalize_expert_source(frame: pd.DataFrame, *, case_id_column: str | None) -> pd.DataFrame:
    case_column = case_id_column if case_id_column and case_id_column in frame.columns else frame.columns[0]
    normalized = frame.copy()
    normalized[case_column] = normalized[case_column].astype(str)
    normalized = normalized.rename(columns={case_column: "case_id"})
    expert_columns = _non_oracle_expert_columns(normalized)
    if not expert_columns:
        return pd.DataFrame(columns=["case_id"])
    keep = ["case_id"] + expert_columns
    return normalized[keep].drop_duplicates(subset=["case_id"], keep="first")


def _merge_expert_sources(reference_case_ids: pd.Series, expert_sources: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.DataFrame({"case_id": reference_case_ids.astype(str).drop_duplicates().tolist()})
    for source in expert_sources:
        if source is None or source.empty:
            continue
        normalized = source.copy()
        duplicate_experts = [column for column in normalized.columns if column != "case_id" and column in merged.columns]
        if not duplicate_experts:
            merged = merged.merge(normalized, on="case_id", how="left")
            continue
        fresh_columns = [column for column in normalized.columns if column == "case_id" or column not in merged.columns]
        if len(fresh_columns) > 1:
            merged = merged.merge(normalized[fresh_columns], on="case_id", how="left")
        overlapping = merged[["case_id"] + duplicate_experts].merge(
            normalized[["case_id"] + duplicate_experts],
            on="case_id",
            how="left",
            suffixes=("", "_src"),
        )
        for column in duplicate_experts:
            merged[column] = overlapping[column].combine_first(overlapping[f"{column}_src"])
    return merged


def _fallback_align_by_row_order(reference: pd.DataFrame, expert_source: pd.DataFrame) -> pd.DataFrame:
    n_rows = min(len(reference), len(expert_source))
    if n_rows == 0:
        return pd.DataFrame({"case_id": reference["case_id"].astype(str)})
    aligned = pd.concat(
        [
            reference[["case_id"]].reset_index(drop=True).iloc[:n_rows],
            expert_source.drop(columns=["case_id"], errors="ignore").reset_index(drop=True).iloc[:n_rows],
        ],
        axis=1,
    )
    if n_rows < len(reference):
        tail = pd.DataFrame({"case_id": reference["case_id"].reset_index(drop=True).iloc[n_rows:]})
        aligned = pd.concat([aligned, tail], ignore_index=True)
    return aligned


def build_expert_reliability_frame(config_path: str | Path) -> tuple[pd.DataFrame, bool]:
    resolved_config_path = Path(config_path).resolve()
    loaded = load_fifar_data(resolved_config_path)
    reference = _load_train_validation_reference(resolved_config_path)
    expert_df, historical_df = _load_expert_tables(resolved_config_path)

    expert_sources: list[pd.DataFrame] = []
    normalized_historical: pd.DataFrame | None = None
    if historical_df is not None and not historical_df.empty:
        normalized_historical = _normalize_expert_source(historical_df, case_id_column=loaded.case_id_column)
        expert_sources.append(normalized_historical)
    if expert_df is not None and not expert_df.empty:
        expert_sources.append(_normalize_expert_source(expert_df, case_id_column=loaded.expert_case_id_column))

    merged_experts = _merge_expert_sources(reference["case_id"], expert_sources)
    expert_value_columns = [column for column in merged_experts.columns if column != "case_id"]
    if expert_value_columns and merged_experts[expert_value_columns].notna().sum().sum() == 0 and normalized_historical is not None:
        LOGGER.warning("Expert reliability: no case_id overlap with historical expert predictions; falling back to row-order alignment.")
        merged_experts = _fallback_align_by_row_order(reference, normalized_historical)
    merged = reference.merge(merged_experts, on="case_id", how="left")
    capacity_by_expert, capacity_totals = _remaining_capacity_summary(resolved_config_path)
    capacity_enabled = bool(capacity_totals)
    if not capacity_enabled:
        LOGGER.info("Expert reliability: capacity_enabled=false")

    cfg = _routing_config(load_yaml(resolved_config_path))
    expert_columns = [column for column in merged.columns if column not in {"case_id", "split", "y_true", *loaded.sensitive_attributes}]
    expert_columns = [column for column in expert_columns if "oracle" not in column.lower()]
    reliability = estimate_expert_reliability(
        merged,
        expert_columns=expert_columns,
        label_column="y_true",
        sensitive_attributes=loaded.sensitive_attributes,
        fp_cost=cfg["fp_cost"],
        fn_cost=cfg["fn_cost"],
        capacity_by_expert=capacity_by_expert,
        capacity_totals=capacity_totals,
        fairness_penalty_weight=cfg["fairness_penalty_weight"],
        capacity_penalty_weight=cfg["capacity_penalty_weight"],
        capacity_enabled=capacity_enabled,
    )
    return reliability, capacity_enabled


def estimate_expert_reliability(
    merged: pd.DataFrame,
    *,
    expert_columns: list[str],
    label_column: str,
    sensitive_attributes: list[str],
    fp_cost: float,
    fn_cost: float,
    capacity_by_expert: dict[str, dict[str, int]] | None = None,
    capacity_totals: dict[str, int] | None = None,
    fairness_penalty_weight: float = 0.0,
    capacity_penalty_weight: float = 0.0,
    capacity_enabled: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    capacity_by_expert = capacity_by_expert or {}
    capacity_totals = capacity_totals or {}

    for expert in expert_columns:
        if "oracle" in expert.lower():
            continue
        if expert not in merged.columns:
            continue
        pred = merged[expert]
        valid = pred.notna()
        if valid.sum() == 0:
            continue
        y = merged.loc[valid, label_column].astype(int)
        p = pred.loc[valid].astype(int)
        accuracy = float((p == y).mean())
        fpr, fnr = _group_rates(y, p)
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        cost_sensitive_loss = float(fp_cost * fp + fn_cost * fn)

        fairness_available = False
        disparity_summary = ""
        groupwise_fpr = 0.0
        groupwise_fnr = 0.0
        fpr_disparity = 0.0
        fnr_disparity = 0.0
        bias_risk = 0.0
        for sensitive in sensitive_attributes:
            if sensitive not in merged.columns:
                continue
            rates = compute_group_error_rates(
                merged.loc[valid].assign(_expert_pred=merged.loc[valid, expert].astype(int)),
                sensitive_column=sensitive,
                y_true_column=label_column,
                y_pred_column="_expert_pred",
            )
            if rates.empty:
                continue
            disparity = summarize_disparity(rates)
            fairness_available = True
            groupwise_fpr = max(groupwise_fpr, float(disparity["groupwise_false_positive_rate"]))
            groupwise_fnr = max(groupwise_fnr, float(disparity["groupwise_false_negative_rate"]))
            fpr_disparity = max(fpr_disparity, float(disparity["false_positive_rate_disparity"]))
            fnr_disparity = max(fnr_disparity, float(disparity["false_negative_rate_disparity"]))
            bias_risk = max(bias_risk, float(disparity["bias_risk"]))
            disparity_summary = str(disparity["disparity_summary"])

        remaining_capacity_total = float(capacity_totals.get(expert, 0))
        capacity_penalty = 1.0 if capacity_enabled and remaining_capacity_total <= 0 else 0.0
        expected_cost = (
            fp_cost * fpr
            + fn_cost * fnr
            + fairness_penalty_weight * bias_risk
            + capacity_penalty_weight * capacity_penalty
        )

        rows.append(
            {
                "expert": expert,
                "accuracy": accuracy,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "cost_sensitive_loss": cost_sensitive_loss,
                "groupwise_false_positive_rate": float(groupwise_fpr),
                "groupwise_false_negative_rate": float(groupwise_fnr),
                "false_positive_rate_disparity": float(fpr_disparity),
                "false_negative_rate_disparity": float(fnr_disparity),
                "disparity_summary": disparity_summary,
                "bias_risk": bias_risk,
                "fairness_available": bool(fairness_available),
                "remaining_capacity_total": remaining_capacity_total,
                "remaining_capacity_by_batch": json.dumps(capacity_by_expert.get(expert, {}), sort_keys=True),
                "capacity_enabled": bool(capacity_enabled),
                "capacity_penalty": float(capacity_penalty),
                "expected_cost": float(expected_cost),
            }
        )

    columns = [
        "expert",
        "accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "cost_sensitive_loss",
        "groupwise_false_positive_rate",
        "groupwise_false_negative_rate",
        "false_positive_rate_disparity",
        "false_negative_rate_disparity",
        "disparity_summary",
        "bias_risk",
        "fairness_available",
        "remaining_capacity_total",
        "remaining_capacity_by_batch",
        "capacity_enabled",
        "capacity_penalty",
        "expected_cost",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["expected_cost", "cost_sensitive_loss", "bias_risk", "false_negative_rate", "false_positive_rate", "accuracy"],
        ascending=[True, True, True, True, True, False],
    ).reset_index(drop=True)


def run_expert_reliability(config_path: str | Path) -> ExpertRoutingArtifacts:
    resolved_config_path = Path(config_path).resolve()
    output_dir = _resolve_output_dir(resolved_config_path)
    paper_tables_dir = _resolve_paper_tables_dir(resolved_config_path)
    expert_metrics, _ = build_expert_reliability_frame(resolved_config_path)
    expert_metrics.to_csv(output_dir / "expert_reliability.csv", index=False)
    summary_columns = [
        "expert",
        "accuracy",
        "false_positive_rate",
        "false_negative_rate",
        "cost_sensitive_loss",
        "bias_risk",
        "expected_cost",
        "remaining_capacity_total",
        "capacity_enabled",
        "fairness_available",
    ]
    summary = expert_metrics[summary_columns].copy() if not expert_metrics.empty else pd.DataFrame(columns=summary_columns)
    summary.to_csv(paper_tables_dir / "expert_reliability_summary.csv", index=False)
    return ExpertRoutingArtifacts(
        expert_reliability=expert_metrics,
        routing_decisions=pd.DataFrame(),
        output_dir=output_dir,
    )


def _ai_expected_cost(row: pd.Series, fp_cost: float, fn_cost: float) -> float:
    ai_score = float(row["ai_score"])
    ai_pred = int(row["ai_pred"])
    estimated_ai_fp_risk = (1.0 - ai_score) if ai_pred == 1 else 0.0
    estimated_ai_fn_risk = ai_score if ai_pred == 0 else 0.0
    return float(fp_cost * estimated_ai_fp_risk + fn_cost * estimated_ai_fn_risk + float(row["magd_assurance_risk"]))


def _expert_expected_cost(
    case_row: pd.Series,
    expert_name: str,
    expert_metrics_lookup: dict[str, dict[str, float]],
    capacity_state: CapacityState,
    *,
    fp_cost: float,
    fn_cost: float,
    fairness_penalty_weight: float,
    capacity_penalty_weight: float,
) -> tuple[float, float]:
    stats = expert_metrics_lookup.get(
        expert_name,
        {
            "false_positive_rate": 0.5,
            "false_negative_rate": 0.5,
            "bias_risk": 0.0,
        },
    )
    fairness_penalty = fairness_penalty_weight * float(stats.get("bias_risk", 0.0))
    capacity_penalty = capacity_penalty_for_expert(case_row, expert_name, capacity_state, capacity_penalty_weight)
    expected_cost = (
        fp_cost * float(stats["false_positive_rate"])
        + fn_cost * float(stats["false_negative_rate"])
        + fairness_penalty
        + capacity_penalty
    )
    return float(expected_cost), float(fairness_penalty)


def _best_available_expert_candidate(
    case_row: pd.Series,
    ranked_experts: list[str],
    expert_metrics_lookup: dict[str, dict[str, float]],
    capacity_state: CapacityState,
    *,
    fp_cost: float,
    fn_cost: float,
    fairness_penalty_weight: float,
    capacity_penalty_weight: float,
    human_review_cost: float,
) -> tuple[str | None, int | None, float | None, float]:
    best_choice: tuple[str, int, float, float] | None = None
    for expert in ranked_experts:
        if expert not in case_row or pd.isna(case_row[expert]):
            continue
        expert_cost, fairness_penalty = _expert_expected_cost(
            case_row,
            expert,
            expert_metrics_lookup,
            capacity_state,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            fairness_penalty_weight=fairness_penalty_weight,
            capacity_penalty_weight=capacity_penalty_weight,
        )
        total_cost = expert_cost + human_review_cost
        pred = int(case_row[expert])
        if best_choice is None or total_cost < best_choice[2]:
            best_choice = (expert, pred, float(total_cost), float(fairness_penalty))
    if best_choice is None:
        return None, None, None, 0.0
    expert, pred, cost, fairness_penalty = best_choice
    return expert, pred, cost, fairness_penalty


def _assign_expert_candidate(
    case_row: pd.Series,
    expert_name: str | None,
    expert_pred: int | None,
    expert_cost: float | None,
    fairness_penalty: float,
    capacity_state: CapacityState,
) -> tuple[str | None, int | None, float | None, float, str]:
    if expert_name is None or expert_pred is None or expert_cost is None:
        return None, None, None, 0.0, "no_expert_prediction"
    if not allocate_expert(case_row, expert_name, capacity_state):
        return None, None, None, 0.0, "capacity_exhausted"
    return expert_name, expert_pred, expert_cost, fairness_penalty, "assigned"


def _escalation_majority_vote(
    case_row: pd.Series,
    ranked_experts: list[str],
    expert_metrics_lookup: dict[str, dict[str, float]],
    capacity_state: CapacityState,
    *,
    top_k: int,
    fp_cost: float,
    fn_cost: float,
    fairness_penalty_weight: float,
    capacity_penalty_weight: float,
    escalation_cost: float,
) -> tuple[int | None, list[str], float | None, float, str]:
    chosen_experts: list[str] = []
    votes: list[int] = []
    costs: list[float] = []
    fairness_penalties: list[float] = []

    for expert in ranked_experts:
        if len(chosen_experts) >= top_k:
            break
        if expert not in case_row or pd.isna(case_row[expert]):
            continue
        if not allocate_expert(case_row, expert, capacity_state):
            continue
        expert_cost, fairness_penalty = _expert_expected_cost(
            case_row,
            expert,
            expert_metrics_lookup,
            capacity_state,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            fairness_penalty_weight=fairness_penalty_weight,
            capacity_penalty_weight=capacity_penalty_weight,
        )
        chosen_experts.append(expert)
        votes.append(int(case_row[expert]))
        costs.append(float(expert_cost))
        fairness_penalties.append(float(fairness_penalty))

    if not votes:
        return None, [], None, 0.0, "no_expert_available_for_escalation"
    final_pred = 1 if sum(votes) >= (len(votes) / 2.0) else 0
    expected_cost = float(np.mean(costs) + escalation_cost) if costs else float(escalation_cost)
    fairness_penalty = float(np.mean(fairness_penalties)) if fairness_penalties else 0.0
    return final_pred, chosen_experts, expected_cost, fairness_penalty, "majority_vote"


def route_cases(
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
) -> pd.DataFrame:
    expert_lookup = expert_metrics.set_index("expert").to_dict(orient="index") if not expert_metrics.empty else {}
    decisions: list[dict[str, object]] = []

    for _, row in core.iterrows():
        ai_expected_cost = _ai_expected_cost(row, fp_cost, fn_cost)
        passes_threshold = bool(row.get("passes_threshold", False))
        selected_route = "AI"
        selected_expert = ""
        final_prediction = int(row["ai_pred"])
        capacity_status = "not_applicable"
        fairness_penalty = 0.0
        decision_reason = "adaptive_threshold_ai_low_cost"
        escalation_expected_cost = None

        risk_category = str(row.get("risk_category", ""))

        if risk_category == "high":
            escalation_pred, escalation_experts, escalation_expected_cost, escalation_fairness_penalty, escalation_status = _escalation_majority_vote(
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
            if escalation_pred is not None:
                selected_route = "Escalate"
                selected_expert = "|".join(escalation_experts)
                final_prediction = int(escalation_pred)
                capacity_status = escalation_status
                fairness_penalty = float(escalation_fairness_penalty)
                decision_reason = "high_risk_majority_vote_escalation"
            else:
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
                best_expert_name, best_expert_pred, best_expert_cost, best_fairness_penalty, best_status = _assign_expert_candidate(
                    row,
                    cand_name,
                    cand_pred,
                    cand_cost,
                    cand_fairness,
                    capacity_state,
                )
                if best_expert_name is not None and best_expert_pred is not None:
                    selected_route = "Human Expert"
                    selected_expert = best_expert_name
                    final_prediction = int(best_expert_pred)
                    capacity_status = best_status
                    fairness_penalty = float(best_fairness_penalty)
                    decision_reason = "high_risk_best_expert_fallback"
                else:
                    capacity_status = escalation_status if escalation_status != "not_attempted" else best_status
                    decision_reason = "high_risk_no_expert_available_ai_fallback"
        else:
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
            best_expert_cost = cand_cost
            if passes_threshold and (cand_cost is None or ai_expected_cost <= cand_cost):
                selected_route = "AI"
                final_prediction = int(row["ai_pred"])
                capacity_status = "threshold_passed"
                fairness_penalty = 0.0
                decision_reason = "adaptive_threshold_ai_low_cost"
            else:
                best_expert_name, best_expert_pred, best_expert_cost, best_fairness_penalty, best_status = _assign_expert_candidate(
                    row,
                    cand_name,
                    cand_pred,
                    cand_cost,
                    cand_fairness,
                    capacity_state,
                )
                if best_expert_name is not None and best_expert_pred is not None:
                    selected_route = "Human Expert"
                    selected_expert = best_expert_name
                    final_prediction = int(best_expert_pred)
                    capacity_status = best_status
                    fairness_penalty = float(best_fairness_penalty)
                    decision_reason = "best_available_expert_lower_cost_or_threshold_fail"
                else:
                    capacity_status = best_status
                    decision_reason = "no_expert_available_ai_fallback"

        decisions.append(
            {
                "case_id": str(row["case_id"]),
                "y_true": int(row["y_true"]) if "y_true" in row and pd.notna(row["y_true"]) else np.nan,
                "ai_pred": int(row["ai_pred"]),
                "final_prediction": int(final_prediction),
                "selected_route": selected_route,
                "selected_expert": selected_expert,
                "ai_expected_cost": float(ai_expected_cost),
                "best_expert_expected_cost": float(best_expert_cost) if best_expert_cost is not None else np.nan,
                "escalation_expected_cost": float(escalation_expected_cost) if escalation_expected_cost is not None else np.nan,
                "capacity_status": capacity_status,
                "fairness_penalty": float(fairness_penalty),
                "decision_reason": decision_reason,
                "magd_assurance_risk": float(row["magd_assurance_risk"]),
                "adaptive_threshold": float(row["adaptive_threshold"]),
                "passes_threshold": passes_threshold,
                "is_correct": int(int(final_prediction) == int(row["y_true"])) if "y_true" in row and pd.notna(row["y_true"]) else np.nan,
            }
        )

    return pd.DataFrame(decisions)


def _load_routing_inputs(config_path: Path) -> pd.DataFrame:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = (config_path.parent / processed_dir).resolve()

    model_test = pd.read_csv(outputs_root / "model" / "test_predictions.csv")
    adaptive = pd.read_csv(outputs_root / "assurance" / "adaptive_thresholds.csv")
    magd = pd.read_csv(outputs_root / "assurance" / "magd_risk.csv")
    test_metadata = pd.read_csv(processed_dir / "test_metadata.csv")

    for frame in [model_test, adaptive, magd, test_metadata]:
        if "case_id" in frame.columns:
            frame["case_id"] = frame["case_id"].astype(str)

    return model_test.merge(
        adaptive[["case_id", "adaptive_threshold", "passes_threshold"]],
        on="case_id",
        how="inner",
    ).merge(
        magd[["case_id", "magd_assurance_risk", "risk_category", "business_risk"]],
        on="case_id",
        how="inner",
    ).merge(
        test_metadata,
        on="case_id",
        how="left",
    )


def run_expert_routing(config_path: str | Path) -> ExpertRoutingArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    cfg = _routing_config(config)
    output_dir = _resolve_output_dir(resolved_config_path)

    core = _load_routing_inputs(resolved_config_path)
    expert_df, historical_df = _load_expert_tables(resolved_config_path)
    loaded = load_fifar_data(resolved_config_path)
    if expert_df is not None:
        expert_df = expert_df.rename(columns={expert_df.columns[0]: "case_id"})
        expert_df["case_id"] = expert_df["case_id"].astype(str)
        core = core.merge(expert_df, on="case_id", how="left")

    capacity_by_expert, capacity_totals = _remaining_capacity_summary(resolved_config_path)

    expert_metrics, _ = build_expert_reliability_frame(resolved_config_path)
    ranked_experts: list[str] = expert_metrics["expert"].tolist() if not expert_metrics.empty else []

    if expert_df is not None:
        available_columns = set(expert_df.columns)
        ranked_experts = [expert for expert in ranked_experts if expert in available_columns]
    else:
        ranked_experts = []

    expert_metrics.to_csv(output_dir / "expert_reliability.csv", index=False)

    capacity_remaining, batch_column = _build_capacity_state(core, resolved_config_path)
    capacity_state = CapacityState(remaining=capacity_remaining, batch_column=batch_column)
    decisions = route_cases(
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
    )
    decisions.to_csv(output_dir / "magd_routing_decisions.csv", index=False)

    return ExpertRoutingArtifacts(
        expert_reliability=expert_metrics,
        routing_decisions=decisions,
        output_dir=output_dir,
    )
