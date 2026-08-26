from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_fifar import load_fifar_data, read_table, resolve_config_path
from src.deferral.baselines import (
    _build_capacity_state,
    _candidate_expert_columns,
    _historical_best_experts,
)
from src.deferral.capacity_assignment import CapacityState, allocate_expert, capacity_penalty_for_expert
from src.utils.io import load_yaml


@dataclass
class AssuranceDeferralArtifacts:
    decisions: pd.DataFrame
    metrics: pd.DataFrame
    output_dir: Path


def _resolve_output_dirs(config_path: Path) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    output_dir = outputs_root / "assurance_deferral"
    plots_dir = outputs_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, plots_dir


def _core_inputs(config_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    assurance_dir = outputs_root / "assurance"
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = (config_path.parent / processed_dir).resolve()

    wrong_conf = pd.read_csv(assurance_dir / "wrong_confident_risk.csv")
    assurance_risk = pd.read_csv(assurance_dir / "assurance_risk.csv")
    test_metadata = pd.read_csv(processed_dir / "test_metadata.csv")
    for frame in [wrong_conf, assurance_risk, test_metadata]:
        if "case_id" in frame.columns:
            frame["case_id"] = frame["case_id"].astype(str)
    merged = wrong_conf.merge(
        assurance_risk[["case_id", "assurance_risk", "risk_category"]],
        on="case_id",
        how="inner",
    ).merge(
        test_metadata,
        on="case_id",
        how="left",
    )
    return merged, assurance_risk


def _deferral_config(config: dict) -> dict:
    cfg = config.get("assurance_deferral", {})
    model_cfg = config.get("model", {})
    return {
        "fp_cost": float(model_cfg.get("false_positive_cost", 1.0)),
        "fn_cost": float(model_cfg.get("false_negative_cost", 5.0)),
        "top_k": int(cfg.get("escalation_top_k_experts", 5)),
        "assurance_risk_penalty": float(cfg.get("assurance_risk_penalty", 1.0)),
        "bias_penalty_weight": float(cfg.get("bias_penalty_weight", 1.0)),
        "capacity_penalty_weight": float(cfg.get("capacity_penalty_weight", 1.0)),
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


def _expert_reliability(config_path: Path, historical_df: pd.DataFrame | None) -> pd.DataFrame:
    if historical_df is None:
        return pd.DataFrame(columns=["expert", "accuracy", "false_positive_rate", "false_negative_rate", "cost_sensitive_loss", "bias_penalty"])

    loaded = load_fifar_data(config_path)
    train_label_column = loaded.config.get("columns", {}).get("train_label") or loaded.config.get("columns", {}).get("label")
    train_file = resolve_config_path(
        loaded.config_path,
        loaded.config.get("dataset", {}).get("train_file"),
        required=False,
        field_name="dataset.train_file",
    )
    if train_file is None:
        return pd.DataFrame(columns=["expert", "accuracy", "false_positive_rate", "false_negative_rate", "cost_sensitive_loss", "bias_penalty"])

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

    fp_cost = float(loaded.config.get("model", {}).get("false_positive_cost", 1.0))
    fn_cost = float(loaded.config.get("model", {}).get("false_negative_cost", 5.0))
    rows: list[dict[str, float | str]] = []
    for expert in _candidate_expert_columns(hist):
        if expert not in merged.columns:
            continue
        pred = merged[expert]
        valid = pred.notna()
        if valid.sum() == 0:
            continue
        y = merged.loc[valid, train_label_column].astype(int)
        p = pred.loc[valid].astype(int)
        tp = int(((p == 1) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        accuracy = float((p == y).mean())
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        cost_loss = fp_cost * fp + fn_cost * fn

        bias_penalty = 0.0
        if loaded.sensitive_attributes:
            group_errors = []
            for sensitive in loaded.sensitive_attributes:
                if sensitive not in merged.columns:
                    continue
                for _, group in merged.loc[valid].groupby(sensitive, dropna=False):
                    if len(group) > 0:
                        group_errors.append(float((group[expert].astype(int) != group[train_label_column].astype(int)).mean()))
            if group_errors:
                bias_penalty = float(max(group_errors) - min(group_errors))

        rows.append(
            {
                "expert": expert,
                "accuracy": accuracy,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "cost_sensitive_loss": float(cost_loss),
                "bias_penalty": bias_penalty,
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def _ai_expected_cost(row: pd.Series, cfg: dict) -> float:
    ai_score = float(row["ai_score"])
    ai_pred = int(row["ai_pred"])
    estimated_fp_risk = (1.0 - ai_score) if ai_pred == 1 else 0.0
    estimated_fn_risk = ai_score if ai_pred == 0 else 0.0
    return (
        cfg["fp_cost"] * estimated_fp_risk
        + cfg["fn_cost"] * estimated_fn_risk
        + cfg["assurance_risk_penalty"] * float(row["assurance_risk"])
    )


def _best_available_expert(
    case_row: pd.Series,
    ranked_experts: list[str],
    expert_metrics: pd.DataFrame,
    capacity_state: CapacityState,
    cfg: dict,
) -> tuple[str | None, int | None, float | None, str]:
    if not ranked_experts:
        return None, None, None, "no_expert_available"
    metrics_lookup = expert_metrics.set_index("expert").to_dict(orient="index") if not expert_metrics.empty else {}
    best_choice: tuple[str, int, float] | None = None
    for expert in ranked_experts:
        if expert not in case_row or pd.isna(case_row[expert]):
            continue
        stats = metrics_lookup.get(
            expert,
            {
                "false_positive_rate": 0.5,
                "false_negative_rate": 0.5,
                "bias_penalty": 0.0,
            },
        )
        expert_cost = (
            cfg["fp_cost"] * float(stats["false_positive_rate"])
            + cfg["fn_cost"] * float(stats["false_negative_rate"])
            + cfg["bias_penalty_weight"] * float(stats.get("bias_penalty", 0.0))
            + capacity_penalty_for_expert(case_row, expert, capacity_state, cfg["capacity_penalty_weight"])
        )
        pred = int(case_row[expert])
        if best_choice is None or expert_cost < best_choice[2]:
            best_choice = (expert, pred, float(expert_cost))
    if best_choice is None:
        return None, None, None, "no_expert_prediction"
    expert, pred, cost = best_choice
    if not allocate_expert(case_row, expert, capacity_state):
        return None, None, None, "capacity_exhausted"
    return expert, pred, cost, "assigned"


def _escalation_majority_vote(
    case_row: pd.Series,
    ranked_experts: list[str],
    capacity_state: CapacityState,
    top_k: int,
) -> tuple[int | None, list[str], str]:
    chosen_experts: list[str] = []
    votes: list[int] = []
    for expert in ranked_experts:
        if len(chosen_experts) >= top_k:
            break
        if expert not in case_row or pd.isna(case_row[expert]):
            continue
        if allocate_expert(case_row, expert, capacity_state):
            chosen_experts.append(expert)
            votes.append(int(case_row[expert]))
    if not votes:
        return None, [], "no_expert_available_for_escalation"
    final_pred = 1 if sum(votes) >= (len(votes) / 2.0) else 0
    return final_pred, chosen_experts, "majority_vote"


def _plot_route_counts(decisions: pd.DataFrame, plot_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to generate the deferral route counts plot.") from exc

    counts = decisions["selected_route"].value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.xlabel("Selected Route")
    plt.ylabel("Case Count")
    plt.title("Assurance-Guided Deferral Route Counts")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def _static_expert_costs(expert_metrics: pd.DataFrame, cfg: dict) -> dict[str, float]:
    if expert_metrics.empty:
        return {}
    costs: dict[str, float] = {}
    for _, row in expert_metrics.iterrows():
        costs[str(row["expert"])] = (
            cfg["fp_cost"] * float(row["false_positive_rate"])
            + cfg["fn_cost"] * float(row["false_negative_rate"])
            + cfg["bias_penalty_weight"] * float(row.get("bias_penalty", 0.0))
        )
    return costs


def _vectorized_best_available_expert(
    core: pd.DataFrame,
    ranked_experts: list[str],
    expert_costs: dict[str, float],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    available_experts = [expert for expert in ranked_experts if expert in core.columns]
    if not available_experts:
        empty_name = pd.Series([""] * len(core), index=core.index, dtype=object)
        empty_pred = pd.Series([np.nan] * len(core), index=core.index, dtype=float)
        empty_cost = pd.Series([np.nan] * len(core), index=core.index, dtype=float)
        return empty_name, empty_pred, empty_cost

    availability = core[available_experts].notna()
    sorted_by_cost = sorted(available_experts, key=lambda expert: expert_costs.get(expert, float("inf")))
    chosen_expert = pd.Series(index=core.index, dtype=object)
    chosen_pred = pd.Series(index=core.index, dtype=float)
    chosen_cost = pd.Series(index=core.index, dtype=float)

    unassigned = pd.Series(True, index=core.index)
    for expert in sorted_by_cost:
        mask = unassigned & availability[expert]
        if mask.any():
            chosen_expert.loc[mask] = expert
            chosen_pred.loc[mask] = core.loc[mask, expert].astype(float)
            chosen_cost.loc[mask] = float(expert_costs.get(expert, float("inf")))
            unassigned.loc[mask] = False
    chosen_expert = chosen_expert.fillna("")
    return chosen_expert, chosen_pred, chosen_cost


def _vectorized_escalation(
    core: pd.DataFrame,
    ranked_experts: list[str],
    top_k: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    available_experts = [expert for expert in ranked_experts if expert in core.columns][:top_k]
    if not available_experts:
        empty_pred = pd.Series([np.nan] * len(core), index=core.index, dtype=float)
        empty_names = pd.Series([""] * len(core), index=core.index, dtype=object)
        empty_status = pd.Series(["no_expert_available_for_escalation"] * len(core), index=core.index, dtype=object)
        return empty_pred, empty_names, empty_status

    expert_votes = core[available_experts].apply(pd.to_numeric, errors="coerce")
    vote_counts = expert_votes.notna().sum(axis=1)
    vote_sums = expert_votes.fillna(0.0).sum(axis=1)
    final_pred = pd.Series(np.where(vote_counts > 0, (vote_sums >= (vote_counts / 2.0)).astype(int), np.nan), index=core.index)

    chosen_names = []
    for _, row in expert_votes.notna().iterrows():
        names = [expert for expert in available_experts if row[expert]]
        chosen_names.append("|".join(names[:top_k]))
    status = pd.Series(
        np.where(vote_counts > 0, "majority_vote", "no_expert_available_for_escalation"),
        index=core.index,
    )
    return final_pred, pd.Series(chosen_names, index=core.index), status


def run_assurance_deferral(config_path: str | Path) -> AssuranceDeferralArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    cfg = _deferral_config(config)
    output_dir, plots_dir = _resolve_output_dirs(resolved_config_path)

    core, _ = _core_inputs(resolved_config_path)
    expert_df, historical_df = _load_expert_tables(resolved_config_path)
    expert_metrics = _expert_reliability(resolved_config_path, historical_df)
    ranked_experts = _historical_best_experts(resolved_config_path, historical_df) if historical_df is not None else []

    loaded = load_fifar_data(resolved_config_path)
    if expert_df is not None:
        expert_df = expert_df.rename(columns={expert_df.columns[0]: "case_id"})
        expert_df["case_id"] = expert_df["case_id"].astype(str)
        core = core.merge(expert_df, on="case_id", how="left")
        ranked_experts = [expert for expert in ranked_experts if expert in expert_df.columns]

    capacity_remaining, batch_column = _build_capacity_state(core, resolved_config_path)
    capacity_state = CapacityState(remaining=capacity_remaining, batch_column=batch_column)

    if capacity_state.remaining is None:
        expert_costs = _static_expert_costs(expert_metrics, cfg)
        best_expert_name, best_expert_pred, best_expert_cost = _vectorized_best_available_expert(core, ranked_experts, expert_costs)
        escalation_pred, escalation_experts, escalation_status = _vectorized_escalation(core, ranked_experts, cfg["top_k"])

        result = core.copy()
        result["selected_route"] = "AI"
        result["selected_expert"] = ""
        result["final_prediction"] = result["ai_pred"].astype(int)
        result["decision_reason"] = "low_risk_ai_lower_cost"
        result["capacity_status"] = "not_applicable"

        ai_cost = result.apply(lambda row: _ai_expected_cost(row, cfg), axis=1)
        expert_available = best_expert_name != ""
        expert_better = expert_available & best_expert_cost.notna() & (ai_cost > best_expert_cost)

        low_mask = result["risk_category"] == "low"
        medium_mask = result["risk_category"] == "medium"
        high_mask = result["risk_category"] == "high"

        low_use_expert = low_mask & expert_better
        result.loc[low_use_expert, "selected_route"] = "Human Expert"
        result.loc[low_use_expert, "selected_expert"] = best_expert_name[low_use_expert]
        result.loc[low_use_expert, "final_prediction"] = best_expert_pred[low_use_expert].astype(int)
        result.loc[low_use_expert, "decision_reason"] = "low_risk_expert_lower_cost"

        medium_use_expert = medium_mask & expert_available
        result.loc[medium_use_expert, "selected_route"] = "Human Expert"
        result.loc[medium_use_expert, "selected_expert"] = best_expert_name[medium_use_expert]
        result.loc[medium_use_expert, "final_prediction"] = best_expert_pred[medium_use_expert].astype(int)
        result.loc[medium_use_expert, "decision_reason"] = "medium_risk_best_available_expert"

        medium_fallback_ai = medium_mask & (~expert_available)
        result.loc[medium_fallback_ai, "decision_reason"] = "medium_risk_expert_unavailable_ai_fallback"
        result.loc[medium_fallback_ai, "capacity_status"] = "no_expert_available"

        high_use_escalation = high_mask & escalation_pred.notna()
        result.loc[high_use_escalation, "selected_route"] = "Escalate"
        result.loc[high_use_escalation, "selected_expert"] = escalation_experts[high_use_escalation]
        result.loc[high_use_escalation, "final_prediction"] = escalation_pred[high_use_escalation].astype(int)
        result.loc[high_use_escalation, "decision_reason"] = "high_risk_majority_vote_escalation"
        result.loc[high_use_escalation, "capacity_status"] = escalation_status[high_use_escalation]

        high_fallback_expert = high_mask & escalation_pred.isna() & expert_available
        result.loc[high_fallback_expert, "selected_route"] = "Human Expert"
        result.loc[high_fallback_expert, "selected_expert"] = best_expert_name[high_fallback_expert]
        result.loc[high_fallback_expert, "final_prediction"] = best_expert_pred[high_fallback_expert].astype(int)
        result.loc[high_fallback_expert, "decision_reason"] = "high_risk_escalation_unavailable_best_expert_fallback"
        result.loc[high_fallback_expert, "capacity_status"] = "no_expert_available_for_escalation"

        high_fallback_ai = high_mask & escalation_pred.isna() & (~expert_available)
        result.loc[high_fallback_ai, "decision_reason"] = "high_risk_no_expert_available_ai_fallback"
        result.loc[high_fallback_ai, "capacity_status"] = "no_expert_available_for_escalation"

        decision_df = result[
            [
                "case_id",
                "y_true",
                "ai_score",
                "ai_pred",
                "numerical_confidence",
                "distance_confidence",
                "distance_uncertainty",
                "calibration_risk",
                "neighbor_error_rate",
                "wrong_confident_risk",
                "assurance_risk",
                "risk_category",
                "selected_route",
                "selected_expert",
                "final_prediction",
                "decision_reason",
                "capacity_status",
            ]
        ].copy()
        decision_df["is_correct"] = (decision_df["final_prediction"].astype(int) == decision_df["y_true"].astype(int)).astype(int)
    else:
        decisions: list[dict[str, object]] = []
        for _, row in core.iterrows():
            ai_cost = _ai_expected_cost(row, cfg)
            selected_route = "AI"
            selected_expert = ""
            final_prediction = int(row["ai_pred"])
            decision_reason = "low_risk_ai_lower_cost"
            capacity_status = "not_applicable"

            if row["risk_category"] == "low":
                expert_name, expert_pred, expert_cost, status = _best_available_expert(
                    row, ranked_experts, expert_metrics, capacity_state, cfg
                )
                if expert_name is not None and expert_cost is not None and ai_cost > expert_cost:
                    selected_route = "Human Expert"
                    selected_expert = expert_name
                    final_prediction = int(expert_pred)
                    decision_reason = "low_risk_expert_lower_cost"
                    capacity_status = status
                else:
                    capacity_status = status
            elif row["risk_category"] == "medium":
                expert_name, expert_pred, _, status = _best_available_expert(
                    row, ranked_experts, expert_metrics, capacity_state, cfg
                )
                if expert_name is not None and expert_pred is not None:
                    selected_route = "Human Expert"
                    selected_expert = expert_name
                    final_prediction = int(expert_pred)
                    decision_reason = "medium_risk_best_available_expert"
                    capacity_status = status
                else:
                    selected_route = "AI"
                    final_prediction = int(row["ai_pred"])
                    decision_reason = "medium_risk_expert_unavailable_ai_fallback"
                    capacity_status = status
            else:
                final_pred, chosen_experts, status = _escalation_majority_vote(
                    row, ranked_experts, capacity_state, cfg["top_k"]
                )
                if final_pred is not None:
                    selected_route = "Escalate"
                    selected_expert = "|".join(chosen_experts)
                    final_prediction = int(final_pred)
                    decision_reason = "high_risk_majority_vote_escalation"
                    capacity_status = status
                else:
                    expert_name, expert_pred, _, fallback_status = _best_available_expert(
                        row, ranked_experts, expert_metrics, capacity_state, cfg
                    )
                    if expert_name is not None and expert_pred is not None:
                        selected_route = "Human Expert"
                        selected_expert = expert_name
                        final_prediction = int(expert_pred)
                        decision_reason = "high_risk_escalation_unavailable_best_expert_fallback"
                        capacity_status = fallback_status
                    else:
                        selected_route = "AI"
                        final_prediction = int(row["ai_pred"])
                        decision_reason = "high_risk_no_expert_available_ai_fallback"
                        capacity_status = fallback_status

            decisions.append(
                {
                    "case_id": row["case_id"],
                    "y_true": int(row["y_true"]),
                    "ai_score": float(row["ai_score"]),
                    "ai_pred": int(row["ai_pred"]),
                    "numerical_confidence": float(row["numerical_confidence"]),
                    "distance_confidence": float(row["distance_confidence"]),
                    "distance_uncertainty": float(row["distance_uncertainty"]),
                    "calibration_risk": float(row["calibration_risk"]),
                    "neighbor_error_rate": float(row["neighbor_error_rate"]),
                    "wrong_confident_risk": float(row["wrong_confident_risk"]),
                    "assurance_risk": float(row["assurance_risk"]),
                    "risk_category": row["risk_category"],
                    "selected_route": selected_route,
                    "selected_expert": selected_expert,
                    "final_prediction": int(final_prediction),
                    "is_correct": int(int(final_prediction) == int(row["y_true"])),
                    "decision_reason": decision_reason,
                    "capacity_status": capacity_status,
                }
            )

        decision_df = pd.DataFrame(decisions)
    decision_df.to_csv(output_dir / "assurance_guided_decisions.csv", index=False)

    fp = int(((decision_df["final_prediction"] == 1) & (decision_df["y_true"] == 0)).sum())
    fn = int(((decision_df["final_prediction"] == 0) & (decision_df["y_true"] == 1)).sum())
    tp = int(((decision_df["final_prediction"] == 1) & (decision_df["y_true"] == 1)).sum())
    tn = int(((decision_df["final_prediction"] == 0) & (decision_df["y_true"] == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    metrics = pd.DataFrame(
        [
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cost_sensitive_loss": cfg["fp_cost"] * fp + cfg["fn_cost"] * fn,
                "ai_route_rate": float((decision_df["selected_route"] == "AI").mean()),
                "expert_route_rate": float((decision_df["selected_route"] == "Human Expert").mean()),
                "escalate_route_rate": float((decision_df["selected_route"] == "Escalate").mean()),
                "total_cases": float(len(decision_df)),
            }
        ]
    )
    metrics.to_csv(output_dir / "assurance_guided_metrics.csv", index=False)
    _plot_route_counts(decision_df, plots_dir / "deferral_route_counts.png")
    return AssuranceDeferralArtifacts(decisions=decision_df, metrics=metrics, output_dir=output_dir)
