from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assurance.magd_risk import add_risk_categories_and_actions, compute_magd_risk
from src.deferral.adaptive_threshold import compute_adaptive_thresholds
from src.deferral.baselines import _build_capacity_state
from src.deferral.capacity_assignment import CapacityState
from src.deferral.expert_routing import _load_expert_tables, _resolve_output_dir, _routing_config
from src.deferral.magd_constrained import MagdConstrainedArtifacts, run_magd_constrained
from src.deferral.magd_deferral import _build_expert_reliability, _load_final_inputs, route_magd_cases
from src.deferral.magd_policy import load_policy_weights
from src.evaluation.ablation_utils import (
    ABLATION_VARIANTS,
    REQUIRED_AUDIT_COLUMNS,
    VariantSpec,
    compute_ablation_metrics,
    ensure_variant_allowed,
    outputs_root_for_config,
    required_plot_path,
    variant_specs,
    zero_out_unselected_signals,
)
from src.utils.io import load_yaml
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class AblationArtifacts:
    metrics: pd.DataFrame
    ablation_dir: Path
    paper_dir: Path
    plot_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAGD ablation studies.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def _ablation_dir(config_path: Path) -> Path:
    output_dir = outputs_root_for_config(config_path) / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _paper_dir(config_path: Path) -> Path:
    output_dir = outputs_root_for_config(config_path) / "paper_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _fairness_risk(core: pd.DataFrame, config_path: Path) -> pd.Series:
    config = load_yaml(config_path)
    outputs_root = outputs_root_for_config(config_path)
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = (config_path.parent / processed_dir).resolve()

    train_metadata_path = processed_dir / "train_metadata.csv"
    model_train_path = outputs_root / "model" / "train_predictions.csv"
    if not train_metadata_path.exists() or not model_train_path.exists():
        return pd.Series([0.0] * len(core), index=core.index, dtype=float)

    train_metadata = pd.read_csv(train_metadata_path)
    model_train = pd.read_csv(model_train_path)
    train_metadata["case_id"] = train_metadata["case_id"].astype(str)
    model_train["case_id"] = model_train["case_id"].astype(str)
    merged = model_train.merge(train_metadata, on="case_id", how="left")
    merged["ai_error"] = (merged["ai_pred"].astype(int) != merged["y_true"].astype(int)).astype(int)

    risk = pd.Series([0.0] * len(core), index=core.index, dtype=float)
    contributing = 0
    for sensitive in load_yaml(config_path)["columns"].get("sensitive_attributes", []):
        if sensitive not in merged.columns or sensitive not in core.columns:
            continue
        group_error = merged.groupby(sensitive, dropna=False)["ai_error"].mean().to_dict()
        overall_error = float(merged["ai_error"].mean()) if len(merged) else 0.0
        risk = risk + core[sensitive].map(lambda value: abs(float(group_error.get(value, overall_error)) - overall_error)).fillna(0.0).astype(float)
        contributing += 1
    if contributing == 0:
        return pd.Series([0.0] * len(core), index=core.index, dtype=float)
    return (risk / float(contributing)).clip(0.0, 1.0)


def _prepare_variant_core(
    *,
    config_path: Path,
    base_core: pd.DataFrame,
    weights: dict[str, float],
    low_risk: float,
    high_risk: float,
    base_threshold: float,
    fairness_enabled: bool,
    capacity_enabled: bool,
) -> pd.DataFrame:
    config = load_yaml(config_path)
    core = base_core.copy()
    core["business_risk"] = pd.to_numeric(core.get("business_risk", 0.0), errors="coerce").fillna(0.0)
    core["fairness_risk"] = _fairness_risk(core, config_path) if fairness_enabled else 0.0
    if capacity_enabled:
        capacity_remaining, batch_column = _build_capacity_state(core, config_path)
        capacity_state = CapacityState(remaining=capacity_remaining, batch_column=batch_column)
        if capacity_state.remaining:
            # Capacity pressure proxy: any batch-expert exhaustion evidence mapped per row.
            batch_values = core[batch_column].astype(str) if batch_column and batch_column in core.columns else pd.Series([""] * len(core))
            core["capacity_pressure"] = batch_values.map(
                lambda batch: float(any(key[0] == batch and value <= 0 for key, value in capacity_state.remaining.items()))
            ).fillna(0.0)
        else:
            core["capacity_pressure"] = 0.0
    else:
        core["capacity_pressure"] = 0.0

    core = compute_magd_risk(
        core,
        weights,
        use_drift_risk=bool(config["magd"]["signals"].get("use_drift_risk", False)),
        use_business_risk=bool(config["magd"]["signals"].get("use_business_risk", False)),
    )
    core = add_risk_categories_and_actions(core, low_risk=low_risk, high_risk=high_risk)
    thresholds = compute_adaptive_thresholds(
        core[["case_id", "magd_assurance_risk", "business_risk", "fairness_risk", "capacity_pressure"]],
        base_threshold=base_threshold,
        value_weight=float(config["magd"]["adaptive_threshold"]["value_weight"]),
        fairness_weight=float(config["magd"]["adaptive_threshold"]["fairness_weight"]) if fairness_enabled else 0.0,
        capacity_weight=float(config["magd"]["adaptive_threshold"]["capacity_weight"]) if capacity_enabled else 0.0,
    )
    return core.drop(columns=["base_threshold", "adaptive_threshold", "passes_threshold"], errors="ignore").merge(
        thresholds,
        on=["case_id", "magd_assurance_risk", "business_risk", "fairness_risk", "capacity_pressure"],
        how="left",
    )


def _normalize_decisions(decisions: pd.DataFrame, core: pd.DataFrame, variant: str) -> pd.DataFrame:
    join_columns = ["case_id", "y_true"]
    if "wrong_confident_label_offline" in core.columns:
        join_columns.append("wrong_confident_label_offline")
    merged = decisions.merge(core[join_columns], on="case_id", how="left")
    if "wrong_confident_label_offline" not in merged.columns:
        merged["wrong_confident_label_offline"] = 0
    merged["used_ai"] = merged["selected_route"].astype(str).eq("AI")
    merged["ai_correct"] = (merged["ai_pred"].astype(int) == merged["y_true"].astype(int)).astype(int)
    merged["is_correct"] = (merged["final_prediction"].astype(int) == merged["y_true"].astype(int)).astype(int)
    merged["method"] = variant
    return merged


def _plot_ablation_cost_vs_overreliance(metrics: pd.DataFrame, plot_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to generate ablation plots.") from exc

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(metrics["overreliance"], metrics["cost_sensitive_loss"])
    for _, row in metrics.iterrows():
        plt.annotate(str(row["variant"]), (row["overreliance"], row["cost_sensitive_loss"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Overreliance")
    plt.ylabel("Cost-Sensitive Loss")
    plt.title("MAGD Ablation Cost vs Overreliance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def _cached_ablation_artifacts_if_complete(config_path: Path, ablation_dir: Path, paper_dir: Path, plot_path: Path) -> AblationArtifacts | None:
    metrics_path = ablation_dir / "ablation_metrics.csv"
    paper_path = paper_dir / "ablation.csv"
    if not metrics_path.exists() or not paper_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    if metrics.empty or "variant" not in metrics.columns or "status" not in metrics.columns:
        return None

    status_by_variant = dict(zip(metrics["variant"].astype(str), metrics["status"].astype(str)))
    for spec in variant_specs():
        allowed, expected_status = ensure_variant_allowed(spec, config_path)
        observed = status_by_variant.get(spec.name)
        if allowed:
            if observed != "completed":
                return None
            if not (ablation_dir / f"ablation_decisions_{spec.name}.csv").exists():
                return None
        elif observed != expected_status:
            return None
    LOGGER.info("Using cached complete MAGD ablation artifacts from %s.", ablation_dir)
    return AblationArtifacts(metrics=metrics, ablation_dir=ablation_dir, paper_dir=paper_dir, plot_path=plot_path)


def _variant_from_constrained_artifacts(spec: VariantSpec, artifacts: MagdConstrainedArtifacts) -> pd.DataFrame:
    if spec.constrained_source == "initial":
        frame = artifacts.initial_decisions.copy()
    else:
        frame = artifacts.calibrated_decisions.copy()
    frame["method"] = spec.name
    return frame


def run_magd_ablations(config_path: str | Path) -> AblationArtifacts:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    cfg = _routing_config(config)
    ablation_dir = _ablation_dir(resolved_config_path)
    paper_dir = _paper_dir(resolved_config_path)
    plot_path = required_plot_path(resolved_config_path)
    cached = _cached_ablation_artifacts_if_complete(resolved_config_path, ablation_dir, paper_dir, plot_path)
    if cached is not None:
        return cached

    base_core = _load_final_inputs(resolved_config_path)
    outputs_root = outputs_root_for_config(resolved_config_path)
    wrong_conf_path = outputs_root / "assurance" / "wrong_confident_risk.csv"
    if wrong_conf_path.exists():
        wrong_conf = pd.read_csv(wrong_conf_path)
        wrong_conf["case_id"] = wrong_conf["case_id"].astype(str)
        if "wrong_confident_label_offline" in wrong_conf.columns:
            base_core = base_core.merge(wrong_conf[["case_id", "wrong_confident_label_offline"]], on="case_id", how="left")
    expert_df, _ = _load_expert_tables(resolved_config_path)
    if expert_df is not None:
        expert_df = expert_df.rename(columns={expert_df.columns[0]: "case_id"})
        expert_df["case_id"] = expert_df["case_id"].astype(str)
        base_core = base_core.merge(expert_df, on="case_id", how="left")

    expert_metrics, ranked_experts = _build_expert_reliability(resolved_config_path)
    if expert_df is not None:
        available_columns = set(expert_df.columns)
        ranked_experts = [expert for expert in ranked_experts if expert in available_columns]

    constrained_config_path = outputs_root / "magd_policy" / "constrained_policy_config.json"
    constrained_policy = json.loads(constrained_config_path.read_text(encoding="utf-8")) if constrained_config_path.exists() else {}
    constrained_artifacts: MagdConstrainedArtifacts | None = None

    rows: list[dict[str, object]] = []
    for spec in variant_specs():
        allowed, status = ensure_variant_allowed(spec, resolved_config_path)
        if not allowed:
            LOGGER.info("Skipping ablation variant `%s`: %s", spec.name, status)
            rows.append({"variant": spec.name, "status": status, "signals_used": ",".join(spec.selected_signals)})
            continue

        if spec.mode == "constrained":
            if constrained_artifacts is None:
                constrained_artifacts = run_magd_constrained(resolved_config_path)
            decisions = _variant_from_constrained_artifacts(spec, constrained_artifacts)
            if spec.requires_fairness or spec.requires_capacity:
                weights = load_policy_weights(resolved_config_path, variant="constrained")
                selected_params = constrained_policy.get("selected_params", {})
                variant_core = _prepare_variant_core(
                    config_path=resolved_config_path,
                    base_core=base_core,
                    weights=weights,
                    low_risk=float(selected_params.get("low_risk", config["magd"]["thresholds"]["low_risk"])),
                    high_risk=float(selected_params.get("high_risk", config["magd"]["thresholds"]["high_risk"])),
                    base_threshold=float(selected_params.get("base_threshold", config["magd"]["adaptive_threshold"]["base_threshold"])),
                    fairness_enabled=spec.requires_fairness,
                    capacity_enabled=spec.requires_capacity,
                )
                capacity_remaining, batch_column = _build_capacity_state(variant_core, resolved_config_path) if spec.requires_capacity else (None, None)
                decisions = route_magd_cases(
                    variant_core,
                    ranked_experts=ranked_experts,
                    expert_metrics=expert_metrics,
                    capacity_state=CapacityState(remaining=capacity_remaining, batch_column=batch_column),
                    fp_cost=cfg["fp_cost"],
                    fn_cost=cfg["fn_cost"],
                    fairness_penalty_weight=cfg["fairness_penalty_weight"] if spec.requires_fairness else 0.0,
                    capacity_penalty_weight=cfg["capacity_penalty_weight"] if spec.requires_capacity else 0.0,
                    human_review_cost=cfg["human_review_cost"],
                    escalation_cost=cfg["escalation_cost"],
                    top_k=cfg["top_k"],
                    wrong_confident_threshold=float(selected_params.get("wrong_confident_threshold", config["magd"]["thresholds"]["wrong_confident"])),
                    ai_cost_margin=float(selected_params.get("ai_cost_margin", config.get("magd", {}).get("ai_cost_margin", 0.01))),
                    ai_cost_scale=float(selected_params.get("ai_cost_scale", 1.0)),
                )
                decisions = _normalize_decisions(decisions, variant_core, spec.name)
            else:
                decisions["method"] = spec.name
        else:
            variant_name = "heuristic" if spec.name == "full_magd_heuristic" else "learned" if spec.name == "full_magd_learned" else None
            weights = load_policy_weights(resolved_config_path, variant=variant_name) if variant_name else zero_out_unselected_signals(pd.DataFrame([config["magd"]["risk_weights"]]), spec.selected_signals).iloc[0].to_dict()
            if variant_name is None:
                weights = {key: float(value) for key, value in weights.items()}
            variant_core = _prepare_variant_core(
                config_path=resolved_config_path,
                base_core=zero_out_unselected_signals(base_core, spec.selected_signals),
                weights=weights,
                low_risk=float(config["magd"]["thresholds"]["low_risk"]),
                high_risk=float(config["magd"]["thresholds"]["high_risk"]),
                base_threshold=float(config["magd"]["adaptive_threshold"]["base_threshold"]),
                fairness_enabled=False,
                capacity_enabled=False,
            )
            decisions = route_magd_cases(
                variant_core,
                ranked_experts=ranked_experts,
                expert_metrics=expert_metrics,
                capacity_state=CapacityState(remaining=None, batch_column=None),
                fp_cost=cfg["fp_cost"],
                fn_cost=cfg["fn_cost"],
                fairness_penalty_weight=0.0,
                capacity_penalty_weight=0.0,
                human_review_cost=cfg["human_review_cost"],
                escalation_cost=cfg["escalation_cost"],
                top_k=cfg["top_k"],
                wrong_confident_threshold=float(config["magd"]["thresholds"]["wrong_confident"]) if "wrong_confident_risk" in spec.selected_signals or spec.name.startswith("full_") else 1.1,
            )
            decisions = _normalize_decisions(decisions, variant_core, spec.name)

        decisions.to_csv(ablation_dir / f"ablation_decisions_{spec.name}.csv", index=False)
        metrics = compute_ablation_metrics(decisions, resolved_config_path)
        rows.append(
            {
                "variant": spec.name,
                "status": "completed",
                "signals_used": ",".join(spec.selected_signals),
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(ablation_dir / "ablation_metrics.csv", index=False)
    metrics_df.to_csv(paper_dir / "ablation.csv", index=False)
    completed = metrics_df.loc[metrics_df["status"].astype(str) == "completed"].copy()
    if not completed.empty:
        _plot_ablation_cost_vs_overreliance(completed, plot_path)
    return AblationArtifacts(metrics=metrics_df, ablation_dir=ablation_dir, paper_dir=paper_dir, plot_path=plot_path)


def main() -> None:
    args = parse_args()
    artifacts = run_magd_ablations(args.config)
    print(f"Saved ablation metrics to: {artifacts.ablation_dir / 'ablation_metrics.csv'}")
    print(f"Variants evaluated: {len(artifacts.metrics)}")


if __name__ == "__main__":
    main()
