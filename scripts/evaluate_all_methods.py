from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_fifar import load_fifar_data
from src.evaluation.assurance_metrics import compute_assurance_metrics
from src.evaluation.audit_metrics import compute_audit_metrics
from src.evaluation.deferral_metrics import compute_deferral_metrics
from src.evaluation.fairness_metrics import compute_fairness_metrics
from src.evaluation.fraud_metrics import compute_fraud_metrics
from src.evaluation.reliance_metrics import compute_reliance_metrics
from src.deferral.magd_constrained import CONSTRAINT_DIAGNOSTIC_COLUMNS, ensure_constraint_diagnostics
from src.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all methods.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def _paths(config_path: Path) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    final_dir = outputs_root / "final_metrics"
    final_dir.mkdir(parents=True, exist_ok=True)
    return outputs_root, final_dir


def _paper_dir(outputs_root: Path) -> Path:
    paper_dir = outputs_root / "paper_tables"
    paper_dir.mkdir(parents=True, exist_ok=True)
    return paper_dir


def _load_constrained_diagnostic_defaults(outputs_root: Path) -> dict[str, bool]:
    defaults = {column: False for column in CONSTRAINT_DIAGNOSTIC_COLUMNS}
    results_path = outputs_root / "paper_tables" / "intervention_calibrated_results.csv"
    if results_path.exists():
        existing = pd.read_csv(results_path)
        if set(CONSTRAINT_DIAGNOSTIC_COLUMNS).issubset(existing.columns):
            if "stage" in existing.columns:
                calibrated = existing.loc[existing["stage"].astype(str).eq("calibrated_test")]
            else:
                calibrated = pd.DataFrame()
            source = calibrated.iloc[-1] if not calibrated.empty else existing.iloc[-1] if not existing.empty else None
            if source is not None:
                return {column: bool(source.get(column, False)) for column in CONSTRAINT_DIAGNOSTIC_COLUMNS}

    diagnostics_path = outputs_root / "magd_policy" / "constrained_policy_diagnostics.json"
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        for key in ["test_metrics", "selected_constraint_status", "selected_validation_metrics"]:
            metrics = diagnostics.get(key, {})
            if isinstance(metrics, dict) and any(column in metrics for column in CONSTRAINT_DIAGNOSTIC_COLUMNS):
                return {column: bool(metrics.get(column, False)) for column in CONSTRAINT_DIAGNOSTIC_COLUMNS}
    return defaults


def _load_method_logs(outputs_root: Path) -> dict[str, tuple[Path, str | None]]:
    def first_existing(*candidates: Path) -> Path | None:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    logs = {
        "AI-only": (outputs_root / "baselines" / "ai_only_decisions.csv", "ai_score"),
        "best expert only": (outputs_root / "baselines" / "best_expert_decisions.csv", None),
        "random expert": (outputs_root / "baselines" / "random_expert_decisions.csv", None),
        "numerical threshold": (outputs_root / "baselines" / "numerical_threshold_decisions.csv", None),
        "distance threshold": (outputs_root / "baselines" / "distance_threshold_decisions.csv", None),
        "oracle upper bound": (outputs_root / "baselines" / "oracle_upper_bound_decisions.csv", None),
    }
    optional_logs = {
        # Legacy pre-MAGD comparator: not produced by any stage of the current MAGD
        # pipeline, so - like the MAGD variants below - it is included only if a decision
        # log genuinely exists on disk, never required.
        "assurance-guided deferral": first_existing(outputs_root / "assurance_deferral" / "assurance_guided_decisions.csv"),
        "learning-to-defer baseline": first_existing(outputs_root / "baselines" / "learning_to_defer_decisions.csv"),
        "MAGD-Heuristic": first_existing(
            outputs_root / "assurance_deferral" / "magd_heuristic_decisions.csv",
            outputs_root / "ablations" / "ablation_decisions_full_magd_heuristic.csv",
        ),
        "MAGD-Learned": first_existing(
            outputs_root / "assurance_deferral" / "magd_learned_decisions.csv",
            outputs_root / "ablations" / "ablation_decisions_full_magd_learned.csv",
        ),
        "MAGD-Fraud-ValidationTuned": first_existing(
            outputs_root / "assurance_deferral" / "magd_validation_tuned_decisions.csv",
        ),
        "MAGD-Constrained": first_existing(
            outputs_root / "assurance_deferral" / "magd_fraud_decisions.csv",
            outputs_root / "assurance_deferral" / "magd_constrained_decisions.csv",
            outputs_root / "ablations" / "ablation_decisions_full_magd_constrained.csv",
            outputs_root / "ablations" / "ablation_decisions_full_magd_constrained_capacity_fairness.csv",
            outputs_root / "ablations" / "ablation_decisions_full_magd_with_capacity_and_fairness.csv",
        ),
    }
    for method_name, path in optional_logs.items():
        if path is not None:
            logs[method_name] = (path, "ai_score")
    return logs


def _core_join_tables(outputs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assurance = pd.read_csv(outputs_root / "assurance" / "wrong_confident_risk.csv")
    model_test = pd.read_csv(outputs_root / "model" / "test_predictions.csv")
    magd_path = outputs_root / "assurance" / "magd_risk.csv"
    magd = pd.read_csv(magd_path) if magd_path.exists() else pd.DataFrame(columns=["case_id", "magd_assurance_risk", "risk_category"])
    assurance["case_id"] = assurance["case_id"].astype(str)
    model_test["case_id"] = model_test["case_id"].astype(str)
    if "case_id" in magd.columns:
        magd["case_id"] = magd["case_id"].astype(str)
    return assurance, model_test, magd


def _fairness_summary(method_name: str, fairness: pd.DataFrame) -> dict[str, float]:
    if fairness.empty:
        return {
            "method": method_name,
            "fairness_disparity": 0.0,
            "fairness_cost_disparity": 0.0,
            "fairness_available": 0.0,
        }
    disparity_columns = [
        column
        for column in [
            "false_positive_rate_disparity",
            "false_negative_rate_disparity",
            "cost_disparity",
        ]
        if column in fairness.columns
    ]
    fairness_disparity = 0.0
    if disparity_columns:
        fairness_disparity = float(
            fairness[disparity_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy()
            .max()
        )
    fairness_cost_disparity = float(pd.to_numeric(fairness.get("cost_disparity", pd.Series([0.0])), errors="coerce").fillna(0.0).max())
    return {
        "method": method_name,
        "fairness_disparity": fairness_disparity,
        "fairness_cost_disparity": fairness_cost_disparity,
        "fairness_available": 1.0,
    }


def _normalize_log(
    frame: pd.DataFrame,
    method_name: str,
    assurance: pd.DataFrame,
    model_test: pd.DataFrame,
    magd: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    working = frame.copy()
    working["case_id"] = working["case_id"].astype(str)
    if "selected_route" not in working.columns:
        source = working.get("decision_source", pd.Series(["AI"] * len(working), index=working.index)).astype(str)
        mapped = source.copy()
        mapped[source.str.contains("Expert")] = "Human Expert"
        mapped[source.str.contains("Escalate")] = "Escalate"
        mapped[~(source.str.contains("Expert") | source.str.contains("Escalate"))] = "AI"
        working["selected_route"] = mapped
    if "selected_expert" not in working.columns:
        working["selected_expert"] = working.get("assigned_expert", "")
    if "decision_reason" not in working.columns:
        working["decision_reason"] = working.get("decision_source", "")
    if "capacity_status" not in working.columns:
        working["capacity_status"] = "not_available"
    if "is_correct" not in working.columns:
        working["is_correct"] = (working["final_prediction"].astype(int) == working["y_true"].astype(int)).astype(int)

    if "offline_wrong_confident_label" in assurance.columns and "wrong_confident_label_offline" not in assurance.columns:
        assurance = assurance.rename(columns={"offline_wrong_confident_label": "wrong_confident_label_offline"})
    assurance_columns = [
        "case_id",
        "ai_score",
        "ai_pred",
        "numerical_confidence",
        "distance_confidence",
        "distance_uncertainty",
        "calibration_risk",
        "neighbor_error_rate",
        "wrong_confident_risk",
        "wrong_confident_label_offline",
    ]
    working = working.merge(
        assurance[[column for column in assurance_columns if column in assurance.columns]],
        on="case_id",
        how="left",
        suffixes=("", "_assurance"),
    ).merge(
        magd[["case_id", "magd_assurance_risk", "risk_category"]],
        on="case_id",
        how="left",
    ).merge(
        metadata,
        on="case_id",
        how="left",
    )
    working["used_ai"] = working["selected_route"].astype(str).eq("AI")
    working["ai_correct"] = (working["ai_pred"].astype(int) == working["y_true"].astype(int)).astype(int)
    working["method"] = method_name
    return working


def run_evaluation(config_path: str | Path) -> tuple[Path, int]:
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    outputs_root, final_dir = _paths(config_path)
    paper_dir = _paper_dir(outputs_root)
    assurance, model_test, magd = _core_join_tables(outputs_root)
    loaded = load_fifar_data(config_path)

    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = (config_path.parent / processed_dir).resolve()
    metadata = pd.read_csv(processed_dir / "test_metadata.csv")
    metadata["case_id"] = metadata["case_id"].astype(str)

    fp_cost = float(config.get("costs", {}).get("false_positive", 1.0))
    fn_cost = float(config.get("costs", {}).get("false_negative", 5.0))

    method_logs = _load_method_logs(outputs_root)
    oracle_frame = None
    fraud_rows = []
    reliance_rows = []
    deferral_rows = []
    fairness_frames = []
    fairness_rows = []
    audit_rows = []
    assurance_rows = []

    normalized_logs: dict[str, pd.DataFrame] = {}
    for method_name, (path, score_column) in method_logs.items():
        frame = pd.read_csv(path)
        normalized = _normalize_log(frame, method_name, assurance, model_test, magd, metadata)
        normalized_logs[method_name] = normalized
        if method_name == "oracle upper bound":
            oracle_frame = normalized

    ai_only_reference = normalized_logs["AI-only"]
    required_assurance_evidence = [
        "numerical_confidence",
        "distance_uncertainty",
        "calibration_risk",
        "neighbor_error_rate",
        "wrong_confident_risk",
        "magd_assurance_risk",
        "selected_route",
        "final_prediction",
        "decision_reason",
    ]

    for method_name, (_, score_column) in method_logs.items():
        normalized = normalized_logs[method_name]
        fraud_rows.append(
            {"method": method_name, **compute_fraud_metrics(normalized, score_column=score_column, fp_cost=fp_cost, fn_cost=fn_cost)}
        )
        reliance_rows.append({"method": method_name, **compute_reliance_metrics(normalized)})
        deferral_rows.append(
            {
                "method": method_name,
                **compute_deferral_metrics(normalized, oracle_reference=oracle_frame),
            }
        )
        fairness = compute_fairness_metrics(normalized, loaded.sensitive_attributes, fp_cost=fp_cost, fn_cost=fn_cost)
        if not fairness.empty:
            fairness["method"] = method_name
            fairness_frames.append(fairness)
        fairness_rows.append(_fairness_summary(method_name, fairness))
        audit_rows.append(
            {
                "method": method_name,
                **compute_audit_metrics(
                    normalized,
                    required_evidence_columns=[
                        "numerical_confidence",
                        "distance_uncertainty",
                        "calibration_risk",
                        "neighbor_error_rate",
                        "wrong_confident_risk",
                    ],
                ),
            }
        )
        assurance_rows.append(
            {
                "method": method_name,
                **compute_assurance_metrics(
                    normalized,
                    ai_only_reference=ai_only_reference,
                    fp_cost=fp_cost,
                    fn_cost=fn_cost,
                    required_evidence_columns=required_assurance_evidence,
                ),
            }
        )

    fraud_df = pd.DataFrame(fraud_rows)
    reliance_df = pd.DataFrame(reliance_rows)
    deferral_df = pd.DataFrame(deferral_rows)
    fairness_df = pd.concat(fairness_frames, ignore_index=True) if fairness_frames else pd.DataFrame()
    fairness_summary_df = pd.DataFrame(fairness_rows)
    audit_df = pd.DataFrame(audit_rows)
    assurance_df = pd.DataFrame(assurance_rows)

    all_method_metrics = (
        fraud_df
        .merge(reliance_df, on="method", how="left")
        .merge(deferral_df, on="method", how="left")
        .merge(audit_df, on="method", how="left")
        .merge(assurance_df, on="method", how="left", suffixes=("", "_assurance"))
        .merge(fairness_summary_df, on="method", how="left")
    )

    all_method_metrics.to_csv(final_dir / "all_method_metrics.csv", index=False)
    reliance_df.to_csv(final_dir / "reliance_metrics.csv", index=False)
    deferral_df.to_csv(final_dir / "deferral_metrics.csv", index=False)
    fairness_df.to_csv(final_dir / "fairness_metrics.csv", index=False)
    audit_df.to_csv(final_dir / "audit_metrics.csv", index=False)
    assurance_df.to_csv(final_dir / "assurance_metrics.csv", index=False)
    all_method_metrics.to_csv(paper_dir / "human_ai_metrics.csv", index=False)

    baseline_methods = [
        "AI-only",
        "best expert only",
        "random expert",
        "numerical threshold",
        "distance threshold",
        "learning-to-defer baseline",
        "oracle upper bound",
    ]
    baseline_comparison = all_method_metrics.loc[all_method_metrics["method"].isin(baseline_methods)].copy()
    baseline_comparison.to_csv(paper_dir / "baseline_comparison.csv", index=False)

    intervention_results = all_method_metrics.loc[all_method_metrics["method"].astype(str).eq("MAGD-Constrained")].copy()
    if not intervention_results.empty:
        intervention_results.insert(0, "stage", "evaluation_test")
    intervention_results = ensure_constraint_diagnostics(
        intervention_results,
        defaults=_load_constrained_diagnostic_defaults(outputs_root),
    )
    intervention_results.to_csv(paper_dir / "intervention_calibrated_results.csv", index=False)
    return final_dir, len(method_logs)


def main() -> None:
    args = parse_args()
    final_dir, method_count = run_evaluation(args.config)
    print(f"Saved outputs to: {final_dir}")
    print(f"Methods evaluated: {method_count}")


if __name__ == "__main__":
    main()
