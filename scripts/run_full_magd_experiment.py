from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_all_methods import run_evaluation as evaluate_all_methods
from scripts.make_paper_tables import generate_paper_tables
from scripts.run_magd_ablations import run_magd_ablations
from scripts.run_statistical_tests import run_statistical_tests
from src.assurance.claim_evidence_matrix import (
    build_auditability_table,
    build_claim_evidence_matrix,
    build_decision_audit_log,
    markdown_table,
    REQUIRED_EVIDENCE_COLUMNS,
    run_claim_evidence_matrix,
)
from src.assurance.distance_uncertainty import run_distance_uncertainty
from src.assurance.local_reliability import run_local_reliability
from src.assurance.magd_risk import run_magd_risk
from src.assurance.wrong_confident_detector import run_wrong_confident_detection
from src.data.validate_splits import validate_data_splits
from src.deferral.baselines import run_baselines
from src.deferral.expert_routing import run_expert_reliability
from src.deferral.learning_to_defer_baseline import run_learning_to_defer_baseline
from src.deferral.magd_constrained import run_magd_constrained
from src.deferral.magd_deferral import run_magd_deferral
from src.deferral.magd_validation_tuned import run_magd_validation_tuned
from src.evaluation.magd_risk_calibration import run_magd_risk_calibration
from src.evaluation.paper_evaluation_outputs import (
    build_budget_matched_results,
    build_constraint_sensitivity,
    build_magd_risk_calibration as build_required_magd_risk_calibration,
    build_per_case_predictions,
    build_statistical_tests as build_required_statistical_tests,
)
from src.models.calibrate_model import run_calibration
from src.models.train_model import run_model_predictions
from src.evaluation.audit_metrics import compute_audit_metrics
from src.utils.io import load_yaml
from src.utils.reporting import (
    PLOT_FILES,
    build_assurance_summary,
    build_audit_coverage,
    build_audit_report_markdown,
    copy_artifact,
    ensure_exists,
    resolve_audit_pack_dirs,
    write_json,
)
from src.utils.scientific_checks import ScientificCheckError, run_scientific_checks


STAGE_COUNT = 29


@dataclass(frozen=True)
class PipelineStage:
    number: int
    label: str
    runner_name: str


@dataclass
class PipelineContext:
    project_root: Path
    config_path: Path
    config: dict[str, Any] = field(default_factory=dict)
    outputs_root: Path | None = None
    final_metrics_dir: Path | None = None
    paper_tables_dir: Path | None = None
    audit_pack_dir: Path | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    stage_records: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_optional_components: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full MAGD-Fraud experiment pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate the pipeline and print the planned stages without executing them.")
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_config(project_root: Path, config_arg: str) -> Path:
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = (project_root / config_arg).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return config_path


def _resolve_path(config_path: Path, raw_path: str | Path | None) -> Path:
    if raw_path is None:
        raise ValueError("A required path is missing from the configuration.")
    path = Path(raw_path)
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _resolve_outputs_root(config: dict[str, Any], config_path: Path) -> Path:
    raw = config.get("paths", {}).get("outputs_dir", config.get("experiment", {}).get("output_dir", "data/outputs"))
    return _resolve_path(config_path, raw)


def _resolve_processed_root(config: dict[str, Any], config_path: Path) -> Path:
    raw = config.get("paths", {}).get("processed_data_dir", "data/processed")
    return _resolve_path(config_path, raw)


def _planned_stages() -> list[PipelineStage]:
    return [
        PipelineStage(1, "validate config", "_stage_validate_config"),
        PipelineStage(2, "validate data splits", "_stage_validate_data_splits"),
        PipelineStage(3, "run/load base model predictions", "_stage_run_model_predictions"),
        PipelineStage(4, "compute numerical confidence and calibration risk", "_stage_run_calibration"),
        PipelineStage(5, "compute distance uncertainty", "_stage_run_distance_uncertainty"),
        PipelineStage(6, "compute local neighbour reliability", "_stage_run_local_reliability"),
        PipelineStage(7, "compute wrong-confident AI risk", "_stage_run_wrong_confident_detection"),
        PipelineStage(8, "compute MAGD assurance risk", "_stage_run_magd_risk"),
        PipelineStage(9, "compute MAGD risk calibration", "_stage_run_magd_risk_calibration"),
        PipelineStage(10, "estimate expert reliability", "_stage_run_expert_reliability"),
        PipelineStage(11, "run baselines", "_stage_run_baselines"),
        PipelineStage(12, "run learning-to-defer baseline", "_stage_run_learning_to_defer_baseline"),
        PipelineStage(13, "run MAGD-Heuristic", "_stage_run_magd_heuristic"),
        PipelineStage(14, "run MAGD-Learned", "_stage_run_magd_learned"),
        PipelineStage(15, "run MAGD-Fraud validation-tuned", "_stage_run_magd_validation_tuned"),
        PipelineStage(16, "run MAGD-Constrained initial", "_stage_run_magd_constrained"),
        PipelineStage(17, "run MAGD-Constrained intervention-calibrated", "_stage_verify_magd_constrained_intervention"),
        PipelineStage(18, "run ablations", "_stage_run_magd_ablations"),
        PipelineStage(19, "evaluate all methods", "_stage_run_evaluation"),
        PipelineStage(20, "run budget-matched deferral analysis", "_stage_run_budget_matched_analysis"),
        PipelineStage(21, "run required MAGD risk calibration", "_stage_run_required_magd_risk_calibration"),
        PipelineStage(22, "run constraint sensitivity analysis", "_stage_run_constraint_sensitivity"),
        PipelineStage(23, "run paired statistical tests for paper", "_stage_run_required_statistical_tests"),
        PipelineStage(24, "run statistical tests", "_stage_run_statistical_tests"),
        PipelineStage(25, "generate audit pack", "_stage_generate_audit_pack"),
        PipelineStage(26, "generate claim-evidence matrix", "_stage_generate_claim_evidence_matrix"),
        PipelineStage(27, "generate paper-ready tables", "_stage_generate_paper_tables"),
        PipelineStage(28, "run scientific checks", "_stage_run_scientific_checks"),
        PipelineStage(29, "write final run summary", "_stage_write_final_run_summary"),
    ]


def _describe_result(result: Any) -> str:
    if result is None:
        return "completed"
    if isinstance(result, tuple):
        parts: list[str] = []
        for item in result:
            if isinstance(item, Path):
                parts.append(str(item))
            elif isinstance(item, pd.DataFrame):
                parts.append(f"{len(item)} rows")
            elif isinstance(item, dict):
                parts.append(", ".join(sorted(item.keys())) if item else "empty mapping")
            else:
                parts.append(str(item))
        return "; ".join(parts)

    details: list[str] = []
    for attribute in [
        "output_dir",
        "assurance_dir",
        "paper_tables_dir",
        "final_dir",
        "plots_dir",
        "ablation_dir",
        "audit_dir",
        "matrix",
        "decisions",
        "metrics",
        "calibration_table",
        "summary_table",
        "local_reliability",
        "distance_frame",
        "risk_frame",
        "expert_reliability",
        "learning_to_defer",
        "baseline_metrics",
        "routing_decisions",
    ]:
        if hasattr(result, attribute):
            value = getattr(result, attribute)
            if isinstance(value, Path):
                details.append(f"{attribute}={value}")
            elif isinstance(value, pd.DataFrame):
                details.append(f"{attribute}={len(value)} rows")
            else:
                details.append(f"{attribute}={type(value).__name__}")
    if isinstance(result, dict):
        if "status" in result:
            details.append(f"status={result['status']}")
        if "critical_failures" in result:
            details.append(f"critical_failures={len(result['critical_failures'])}")
        if "warnings" in result:
            details.append(f"warnings={len(result['warnings'])}")
    return ", ".join(details) if details else type(result).__name__


def _record_stage(context: PipelineContext, stage: PipelineStage, status: str, result: Any | None = None, message: str = "") -> None:
    context.stage_records.append(
        {
            "number": stage.number,
            "label": stage.label,
            "status": status,
            "message": message or _describe_result(result),
        }
    )


def _run_stage(stage: PipelineStage, context: PipelineContext, dry_run: bool) -> Any:
    print(f"[{stage.number}/{STAGE_COUNT}] {stage.label}")
    if dry_run:
        _record_stage(context, stage, "planned", message="dry run")
        return None

    runner = globals()[stage.runner_name]
    try:
        result = runner(context.config_path, context)
    except (FileNotFoundError, ValueError, ScientificCheckError) as exc:
        _record_stage(context, stage, "failed", message=str(exc))
        raise RuntimeError(f"Pipeline stopped at step {stage.number} ({stage.label}): {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive path
        _record_stage(context, stage, "failed", message=str(exc))
        raise RuntimeError(f"Pipeline stopped at step {stage.number} ({stage.label}): {exc}") from exc

    _record_stage(context, stage, "completed", result=result)
    return result


def _stage_validate_config(config_path: Path, context: PipelineContext) -> dict[str, Any]:
    config = load_yaml(config_path)
    outputs_root = _resolve_outputs_root(config, config_path)
    processed_root = _resolve_processed_root(config, config_path)
    context.config = config
    context.outputs_root = outputs_root
    context.final_metrics_dir = outputs_root / "final_metrics"
    context.paper_tables_dir = outputs_root / "paper_tables"
    context.audit_pack_dir = outputs_root / "audit_pack"
    context.artifacts["config"] = config
    context.artifacts["outputs_root"] = outputs_root
    context.artifacts["processed_root"] = processed_root
    return {
        "config_path": str(config_path),
        "outputs_root": str(outputs_root),
        "processed_root": str(processed_root),
    }


def _stage_validate_data_splits(config_path: Path, context: PipelineContext):
    artifacts = validate_data_splits(config_path)
    context.artifacts["dataset_validation"] = artifacts
    context.artifacts["dataset_summary"] = artifacts.summary_table.iloc[0].to_dict()
    if not artifacts.statistics.capacity_configured:
        context.warnings.append("Capacity is not configured; validation will continue.")
    return artifacts


def _stage_run_model_predictions(config_path: Path, context: PipelineContext):
    artifacts = run_model_predictions(config_path)
    context.artifacts["model_predictions"] = artifacts
    return artifacts


def _stage_run_calibration(config_path: Path, context: PipelineContext):
    artifacts = run_calibration(config_path)
    context.artifacts["calibration"] = artifacts
    return artifacts


def _stage_run_distance_uncertainty(config_path: Path, context: PipelineContext):
    artifacts = run_distance_uncertainty(config_path)
    context.artifacts["distance_uncertainty"] = artifacts
    return artifacts


def _stage_run_local_reliability(config_path: Path, context: PipelineContext):
    artifacts = run_local_reliability(config_path)
    context.artifacts["local_reliability"] = artifacts
    return artifacts


def _stage_run_wrong_confident_detection(config_path: Path, context: PipelineContext):
    artifacts = run_wrong_confident_detection(config_path)
    context.artifacts["wrong_confident_detection"] = artifacts
    return artifacts


def _stage_run_magd_risk(config_path: Path, context: PipelineContext):
    artifacts = run_magd_risk(config_path)
    context.artifacts["magd_risk"] = artifacts
    return artifacts


def _stage_run_magd_risk_calibration(config_path: Path, context: PipelineContext):
    artifacts = run_magd_risk_calibration(config_path)
    context.artifacts["magd_risk_calibration"] = artifacts
    return artifacts


def _stage_run_expert_reliability(config_path: Path, context: PipelineContext):
    artifacts = run_expert_reliability(config_path)
    context.artifacts["expert_reliability"] = artifacts
    return artifacts


def _stage_run_baselines(config_path: Path, context: PipelineContext):
    artifacts = run_baselines(config_path)
    context.artifacts["baselines"] = artifacts
    return artifacts


def _stage_run_learning_to_defer_baseline(config_path: Path, context: PipelineContext):
    # L2D-Standard (feature_set="standard") is the official, MAGD-independent baseline
    # for the canonical pipeline. L2D+MAGD (augmented) is a separate, explicitly-labeled
    # optional experiment - see scripts/run_learning_to_defer_augmented_experiment.py.
    artifacts = run_learning_to_defer_baseline(config_path, feature_set="standard")
    context.artifacts["learning_to_defer"] = artifacts
    return artifacts


def _stage_run_magd_heuristic(config_path: Path, context: PipelineContext):
    artifacts = run_magd_deferral(config_path, mode="heuristic")
    context.artifacts["magd_heuristic"] = artifacts
    return artifacts


def _stage_run_magd_learned(config_path: Path, context: PipelineContext):
    artifacts = run_magd_deferral(config_path, mode="learned")
    context.artifacts["magd_learned"] = artifacts
    return artifacts


def _stage_run_magd_validation_tuned(config_path: Path, context: PipelineContext):
    artifacts = run_magd_validation_tuned(config_path)
    context.artifacts["magd_validation_tuned"] = artifacts
    return artifacts


def _stage_run_magd_constrained(config_path: Path, context: PipelineContext):
    artifacts = run_magd_constrained(config_path)
    context.artifacts["magd_constrained"] = artifacts
    return artifacts


def _stage_verify_magd_constrained_intervention(config_path: Path, context: PipelineContext) -> dict[str, str]:
    outputs_root = context.outputs_root or _resolve_outputs_root(context.config or load_yaml(config_path), config_path)
    decisions_path = ensure_exists(
        outputs_root / "assurance_deferral" / "magd_constrained_calibrated_decisions.csv",
        description="MAGD-Constrained intervention-calibrated decisions",
    )
    diagnostics_path = ensure_exists(
        outputs_root / "magd_policy" / "constrained_policy_diagnostics.json",
        description="MAGD-Constrained policy diagnostics",
    )
    result = {
        "decisions_path": str(decisions_path),
        "diagnostics_path": str(diagnostics_path),
    }
    context.artifacts["magd_constrained_intervention"] = result
    return result


def _stage_run_magd_ablations(config_path: Path, context: PipelineContext):
    artifacts = run_magd_ablations(config_path)
    context.artifacts["magd_ablations"] = artifacts
    return artifacts


def _stage_run_evaluation(config_path: Path, context: PipelineContext) -> tuple[Path, int]:
    result = evaluate_all_methods(config_path)
    context.artifacts["evaluation"] = {"final_dir": str(result[0]), "method_count": int(result[1])}
    return result


def _write_artifact_table_map(paper_dir: Path) -> Path:
    rows = [
        {
            "paper_table_or_figure": "budget-matched table/figure",
            "source_csv": "budget_matched_results.csv",
            "description": "Budget-matched deferral comparison across MAGD-Fraud and baseline routing scores.",
        },
        {
            "paper_table_or_figure": "MAGD risk calibration table/figure",
            "source_csv": "magd_risk_calibration.csv",
            "description": "Held-out test MAGD risk quartiles with fraud prevalence, AI error, wrong-confident error, and MAGD deferral.",
        },
        {
            "paper_table_or_figure": "constraint sensitivity table/figure",
            "source_csv": "constraint_sensitivity.csv",
            "description": "Validation-selected MAGD risk thresholds under strict, moderate, relaxed, and high-review constraints.",
        },
        {
            "paper_table_or_figure": "statistical comparison table",
            "source_csv": "statistical_tests.csv",
            "description": "Paired bootstrap confidence intervals and McNemar tests over held-out test cases.",
        },
    ]
    path = paper_dir / "artifact_table_map.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _stage_run_budget_matched_analysis(config_path: Path, context: PipelineContext) -> tuple[pd.DataFrame, Path]:
    outputs_root = context.outputs_root or _resolve_outputs_root(context.config or load_yaml(config_path), config_path)
    paper_dir = outputs_root / "paper_tables"
    paper_dir.mkdir(parents=True, exist_ok=True)
    results = build_budget_matched_results(config_path)
    path = paper_dir / "budget_matched_results.csv"
    results.to_csv(path, index=False)
    context.artifacts["budget_matched_results"] = {"rows": int(len(results)), "path": str(path)}
    return results, path


def _stage_run_required_magd_risk_calibration(config_path: Path, context: PipelineContext) -> tuple[pd.DataFrame, Path]:
    outputs_root = context.outputs_root or _resolve_outputs_root(context.config or load_yaml(config_path), config_path)
    paper_dir = outputs_root / "paper_tables"
    paper_dir.mkdir(parents=True, exist_ok=True)
    calibration, thresholds = build_required_magd_risk_calibration(config_path)
    path = paper_dir / "magd_risk_calibration.csv"
    thresholds_path = paper_dir / "magd_risk_calibration_thresholds.json"
    calibration.to_csv(path, index=False)
    thresholds_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    context.artifacts["required_magd_risk_calibration"] = {
        "rows": int(len(calibration)),
        "path": str(path),
        "thresholds_path": str(thresholds_path),
    }
    return calibration, path


def _stage_run_constraint_sensitivity(config_path: Path, context: PipelineContext) -> tuple[pd.DataFrame, Path]:
    outputs_root = context.outputs_root or _resolve_outputs_root(context.config or load_yaml(config_path), config_path)
    paper_dir = outputs_root / "paper_tables"
    policy_dir = outputs_root / "magd_policy"
    paper_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)
    sensitivity, diagnostics = build_constraint_sensitivity(config_path)
    path = paper_dir / "constraint_sensitivity.csv"
    diagnostics_path = policy_dir / "constraint_sensitivity_diagnostics.json"
    sensitivity.to_csv(path, index=False)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    context.artifacts["constraint_sensitivity"] = {
        "rows": int(len(sensitivity)),
        "path": str(path),
        "diagnostics_path": str(diagnostics_path),
    }
    return sensitivity, path


def _stage_run_required_statistical_tests(config_path: Path, context: PipelineContext) -> tuple[pd.DataFrame, Path]:
    outputs_root = context.outputs_root or _resolve_outputs_root(context.config or load_yaml(config_path), config_path)
    final_dir = outputs_root / "final_metrics"
    paper_dir = outputs_root / "paper_tables"
    final_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    per_case, _ = build_per_case_predictions(config_path)
    per_case_path = final_dir / "per_case_predictions.csv"
    per_case.to_csv(per_case_path, index=False)
    results = build_required_statistical_tests(config_path, per_case)
    path = paper_dir / "statistical_tests.csv"
    results.to_csv(path, index=False)
    artifact_map_path = _write_artifact_table_map(paper_dir)
    context.artifacts["required_statistical_tests"] = {
        "rows": int(len(results)),
        "path": str(path),
        "per_case_path": str(per_case_path),
        "artifact_map_path": str(artifact_map_path),
    }
    return results, path


def _stage_run_statistical_tests(config_path: Path, context: PipelineContext) -> tuple[pd.DataFrame, Path]:
    result = run_statistical_tests(config_path)
    context.artifacts["statistical_tests"] = {"rows": int(len(result[0])), "final_dir": str(result[1])}
    return result


def _generate_audit_pack(config_path: Path, context: PipelineContext) -> Path:
    config, outputs_root, audit_pack_dir = resolve_audit_pack_dirs(config_path)
    paper_dir = outputs_root / "paper_tables"
    paper_dir.mkdir(parents=True, exist_ok=True)

    threshold_src = ensure_exists(outputs_root / "assurance" / "threshold_exploration_distance.csv", description="threshold exploration output")
    final_metrics_src = ensure_exists(outputs_root / "final_metrics" / "all_method_metrics.csv", description="final metrics output")
    reliance_metrics_src = ensure_exists(outputs_root / "final_metrics" / "reliance_metrics.csv", description="reliance metrics output")
    deferral_metrics_src = ensure_exists(outputs_root / "final_metrics" / "deferral_metrics.csv", description="deferral metrics output")
    audit_metrics_src = ensure_exists(outputs_root / "final_metrics" / "audit_metrics.csv", description="audit metrics output")

    fairness_src = outputs_root / "final_metrics" / "fairness_metrics.csv"
    fairness_available = fairness_src.exists()

    decision_logs = build_decision_audit_log(config_path)
    final_metrics = pd.read_csv(final_metrics_src)
    reliance_metrics = pd.read_csv(reliance_metrics_src)
    deferral_metrics = pd.read_csv(deferral_metrics_src)
    audit_metrics = pd.read_csv(audit_metrics_src)
    fairness_metrics = pd.read_csv(fairness_src) if fairness_available else pd.DataFrame()

    decision_logs.to_csv(audit_pack_dir / "decision_audit_log.csv", index=False)
    decision_logs.to_csv(audit_pack_dir / "decision_logs.csv", index=False)

    assurance_summary = build_assurance_summary(decision_logs)
    assurance_summary.to_csv(audit_pack_dir / "assurance_summary.csv", index=False)

    claim_matrix = build_claim_evidence_matrix(decision_logs)
    claim_matrix.to_csv(audit_pack_dir / "claim_evidence_matrix.csv", index=False)
    (audit_pack_dir / "claim_evidence_matrix.md").write_text(markdown_table(claim_matrix.round({"coverage_score": 4})), encoding="utf-8")

    auditability = build_auditability_table(decision_logs, claim_matrix)
    auditability.to_csv(paper_dir / "auditability.csv", index=False)

    copy_artifact(threshold_src, audit_pack_dir / "threshold_exploration.csv")
    copy_artifact(final_metrics_src, audit_pack_dir / "final_metrics.csv")

    if fairness_available:
        fairness_metrics.to_csv(audit_pack_dir / "fairness_summary.csv", index=False)

    fresh_audit_metrics = compute_audit_metrics(decision_logs, REQUIRED_EVIDENCE_COLUMNS)
    method_name = str(decision_logs["method"].iloc[0]) if not decision_logs.empty else "unknown"
    audit_metrics = pd.concat([audit_metrics, pd.DataFrame([{"method": method_name, **fresh_audit_metrics}])], ignore_index=True)

    audit_coverage = build_audit_coverage(
        decision_logs=decision_logs,
        audit_metrics=audit_metrics,
        fairness_available=fairness_available,
        capacity_available=bool(config.get("dataset", {}).get("capacity_file")),
    )
    write_json(audit_coverage, audit_pack_dir / "audit_coverage.json")

    plots_root = outputs_root / "plots"
    for plot_name in PLOT_FILES:
        plot_path = plots_root / plot_name
        if plot_path.exists():
            copy_artifact(plot_path, audit_pack_dir / "plots" / plot_name)

    report_text = build_audit_report_markdown(
        config=config,
        final_metrics=final_metrics,
        reliance_metrics=reliance_metrics,
        deferral_metrics=deferral_metrics,
        audit_coverage=audit_coverage,
        fairness_available=fairness_available,
    )
    (audit_pack_dir / "audit_report.md").write_text(report_text, encoding="utf-8")
    context.artifacts["audit_pack"] = {"audit_pack_dir": str(audit_pack_dir), "fairness_available": fairness_available}
    return audit_pack_dir


def _stage_generate_audit_pack(config_path: Path, context: PipelineContext) -> Path:
    return _generate_audit_pack(config_path, context)


def _stage_generate_claim_evidence_matrix(config_path: Path, context: PipelineContext):
    artifacts = run_claim_evidence_matrix(config_path)
    context.artifacts["claim_evidence_matrix"] = artifacts
    return artifacts


def _detect_placeholder_table(frame: pd.DataFrame) -> bool:
    if len(frame) != 1:
        return False
    row = frame.iloc[0].fillna("NA")
    return all(str(value) == "NA" for value in row.tolist())


def _stage_generate_paper_tables(config_path: Path, context: PipelineContext) -> tuple[Path, dict[str, pd.DataFrame]]:
    paper_dir, generated = generate_paper_tables(config_path)
    context.artifacts["paper_tables"] = {"paper_dir": str(paper_dir), "tables": {name: len(frame) for name, frame in generated.items()}}
    for name, frame in generated.items():
        if _detect_placeholder_table(frame):
            context.missing_optional_components.append(f"paper_table:{name}")
    return paper_dir, generated


def _stage_run_scientific_checks(config_path: Path, context: PipelineContext) -> dict[str, Any]:
    payload = run_scientific_checks(config_path, project_root=context.project_root)
    context.artifacts["scientific_checks"] = payload
    context.warnings.extend(payload.get("warnings", []))
    return payload


def _metrics_file_paths(outputs_root: Path) -> list[str]:
    final_dir = outputs_root / "final_metrics"
    paths = [path for path in sorted(final_dir.iterdir()) if path.is_file() and path.suffix in {".csv", ".json"}]
    return [str(path) for path in paths]


def _paper_table_paths(outputs_root: Path) -> list[str]:
    paper_dir = outputs_root / "paper_tables"
    paths = [path for path in sorted(paper_dir.iterdir()) if path.is_file() and path.suffix in {".csv", ".md"}]
    return [str(path) for path in paths]


def _dataset_summary(context: PipelineContext, outputs_root: Path) -> dict[str, Any]:
    summary = context.artifacts.get("dataset_summary")
    if summary is not None:
        return dict(summary)
    path = outputs_root / "paper_tables" / "dataset_summary.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return frame.iloc[0].to_dict() if not frame.empty else {}


def _methods_run(outputs_root: Path) -> list[str]:
    path = outputs_root / "final_metrics" / "all_method_metrics.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if "method" not in frame.columns:
        return []
    return list(dict.fromkeys(frame["method"].dropna().astype(str).tolist()))


def _missing_optional_components(context: PipelineContext, summary: dict[str, Any]) -> list[str]:
    missing = list(dict.fromkeys(context.missing_optional_components))
    scientific = summary.get("scientific_checks", {})
    for warning in scientific.get("warnings", []):
        if warning not in missing:
            missing.append(warning)
    return missing


def _markdown_summary(summary: dict[str, Any]) -> str:
    dataset_summary = summary.get("dataset_summary", {})
    dataset_rows = [
        f"| {key} | {value} |"
        for key, value in dataset_summary.items()
    ]
    warnings = summary.get("warnings", [])
    missing = summary.get("missing_optional_components", [])
    stages = summary.get("stage_records", [])

    lines = [
        "# Full MAGD-Fraud Run Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Scientific checks passed: `{summary['all_scientific_checks_passed']}`",
        f"- Outputs root: `{summary['outputs_root']}`",
        "",
        "## Dataset Summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    lines.extend(dataset_rows if dataset_rows else ["| NA | NA |"])
    lines.extend(
        [
            "",
            "## Methods Run",
            "",
        ]
    )
    lines.extend([f"- {method}" for method in summary.get("methods_run", [])] or ["- None recorded"])
    lines.extend(
        [
            "",
            "## Metrics Files",
            "",
        ]
    )
    lines.extend([f"- `{path}`" for path in summary.get("metrics_file_paths", [])] or ["- None recorded"])
    lines.extend(
        [
            "",
            "## Paper Tables",
            "",
        ]
    )
    lines.extend([f"- `{path}`" for path in summary.get("paper_table_paths", [])] or ["- None recorded"])
    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )
    lines.extend([f"- {warning}" for warning in warnings] or ["- None"])
    lines.extend(
        [
            "",
            "## Missing Optional Components",
            "",
        ]
    )
    lines.extend([f"- {component}" for component in missing] or ["- None"])
    lines.extend(
        [
            "",
            "## Stage Log",
            "",
            "| Step | Stage | Status | Message |",
            "| --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        [
            f"| {stage['number']} | {stage['label']} | {stage['status']} | {stage.get('message', '')} |"
            for stage in stages
        ]
    )
    lines.append("")
    return "\n".join(lines)


def _write_final_run_summary(config_path: Path, context: PipelineContext) -> dict[str, Any]:
    outputs_root = context.outputs_root or _resolve_outputs_root(context.config or load_yaml(config_path), config_path)
    summary_path = outputs_root / "run_summary.json"
    summary_md_path = outputs_root / "run_summary.md"
    scientific_checks_path = outputs_root / "final_metrics" / "scientific_checks.json"
    scientific_checks: dict[str, Any] = {}
    if scientific_checks_path.exists():
        scientific_checks = json.loads(scientific_checks_path.read_text(encoding="utf-8"))
    elif "scientific_checks" in context.artifacts:
        scientific_checks = context.artifacts["scientific_checks"]

    summary = {
        "config_path": str(config_path),
        "outputs_root": str(outputs_root),
        "dataset_summary": _dataset_summary(context, outputs_root),
        "methods_run": _methods_run(outputs_root),
        "metrics_file_paths": _metrics_file_paths(outputs_root),
        "paper_table_paths": _paper_table_paths(outputs_root),
        "warnings": list(dict.fromkeys(context.warnings)),
        "missing_optional_components": _missing_optional_components(context, {"scientific_checks": scientific_checks}),
        "scientific_checks": {
            "status": scientific_checks.get("status", "unknown"),
            "passed": scientific_checks.get("status") == "passed",
            "path": str(scientific_checks_path),
        },
        "all_scientific_checks_passed": scientific_checks.get("status") == "passed",
        "stage_records": context.stage_records,
    }

    summary["status"] = "passed"
    if summary["warnings"]:
        summary["status"] = "passed_with_warnings"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md_path.write_text(_markdown_summary(summary), encoding="utf-8")
    context.artifacts["run_summary"] = summary
    return summary


def _stage_write_final_run_summary(config_path: Path, context: PipelineContext) -> dict[str, Any]:
    return _write_final_run_summary(config_path, context)


def run_full_magd_experiment(config_path: str | Path, *, project_root: str | Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    resolved_project_root = Path(project_root).resolve() if project_root is not None else _project_root()
    resolved_config = _resolve_config(resolved_project_root, str(config_path))
    context = PipelineContext(project_root=resolved_project_root, config_path=resolved_config)
    stages = _planned_stages()

    print("Starting full MAGD-Fraud experiment pipeline")
    print(f"Project root: {resolved_project_root}")
    print(f"Config: {resolved_config}")
    print(f"Dry run: {dry_run}")

    result: dict[str, Any] | None = None
    for stage in stages:
        stage_result = _run_stage(stage, context, dry_run=dry_run)
        if stage.runner_name == "_stage_run_scientific_checks" and not dry_run:
            context.artifacts["scientific_checks"] = stage_result
        if stage.runner_name == "_stage_write_final_run_summary" and not dry_run:
            result = stage_result

    if dry_run:
        result = {
            "config_path": str(resolved_config),
            "outputs_root": str(_resolve_outputs_root(load_yaml(resolved_config), resolved_config)),
            "status": "dry_run",
            "stage_records": context.stage_records,
            "planned_stages": [stage.label for stage in stages],
        }

    assert result is not None
    print(f"Run summary written to: {Path(result['outputs_root']) / 'run_summary.json' if 'outputs_root' in result else 'not written in dry run'}")
    if not dry_run:
        print("Full MAGD-Fraud experiment pipeline completed successfully.")
    return result


def main() -> None:
    args = parse_args()
    try:
        payload = run_full_magd_experiment(args.config, dry_run=args.dry_run)
    except RuntimeError as exc:
        print(str(exc))
        raise
    if args.dry_run:
        print(f"Dry run planned {len(payload['planned_stages'])} stages.")
    else:
        print(f"Artifacts are available under: {payload['outputs_root']}")


if __name__ == "__main__":
    main()
