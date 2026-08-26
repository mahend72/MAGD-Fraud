from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.assurance.wrong_confident_detector import compute_deployable_wrong_confident_risk
from src.deferral.expert_routing import _ai_expected_cost, _expert_expected_cost, route_cases
from src.deferral.magd_deferral import route_magd_cases
from src.utils.io import load_yaml


class ScientificCheckError(RuntimeError):
    """Raised when a scientific guardrail check fails."""


@dataclass
class CheckRecord:
    name: str
    severity: str
    passed: bool
    message: str
    evidence: dict[str, Any]


def _resolve_outputs_root(config_path: Path) -> Path:
    config = load_yaml(config_path)
    output_dir = config.get("paths", {}).get("outputs_dir", config.get("experiment", {}).get("output_dir", "data/outputs"))
    outputs_root = Path(output_dir)
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    return outputs_root


def _decision_logic_source(function) -> str:
    source = inspect.getsource(function)
    if "decisions.append(" in source:
        return source.split("decisions.append(", 1)[0]
    return source


def _make_record(name: str, severity: str, passed: bool, message: str, **evidence: Any) -> CheckRecord:
    return CheckRecord(name=name, severity=severity, passed=passed, message=message, evidence=evidence)


def _read_csv_required(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return pd.read_csv(path)


def _ensure_deployable_methods_do_not_use_y_true() -> CheckRecord:
    deployable_functions = {
        "route_magd_cases": _decision_logic_source(route_magd_cases),
        "route_cases": _decision_logic_source(route_cases),
        "_ai_expected_cost": inspect.getsource(_ai_expected_cost),
        "_expert_expected_cost": inspect.getsource(_expert_expected_cost),
        "compute_deployable_wrong_confident_risk": inspect.getsource(compute_deployable_wrong_confident_risk),
    }
    offenders = [name for name, source in deployable_functions.items() if "y_true" in source]
    if offenders:
        return _make_record(
            "deployable_y_true_guard",
            "critical",
            False,
            "Deployable routing or deployable risk code references `y_true` inside decision logic.",
            offenders=offenders,
        )
    return _make_record(
        "deployable_y_true_guard",
        "critical",
        True,
        "Deployable routing and deployable risk code do not reference `y_true` inside decision logic.",
        inspected_functions=sorted(deployable_functions.keys()),
    )


def _ensure_oracle_only_upper_bound(project_root: Path, outputs_root: Path) -> CheckRecord:
    oracle_baseline = outputs_root / "baselines" / "oracle_upper_bound_decisions.csv"
    deployable_oracle_outputs = list((outputs_root / "assurance_deferral").glob("*oracle*"))
    deployable_decision_logs = list((outputs_root / "assurance_deferral").glob("*decisions*.csv"))
    oracle_selected_in_deployable: list[str] = []
    for path in deployable_decision_logs:
        frame = pd.read_csv(path, usecols=lambda column: column in {"selected_expert", "method"})
        for column in ["selected_expert", "method"]:
            if column in frame.columns and frame[column].astype(str).str.contains("oracle", case=False, na=False).any():
                oracle_selected_in_deployable.append(str(path))
                break
    failures: list[str] = []
    if not oracle_baseline.exists():
        failures.append("Oracle upper-bound baseline artifact is missing.")
    if deployable_oracle_outputs:
        failures.append("Oracle artifacts were written under deployable assurance outputs.")
    if oracle_selected_in_deployable:
        failures.append("Deployable decision logs reference oracle experts or oracle methods.")
    return _make_record(
        "oracle_upper_bound_only",
        "critical",
        not failures,
        "Oracle is restricted to the upper-bound baseline only." if not failures else " ".join(failures),
        oracle_baseline_exists=oracle_baseline.exists(),
        deployable_oracle_outputs=[str(path) for path in deployable_oracle_outputs],
        deployable_decision_logs_with_oracle=oracle_selected_in_deployable,
    )


def _ensure_accuracy_not_main_metric(outputs_root: Path, project_root: Path) -> CheckRecord:
    all_method_metrics = _read_csv_required(outputs_root / "final_metrics" / "all_method_metrics.csv", "all method metrics")
    accuracy_columns = [column for column in all_method_metrics.columns if "accuracy" in column.lower()]
    readme_text = (project_root / "README.md").read_text(encoding="utf-8").lower()
    forbidden_phrases = [
        "accuracy is the main metric",
        "main metric is accuracy",
        "primary metric is accuracy",
    ]
    failures: list[str] = []
    if accuracy_columns:
        failures.append("Accuracy appears as a first-class reported metric.")
    if any(phrase in readme_text for phrase in forbidden_phrases):
        failures.append("README describes accuracy as the main or primary metric.")
    return _make_record(
        "accuracy_not_headline_metric",
        "critical",
        not failures,
        "Accuracy is not treated as the headline metric." if not failures else " ".join(failures),
        accuracy_columns=accuracy_columns,
    )


def _ensure_required_metrics_reported(outputs_root: Path) -> list[CheckRecord]:
    all_method_metrics = _read_csv_required(outputs_root / "final_metrics" / "all_method_metrics.csv", "all method metrics")
    reliance_metrics = _read_csv_required(outputs_root / "final_metrics" / "reliance_metrics.csv", "reliance metrics")
    audit_metrics = _read_csv_required(outputs_root / "final_metrics" / "audit_metrics.csv", "audit metrics")
    checks: list[CheckRecord] = []
    required = [
        ("cost_sensitive_loss_reported", "critical", "cost_sensitive_loss", all_method_metrics, "all_method_metrics.csv"),
        ("overreliance_reported", "critical", "overreliance", reliance_metrics, "reliance_metrics.csv"),
        ("correct_rejection_reported", "critical", "correct_rejection", reliance_metrics, "reliance_metrics.csv"),
        ("wrong_confident_avoidance_reported", "critical", "wrong_confident_avoidance_rate", reliance_metrics, "reliance_metrics.csv"),
        ("audit_coverage_reported", "critical", "audit_coverage", audit_metrics, "audit_metrics.csv"),
    ]
    for name, severity, column, frame, source_name in required:
        checks.append(
            _make_record(
                name,
                severity,
                column in frame.columns,
                f"`{column}` is reported in `{source_name}`." if column in frame.columns else f"`{column}` is missing from `{source_name}`.",
                source_file=source_name,
                column=column,
            )
        )
    return checks


def _ensure_synthetic_experts_described_correctly(project_root: Path) -> CheckRecord:
    readme_text = (project_root / "README.md").read_text(encoding="utf-8").lower()
    mentions_synthetic = "synthetic expert" in readme_text or "synthetic experts" in readme_text
    wrongly_claims_real = "real humans" in readme_text or "real human experts" in readme_text
    passed = mentions_synthetic and not wrongly_claims_real
    message = "README labels FiFAR experts as synthetic." if passed else "README does not clearly label FiFAR experts as synthetic."
    return _make_record(
        "synthetic_experts_labelled",
        "critical",
        passed,
        message,
        mentions_synthetic=mentions_synthetic,
        wrongly_claims_real=wrongly_claims_real,
    )


def _ensure_dashboard_described_as_prototype(project_root: Path) -> CheckRecord:
    readme_text = (project_root / "README.md").read_text(encoding="utf-8").lower()
    dashboard_text = (project_root / "src" / "dashboard" / "app.py").read_text(encoding="utf-8").lower()
    has_prototype_readme = "prototype" in readme_text
    has_prototype_dashboard = "research prototype" in dashboard_text
    forbidden_phrases = ["validated with human subjects"]
    positive_validation_markers = ["human-subject validation", "human subject validation"]
    negative_context_markers = [
        "not a real human-subject study",
        "not yet a real human-subject study",
        "not evidence of human-subject validation",
        "not human-subject validation",
    ]
    invalid_validation_claim = (
        any(phrase in readme_text for phrase in forbidden_phrases)
        or any(phrase in dashboard_text for phrase in forbidden_phrases)
        or (any(marker in readme_text for marker in positive_validation_markers) and not any(marker in readme_text for marker in negative_context_markers))
        or (any(marker in dashboard_text for marker in positive_validation_markers) and not any(marker in dashboard_text for marker in negative_context_markers))
    )
    passed = has_prototype_readme and has_prototype_dashboard and not invalid_validation_claim
    return _make_record(
        "dashboard_labelled_prototype",
        "critical",
        passed,
        "README and dashboard label the interface as a research prototype." if passed else "Prototype labelling or validation disclaimers are insufficient.",
        readme_mentions_prototype=has_prototype_readme,
        dashboard_mentions_prototype=has_prototype_dashboard,
        invalid_validation_claim=invalid_validation_claim,
    )


def _optional_signal_status(outputs_root: Path, config_path: Path) -> CheckRecord:
    config = load_yaml(config_path)
    warnings: list[str] = []
    evidence: dict[str, Any] = {}
    magd_risk_path = outputs_root / "assurance" / "magd_risk.csv"
    expert_reliability_path = outputs_root / "assurance_deferral" / "expert_reliability.csv"

    if magd_risk_path.exists():
        magd = pd.read_csv(magd_risk_path, nrows=10)
        for signal, availability in [("drift_risk", "drift_available"), ("business_risk", "business_available")]:
            configured = bool(config.get("magd", {}).get("signals", {}).get(f"use_{signal}", config.get("magd", {}).get("use_" + signal, False)))
            evidence[f"{signal}_configured"] = configured
            if not configured:
                if availability not in magd.columns:
                    warnings.append(f"`{availability}` is missing from `magd_risk.csv`.")
                elif not magd[availability].isin([False, 0]).all():
                    warnings.append(f"`{availability}` is not logged as unavailable when `{signal}` is disabled.")
    else:
        warnings.append("`magd_risk.csv` is missing, so optional drift/business availability could not be checked.")

    if expert_reliability_path.exists():
        expert = pd.read_csv(expert_reliability_path, nrows=10)
        if bool(config.get("expert_routing", {}).get("capacity_enabled", False)) is False and "capacity_enabled" in expert.columns:
            if not expert["capacity_enabled"].isin([False, 0]).all():
                warnings.append("`capacity_enabled` is not logged as false in `expert_reliability.csv` when capacity is disabled.")
        elif "capacity_enabled" not in expert.columns:
            warnings.append("`capacity_enabled` is missing from `expert_reliability.csv`.")
        if "fairness_available" not in expert.columns:
            warnings.append("`fairness_available` is missing from `expert_reliability.csv`.")
    else:
        warnings.append("`expert_reliability.csv` is missing, so optional fairness/capacity availability could not be checked.")

    passed = not warnings
    return _make_record(
        "optional_signals_logged_when_missing",
        "warning",
        passed,
        "Optional capacity/fairness/drift/business signals are explicitly logged as unavailable when missing." if passed else " ".join(warnings),
        warnings=warnings,
        **evidence,
    )


def _ensure_intervention_calibrated_diagnostics(outputs_root: Path) -> CheckRecord:
    diagnostics_path = outputs_root / "magd_policy" / "constrained_policy_diagnostics.json"
    results_path = outputs_root / "paper_tables" / "intervention_calibrated_results.csv"
    failures: list[str] = []
    evidence: dict[str, Any] = {}
    if not diagnostics_path.exists():
        failures.append("Constrained-policy diagnostics JSON is missing.")
    else:
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        evidence["diagnostic_keys"] = sorted(diagnostics.keys())
        required_keys = {"selected_constraint_status", "selected_params", "test_metrics"}
        missing_keys = sorted(required_keys - set(diagnostics.keys()))
        if missing_keys:
            failures.append("Constrained-policy diagnostics JSON is missing required keys: " + ", ".join(missing_keys))
    if not results_path.exists():
        failures.append("Intervention-calibrated results table is missing.")
    else:
        results = pd.read_csv(results_path)
        required_columns = {"stage", "feasible", "overreliance_satisfied", "wrong_confident_avoidance_satisfied", "audit_coverage_satisfied"}
        missing_columns = sorted(required_columns - set(results.columns))
        if missing_columns:
            failures.append("Intervention-calibrated results are missing required diagnostic columns: " + ", ".join(missing_columns))
    return _make_record(
        "intervention_calibrated_diagnostics_present",
        "critical",
        not failures,
        "Intervention-calibrated MAGD-Constrained diagnostics are present." if not failures else " ".join(failures),
        diagnostics_path=str(diagnostics_path),
        results_path=str(results_path),
        **evidence,
    )


def _ensure_required_paper_evaluation_outputs(outputs_root: Path) -> list[CheckRecord]:
    paper_dir = outputs_root / "paper_tables"
    required_files: dict[str, set[str]] = {
        "budget_matched_results.csv": {
            "method",
            "budget",
            "threshold",
            "recall",
            "precision",
            "f1",
            "cost_loss",
            "ai_coverage",
            "human_deferral",
            "wca",
            "correct_rejection",
            "overreliance",
        },
        "magd_risk_calibration.csv": {
            "risk_bin",
            "cases",
            "fraud_prevalence",
            "ai_error",
            "wc_error",
            "deferral",
        },
        "constraint_sensitivity.csv": {
            "setting",
            "deferral_budget",
            "wca_target",
            "overreliance_bound",
            "feasible",
            "recall",
            "precision",
            "f1",
            "cost_loss",
            "wca",
            "correct_rejection",
            "overreliance",
            "human_deferral",
            "ai_coverage",
        },
        "statistical_tests.csv": {
            "comparison",
            "delta_f1",
            "delta_f1_ci_low",
            "delta_f1_ci_high",
            "delta_cost",
            "delta_cost_ci_low",
            "delta_cost_ci_high",
            "mcnemar_p",
            "n_bootstrap",
        },
    }
    records: list[CheckRecord] = []
    for file_name, required_columns in required_files.items():
        path = paper_dir / file_name
        failures: list[str] = []
        evidence: dict[str, Any] = {"path": str(path)}
        if not path.exists():
            failures.append(f"{file_name} is missing.")
            records.append(_make_record(f"required_{path.stem}_present", "critical", False, " ".join(failures), **evidence))
            continue
        frame = pd.read_csv(path)
        evidence["rows"] = int(len(frame))
        evidence["columns"] = list(frame.columns)
        if frame.empty:
            failures.append(f"{file_name} is empty.")
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            failures.append(f"{file_name} is missing required columns: {', '.join(missing)}.")
        if frame.astype(str).eq("--").any().any():
            failures.append(f"{file_name} contains placeholder string `--`.")
        if file_name == "budget_matched_results.csv" and "budget" in frame.columns:
            observed = sorted({round(float(value), 2) for value in pd.to_numeric(frame["budget"], errors="coerce").dropna().tolist()})
            expected = [0.01, 0.02, 0.05, 0.10, 0.20]
            evidence["observed_budgets"] = observed
            if observed != expected:
                failures.append("budget_matched_results.csv does not include exactly the required budgets.")
            if {"ai_coverage", "human_deferral"}.issubset(frame.columns):
                sums = pd.to_numeric(frame["ai_coverage"], errors="coerce") + pd.to_numeric(frame["human_deferral"], errors="coerce")
                max_gap = float((sums.dropna() - 1.0).abs().max()) if sums.notna().any() else 0.0
                evidence["max_coverage_sum_gap"] = max_gap
                if max_gap > 1e-6:
                    failures.append("budget_matched_results.csv has rows where ai_coverage + human_deferral is not approximately 1.0.")
        if file_name == "statistical_tests.csv":
            for column in ["delta_f1_ci_low", "delta_f1_ci_high", "delta_cost_ci_low", "delta_cost_ci_high"]:
                if column in frame.columns and pd.to_numeric(frame[column], errors="coerce").isna().any():
                    failures.append(f"statistical_tests.csv column {column} is not fully numeric.")
        records.append(
            _make_record(
                f"required_{path.stem}_present",
                "critical",
                not failures,
                f"{file_name} is present, non-empty, and schema-valid." if not failures else " ".join(failures),
                **evidence,
            )
        )

    sidecars = [
        paper_dir / "magd_risk_calibration_thresholds.json",
        outputs_root / "magd_policy" / "constraint_sensitivity_diagnostics.json",
        paper_dir / "artifact_table_map.csv",
    ]
    for path in sidecars:
        passed = path.exists() and path.stat().st_size > 0
        records.append(
            _make_record(
                f"required_{path.stem}_present",
                "critical",
                passed,
                f"{path.name} is present and non-empty." if passed else f"{path.name} is missing or empty.",
                path=str(path),
            )
        )
    placeholder_files: list[str] = []
    if paper_dir.exists():
        for path in sorted(paper_dir.glob("*.csv")):
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if frame.astype(str).eq("--").any().any():
                placeholder_files.append(path.name)
    records.append(
        _make_record(
            "paper_table_placeholder_strings_absent",
            "critical",
            not placeholder_files,
            "No paper-table CSV contains placeholder string `--`."
            if not placeholder_files
            else "Paper-table CSVs contain placeholder string `--`: " + ", ".join(placeholder_files),
            placeholder_files=placeholder_files,
        )
    )
    return records


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Scientific Guardrails Report",
        "",
        f"- Status: `{payload['status']}`",
        f"- Critical failures: `{len(payload['critical_failures'])}`",
        f"- Optional warnings: `{len(payload['warnings'])}`",
        "",
        "## Checks",
        "",
        "| Check | Severity | Passed | Message |",
        "| --- | --- | --- | --- |",
    ]
    for check in payload["checks"]:
        lines.append(
            f"| {check['name']} | {check['severity']} | {'yes' if check['passed'] else 'no'} | {check['message']} |"
        )
    if payload["critical_failures"]:
        lines.extend(["", "## Critical Failures", ""])
        lines.extend([f"- {message}" for message in payload["critical_failures"]])
    if payload["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {message}" for message in payload["warnings"]])
    return "\n".join(lines) + "\n"


def _serialize_payload(*, config_path: Path, project_root: Path, outputs_root: Path, records: list[CheckRecord]) -> dict[str, Any]:
    critical_failures = [record.message for record in records if record.severity == "critical" and not record.passed]
    warnings = [record.message for record in records if record.severity != "critical" and not record.passed]
    return {
        "config_path": str(config_path),
        "project_root": str(project_root),
        "outputs_root": str(outputs_root),
        "status": "failed" if critical_failures else "passed_with_warnings" if warnings else "passed",
        "critical_failures": critical_failures,
        "warnings": warnings,
        "checks": [asdict(record) for record in records],
    }


def _safe_run(name: str, severity: str, fn, *args) -> CheckRecord | list[CheckRecord]:
    try:
        return fn(*args)
    except FileNotFoundError as exc:
        return _make_record(name, severity, False, str(exc))
    except Exception as exc:  # pragma: no cover - defensive path
        return _make_record(name, severity, False, f"Unexpected scientific check error in `{name}`: {exc}")


def run_scientific_checks(config_path: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    resolved_config = Path(config_path).resolve()
    resolved_project_root = Path(project_root).resolve() if project_root is not None else resolved_config.parent
    outputs_root = _resolve_outputs_root(resolved_config)

    raw_records: list[CheckRecord | list[CheckRecord]] = [
        _safe_run("deployable_y_true_guard", "critical", _ensure_deployable_methods_do_not_use_y_true),
        _safe_run("oracle_upper_bound_only", "critical", _ensure_oracle_only_upper_bound, resolved_project_root, outputs_root),
        _safe_run("accuracy_not_headline_metric", "critical", _ensure_accuracy_not_main_metric, outputs_root, resolved_project_root),
        _safe_run("required_metrics_reported", "critical", _ensure_required_metrics_reported, outputs_root),
        _safe_run("synthetic_experts_labelled", "critical", _ensure_synthetic_experts_described_correctly, resolved_project_root),
        _safe_run("dashboard_labelled_prototype", "critical", _ensure_dashboard_described_as_prototype, resolved_project_root),
        _safe_run("optional_signals_logged_when_missing", "warning", _optional_signal_status, outputs_root, resolved_config),
        _safe_run("intervention_calibrated_diagnostics_present", "critical", _ensure_intervention_calibrated_diagnostics, outputs_root),
        _safe_run("required_paper_evaluation_outputs", "critical", _ensure_required_paper_evaluation_outputs, outputs_root),
    ]
    records: list[CheckRecord] = []
    for item in raw_records:
        if isinstance(item, list):
            records.extend(item)
        else:
            records.append(item)

    payload = _serialize_payload(
        config_path=resolved_config,
        project_root=resolved_project_root,
        outputs_root=outputs_root,
        records=records,
    )

    final_dir = outputs_root / "final_metrics"
    final_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = resolved_project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "scientific_checks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (docs_dir / "scientific_guardrails_report.md").write_text(_markdown_report(payload), encoding="utf-8")

    if payload["critical_failures"]:
        raise ScientificCheckError("Scientific guardrail checks failed:\n- " + "\n- ".join(payload["critical_failures"]))
    return payload
