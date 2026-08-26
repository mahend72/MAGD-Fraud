from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.utils.scientific_checks import ScientificCheckError, run_scientific_checks


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _setup_scientific_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path
    config_path = repo / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  output_dir: data/outputs",
                "paths:",
                "  outputs_dir: data/outputs",
                "magd:",
                "  signals:",
                "    use_drift_risk: false",
                "    use_business_risk: false",
                "expert_routing:",
                "  capacity_enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    (repo / "README.md").write_text(
        "\n".join(
            [
                "This repository is a research prototype.",
                "The FiFAR panels are synthetic experts from a benchmark environment.",
                "The dashboard is a research prototype. It is not evidence of human-subject validation.",
            ]
        ),
        encoding="utf-8",
    )
    dashboard_dir = repo / "src" / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    (dashboard_dir / "app.py").write_text(
        'PROTOTYPE_NOTICE = "This dashboard is a research prototype and not a validated operational fraud-review tool."\n',
        encoding="utf-8",
    )

    outputs = repo / "data" / "outputs"
    _write_csv(outputs / "final_metrics" / "all_method_metrics.csv", pd.DataFrame({"method": ["MAGD-Constrained"], "cost_sensitive_loss": [1.0], "precision": [0.5]}))
    _write_csv(outputs / "final_metrics" / "reliance_metrics.csv", pd.DataFrame({"method": ["MAGD-Constrained"], "overreliance": [0.1], "correct_rejection": [0.9], "wrong_confident_avoidance_rate": [0.7]}))
    _write_csv(outputs / "final_metrics" / "audit_metrics.csv", pd.DataFrame({"method": ["MAGD-Constrained"], "audit_coverage": [1.0]}))
    _write_csv(outputs / "baselines" / "oracle_upper_bound_decisions.csv", pd.DataFrame({"case_id": ["c1"]}))
    _write_csv(
        outputs / "assurance" / "magd_risk.csv",
        pd.DataFrame(
            {
                "case_id": ["c1"],
                "drift_risk": [0.0],
                "business_risk": [0.0],
                "drift_available": [False],
                "business_available": [False],
            }
        ),
    )
    _write_csv(
        outputs / "assurance_deferral" / "expert_reliability.csv",
        pd.DataFrame(
            {
                "expert": ["expert_a"],
                "capacity_enabled": [False],
                "fairness_available": [False],
            }
        ),
    )
    (outputs / "magd_policy").mkdir(parents=True, exist_ok=True)
    (outputs / "magd_policy" / "constrained_policy_diagnostics.json").write_text(
        json.dumps(
            {
                "selected_constraint_status": {"audit_coverage_satisfied": True},
                "selected_params": {"high_risk": 0.7},
                "test_metrics": {"audit_coverage": 1.0},
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        outputs / "paper_tables" / "intervention_calibrated_results.csv",
        pd.DataFrame(
            {
                "stage": ["calibrated_test"],
                "feasible": [True],
                "overreliance_satisfied": [True],
                "wrong_confident_avoidance_satisfied": [True],
                "audit_coverage_satisfied": [True],
            }
        ),
    )
    _write_csv(
        outputs / "paper_tables" / "budget_matched_results.csv",
        pd.DataFrame(
            {
                "method": ["MAGD-Fraud"] * 5,
                "budget": [0.01, 0.02, 0.05, 0.10, 0.20],
                "threshold": [0.9, 0.8, 0.7, 0.6, 0.5],
                "recall": [0.1] * 5,
                "precision": [0.2] * 5,
                "f1": [0.13] * 5,
                "cost_loss": [1.0] * 5,
                "ai_coverage": [0.99, 0.98, 0.95, 0.90, 0.80],
                "human_deferral": [0.01, 0.02, 0.05, 0.10, 0.20],
                "wca": [0.5] * 5,
                "correct_rejection": [0.1] * 5,
                "overreliance": [0.1] * 5,
            }
        ),
    )
    _write_csv(
        outputs / "paper_tables" / "magd_risk_calibration.csv",
        pd.DataFrame(
            {
                "risk_bin": ["Low", "Medium", "High", "Very high"],
                "cases": [1, 1, 1, 1],
                "fraud_prevalence": [0.0, 0.0, 1.0, 1.0],
                "ai_error": [0.0, 0.1, 0.2, 0.3],
                "wc_error": [0.0, 0.0, 0.1, 0.2],
                "deferral": [0.0, 0.1, 0.2, 0.3],
            }
        ),
    )
    _write_csv(
        outputs / "paper_tables" / "constraint_sensitivity.csv",
        pd.DataFrame(
            {
                "setting": ["Strict"],
                "deferral_budget": [0.01],
                "wca_target": [0.5],
                "overreliance_bound": [0.5],
                "feasible": [True],
                "recall": [0.1],
                "precision": [0.2],
                "f1": [0.13],
                "cost_loss": [1.0],
                "wca": [0.5],
                "correct_rejection": [0.1],
                "overreliance": [0.1],
                "human_deferral": [0.01],
                "ai_coverage": [0.99],
            }
        ),
    )
    _write_csv(
        outputs / "paper_tables" / "statistical_tests.csv",
        pd.DataFrame(
            {
                "comparison": ["MAGD-Fraud-ValidationTuned vs AI-only"],
                "delta_f1": [0.01],
                "delta_f1_ci_low": [0.0],
                "delta_f1_ci_high": [0.02],
                "delta_cost": [-1.0],
                "delta_cost_ci_low": [-2.0],
                "delta_cost_ci_high": [0.0],
                "mcnemar_p": [1.0],
                "n_bootstrap": [1000],
            }
        ),
    )
    (outputs / "paper_tables" / "magd_risk_calibration_thresholds.json").write_text(
        json.dumps({"binning": "fixture", "bins": []}),
        encoding="utf-8",
    )
    (outputs / "magd_policy" / "constraint_sensitivity_diagnostics.json").write_text(
        json.dumps([{"setting": "Strict", "optimizer_status": "feasible"}]),
        encoding="utf-8",
    )
    _write_csv(
        outputs / "paper_tables" / "artifact_table_map.csv",
        pd.DataFrame(
            {
                "paper_table_or_figure": ["budget-matched table/figure"],
                "source_csv": ["budget_matched_results.csv"],
                "description": ["Fixture mapping row."],
            }
        ),
    )
    return config_path


def test_scientific_checks_run_and_write_outputs(tmp_path: Path) -> None:
    config_path = _setup_scientific_fixture_repo(tmp_path)
    payload = run_scientific_checks(config_path, project_root=config_path.parent)
    assert payload["status"] == "passed"
    assert (config_path.parent / "data" / "outputs" / "final_metrics" / "scientific_checks.json").exists()
    assert (config_path.parent / "docs" / "scientific_guardrails_report.md").exists()


def test_scientific_check_failure_messages_are_clear(tmp_path: Path) -> None:
    config_path = _setup_scientific_fixture_repo(tmp_path)
    (config_path.parent / "data" / "outputs" / "final_metrics" / "all_method_metrics.csv").unlink()
    with pytest.raises(ScientificCheckError, match="Missing all method metrics"):
        run_scientific_checks(config_path, project_root=config_path.parent)


def test_required_output_missing_does_not_false_pass(tmp_path: Path) -> None:
    config_path = _setup_scientific_fixture_repo(tmp_path)
    (config_path.parent / "data" / "outputs" / "magd_policy" / "constrained_policy_diagnostics.json").unlink()
    with pytest.raises(ScientificCheckError):
        run_scientific_checks(config_path, project_root=config_path.parent)
    payload = json.loads((config_path.parent / "data" / "outputs" / "final_metrics" / "scientific_checks.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert any(check["name"] == "intervention_calibrated_diagnostics_present" and not check["passed"] for check in payload["checks"])


def test_scientific_checks_fail_on_placeholder_in_any_paper_table(tmp_path: Path) -> None:
    config_path = _setup_scientific_fixture_repo(tmp_path)
    _write_csv(
        config_path.parent / "data" / "outputs" / "paper_tables" / "unrelated_table.csv",
        pd.DataFrame({"column": ["--"]}),
    )

    with pytest.raises(ScientificCheckError, match="placeholder string"):
        run_scientific_checks(config_path, project_root=config_path.parent)
