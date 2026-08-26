from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.make_paper_tables import EXPECTED_TABLES, generate_paper_tables
from tests.test_run_evaluation import _setup_evaluation_fixture_repo


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_make_paper_tables_generates_expected_outputs_and_manifest(tmp_path: Path) -> None:
    config_path = _setup_evaluation_fixture_repo(tmp_path)
    repo_root = config_path.parent
    outputs_root = repo_root / "data" / "outputs"
    paper_dir = outputs_root / "paper_tables"
    assurance_dir = outputs_root / "assurance"
    ablations_dir = outputs_root / "ablations"
    audit_pack_dir = outputs_root / "audit_pack"
    final_dir = outputs_root / "final_metrics"

    _write_csv(
        paper_dir / "dataset_summary.csv",
        pd.DataFrame(
            [
                {
                    "total_cases": 4,
                    "train_cases": 1,
                    "validation_cases": 1,
                    "test_cases": 2,
                    "train_fraud_prevalence": 0.0,
                    "validation_fraud_prevalence": 1.0,
                    "test_fraud_prevalence": 0.5,
                    "number_of_synthetic_experts": 2,
                    "sensitive_attribute_used": "customer_age",
                    "capacity_configured": "no",
                }
            ]
        ),
    )
    _write_csv(
        paper_dir / "ai_assurance.csv",
        pd.DataFrame(
            [
                {
                    "split": "test",
                    "pr_auc": 0.5,
                    "f1": 0.4,
                    "expected_calibration_error": 0.1,
                    "mean_numerical_confidence": 0.8,
                    "mean_calibration_risk": 0.1,
                }
            ]
        ),
    )
    _write_csv(
        paper_dir / "baseline_comparison.csv",
        pd.DataFrame([{"method": "AI-only", "precision": 0.5, "recall": 0.5, "f1": 0.5, "cost_sensitive_loss": 1.0}]),
    )
    _write_csv(
        paper_dir / "intervention_calibrated_results.csv",
        pd.DataFrame([{"stage": "calibrated_test", "phase": "test", "fraud_loss": 1.0, "audit_coverage": 1.0, "feasible": False}]),
    )
    _write_csv(
        paper_dir / "human_ai_metrics.csv",
        pd.DataFrame([{"method": "AI-only", "precision": 0.5, "recall": 0.5, "f1": 0.5, "cost_sensitive_loss": 1.0, "audit_coverage": 1.0}]),
    )
    _write_csv(
        paper_dir / "magd_risk_calibration.csv",
        pd.DataFrame([{"magd_risk_bin": "0.0-0.2", "cases": 2, "ai_error_rate": 0.0, "wrong_confident_rate": 0.0, "deferral_rate": None}]),
    )
    _write_csv(
        paper_dir / "ablation.csv",
        pd.DataFrame([{"variant": "distance_only", "status": "completed", "signals_used": "distance_uncertainty", "f1": 0.5, "audit_coverage": 1.0}]),
    )
    _write_csv(
        paper_dir / "statistical_comparison.csv",
        pd.DataFrame([{"method_A": "AI-only", "method_B": "distance-threshold", "metric": "f1", "test": "paired_bootstrap", "statistic": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0, "interpretation": "NA"}]),
    )
    _write_csv(
        assurance_dir / "threshold_exploration_distance.csv",
        pd.DataFrame({"threshold": [0.1], "cost_sensitive_loss": [1.0]}),
    )
    _write_csv(
        outputs_root / "assurance_deferral" / "magd_constrained_calibrated_decisions.csv",
        pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "y_true": [0, 1],
                "ai_score": [0.8, 0.2],
                "ai_pred": [1, 0],
                "numerical_confidence": [0.8, 0.8],
                "calibration_risk": [0.1, 0.2],
                "distance_uncertainty": [0.3, 0.4],
                "neighbor_error_rate": [0.2, 0.3],
                "wrong_confident_risk": [0.4, 0.1],
                "magd_assurance_risk": [0.7, 0.2],
                "risk_category": ["high", "low"],
                "selected_route": ["Escalate", "AI"],
                "selected_expert": ["expert_a|expert_b", ""],
                "final_prediction": [0, 0],
                "decision_reason": ["high_risk", "low_risk"],
                "method": ["magd_constrained_calibrated", "magd_constrained_calibrated"],
            }
        ),
    )
    _write_csv(
        audit_pack_dir / "claim_evidence_matrix.csv",
        pd.DataFrame([{"claim_id": "C1", "claim_text": "AI confidence is reliable enough for use.", "evidence_fields": "numerical_confidence, calibration_risk", "coverage_score": 1.0, "missing_evidence_count": 0, "example_case_ids": "c1"}]),
    )
    _write_csv(
        final_dir / "all_method_metrics.csv",
        pd.DataFrame([{"method": "AI-only", "precision": 0.5}]),
    )

    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "implementation_coverage_report.md").write_text(
        "\n".join(
            [
                "| Component | Status | Notes |",
                "| --- | --- | --- |",
                "| 1. Load data | IMPLEMENTED | test note |",
                "| 2. Drift risk | MISSING | test missing |",
            ]
        ),
        encoding="utf-8",
    )

    generated_dir, generated = generate_paper_tables(config_path)

    assert generated_dir == paper_dir
    assert set(generated.keys()) == set(EXPECTED_TABLES)
    for table_name in EXPECTED_TABLES:
        csv_path = paper_dir / f"{table_name}.csv"
        md_path = paper_dir / f"{table_name}.md"
        assert csv_path.exists()
        assert md_path.exists()
        frame = pd.read_csv(csv_path)
        assert not frame.empty

    manifest_path = paper_dir / "table_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest) == len(EXPECTED_TABLES)
    assert any(entry["table_name"] == "magd_risk_calibration" for entry in manifest)
    for entry in manifest:
        assert "source_files" in entry
        assert "generated_timestamp" in entry
        assert "missing_values" in entry
        assert "warnings" in entry
