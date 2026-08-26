from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from scripts.evaluate_all_methods import run_evaluation
from scripts.generate_audit_pack import main as generate_audit_pack_main
from scripts.generate_claim_evidence_matrix import main as generate_claim_matrix_main
from tests.test_run_evaluation import _setup_evaluation_fixture_repo


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_audit_pack_and_claim_matrix_outputs_created(tmp_path: Path, monkeypatch) -> None:
    config_path = _setup_evaluation_fixture_repo(tmp_path)
    outputs_root = config_path.parent / "data" / "outputs"
    _write_csv(
        outputs_root / "assurance" / "threshold_exploration_distance.csv",
        pd.DataFrame({"threshold": [0.2, 0.4], "cost_sensitive_loss": [1.0, 0.8]}),
    )
    run_evaluation(config_path)

    monkeypatch.setattr(sys, "argv", ["generate_audit_pack.py", "--config", str(config_path)])
    generate_audit_pack_main()
    monkeypatch.setattr(sys, "argv", ["generate_claim_evidence_matrix.py", "--config", str(config_path)])
    generate_claim_matrix_main()

    audit_pack_dir = outputs_root / "audit_pack"
    paper_dir = outputs_root / "paper_tables"
    assert (audit_pack_dir / "decision_audit_log.csv").exists()
    assert (audit_pack_dir / "claim_evidence_matrix.csv").exists()
    assert (audit_pack_dir / "claim_evidence_matrix.md").exists()
    assert (paper_dir / "auditability.csv").exists()

    auditability = pd.read_csv(paper_dir / "auditability.csv")
    assert {"audit_coverage", "evidence_completeness", "missing_rationale_rate"}.issubset(auditability.columns)

    claim_matrix = pd.read_csv(audit_pack_dir / "claim_evidence_matrix.csv")
    assert len(claim_matrix) == 7
    assert claim_matrix["coverage_score"].between(0.0, 1.0).all()
    assert {"coverage_score", "example_case_ids"}.issubset(claim_matrix.columns)
