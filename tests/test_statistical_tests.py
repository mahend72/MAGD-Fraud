from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.run_statistical_tests import run_statistical_tests
from src.evaluation.statistical_tests import paired_bootstrap_interval
from tests.test_magd_constrained import _setup_constrained_fixture_repo


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _setup_statistical_fixture_repo(tmp_path: Path) -> Path:
    config_path = _setup_constrained_fixture_repo(tmp_path)
    outputs_root = config_path.parent / "data" / "outputs"
    baselines_dir = outputs_root / "baselines"
    assurance_deferral_dir = outputs_root / "assurance_deferral"

    ai_only = pd.DataFrame(
        {
            "case_id": ["s1", "s2", "s3"],
            "y_true": [0, 1, 1],
            "ai_score": [0.9, 0.25, 0.55],
            "ai_pred": [1, 0, 1],
            "selected_route": ["AI", "AI", "AI"],
            "selected_expert": ["", "", ""],
            "final_prediction": [1, 0, 1],
            "decision_reason": ["ai_only", "ai_only", "ai_only"],
        }
    )
    distance = pd.DataFrame(
        {
            "case_id": ["s1", "s2", "s3"],
            "y_true": [0, 1, 1],
            "ai_score": [0.9, 0.25, 0.55],
            "ai_pred": [1, 0, 1],
            "selected_route": ["Escalate", "AI", "Human Expert"],
            "selected_expert": ["expert_a|expert_b", "", "expert_a"],
            "final_prediction": [0, 0, 1],
            "decision_reason": ["distance", "distance", "distance"],
        }
    )
    ltd = pd.DataFrame(
        {
            "case_id": ["s1", "s2", "s3"],
            "y_true": [0, 1, 1],
            "ai_score": [0.9, 0.25, 0.55],
            "ai_pred": [1, 0, 1],
            "selected_route": ["Human Expert", "AI", "Human Expert"],
            "selected_expert": ["expert_a", "", "expert_a"],
            "final_prediction": [0, 0, 1],
            "decision_reason": ["ltd", "ltd", "ltd"],
        }
    )
    constrained_initial = pd.DataFrame(
        {
            "case_id": ["s1", "s2", "s3"],
            "y_true": [0, 1, 1],
            "ai_score": [0.9, 0.25, 0.55],
            "ai_pred": [1, 0, 1],
            "selected_route": ["Escalate", "AI", "AI"],
            "selected_expert": ["expert_a|expert_b", "", ""],
            "final_prediction": [0, 0, 1],
            "decision_reason": ["initial", "initial", "initial"],
        }
    )
    constrained_calibrated = pd.DataFrame(
        {
            "case_id": ["s1", "s2", "s3"],
            "y_true": [0, 1, 1],
            "ai_score": [0.9, 0.25, 0.55],
            "ai_pred": [1, 0, 1],
            "selected_route": ["Escalate", "Human Expert", "Human Expert"],
            "selected_expert": ["expert_a|expert_b", "expert_a", "expert_a"],
            "final_prediction": [0, 1, 1],
            "decision_reason": ["calibrated", "calibrated", "calibrated"],
        }
    )

    _write_csv(baselines_dir / "ai_only_decisions.csv", ai_only)
    _write_csv(baselines_dir / "distance_threshold_decisions.csv", distance)
    _write_csv(baselines_dir / "learning_to_defer_decisions.csv", ltd)
    _write_csv(assurance_deferral_dir / "magd_constrained_initial_decisions.csv", constrained_initial)
    _write_csv(assurance_deferral_dir / "magd_constrained_calibrated_decisions.csv", constrained_calibrated)
    return config_path


def test_bootstrap_reproducible_with_seed() -> None:
    frame_a = pd.DataFrame({"y_true": [0, 1, 1], "final_prediction": [1, 0, 1], "used_ai": [1, 1, 1], "ai_correct": [0, 0, 1], "wrong_confident_label_offline": [1, 0, 1]})
    frame_b = pd.DataFrame({"y_true": [0, 1, 1], "final_prediction": [0, 1, 1], "used_ai": [0, 1, 0], "ai_correct": [0, 0, 1], "wrong_confident_label_offline": [1, 0, 1]})
    first = paired_bootstrap_interval(frame_a, frame_b, metric="cost_sensitive_loss", fp_cost=2.0, fn_cost=7.0, n_bootstrap=100, random_state=42)
    second = paired_bootstrap_interval(frame_a, frame_b, metric="cost_sensitive_loss", fp_cost=2.0, fn_cost=7.0, n_bootstrap=100, random_state=42)
    assert first == second


def test_missing_methods_raise_clear_error(tmp_path: Path) -> None:
    config_path = _setup_constrained_fixture_repo(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing required method outputs"):
        run_statistical_tests(config_path)


def test_output_table_created_and_p_values_valid(tmp_path: Path) -> None:
    config_path = _setup_statistical_fixture_repo(tmp_path)
    results, final_dir = run_statistical_tests(config_path)
    assert not results.empty
    valid_p = results["p_value"].dropna()
    assert ((valid_p >= 0.0) & (valid_p <= 1.0)).all()
    assert (final_dir / "statistical_comparisons.csv").exists()
    assert (config_path.parent / "data" / "outputs" / "paper_tables" / "statistical_comparison.csv").exists()
