from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_magd_ablations import run_magd_ablations
from tests.test_magd_constrained import _setup_constrained_fixture_repo


def test_all_variants_produce_metrics_or_skip_with_status(tmp_path: Path) -> None:
    config_path = _setup_constrained_fixture_repo(tmp_path)
    artifacts = run_magd_ablations(config_path)
    expected = {
        "distance_only",
        "distance_plus_calibration",
        "distance_plus_neighbor_error",
        "distance_plus_wrong_confident",
        "full_magd_heuristic",
        "full_magd_learned",
        "full_magd_constrained_initial",
        "full_magd_constrained_intervention_calibrated",
        "full_magd_constrained_fairness_if_available",
        "full_magd_constrained_capacity_if_available",
    }
    assert expected == set(artifacts.metrics["variant"].astype(str))
    assert artifacts.metrics["status"].astype(str).isin(["completed", "skipped_no_sensitive_attributes", "skipped_no_capacity_data"]).all()


def test_variants_use_only_selected_signals(tmp_path: Path) -> None:
    config_path = _setup_constrained_fixture_repo(tmp_path)
    run_magd_ablations(config_path)
    ablation_dir = config_path.parent / "data" / "outputs" / "ablations"

    distance_only = pd.read_csv(ablation_dir / "ablation_decisions_distance_only.csv")
    assert distance_only["calibration_risk"].eq(0.0).all()
    assert distance_only["neighbor_error_rate"].eq(0.0).all()
    assert distance_only["wrong_confident_risk"].eq(0.0).all()

    distance_plus_cal = pd.read_csv(ablation_dir / "ablation_decisions_distance_plus_calibration.csv")
    assert distance_plus_cal["neighbor_error_rate"].eq(0.0).all()
    assert distance_plus_cal["wrong_confident_risk"].eq(0.0).all()
    assert distance_plus_cal["calibration_risk"].gt(0.0).any()


def test_missing_optional_capacity_variant_is_skipped_with_clear_status(tmp_path: Path) -> None:
    config_path = _setup_constrained_fixture_repo(tmp_path)
    artifacts = run_magd_ablations(config_path)
    row = artifacts.metrics.loc[artifacts.metrics["variant"] == "full_magd_constrained_capacity_if_available"].iloc[0]
    assert row["status"] == "skipped_no_capacity_data"
    assert not (artifacts.ablation_dir / "ablation_decisions_full_magd_constrained_capacity_if_available.csv").exists()


def test_paper_table_generated(tmp_path: Path) -> None:
    config_path = _setup_constrained_fixture_repo(tmp_path)
    artifacts = run_magd_ablations(config_path)
    assert (artifacts.paper_dir / "ablation.csv").exists()
    assert (artifacts.ablation_dir / "ablation_metrics.csv").exists()
