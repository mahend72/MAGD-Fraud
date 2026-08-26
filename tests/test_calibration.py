from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import yaml

from src.assurance.calibration import (
    assign_calibration_risk,
    build_calibration_bins,
    expected_calibration_error,
    run_calibration_assurance,
)


def _validation_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["v1", "v2", "v3", "v4"],
            "split": ["val", "val", "val", "val"],
            "y_true": [0, 0, 0, 1],
            "ai_score": [0.1, 0.2, 0.8, 0.9],
            "ai_pred": [0, 0, 1, 1],
        }
    )


def test_expected_calibration_error_matches_manual_value() -> None:
    calibration_bins = build_calibration_bins(_validation_predictions(), n_bins=2)
    ece = expected_calibration_error(calibration_bins)
    assert math.isclose(ece, 0.10, rel_tol=1e-9, abs_tol=1e-9)


def test_confidence_in_zero_one() -> None:
    calibration_bins = build_calibration_bins(_validation_predictions(), n_bins=2)
    scored = assign_calibration_risk(_validation_predictions(), calibration_bins)
    assert scored["numerical_confidence"].between(0.0, 1.0).all()


def test_calibration_risk_in_zero_one() -> None:
    calibration_bins = build_calibration_bins(_validation_predictions(), n_bins=2)
    scored = assign_calibration_risk(_validation_predictions(), calibration_bins)
    assert scored["calibration_risk"].between(0.0, 1.0).all()


def test_calibration_bins_are_learned_from_validation_only() -> None:
    calibration_bins = build_calibration_bins(_validation_predictions(), n_bins=2)
    assert (calibration_bins["split_used_to_learn_bins"] == "validation").all()


def test_test_y_true_is_not_used_to_compute_deployable_calibration_risk() -> None:
    calibration_bins = build_calibration_bins(_validation_predictions(), n_bins=2)
    test_a = pd.DataFrame(
        {
            "case_id": ["t1", "t2"],
            "split": ["test", "test"],
            "y_true": [0, 1],
            "ai_score": [0.15, 0.85],
            "ai_pred": [0, 1],
        }
    )
    test_b = test_a.copy()
    test_b["y_true"] = [1, 0]

    scored_a = assign_calibration_risk(test_a, calibration_bins)
    scored_b = assign_calibration_risk(test_b, calibration_bins)

    pd.testing.assert_series_equal(scored_a["calibration_risk"], scored_b["calibration_risk"])
    pd.testing.assert_series_equal(scored_a["calibration_bin"], scored_b["calibration_bin"])


def test_empty_calibration_bin_yields_zero_risk_not_nan() -> None:
    # All validation confidences fall in the top half of [0,1], so with 4 bins the two
    # lowest bins are empty (count=0, absolute_calibration_gap=NaN in the bins table).
    calibration_bins = build_calibration_bins(_validation_predictions(), n_bins=4)
    empty_bins = calibration_bins.loc[calibration_bins["count"] == 0]
    assert not empty_bins.empty
    assert empty_bins["absolute_calibration_gap"].isna().all()

    test_case_in_empty_bin = pd.DataFrame(
        {
            "case_id": ["low_conf_case"],
            "split": ["test"],
            "y_true": [0],
            "ai_score": [0.51],
            "ai_pred": [1],
        }
    )
    scored = assign_calibration_risk(test_case_in_empty_bin, calibration_bins)
    assert scored["calibration_bin"].iloc[0] in set(empty_bins["calibration_bin"])
    assert not scored["calibration_risk"].isna().any()
    assert float(scored["calibration_risk"].iloc[0]) == 0.0

    ece = expected_calibration_error(calibration_bins)
    assert ece == ece  # not NaN


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_config(path: Path) -> None:
    payload = {
        "experiment": {"output_dir": "data/outputs"},
        "paths": {"outputs_dir": "data/outputs"},
        "model": {"threshold": 0.5},
        "costs": {"false_positive": 0.057, "false_negative": 1.0},
        "assurance": {"n_bins_calibration": 2},
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def test_run_calibration_assurance_writes_required_outputs(tmp_path: Path) -> None:
    model_dir = tmp_path / "data/outputs/model"
    train = pd.DataFrame(
        {
            "case_id": ["c1", "c2"],
            "y_true": [0, 1],
            "ai_score": [0.2, 0.8],
            "ai_pred": [0, 1],
            "split": ["train", "train"],
        }
    )
    val = _validation_predictions()
    test = pd.DataFrame(
        {
            "case_id": ["t1", "t2"],
            "y_true": [0, 1],
            "ai_score": [0.3, 0.7],
            "ai_pred": [0, 1],
            "split": ["test", "test"],
        }
    )
    _write_csv(model_dir / "train_predictions.csv", train)
    _write_csv(model_dir / "val_predictions.csv", val)
    _write_csv(model_dir / "test_predictions.csv", test)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)

    artifacts = run_calibration_assurance(config_path)

    assert (artifacts.assurance_dir / "numerical_confidence.csv").exists()
    assert (artifacts.assurance_dir / "calibration_bins.csv").exists()
    assert (artifacts.assurance_dir / "calibration_risk.csv").exists()
    assert (artifacts.paper_tables_dir / "ai_assurance.csv").exists()
