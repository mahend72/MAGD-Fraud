from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assurance.magd_risk import (
    _magd_config,
    add_risk_categories_and_actions,
    compute_magd_risk,
    derive_validation_risk_thresholds,
    map_magd_risk_category,
    resolve_magd_risk_thresholds,
)
from src.deferral.magd_policy import _active_weight_keys, _normalize_weights
from src.evaluation.magd_risk_calibration import build_magd_risk_calibration_table


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3"],
            "split": ["val", "test", "test"],
            "distance_uncertainty": [0.1, 0.5, 0.8],
            "calibration_risk": [0.1, 0.4, 0.8],
            "neighbor_error_rate": [0.0, 0.4, 0.9],
            "wrong_confident_risk": [0.1, 0.6, 0.95],
        }
    )


def _weights() -> dict[str, float]:
    return {
        "distance_uncertainty": 0.25,
        "calibration_risk": 0.20,
        "neighbor_error_rate": 0.20,
        "wrong_confident_risk": 0.25,
        "drift_risk": 0.05,
        "business_risk": 0.05,
    }


def test_magd_risk_between_zero_and_one() -> None:
    scored = compute_magd_risk(
        _frame(),
        _weights(),
        use_drift_risk=False,
        use_business_risk=False,
    )
    assert scored["magd_assurance_risk"].between(0.0, 1.0).all()


def test_magd_category_mapping_correct() -> None:
    assert map_magd_risk_category(0.10, 0.35, 0.70) == "low"
    assert map_magd_risk_category(0.35, 0.35, 0.70) == "medium"
    assert map_magd_risk_category(0.69, 0.35, 0.70) == "medium"
    assert map_magd_risk_category(0.70, 0.35, 0.70) == "high"

    enriched = add_risk_categories_and_actions(
        pd.DataFrame({"magd_assurance_risk": [0.10, 0.50, 0.90]}),
        low_risk=0.35,
        high_risk=0.70,
    )
    assert enriched["risk_category"].tolist() == ["low", "medium", "high"]
    assert enriched["recommended_action"].tolist() == ["AI", "Human Expert", "Escalate"]


def test_missing_optional_signals_handled_safely() -> None:
    scored = compute_magd_risk(
        _frame(),
        _weights(),
        use_drift_risk=True,
        use_business_risk=True,
    )
    assert (scored["drift_risk"] == 0.0).all()
    assert (scored["business_risk"] == 0.0).all()
    assert not scored["drift_available"].any()
    assert not scored["business_available"].any()


def test_y_true_not_used_in_risk_computation() -> None:
    frame_a = _frame().assign(y_true=[0, 1, 0])
    frame_b = _frame().assign(y_true=[1, 0, 1])
    scored_a = compute_magd_risk(frame_a, _weights(), use_drift_risk=False, use_business_risk=False)
    scored_b = compute_magd_risk(frame_b, _weights(), use_drift_risk=False, use_business_risk=False)
    pd.testing.assert_series_equal(scored_a["magd_assurance_risk"], scored_b["magd_assurance_risk"])


def test_zero_sum_weights_fail_safely() -> None:
    zero_weights = {
        "distance_uncertainty": 0.0,
        "calibration_risk": 0.0,
        "neighbor_error_rate": 0.0,
        "wrong_confident_risk": 0.0,
        "drift_risk": 0.0,
        "business_risk": 0.0,
    }
    with pytest.raises(ValueError, match="positive"):
        compute_magd_risk(_frame(), zero_weights, use_drift_risk=False, use_business_risk=False)


def test_thresholds_are_derived_from_validation_split_only() -> None:
    frame = pd.DataFrame(
        {
            "case_id": [f"v{i}" for i in range(10)] + [f"t{i}" for i in range(5)],
            "split": ["val"] * 10 + ["test"] * 5,
            "magd_assurance_risk": [0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.60]
            + [0.99, 0.98, 0.97, 0.96, 0.95],
        }
    )
    low_risk, high_risk, metadata = resolve_magd_risk_thresholds(
        frame,
        threshold_mode="validation_quantile",
        low_quantile=0.60,
        high_quantile=0.90,
        fallback_low_risk=0.35,
        fallback_high_risk=0.70,
    )
    assert metadata["mode"] == "validation_quantile"
    assert metadata["validation_rows"] == 10
    # Thresholds must come purely from the val-split distribution (max 0.60), never the
    # much larger test-split values (0.95-0.99) that would otherwise pull them upward.
    assert high_risk < 0.90

    mutated = frame.copy()
    mutated.loc[mutated["split"] == "test", "magd_assurance_risk"] = 0.01
    low_risk_b, high_risk_b, _ = resolve_magd_risk_thresholds(
        mutated,
        threshold_mode="validation_quantile",
        low_quantile=0.60,
        high_quantile=0.90,
        fallback_low_risk=0.35,
        fallback_high_risk=0.70,
    )
    assert low_risk_b == low_risk
    assert high_risk_b == high_risk


def test_medium_and_high_risk_categories_are_reachable_from_validation_quantiles() -> None:
    # A realistic, right-skewed validation risk distribution (most cases low risk, a
    # meaningful tail of higher-risk cases) - the historical bug was that fixed absolute
    # thresholds (0.35/0.70) sat above the entire observed range, so medium/high were
    # unreachable regardless of the actual risk distribution shape.
    validation_risk = pd.Series([0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.20, 0.38])
    low_risk, high_risk = derive_validation_risk_thresholds(validation_risk, low_quantile=0.60, high_quantile=0.90)
    categories = [map_magd_risk_category(value, low_risk, high_risk) for value in validation_risk]
    assert "medium" in categories
    assert "high" in categories


def test_threshold_freezing_and_routing_use_identical_active_signal_normalization() -> None:
    """Regression test for the magd_risk.py vs magd_policy.py normalization mismatch:
    _magd_config() (threshold-freezing, run_magd_risk) and _normalize_weights()
    (routing, load_policy_weights/run_magd_deferral) must score every case identically
    for the same nominal weights and disabled-signal set - disabled signals (drift/
    business, here both off) must not contribute to either function's normalization
    denominator, or the frozen thresholds silently stop matching the scale of the
    scores routing actually compares them against.
    """
    config = {
        "magd": {
            "risk_weights": {
                "distance_uncertainty": 0.25,
                "calibration_risk": 0.20,
                "neighbor_error_rate": 0.20,
                "wrong_confident_risk": 0.25,
                "drift_risk": 0.05,
                "business_risk": 0.05,
            },
            "signals": {
                "use_distance_uncertainty": True,
                "use_calibration_risk": True,
                "use_neighbor_error_rate": True,
                "use_wrong_confident_risk": True,
                "use_drift_risk": False,
                "use_business_risk": False,
            },
            "thresholds": {},
        }
    }
    freeze_weights = _magd_config(config)["weights"]
    routing_weights = _normalize_weights(config["magd"]["risk_weights"], _active_weight_keys(config))

    frame = pd.DataFrame(
        {
            "case_id": ["a", "b", "c"],
            "distance_uncertainty": [0.4, 0.1, 0.9],
            "calibration_risk": [0.3, 0.2, 0.05],
            "neighbor_error_rate": [0.2, 0.4, 0.6],
            "wrong_confident_risk": [0.5, 0.6, 0.1],
        }
    )
    scored_freeze = compute_magd_risk(frame, freeze_weights, use_drift_risk=False, use_business_risk=False)
    scored_routing = compute_magd_risk(frame, routing_weights, use_drift_risk=False, use_business_risk=False)
    pd.testing.assert_series_equal(
        scored_freeze["magd_assurance_risk"],
        scored_routing["magd_assurance_risk"],
        check_exact=False,
        atol=1e-9,
    )


def test_frozen_validation_quantile_thresholds_produce_intended_risk_proportions() -> None:
    """The 60th/90th percentile validation-quantile design should yield approximately
    60% low / 30% medium / 10% high on the SAME validation distribution the thresholds
    were derived from (this is a tautology of quantiles, but guards against a future
    regression - e.g. a scale mismatch like the one fixed here - silently breaking it).
    """
    rng = np.random.default_rng(42)
    validation_risk = pd.Series(rng.beta(a=2.0, b=8.0, size=20000))
    low_risk, high_risk = derive_validation_risk_thresholds(validation_risk, low_quantile=0.60, high_quantile=0.90)

    categories = pd.Series([map_magd_risk_category(v, low_risk, high_risk) for v in validation_risk])
    proportions = categories.value_counts(normalize=True)

    assert abs(proportions.get("low", 0.0) - 0.60) < 0.01
    assert abs(proportions.get("medium", 0.0) - 0.30) < 0.01
    assert abs(proportions.get("high", 0.0) - 0.10) < 0.01


def test_calibration_table_generated(tmp_path: Path) -> None:
    repo_dir = tmp_path
    assurance_dir = repo_dir / "data" / "outputs" / "assurance"
    assurance_dir.mkdir(parents=True, exist_ok=True)

    config_path = repo_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  output_dir: data/outputs",
                "data:",
                "  dataset_name: fifar",
                "  train_path: data/processed/X_train.csv",
                "  val_path: data/processed/X_val.csv",
                "  test_path: data/processed/X_test.csv",
                "  y_train_path: data/processed/y_train.csv",
                "  y_val_path: data/processed/y_val.csv",
                "  y_test_path: data/processed/y_test.csv",
                "  sensitive_attribute: customer_age",
            ]
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "case_id": ["v1", "t1", "t2", "t3"],
            "split": ["val", "test", "test", "test"],
            "magd_assurance_risk": [0.1, 0.25, 0.65, 0.85],
        }
    ).to_csv(assurance_dir / "magd_risk.csv", index=False)
    pd.DataFrame(
        {
            "case_id": ["v1", "t1", "t2", "t3"],
            "split": ["val", "test", "test", "test"],
            "wrong_confident_label_offline": [0, 1, 0, 1],
        }
    ).to_csv(assurance_dir / "wrong_confident_risk.csv", index=False)
    pd.DataFrame(
        {
            "case_id": ["v1", "t1", "t2", "t3"],
            "split": ["val", "test", "test", "test"],
            "y_true": [0, 1, 0, 1],
            "ai_pred": [0, 0, 0, 1],
        }
    ).to_csv(assurance_dir / "calibration_risk.csv", index=False)

    table = build_magd_risk_calibration_table(config_path)
    assert table["magd_risk_bin"].tolist() == ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    assert set(["cases", "ai_error_rate", "wrong_confident_rate", "deferral_rate"]).issubset(table.columns)
