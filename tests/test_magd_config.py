from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.io import CONFIG_DEFAULTS, MAGD_DEFAULTS, load_yaml


def _config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config.yaml"


def _assert_subset(expected, actual) -> None:
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual
            _assert_subset(value, actual[key])
    else:
        assert expected == actual


def test_config_loads_with_new_sections() -> None:
    config = load_yaml(_config_path())

    assert config["experiment"]["seed"] == 42
    assert config["experiment"]["output_dir"] == "data/outputs"
    assert config["experiment"]["test_labels_allowed_for_routing"] is False
    assert config["data"]["dataset_name"] == "fifar"
    assert config["model"]["model_type"] == "xgboost"
    assert config["costs"]["false_positive"] == 0.057
    assert config["statistics"]["bootstrap_iterations"] == 1000


def test_config_loads_with_magd_section() -> None:
    config = load_yaml(_config_path())

    assert "magd" in config
    assert config["magd"]["mode"] == "constrained"
    assert config["magd"]["signals"]["use_distance_uncertainty"] is True
    assert config["magd"]["use_distance_uncertainty"] is True
    assert config["magd"]["expert_routing"]["top_k_for_escalation"] == 5


def test_all_magd_weights_are_numeric() -> None:
    config = load_yaml(_config_path())
    weights = config["magd"]["risk_weights"]

    assert set(weights) == set(MAGD_DEFAULTS["risk_weights"])
    assert all(isinstance(value, (int, float)) for value in weights.values())


def test_all_requested_numeric_weights_are_numeric() -> None:
    config = load_yaml(_config_path())

    assert all(isinstance(value, (int, float)) for value in config["magd"]["weights"].values())
    assert isinstance(config["expert_routing"]["capacity_penalty_weight"], (int, float))
    assert isinstance(config["expert_routing"]["fairness_penalty_weight"], (int, float))


def test_magd_thresholds_are_between_zero_and_one() -> None:
    config = load_yaml(_config_path())
    thresholds = config["magd"]["thresholds"]
    numeric_values = [value for key, value in thresholds.items() if key != "mode"]

    assert thresholds["mode"] in {"fixed", "validation_quantile"}
    assert all(0.0 <= float(value) <= 1.0 for value in numeric_values)


def test_intervention_constraints_are_valid() -> None:
    config = load_yaml(_config_path())
    constraints = config["intervention_constraints"]

    assert constraints["enabled"] is True
    assert 0.0 <= constraints["max_overreliance"] <= 1.0
    assert 0.0 <= constraints["min_wrong_confident_avoidance"] <= 1.0
    assert 0.0 <= constraints["min_deferral_rate"] <= constraints["max_deferral_rate"] <= 1.0
    assert 0.0 <= constraints["min_audit_coverage"] <= 1.0


def test_output_dir_is_created_if_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "nested" / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "experiment:\n"
        f"  output_dir: {output_dir.as_posix()}\n",
        encoding="utf-8",
    )

    assert not output_dir.exists()
    config = load_yaml(config_path)

    assert output_dir.exists()
    assert config["experiment"]["output_dir"] == output_dir.as_posix()


def test_defaults_are_applied_when_sections_are_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project:\n  name: demo\n", encoding="utf-8")

    config = load_yaml(config_path)

    _assert_subset(CONFIG_DEFAULTS["experiment"], config["experiment"])
    _assert_subset(CONFIG_DEFAULTS["data"], config["data"])
    _assert_subset(MAGD_DEFAULTS, config["magd"])


def test_invalid_magd_threshold_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "magd:\n"
        "  thresholds:\n"
        "    low_risk: 1.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="magd.thresholds.low_risk"):
        load_yaml(config_path)


def test_non_numeric_magd_weight_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "magd:\n"
        "  weights:\n"
        "    distance_uncertainty: abc\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="magd.risk_weights.distance_uncertainty"):
        load_yaml(config_path)


def test_invalid_intervention_constraints_raise(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "intervention_constraints:\n"
        "  min_deferral_rate: 0.8\n"
        "  max_deferral_rate: 0.4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="min_deferral_rate"):
        load_yaml(config_path)
