from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.deferral.learning_to_defer_baseline import (
    AUGMENTED_FEATURE_COLUMNS,
    STANDARD_FEATURE_COLUMNS,
    run_learning_to_defer_baseline,
)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _setup_fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path
    model_dir = repo / "data" / "outputs" / "model"
    assurance_dir = repo / "data" / "outputs" / "assurance"
    processed_dir = repo / "data" / "processed"
    raw_dir = repo / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    config_path = repo / "config.yaml"
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
                "paths:",
                "  raw_data_dir: data/raw",
                "  processed_data_dir: data/processed",
                "  outputs_dir: data/outputs",
                "dataset:",
                "  main_file: data/raw/main.csv",
                "  train_file: data/raw/train.csv",
                "  test_file: data/raw/test.csv",
                "  expert_predictions_file: data/raw/expert_predictions.csv",
                "  historical_expert_predictions_file: data/raw/historical_expert_predictions.csv",
                "columns:",
                "  case_id: case_id",
                "  batch_id: batch",
                "  time: month",
                "  label: fraud_label",
                "  train_label: fraud_label",
                "  test_label: fraud_label",
                "  model_score: model_score",
                "  expert_case_id: case_id",
                "  expert_prediction: expert_a",
                "  sensitive_attributes:",
                "    - customer_age",
            ]
        ),
        encoding="utf-8",
    )

    _write_csv(raw_dir / "main.csv", pd.DataFrame({"case_id": [1], "fraud_label": [0], "batch": ["b1"], "month": [1], "customer_age": ["young"], "model_score": [0.5]}))
    _write_csv(
        raw_dir / "train.csv",
        pd.DataFrame(
            {
                "case_id": ["t1", "t2", "v1", "v2"],
                "fraud_label": [0, 1, 0, 1],
                "batch": ["b1", "b1", "b1", "b1"],
                "month": [1, 1, 2, 2],
                "customer_age": ["young", "old", "young", "old"],
                "model_score": [0.2, 0.8, 0.9, 0.4],
            }
        ),
    )
    _write_csv(
        raw_dir / "test.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2"],
                "fraud_label": [0, 1],
                "batch": ["b1", "b1"],
                "month": [3, 3],
                "customer_age": ["young", "old"],
                "model_score": [0.8, 0.3],
            }
        ),
    )
    _write_csv(
        raw_dir / "historical_expert_predictions.csv",
        pd.DataFrame(
            {
                "case_id": ["t1", "t2", "v1", "v2"],
                "expert_a": [0, 1, 0, 1],
                "expert_b": [1, 1, 0, 0],
            }
        ),
    )
    _write_csv(
        raw_dir / "expert_predictions.csv",
        pd.DataFrame(
            {
                "case_id": ["s1", "s2"],
                "expert_a": [0, 1],
                "expert_b": [1, 0],
            }
        ),
    )

    split_rows = {
        "train": [("t1", 0), ("t2", 1)],
        "val": [("v1", 0), ("v2", 1)],
        "test": [("s1", 0), ("s2", 1)],
    }
    for split_name, rows in split_rows.items():
        ids = [row[0] for row in rows]
        labels = [row[1] for row in rows]
        _write_csv(processed_dir / f"X_{split_name}.csv", pd.DataFrame({"f1": [0.0, 1.0][: len(rows)]}))
        _write_csv(processed_dir / f"y_{split_name}.csv", pd.DataFrame({"fraud_label": labels}))
        metadata_name = f"{split_name}_metadata.csv" if split_name != "test" else "test_metadata.csv"
        _write_csv(processed_dir / metadata_name, pd.DataFrame({"case_id": ids, "batch": ["b1", "b1"][: len(rows)], "month": [1, 1][: len(rows)], "customer_age": ["young", "old"][: len(rows)]}))

        _write_csv(
            model_dir / f"{split_name}_predictions.csv",
            pd.DataFrame(
                {
                    "case_id": ids,
                    "y_true": labels,
                    "ai_score": [0.9, 0.4][: len(rows)],
                    "ai_pred": [0, 0][: len(rows)],
                    "split": [split_name] * len(rows),
                }
            ),
        )

    base = pd.DataFrame(
        {
            "case_id": ["t1", "t2", "v1", "v2", "s1", "s2"],
            "split": ["train", "train", "val", "val", "test", "test"],
            "numerical_confidence": [0.9, 0.6, 0.9, 0.6, 0.8, 0.7],
            "distance_uncertainty": [0.1, 0.8, 0.2, 0.7, 0.25, 0.75],
            "calibration_risk": [0.1, 0.7, 0.2, 0.6, 0.2, 0.7],
            "neighbor_error_rate": [0.1, 0.9, 0.2, 0.8, 0.2, 0.85],
            "wrong_confident_risk": [0.1, 0.85, 0.2, 0.8, 0.25, 0.9],
            "magd_assurance_risk": [0.1, 0.85, 0.2, 0.75, 0.2, 0.8],
        }
    )
    _write_csv(
        assurance_dir / "numerical_confidence.csv",
        base[["case_id", "split"]].assign(ai_score=[0.9, 0.4, 0.9, 0.4, 0.8, 0.3], ai_pred=[0, 0, 0, 0, 0, 0], numerical_confidence=base["numerical_confidence"]),
    )
    _write_csv(
        assurance_dir / "distance_uncertainty.csv",
        base[["case_id", "split"]].assign(y_true=[0, 1, 0, 1, 0, 1], ai_score=[0.9, 0.4, 0.9, 0.4, 0.8, 0.3], ai_pred=[0, 0, 0, 0, 0, 0], distance_uncertainty=base["distance_uncertainty"]),
    )
    _write_csv(assurance_dir / "calibration_risk.csv", base[["case_id", "split", "calibration_risk"]])
    _write_csv(assurance_dir / "local_reliability.csv", base[["case_id", "split", "neighbor_error_rate"]])
    _write_csv(assurance_dir / "wrong_confident_risk.csv", base[["case_id", "split", "wrong_confident_risk"]])
    _write_csv(assurance_dir / "magd_risk.csv", base[["case_id", "split", "magd_assurance_risk"]])
    return config_path


def test_test_labels_not_used_to_create_test_routes(tmp_path: Path) -> None:
    config_path = _setup_fixture_repo(tmp_path)
    first = run_learning_to_defer_baseline(config_path)
    first_routes = first.decisions[["case_id", "selected_route", "selected_expert", "final_prediction"]].copy()

    model_dir = config_path.parent / "data" / "outputs" / "model"
    test_predictions = pd.read_csv(model_dir / "test_predictions.csv")
    test_predictions["y_true"] = [1, 0]
    test_predictions.to_csv(model_dir / "test_predictions.csv", index=False)

    second = run_learning_to_defer_baseline(config_path)
    second_routes = second.decisions[["case_id", "selected_route", "selected_expert", "final_prediction"]].copy()
    pd.testing.assert_frame_equal(first_routes, second_routes)


def test_decisions_include_selected_route_and_final_prediction(tmp_path: Path) -> None:
    config_path = _setup_fixture_repo(tmp_path)
    artifacts = run_learning_to_defer_baseline(config_path)
    assert {"selected_route", "selected_expert", "final_prediction", "decision_reason"}.issubset(artifacts.decisions.columns)
    assert len(artifacts.decisions) == 2


def test_metrics_generated(tmp_path: Path) -> None:
    config_path = _setup_fixture_repo(tmp_path)
    artifacts = run_learning_to_defer_baseline(config_path)
    assert not artifacts.metrics.empty
    assert (artifacts.output_dir / "learning_to_defer_metrics.csv").exists()


def test_standard_feature_set_excludes_magd_assurance_risk() -> None:
    assert "magd_assurance_risk" not in STANDARD_FEATURE_COLUMNS
    assert "magd_assurance_risk" in AUGMENTED_FEATURE_COLUMNS
    assert set(STANDARD_FEATURE_COLUMNS) == set(AUGMENTED_FEATURE_COLUMNS) - {"magd_assurance_risk"}


def test_standard_and_augmented_variants_write_distinct_labeled_outputs(tmp_path: Path) -> None:
    config_path = _setup_fixture_repo(tmp_path)
    standard = run_learning_to_defer_baseline(config_path, feature_set="standard")
    augmented = run_learning_to_defer_baseline(config_path, feature_set="augmented")

    # L2D-Standard is now the official baseline and owns the canonical, unsuffixed
    # filenames; L2D+MAGD (augmented) is the explicitly-labeled optional experiment.
    assert (standard.output_dir / "learning_to_defer_decisions.csv").exists()
    assert (standard.output_dir / "learning_to_defer_metrics.csv").exists()
    assert (augmented.output_dir / "learning_to_defer_augmented_decisions.csv").exists()
    assert (augmented.output_dir / "learning_to_defer_augmented_metrics.csv").exists()

    assert standard.decisions["method"].eq("learning_to_defer").all()
    assert augmented.decisions["method"].eq("learning_to_defer_augmented").all()
    assert standard.decisions["feature_set"].eq("standard").all()
    assert augmented.decisions["feature_set"].eq("augmented").all()
    assert "magd_assurance_risk" not in standard.metrics["feature_columns"].iloc[0]
    assert "magd_assurance_risk" in augmented.metrics["feature_columns"].iloc[0]


def test_default_feature_set_is_standard() -> None:
    """The default (no feature_set passed) must be L2D-Standard - it is the official
    independent baseline used by final paper scripts."""
    import inspect

    sig = inspect.signature(run_learning_to_defer_baseline)
    assert sig.parameters["feature_set"].default == "standard"


def test_train_labels_used_only_for_training_not_test_routing(tmp_path: Path) -> None:
    config_path = _setup_fixture_repo(tmp_path)
    first = run_learning_to_defer_baseline(config_path)

    model_dir = config_path.parent / "data" / "outputs" / "model"
    train_predictions = pd.read_csv(model_dir / "train_predictions.csv")
    train_predictions["y_true"] = [1, 0]
    train_predictions.to_csv(model_dir / "train_predictions.csv", index=False)

    second = run_learning_to_defer_baseline(config_path)
    assert len(first.decisions) == len(second.decisions) == 2
