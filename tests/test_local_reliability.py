from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from src.assurance.local_reliability import build_local_reliability_for_queries, run_local_reliability


def _train_embeddings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3", "c4"],
            "split": ["train"] * 4,
            "embedding_0": [0.0, 0.2, 1.0, 1.2],
            "embedding_1": [0.0, 0.1, 1.0, 1.1],
        }
    )


def _val_embeddings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["v1", "v2"],
            "split": ["val"] * 2,
            "embedding_0": [0.1, 1.1],
            "embedding_1": [0.1, 1.0],
        }
    )


def _test_embeddings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["t1", "t2"],
            "split": ["test"] * 2,
            "embedding_0": [0.15, 1.15],
            "embedding_1": [0.05, 1.05],
        }
    )


def _train_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3", "c4"],
            "y_true": [0, 0, 1, 1],
            "ai_pred": [0, 1, 1, 1],
        }
    )


def _query_predictions(case_ids: list[str], ai_pred: list[int], y_true: list[int] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({"case_id": case_ids, "ai_pred": ai_pred})
    if y_true is not None:
        frame["y_true"] = y_true
    return frame


def test_local_reliability_values_in_valid_ranges() -> None:
    summary = build_local_reliability_for_queries(
        train_embeddings=_train_embeddings(),
        train_predictions=_train_predictions(),
        query_embeddings=_test_embeddings(),
        query_predictions=_query_predictions(["t1", "t2"], [0, 1]),
        split_name="test",
        top_k=3,
    )
    for column in ["neighbor_error_rate", "neighbor_fraud_rate", "neighbor_ai_agreement"]:
        assert summary[column].between(0.0, 1.0).all()
    assert (summary["mean_neighbor_distance"] >= 0.0).all()


def test_one_row_per_val_and_test_case() -> None:
    val_summary = build_local_reliability_for_queries(
        train_embeddings=_train_embeddings(),
        train_predictions=_train_predictions(),
        query_embeddings=_val_embeddings(),
        query_predictions=_query_predictions(["v1", "v2"], [0, 1]),
        split_name="val",
        top_k=2,
    )
    test_summary = build_local_reliability_for_queries(
        train_embeddings=_train_embeddings(),
        train_predictions=_train_predictions(),
        query_embeddings=_test_embeddings(),
        query_predictions=_query_predictions(["t1", "t2"], [0, 1]),
        split_name="test",
        top_k=2,
    )
    combined = pd.concat([val_summary, test_summary], ignore_index=True)
    assert combined["case_id"].tolist() == ["v1", "v2", "t1", "t2"]
    assert combined["split"].tolist() == ["val", "val", "test", "test"]
    assert (combined["knn_k"] == 2).all()


def test_local_reliability_does_not_use_query_labels() -> None:
    summary_a = build_local_reliability_for_queries(
        train_embeddings=_train_embeddings(),
        train_predictions=_train_predictions(),
        query_embeddings=_test_embeddings(),
        query_predictions=_query_predictions(["t1", "t2"], [0, 1], y_true=[0, 1]),
        split_name="test",
        top_k=2,
    )
    summary_b = build_local_reliability_for_queries(
        train_embeddings=_train_embeddings(),
        train_predictions=_train_predictions(),
        query_embeddings=_test_embeddings(),
        query_predictions=_query_predictions(["t1", "t2"], [0, 1], y_true=[1, 0]),
        split_name="test",
        top_k=2,
    )
    assert_frame_equal(summary_a, summary_b)


def test_run_local_reliability_reads_k_from_config(tmp_path: Path) -> None:
    repo_dir = tmp_path
    processed_dir = repo_dir / "data" / "processed"
    outputs_model_dir = repo_dir / "data" / "outputs" / "model"
    outputs_assurance_dir = repo_dir / "data" / "outputs" / "assurance"
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_model_dir.mkdir(parents=True, exist_ok=True)
    outputs_assurance_dir.mkdir(parents=True, exist_ok=True)

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
                "assurance:",
                "  knn_k: 3",
            ]
        ),
        encoding="utf-8",
    )

    _train_embeddings().to_csv(outputs_assurance_dir / "embeddings_train.csv", index=False)
    _val_embeddings().to_csv(outputs_assurance_dir / "embeddings_val.csv", index=False)
    _test_embeddings().to_csv(outputs_assurance_dir / "embeddings_test.csv", index=False)

    pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3", "c4"],
            "y_true": [0, 0, 1, 1],
            "ai_score": [0.1, 0.6, 0.8, 0.9],
            "ai_pred": [0, 1, 1, 1],
            "split": ["train"] * 4,
        }
    ).to_csv(outputs_model_dir / "train_predictions.csv", index=False)
    pd.DataFrame(
        {
            "case_id": ["v1", "v2"],
            "y_true": [0, 1],
            "ai_score": [0.2, 0.9],
            "ai_pred": [0, 1],
            "split": ["val"] * 2,
        }
    ).to_csv(outputs_model_dir / "val_predictions.csv", index=False)
    pd.DataFrame(
        {
            "case_id": ["t1", "t2"],
            "y_true": [0, 1],
            "ai_score": [0.3, 0.85],
            "ai_pred": [0, 1],
            "split": ["test"] * 2,
        }
    ).to_csv(outputs_model_dir / "test_predictions.csv", index=False)

    artifacts = run_local_reliability(config_path)
    assert len(artifacts.local_reliability) == 4
    assert (artifacts.local_reliability["knn_k"] == 3).all()
