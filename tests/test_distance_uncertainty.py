from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.assurance.distance_uncertainty import compute_distance_confidence_frame
from src.models.embeddings import EmbeddingArtifacts, load_embedding_artifacts


def _embedding_artifacts() -> EmbeddingArtifacts:
    train_embeddings = pd.DataFrame(
        {
            "embedding_0": [0.0, 0.2, 1.0, 1.2],
            "embedding_1": [0.0, 0.1, 1.0, 1.1],
        }
    )
    val_embeddings = pd.DataFrame(
        {
            "embedding_0": [0.1, 1.1],
            "embedding_1": [0.0, 1.0],
        }
    )
    test_embeddings = pd.DataFrame(
        {
            "embedding_0": [0.1, 1.1, 0.6],
            "embedding_1": [0.1, 1.0, 0.5],
        }
    )
    train_metadata = pd.DataFrame({"case_id": ["c1", "c2", "c3", "c4"]})
    val_metadata = pd.DataFrame({"case_id": ["v1", "v2"]})
    test_metadata = pd.DataFrame({"case_id": ["t1", "t2", "t3"]})
    return EmbeddingArtifacts(
        method="pca",
        n_components=2,
        feature_names=["f1", "f2"],
        train_embeddings=train_embeddings,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        train_labels=pd.Series([0, 0, 1, 1]),
        val_labels=pd.Series([0, 1]),
        test_labels=pd.Series([0, 1, 1]),
        train_metadata=train_metadata,
        val_metadata=val_metadata,
        test_metadata=test_metadata,
    )


def _prediction_frames() -> dict[str, pd.DataFrame]:
    return {
        "train": pd.DataFrame(
            {
                "case_id": ["c1", "c2", "c3", "c4"],
                "y_true": [0, 0, 1, 1],
                "ai_score": [0.10, 0.20, 0.85, 0.90],
                "ai_pred": [0, 0, 1, 1],
                "split": ["train"] * 4,
            }
        ),
        "val": pd.DataFrame(
            {
                "case_id": ["v1", "v2"],
                "y_true": [0, 1],
                "ai_score": [0.15, 0.88],
                "ai_pred": [0, 1],
                "split": ["val"] * 2,
            }
        ),
        "test": pd.DataFrame(
            {
                "case_id": ["t1", "t2", "t3"],
                "y_true": [0, 1, 1],
                "ai_score": [0.12, 0.92, 0.70],
                "ai_pred": [0, 1, 1],
                "split": ["test"] * 3,
            }
        ),
    }


def test_distance_confidence_between_zero_and_one() -> None:
    frame = compute_distance_confidence_frame(_embedding_artifacts(), _prediction_frames())
    assert frame["distance_confidence"].between(0.0, 1.0).all()


def test_distance_uncertainty_between_zero_and_one() -> None:
    frame = compute_distance_confidence_frame(_embedding_artifacts(), _prediction_frames())
    assert frame["distance_uncertainty"].between(0.0, 1.0).all()


def test_distance_uncertainty_is_one_minus_confidence() -> None:
    frame = compute_distance_confidence_frame(_embedding_artifacts(), _prediction_frames())
    difference = (frame["distance_uncertainty"] - (1.0 - frame["distance_confidence"])).abs()
    assert (difference < 1e-12).all()


def test_distance_output_has_one_row_per_case() -> None:
    frames = _prediction_frames()
    frame = compute_distance_confidence_frame(_embedding_artifacts(), frames)
    expected_cases = sum(len(split_frame) for split_frame in frames.values())
    assert len(frame) == expected_cases
    assert frame["case_id"].is_unique


def test_test_labels_are_not_used_in_distance_computation() -> None:
    embeddings = _embedding_artifacts()
    base_predictions = _prediction_frames()
    changed_predictions = _prediction_frames()
    changed_predictions["test"]["y_true"] = [1, 0, 0]

    base = compute_distance_confidence_frame(embeddings, base_predictions)
    changed = compute_distance_confidence_frame(embeddings, changed_predictions)

    columns = ["case_id", "split", "distance_to_predicted_centroid", "distance_confidence", "distance_uncertainty"]
    pd.testing.assert_frame_equal(base[columns], changed[columns])


def test_pca_is_fit_using_training_data_only(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"f1": [0.0, 2.0], "f2": [0.0, 0.0]}).to_csv(processed_dir / "X_train.csv", index=False)
    pd.DataFrame({"fraud_label": [0, 1]}).to_csv(processed_dir / "y_train.csv", index=False)
    pd.DataFrame({"case_id": ["c1", "c2"]}).to_csv(processed_dir / "train_metadata.csv", index=False)

    pd.DataFrame({"f1": [10.0], "f2": [10.0]}).to_csv(processed_dir / "X_val.csv", index=False)
    pd.DataFrame({"fraud_label": [0]}).to_csv(processed_dir / "y_val.csv", index=False)
    pd.DataFrame({"case_id": ["v1"]}).to_csv(processed_dir / "val_metadata.csv", index=False)

    pd.DataFrame({"f1": [20.0], "f2": [20.0]}).to_csv(processed_dir / "X_test.csv", index=False)
    pd.DataFrame({"fraud_label": [1]}).to_csv(processed_dir / "y_test.csv", index=False)
    pd.DataFrame({"case_id": ["t1"]}).to_csv(processed_dir / "test_metadata.csv", index=False)

    config_path = tmp_path / "config.yaml"
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
                "  embedding_method: pca",
                "  pca_components: 1",
            ]
        ),
        encoding="utf-8",
    )

    artifacts = load_embedding_artifacts(config_path)
    assert artifacts.n_components == 1
    assert float(artifacts.train_embeddings.iloc[0, 0]) == -1.0
    assert float(artifacts.train_embeddings.iloc[1, 0]) == 1.0
    assert float(artifacts.val_embeddings.iloc[0, 0]) == 9.0
    assert float(artifacts.test_embeddings.iloc[0, 0]) == 19.0
