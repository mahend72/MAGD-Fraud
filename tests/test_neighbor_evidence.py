from __future__ import annotations

import pandas as pd

from src.assurance.explanation_neighbors import build_neighbor_evidence
from src.models.embeddings import EmbeddingArtifacts


def _embedding_artifacts() -> EmbeddingArtifacts:
    train_embeddings = pd.DataFrame(
        {
            "embedding_0": [0.0, 0.2, 1.0, 1.2],
            "embedding_1": [0.0, 0.1, 1.0, 1.1],
        }
    )
    test_embeddings = pd.DataFrame(
        {
            "embedding_0": [0.1, 1.1],
            "embedding_1": [0.1, 1.0],
        }
    )
    train_metadata = pd.DataFrame({"case_id": ["c1", "c2", "c3", "c4"]})
    test_metadata = pd.DataFrame({"case_id": ["t1", "t2"]})
    return EmbeddingArtifacts(
        method="pca",
        n_components=2,
        feature_names=["f1", "f2"],
        train_embeddings=train_embeddings,
        val_embeddings=train_embeddings.iloc[:1].reset_index(drop=True),
        test_embeddings=test_embeddings,
        train_labels=pd.Series([0, 0, 1, 1]),
        val_labels=pd.Series([0]),
        test_labels=pd.Series([0, 1]),
        train_metadata=train_metadata,
        val_metadata=train_metadata.iloc[:1].reset_index(drop=True),
        test_metadata=test_metadata,
    )


def _train_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3", "c4"],
            "y_true": [0, 0, 1, 1],
            "ai_pred": [0, 1, 1, 1],
            "ai_score": [0.1, 0.6, 0.9, 0.8],
        }
    )


def test_each_test_case_has_k_neighbors() -> None:
    long_frame, summary = build_neighbor_evidence(
        embeddings=_embedding_artifacts(),
        train_predictions=_train_predictions(),
        top_k=2,
    )
    counts = long_frame.groupby("case_id")["neighbor_case_id"].count()
    assert counts.tolist() == [2, 2]
    assert (summary["top_k"] == 2).all()


def test_neighbor_error_rate_between_zero_and_one() -> None:
    _, summary = build_neighbor_evidence(
        embeddings=_embedding_artifacts(),
        train_predictions=_train_predictions(),
        top_k=3,
    )
    assert summary["neighbor_error_rate"].between(0.0, 1.0).all()
