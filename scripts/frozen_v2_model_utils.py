"""Shared utilities for Figures 3 and 4: reconstructing the FROZEN MAGD-Fraud v2
assurance model (fold-local calibration mapping + ECDF normalization + the
7-feature L2 logistic regression, all fit once on validation only) so its
fitted response surface can be visualised. This module fits nothing new that
isn't already part of the documented, frozen v2 pipeline in
src/assurance/magd_v2.py -- it calls that module's own
fit_final_v2_pipeline/apply_final_v2_pipeline unmodified, and verifies the
result reproduces the ALREADY-frozen `magd_assurance_risk` column in
data/outputs/final_reproducible_run/magd_v2_test_decisions.csv to floating-point
precision before any plot is allowed to use it. Test is used only for this
read-only verification step, never for fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assurance.magd_v2 import (
    apply_final_v2_pipeline,
    apply_fold_artifacts,
    build_logistic_features,
    compute_wrong_confident_risk_v2,
    fit_final_v2_pipeline,
)
from src.evaluation.ablation_utils import outputs_root_for_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (ROOT / "config.yaml").resolve()
FROZEN_TEST_DECISIONS = ROOT / "data/outputs/final_reproducible_run/magd_v2_test_decisions.csv"
MAX_ALLOWED_RECONSTRUCTION_DIFF = 1e-8  # far tighter than float roundoff would ever require


@dataclass
class FrozenV2Model:
    artifacts: object  # src.assurance.magd_v2.FinalV2Artifacts
    val_scored: pd.DataFrame  # validation, fold-local-scored (fit_final_v2_pipeline's own fit split)
    medians: dict[str, float]
    max_reconstruction_diff: float


def _build_core(split_name: str) -> pd.DataFrame:
    outputs_root = outputs_root_for_config(CONFIG_PATH)
    assurance_dir = outputs_root / "assurance"
    model_dir = outputs_root / "model"

    pred = pd.read_csv(model_dir / f"{split_name}_predictions.csv")
    dist = pd.read_csv(assurance_dir / "distance_uncertainty.csv")
    dist = dist.loc[dist["split"] == split_name, ["case_id", "split", "distance_uncertainty"]]
    local = pd.read_csv(assurance_dir / "local_reliability.csv")
    local = local.loc[local["split"] == split_name, ["case_id", "split", "neighbor_error_rate"]]
    numconf = pd.read_csv(assurance_dir / "numerical_confidence.csv")
    numconf = numconf.loc[numconf["split"] == split_name, ["case_id", "split", "numerical_confidence"]]

    core = (
        pred[["case_id", "split", "y_true", "ai_score", "ai_pred"]]
        .merge(dist, on=["case_id", "split"], validate="one_to_one")
        .merge(local, on=["case_id", "split"], validate="one_to_one")
        .merge(numconf, on=["case_id", "split"], validate="one_to_one")
    )
    core["case_id"] = core["case_id"].astype(str)
    return core.reset_index(drop=True)


def load_frozen_v2_model() -> FrozenV2Model:
    val_core = _build_core("val")
    test_core = _build_core("test")

    artifacts = fit_final_v2_pipeline(val_core)

    # Verification: apply to test and compare against the already-frozen column.
    scored_test = apply_final_v2_pipeline(test_core, artifacts)
    frozen = pd.read_csv(FROZEN_TEST_DECISIONS)
    frozen["case_id"] = frozen["case_id"].astype(str)
    merged = scored_test[["case_id", "magd_assurance_risk"]].merge(
        frozen[["case_id", "magd_assurance_risk"]], on="case_id", suffixes=("_reconstructed", "_frozen")
    )
    if len(merged) != len(test_core):
        raise AssertionError("Reconstruction verification join dropped rows -- case_id mismatch with frozen file.")
    max_diff = float((merged["magd_assurance_risk_reconstructed"] - merged["magd_assurance_risk_frozen"]).abs().max())
    if max_diff > MAX_ALLOWED_RECONSTRUCTION_DIFF:
        raise AssertionError(
            f"Reconstructed frozen v2 model does not reproduce magd_v2_test_decisions.csv "
            f"(max abs diff {max_diff:.3e} > {MAX_ALLOWED_RECONSTRUCTION_DIFF:.3e}). Refusing to plot from an "
            "unverified model."
        )

    val_scored = apply_fold_artifacts(
        val_core, artifacts.fold_artifacts, beta_calibration=artifacts.beta_calibration, beta_neighbor=artifacts.beta_neighbor
    )
    medians = {
        "distance_uncertainty_norm": float(val_scored["distance_uncertainty_norm"].median()),
        "calibration_risk_norm": float(val_scored["calibration_risk_norm"].median()),
        "neighbor_error_rate_norm": float(val_scored["neighbor_error_rate_norm"].median()),
        "numerical_confidence": float(val_scored["numerical_confidence"].median()),
    }
    return FrozenV2Model(artifacts=artifacts, val_scored=val_scored, medians=medians, max_reconstruction_diff=max_diff)


def predict_risk_grid(
    frozen: FrozenV2Model,
    *,
    calibration_risk_norm,
    distance_uncertainty_norm,
    neighbor_error_rate_norm,
    numerical_confidence,
    beta_calibration: float = 0.5,
    beta_neighbor: float = 0.5,
) -> np.ndarray:
    """Evaluates the frozen 7-feature logistic model at arbitrary (broadcastable)
    arrays of the four raw assurance-signal inputs, using the exact frozen
    feature-construction formula (build_logistic_features / compute_wrong_confident_risk_v2)
    and the exact frozen, already-fit LogisticRegression object. Does not refit
    anything."""
    cal = np.asarray(calibration_risk_norm, dtype=float)
    dist = np.broadcast_to(np.asarray(distance_uncertainty_norm, dtype=float), cal.shape)
    nbr = np.broadcast_to(np.asarray(neighbor_error_rate_norm, dtype=float), cal.shape)
    conf = np.broadcast_to(np.asarray(numerical_confidence, dtype=float), cal.shape)

    grid_frame = pd.DataFrame(
        {
            "calibration_risk_norm": cal.ravel(),
            "neighbor_error_rate_norm": nbr.ravel(),
            "numerical_confidence": conf.ravel(),
        }
    )
    wc_v2 = compute_wrong_confident_risk_v2(
        grid_frame, beta_calibration=beta_calibration, beta_neighbor=beta_neighbor
    ).to_numpy().reshape(cal.shape)

    features = np.column_stack(
        [
            dist.ravel(),
            cal.ravel(),
            nbr.ravel(),
            wc_v2.ravel(),
            (cal * dist).ravel(),
            (cal * nbr).ravel(),
            (conf * cal).ravel(),
        ]
    )
    risk = frozen.artifacts.model.predict_proba(features)[:, 1]
    return risk.reshape(cal.shape)
