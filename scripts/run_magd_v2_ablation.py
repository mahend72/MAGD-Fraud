"""Reproduces the manuscript's MAGD-Fraud assurance-scorer ablation table.

Evaluates four nested cross-validated logistic-regression variants over the
validation split only:

  1. calibration risk alone
  2. normalized distance + calibration + neighbourhood main effects
  3. those three base risks + the revised, confidence-gated wrong-confident risk
  4. the final seven-feature model with the three pre-specified interactions
     (the frozen MAGD-Fraud v2 assurance scorer)

Development-only: uses the same five folds, fold-local calibration mapping, and
fold-local ECDF normalization as the frozen MAGD-Fraud v2 pipeline
(src/assurance/magd_v2.py), by calling that module's unmodified functions
directly rather than reimplementing them. Test is never loaded. Nothing in
src/assurance/magd_v2.py or any other frozen MAGD-Fraud module is modified,
refit outside its existing fold-local contract, or evaluated on test.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assurance.magd_v2 import (  # noqa: E402  (frozen module, imported not modified)
    LOGISTIC_MODEL_PARAMS,
    _stratification_labels,
    apply_fold_artifacts,
    build_logistic_features,
    fit_fold_artifacts,
)
from src.evaluation.ablation_utils import outputs_root_for_config  # noqa: E402

N_SPLITS = 5
SEED = 42

ROW_ORDER = [
    "calibration_risk_alone",
    "distance_calibration_neighbourhood_main_effects",
    "base_risks_plus_revised_wrong_confident",
    "full_interaction_model",
]

ROW_LABELS = {
    "calibration_risk_alone": "Calibration risk only",
    "distance_calibration_neighbourhood_main_effects": "Normalized distance + calibration + neighbourhood main effects",
    "base_risks_plus_revised_wrong_confident": "Base risks + revised gated wrong-confident risk",
    "full_interaction_model": "Full interaction model (frozen MAGD-Fraud v2 scorer)",
}

# Feature columns for the three main-effects-only rows (row 4 uses
# build_logistic_features instead, since it also adds the three pre-specified
# interaction terms).
ROW_FEATURES = {
    "calibration_risk_alone": ["calibration_risk_norm"],
    "distance_calibration_neighbourhood_main_effects": [
        "distance_uncertainty_norm",
        "calibration_risk_norm",
        "neighbor_error_rate_norm",
    ],
    "base_risks_plus_revised_wrong_confident": [
        "distance_uncertainty_norm",
        "calibration_risk_norm",
        "neighbor_error_rate_norm",
        "wrong_confident_risk_v2",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the manuscript MAGD-Fraud v2 assurance-scorer ablation (validation-only).")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def _load_validation_core(assurance_dir: Path, model_dir: Path) -> pd.DataFrame:
    """Builds the validation-only development core. Never reads a test file."""
    val_predictions = pd.read_csv(model_dir / "val_predictions.csv")
    if not (val_predictions["split"] == "val").all():
        raise ValueError("val_predictions.csv contains rows outside the val split.")

    distance = pd.read_csv(assurance_dir / "distance_uncertainty.csv")
    distance = distance.loc[distance["split"] == "val", ["case_id", "split", "distance_uncertainty"]]

    local_reliability = pd.read_csv(assurance_dir / "local_reliability.csv")
    local_reliability = local_reliability.loc[local_reliability["split"] == "val", ["case_id", "split", "neighbor_error_rate"]]

    numerical_confidence = pd.read_csv(assurance_dir / "numerical_confidence.csv")
    numerical_confidence = numerical_confidence.loc[
        numerical_confidence["split"] == "val", ["case_id", "split", "numerical_confidence"]
    ]

    core = (
        val_predictions[["case_id", "split", "y_true", "ai_score", "ai_pred"]]
        .merge(distance, on=["case_id", "split"], how="inner", validate="one_to_one")
        .merge(local_reliability, on=["case_id", "split"], how="inner", validate="one_to_one")
        .merge(numerical_confidence, on=["case_id", "split"], how="inner", validate="one_to_one")
    )
    if core["split"].ne("val").any():
        raise ValueError("Development core must contain only the val split.")
    core["case_id"] = core["case_id"].astype(str)
    core["ai_wrong"] = (core["ai_pred"].astype(int) != core["y_true"].astype(int)).astype(int)
    return core.reset_index(drop=True)


def _fit_eval(X_inner: np.ndarray, y_inner: np.ndarray, X_held: np.ndarray, y_held: np.ndarray) -> tuple[float, float]:
    model = LogisticRegression(**LOGISTIC_MODEL_PARAMS)
    model.fit(X_inner, y_inner)
    scores = model.predict_proba(X_held)[:, 1]
    return float(roc_auc_score(y_held, scores)), float(average_precision_score(y_held, scores))


def run_ablation(core: pd.DataFrame) -> pd.DataFrame:
    strat_labels = _stratification_labels(core, n_splits=N_SPLITS)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Fold-local calibration mapping + ECDF normalization, fit on inner-train only
    # and applied (never refit) to held-out -- identical contract to the frozen
    # MAGD-Fraud v2 pipeline. Computed once per fold, reused across all four rows.
    fold_frames: list[dict[str, object]] = []
    for fold_idx, (inner_idx, held_idx) in enumerate(skf.split(core, strat_labels)):
        inner = core.iloc[inner_idx].reset_index(drop=True)
        held = core.iloc[held_idx].reset_index(drop=True)
        artifacts = fit_fold_artifacts(inner)
        fold_frames.append(
            {
                "fold": fold_idx,
                "inner_scored": apply_fold_artifacts(inner, artifacts),
                "held_scored": apply_fold_artifacts(held, artifacts),
                "n_inner_train": len(inner),
                "n_held_out": len(held),
            }
        )

    records: list[dict[str, object]] = []
    for row_name in ROW_ORDER:
        fold_aurocs, fold_pr_aucs = [], []
        for fold in fold_frames:
            inner_scored, held_scored = fold["inner_scored"], fold["held_scored"]
            y_inner = inner_scored["ai_wrong"].to_numpy()
            y_held = held_scored["ai_wrong"].to_numpy()

            if row_name == "full_interaction_model":
                X_inner = build_logistic_features(inner_scored)
                X_held = build_logistic_features(held_scored)
            else:
                cols = ROW_FEATURES[row_name]
                X_inner = inner_scored[cols].astype(float).to_numpy()
                X_held = held_scored[cols].astype(float).to_numpy()

            auroc, pr_auc = _fit_eval(X_inner, y_inner, X_held, y_held)
            fold_aurocs.append(auroc)
            fold_pr_aucs.append(pr_auc)
            records.append(
                {
                    "variant": row_name,
                    "assurance_representation": ROW_LABELS[row_name],
                    "fold": fold["fold"],
                    "n_inner_train": fold["n_inner_train"],
                    "n_held_out": fold["n_held_out"],
                    "auroc": auroc,
                    "pr_auc": pr_auc,
                }
            )
        records.append(
            {
                "variant": row_name,
                "assurance_representation": ROW_LABELS[row_name],
                "fold": "mean",
                "n_inner_train": np.nan,
                "n_held_out": np.nan,
                "auroc": float(np.mean(fold_aurocs)),
                "pr_auc": float(np.mean(fold_pr_aucs)),
            }
        )

    return pd.DataFrame.from_records(records)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    outputs_root = outputs_root_for_config(config_path)
    assurance_dir = outputs_root / "assurance"
    model_dir = outputs_root / "model"
    paper_tables_dir = outputs_root / "paper_tables"
    paper_tables_dir.mkdir(parents=True, exist_ok=True)

    core = _load_validation_core(assurance_dir, model_dir)
    results = run_ablation(core)

    out_path = paper_tables_dir / "magd_v2_scorer_ablation.csv"
    results.to_csv(out_path, index=False)

    summary = results[results["fold"] == "mean"]
    md_lines = [
        "| Assurance representation | AUROC | PR-AUC |",
        "| --- | --- | --- |",
    ]
    for row_name in ROW_ORDER:
        row = summary[summary["variant"] == row_name].iloc[0]
        md_lines.append(f"| {ROW_LABELS[row_name]} | {row['auroc']:.4f} | {row['pr_auc']:.4f} |")
    md_path = paper_tables_dir / "magd_v2_scorer_ablation.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"Saved per-fold + mean results to: {out_path}")
    print(f"Saved manuscript-ready table to: {md_path}")
    print()
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
