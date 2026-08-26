"""Diagnoses the AUROC~0.8617 result observed when the ORIGINAL (v1) MAGD-Additive
wrong-confident score is added as a fourth feature to the base-risks additive
logistic model (src/assurance/magd_v2.py's row-3-style ablation, but with v1's
`wrong_confident_risk` from src/assurance/wrong_confident_detector.py in place of
the revised, confidence-gated `wrong_confident_risk_v2`).

This is a DIAGNOSTIC, not a manuscript ablation. It is validation-only (never
touches test) and does not modify, refit outside its documented contract, or
otherwise alter any frozen MAGD-Fraud module (magd_v2.py, wrong_confident_detector.py,
calibration.py, magd_risk.py). Every score it evaluates is produced by calling those
modules' existing public functions, with different (still valid) arguments where the
functions already expose a parameter for that purpose (e.g. `weights` for
compute_deployable_wrong_confident_risk).

Checks performed, in order:
  D0. Reproduce the ~0.8617 AUROC / ~0.1449 PR-AUC result.
  D1/D2. Standalone AUROC/PR-AUC of the two components of wrong_confident_risk that
         are NOT present anywhere in the revised four-signal representation:
         numerical_confidence and confidence_disagreement.
  D3. Correlation of the v1 score against every feature in the final model.
  D4. Whether the `calibration_risk` column feeding wrong_confident_risk was fit
      using data that includes the held-out fold (global fit) vs the frozen
      pipeline's fold-local fit -- same held-out rows, same folds, both versions.
  D5. Whether fold-local vs whole-validation ECDF normalization of the v1 score
      itself materially changes the result.
  D6. Whether the AUROC gain survives once every constituent (in particular
      calibration_risk) is reconstructed strictly fold-locally.
  D7. Whether zeroing wrong_confident_risk's distance_uncertainty weight (removing
      the duplication with the standalone distance feature) alone explains the gain.
  D8. Isolated contribution of the two "extra" predictors (numerical_confidence,
      confidence_disagreement) with no overlap against distance/calibration/neighbor.

Reports a final classification: leakage / non-comparable extra information /
duplication effect / genuinely stronger leakage-safe representation -- backed by
the numbers above, without discarding or softening the 0.8617 finding.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
    fit_ecdf,
    fit_fold_artifacts,
)
from src.assurance.wrong_confident_detector import (  # noqa: E402  (frozen module)
    _wrong_confident_config,
    compute_deployable_wrong_confident_risk,
)
from src.evaluation.ablation_utils import outputs_root_for_config  # noqa: E402
from src.utils.io import load_yaml  # noqa: E402

N_SPLITS = 5
SEED = 42
BASE_COLS = ["distance_uncertainty_norm", "calibration_risk_norm", "neighbor_error_rate_norm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose the v1 wrong-confident-score AUROC anomaly (validation-only).")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def _load_core(assurance_dir: Path, model_dir: Path) -> pd.DataFrame:
    val_predictions = pd.read_csv(model_dir / "val_predictions.csv")
    if not (val_predictions["split"] == "val").all():
        raise ValueError("val_predictions.csv contains rows outside the val split.")

    distance = pd.read_csv(assurance_dir / "distance_uncertainty.csv")
    distance = distance.loc[
        distance["split"] == "val", ["case_id", "split", "distance_confidence", "distance_uncertainty"]
    ]
    local_reliability = pd.read_csv(assurance_dir / "local_reliability.csv")
    local_reliability = local_reliability.loc[local_reliability["split"] == "val", ["case_id", "split", "neighbor_error_rate"]]
    numerical_confidence = pd.read_csv(assurance_dir / "numerical_confidence.csv")
    numerical_confidence = numerical_confidence.loc[
        numerical_confidence["split"] == "val", ["case_id", "split", "numerical_confidence"]
    ]
    # The precomputed, GLOBALLY-fit calibration_risk (fit once on the whole
    # validation split by src/assurance/calibration.py -- see check D4).
    calibration_global = pd.read_csv(assurance_dir / "calibration_risk.csv")
    calibration_global = calibration_global.loc[
        calibration_global["split"] == "val", ["case_id", "split", "calibration_risk"]
    ].rename(columns={"calibration_risk": "calibration_risk_global"})
    # The precomputed, GLOBAL v1 wrong_confident_risk (built from the same
    # globally-fit calibration_risk above).
    wc_v1_global = pd.read_csv(assurance_dir / "wrong_confident_risk.csv")
    wc_v1_global = wc_v1_global.loc[wc_v1_global["split"] == "val", ["case_id", "split", "wrong_confident_risk"]].rename(
        columns={"wrong_confident_risk": "wrong_confident_risk_v1_global"}
    )

    core = (
        val_predictions[["case_id", "split", "y_true", "ai_score", "ai_pred"]]
        .merge(distance, on=["case_id", "split"], how="inner", validate="one_to_one")
        .merge(local_reliability, on=["case_id", "split"], how="inner", validate="one_to_one")
        .merge(numerical_confidence, on=["case_id", "split"], how="inner", validate="one_to_one")
        .merge(calibration_global, on=["case_id", "split"], how="inner", validate="one_to_one")
        .merge(wc_v1_global, on=["case_id", "split"], how="inner", validate="one_to_one")
    )
    core["case_id"] = core["case_id"].astype(str)
    core["ai_wrong"] = (core["ai_pred"].astype(int) != core["y_true"].astype(int)).astype(int)
    return core.reset_index(drop=True)


def _cv_folds(core: pd.DataFrame):
    strat_labels = _stratification_labels(core, n_splits=N_SPLITS)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold_idx, (inner_idx, held_idx) in enumerate(skf.split(core, strat_labels)):
        inner = core.iloc[inner_idx].reset_index(drop=True)
        held = core.iloc[held_idx].reset_index(drop=True)
        artifacts = fit_fold_artifacts(inner)
        inner_scored = apply_fold_artifacts(inner, artifacts)
        held_scored = apply_fold_artifacts(held, artifacts)
        yield fold_idx, inner_scored, held_scored


def _logistic_cv(core: pd.DataFrame, feature_fn) -> pd.DataFrame:
    """feature_fn(inner_scored, held_scored) -> (X_inner, X_held). Fits the fixed
    logistic model per fold and returns per-fold + mean AUROC/PR-AUC."""
    rows = []
    for fold_idx, inner_scored, held_scored in _cv_folds(core):
        y_inner = inner_scored["ai_wrong"].to_numpy()
        y_held = held_scored["ai_wrong"].to_numpy()
        X_inner, X_held = feature_fn(inner_scored, held_scored)
        model = LogisticRegression(**LOGISTIC_MODEL_PARAMS).fit(X_inner, y_inner)
        scores = model.predict_proba(X_held)[:, 1]
        rows.append(
            {"fold": fold_idx, "auroc": roc_auc_score(y_held, scores), "pr_auc": average_precision_score(y_held, scores)}
        )
    df = pd.DataFrame(rows)
    return pd.concat([df, pd.DataFrame([{"fold": "mean", "auroc": df["auroc"].mean(), "pr_auc": df["pr_auc"].mean()}])], ignore_index=True)


def _rank_cv(core: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """No-fit variant: ranks an already-constructed score column directly. Used for
    single raw features that need no logistic wrapper."""
    rows = []
    for fold_idx, _inner_scored, held_scored in _cv_folds(core):
        y_held = held_scored["ai_wrong"].to_numpy()
        scores = held_scored[score_col].astype(float).to_numpy()
        rows.append(
            {"fold": fold_idx, "auroc": roc_auc_score(y_held, scores), "pr_auc": average_precision_score(y_held, scores)}
        )
    df = pd.DataFrame(rows)
    return pd.concat([df, pd.DataFrame([{"fold": "mean", "auroc": df["auroc"].mean(), "pr_auc": df["pr_auc"].mean()}])], ignore_index=True)


def d0_reproduce_baseline(core: pd.DataFrame) -> pd.DataFrame:
    """4-signal additive logistic model using the GLOBAL (leaky) v1 score,
    fold-locally ECDF-normalized -- exactly the setup that produced ~0.8617."""

    def feature_fn(inner_scored, held_scored):
        wc_ecdf = fit_ecdf(inner_scored["wrong_confident_risk_v1_global"])
        inner_wc = wc_ecdf.transform(inner_scored["wrong_confident_risk_v1_global"])
        held_wc = wc_ecdf.transform(held_scored["wrong_confident_risk_v1_global"])
        X_inner = np.column_stack([inner_scored[BASE_COLS].astype(float).to_numpy(), inner_wc])
        X_held = np.column_stack([held_scored[BASE_COLS].astype(float).to_numpy(), held_wc])
        return X_inner, X_held

    return _logistic_cv(core, feature_fn)


def d1_d2_standalone_extra_predictors(core: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "numerical_confidence_alone": _rank_cv(core, "numerical_confidence"),
        "confidence_disagreement_alone": _rank_cv(
            core.assign(confidence_disagreement=(core["numerical_confidence"] - core["distance_confidence"]).abs()),
            "confidence_disagreement",
        ),
    }


def d3_correlations(core: pd.DataFrame) -> pd.DataFrame:
    """Correlations computed on a single fold-local scoring of the FULL validation
    set (fit_fold_artifacts/apply_fold_artifacts on all of val at once) -- purely
    descriptive, not a leave-fold-out evaluation."""
    artifacts = fit_fold_artifacts(core)
    scored = apply_fold_artifacts(core, artifacts)
    scored["confidence_disagreement"] = (scored["numerical_confidence"] - scored["distance_confidence"]).abs()
    target_cols = [
        "distance_uncertainty_norm",
        "calibration_risk_norm",
        "neighbor_error_rate_norm",
        "wrong_confident_risk_v2",
        "numerical_confidence",
        "confidence_disagreement",
    ]
    rows = []
    for col in target_cols:
        pearson = float(np.corrcoef(scored["wrong_confident_risk_v1_global"], scored[col])[0, 1])
        spearman = float(spearmanr(scored["wrong_confident_risk_v1_global"], scored[col]).statistic)
        rows.append({"feature": col, "pearson_r": pearson, "spearman_rho": spearman})
    return pd.DataFrame(rows)


def d4_calibration_leakage(core: pd.DataFrame) -> pd.DataFrame:
    """Same held-out rows, same folds: global (whole-validation-fit) calibration_risk
    vs the frozen pipeline's fold-local calibration_risk_norm, ranked directly."""
    rows = []
    for fold_idx, _inner_scored, held_scored in _cv_folds(core):
        y_held = held_scored["ai_wrong"].to_numpy()
        rows.append(
            {
                "fold": fold_idx,
                "auroc_global_calibration_risk": roc_auc_score(y_held, held_scored["calibration_risk_global"]),
                "auroc_fold_local_calibration_risk_norm": roc_auc_score(y_held, held_scored["calibration_risk_norm"]),
            }
        )
    df = pd.DataFrame(rows)
    mean_row = {"fold": "mean"}
    for col in ["auroc_global_calibration_risk", "auroc_fold_local_calibration_risk_norm"]:
        mean_row[col] = df[col].mean()
    return pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)


def d5_normalization_scope(core: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """v1 score kept GLOBAL (leaky) throughout; only the normalization scope
    (fold-local ECDF vs whole-validation ECDF) varies."""

    def fold_local_norm(inner_scored, held_scored):
        wc_ecdf = fit_ecdf(inner_scored["wrong_confident_risk_v1_global"])
        inner_wc = wc_ecdf.transform(inner_scored["wrong_confident_risk_v1_global"])
        held_wc = wc_ecdf.transform(held_scored["wrong_confident_risk_v1_global"])
        X_inner = np.column_stack([inner_scored[BASE_COLS].astype(float).to_numpy(), inner_wc])
        X_held = np.column_stack([held_scored[BASE_COLS].astype(float).to_numpy(), held_wc])
        return X_inner, X_held

    whole_val_ecdf = fit_ecdf(core["wrong_confident_risk_v1_global"])
    core = core.assign(wc_v1_global_norm_whole_val=whole_val_ecdf.transform(core["wrong_confident_risk_v1_global"]))

    def global_norm(inner_scored, held_scored):
        # inner_scored/held_scored are per-fold slices of `core`; re-attach the
        # whole-validation-normalized column via case_id since apply_fold_artifacts
        # returns copies.
        inner_wc = core.set_index("case_id").loc[inner_scored["case_id"], "wc_v1_global_norm_whole_val"].to_numpy()
        held_wc = core.set_index("case_id").loc[held_scored["case_id"], "wc_v1_global_norm_whole_val"].to_numpy()
        X_inner = np.column_stack([inner_scored[BASE_COLS].astype(float).to_numpy(), inner_wc])
        X_held = np.column_stack([held_scored[BASE_COLS].astype(float).to_numpy(), held_wc])
        return X_inner, X_held

    return {
        "fold_local_ecdf_normalization": _logistic_cv(core, fold_local_norm),
        "whole_validation_ecdf_normalization": _logistic_cv(core, global_norm),
    }


def d6_strict_fold_local_reconstruction(core: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Rebuilds wrong_confident_risk from scratch inside each fold using the frozen
    compute_deployable_wrong_confident_risk, fed the FOLD-LOCAL calibration_risk that
    apply_fold_artifacts already produces (leakage-safe), instead of the precomputed
    global calibration_risk.csv value. `weights` is passed straight through to the
    frozen function -- used unmodified for D6, and overridden for D7/D8."""

    def feature_fn(inner_scored, held_scored):
        inner_wc_frame = compute_deployable_wrong_confident_risk(inner_scored, weights)
        held_wc_frame = compute_deployable_wrong_confident_risk(held_scored, weights)
        wc_ecdf = fit_ecdf(inner_wc_frame["wrong_confident_risk"])
        inner_wc = wc_ecdf.transform(inner_wc_frame["wrong_confident_risk"])
        held_wc = wc_ecdf.transform(held_wc_frame["wrong_confident_risk"])
        X_inner = np.column_stack([inner_scored[BASE_COLS].astype(float).to_numpy(), inner_wc])
        X_held = np.column_stack([held_scored[BASE_COLS].astype(float).to_numpy(), held_wc])
        return X_inner, X_held

    return _logistic_cv(core, feature_fn)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    outputs_root = outputs_root_for_config(config_path)
    assurance_dir = outputs_root / "assurance"
    model_dir = outputs_root / "model"
    ablations_dir = outputs_root / "ablations"
    ablations_dir.mkdir(parents=True, exist_ok=True)

    wc_cfg = _wrong_confident_config(config)
    weights = wc_cfg["weights"]

    print("=" * 88)
    print("D-formula: exact wrong_confident_risk (v1) component formula, from config.yaml")
    print("=" * 88)
    print(
        "wrong_confident_risk = ("
        f"{weights['numerical_confidence']:.2f}*numerical_confidence + "
        f"{weights['distance_uncertainty']:.2f}*distance_uncertainty + "
        f"{weights['calibration_risk']:.2f}*calibration_risk + "
        f"{weights['neighbor_error_rate']:.2f}*neighbor_error_rate + "
        f"{weights['confidence_disagreement']:.2f}*confidence_disagreement"
        ") / total_weight, clipped to [0, 1]"
    )
    print("confidence_disagreement = |numerical_confidence - distance_confidence|")
    print(
        "NOTE: distance_uncertainty appears BOTH as its own top-level component AND "
        "inside wrong_confident_risk -> duplicated evidence by construction."
    )
    print(
        "NOTE: numerical_confidence (largest single weight, 0.25) and "
        "confidence_disagreement (0.15) are NOT present anywhere in the base-risks "
        "or revised four-signal representation."
    )
    print()

    core = _load_core(assurance_dir, model_dir)

    print("D0. Reproducing the baseline result (global v1 score, fold-local ECDF-normalized)")
    d0 = d0_reproduce_baseline(core)
    print(d0.to_string(index=False))
    print()

    print("D1/D2. Standalone AUROC/PR-AUC of the two predictors absent from the revised representation")
    extras = d1_d2_standalone_extra_predictors(core)
    for name, df in extras.items():
        print(f"-- {name} --")
        print(df.to_string(index=False))
    print()

    print("D3. Correlation of the global v1 score against every feature in the final model")
    d3 = d3_correlations(core)
    print(d3.to_string(index=False))
    print()

    print("D4. Calibration leakage check: global (whole-val-fit) vs fold-local calibration_risk, same held-out rows")
    d4 = d4_calibration_leakage(core)
    print(d4.to_string(index=False))
    print()

    print("D5. Normalization scope check: fold-local ECDF vs whole-validation ECDF of the (still-global) v1 score")
    d5 = d5_normalization_scope(core)
    for name, df in d5.items():
        print(f"-- {name} --")
        print(df.to_string(index=False))
    print()

    print("D6. Strict fold-local reconstruction: same formula/weights, but calibration_risk rebuilt per fold (leakage removed)")
    d6 = d6_strict_fold_local_reconstruction(core, weights)
    print(d6.to_string(index=False))
    print()

    print("D7. D6 + distance_uncertainty weight zeroed in the WC formula (removes the duplication with the standalone distance feature)")
    weights_no_distance = dict(weights)
    weights_no_distance["distance_uncertainty"] = 0.0
    d7 = d6_strict_fold_local_reconstruction(core, weights_no_distance)
    print(d7.to_string(index=False))
    print()

    print("D8. Isolated 'extra predictors only' bundle: numerical_confidence + confidence_disagreement, no overlap with distance/calibration/neighbor")
    weights_extra_only = {
        "numerical_confidence": weights["numerical_confidence"],
        "distance_uncertainty": 0.0,
        "calibration_risk": 0.0,
        "neighbor_error_rate": 0.0,
        "confidence_disagreement": weights["confidence_disagreement"],
    }
    d8 = d6_strict_fold_local_reconstruction(core, weights_extra_only)
    print(d8.to_string(index=False))
    print()

    summary = pd.DataFrame(
        [
            {"check": "D0_baseline_global_leaky_wc_v1", "auroc_mean": d0.loc[d0["fold"] == "mean", "auroc"].item(), "pr_auc_mean": d0.loc[d0["fold"] == "mean", "pr_auc"].item()},
            {"check": "D1_numerical_confidence_alone", "auroc_mean": extras["numerical_confidence_alone"].loc[extras["numerical_confidence_alone"]["fold"] == "mean", "auroc"].item(), "pr_auc_mean": extras["numerical_confidence_alone"].loc[extras["numerical_confidence_alone"]["fold"] == "mean", "pr_auc"].item()},
            {"check": "D2_confidence_disagreement_alone", "auroc_mean": extras["confidence_disagreement_alone"].loc[extras["confidence_disagreement_alone"]["fold"] == "mean", "auroc"].item(), "pr_auc_mean": extras["confidence_disagreement_alone"].loc[extras["confidence_disagreement_alone"]["fold"] == "mean", "pr_auc"].item()},
            {"check": "D4_global_calibration_risk_alone", "auroc_mean": d4["auroc_global_calibration_risk"].iloc[-1], "pr_auc_mean": np.nan},
            {"check": "D4_fold_local_calibration_risk_alone", "auroc_mean": d4["auroc_fold_local_calibration_risk_norm"].iloc[-1], "pr_auc_mean": np.nan},
            {"check": "D5_fold_local_ecdf_norm", "auroc_mean": d5["fold_local_ecdf_normalization"].loc[d5["fold_local_ecdf_normalization"]["fold"] == "mean", "auroc"].item(), "pr_auc_mean": d5["fold_local_ecdf_normalization"].loc[d5["fold_local_ecdf_normalization"]["fold"] == "mean", "pr_auc"].item()},
            {"check": "D5_whole_val_ecdf_norm", "auroc_mean": d5["whole_validation_ecdf_normalization"].loc[d5["whole_validation_ecdf_normalization"]["fold"] == "mean", "auroc"].item(), "pr_auc_mean": d5["whole_validation_ecdf_normalization"].loc[d5["whole_validation_ecdf_normalization"]["fold"] == "mean", "pr_auc"].item()},
            {"check": "D6_strict_fold_local_full_formula", "auroc_mean": d6.loc[d6["fold"] == "mean", "auroc"].item(), "pr_auc_mean": d6.loc[d6["fold"] == "mean", "pr_auc"].item()},
            {"check": "D7_strict_fold_local_no_distance_weight", "auroc_mean": d7.loc[d7["fold"] == "mean", "auroc"].item(), "pr_auc_mean": d7.loc[d7["fold"] == "mean", "pr_auc"].item()},
            {"check": "D8_extra_predictors_only_no_overlap", "auroc_mean": d8.loc[d8["fold"] == "mean", "auroc"].item(), "pr_auc_mean": d8.loc[d8["fold"] == "mean", "pr_auc"].item()},
        ]
    )
    print("=" * 88)
    print("Summary")
    print("=" * 88)
    print(summary.to_string(index=False))

    out_path = ablations_dir / "wrong_confident_v1_auroc_diagnostic.csv"
    summary.to_csv(out_path, index=False)
    corr_path = ablations_dir / "wrong_confident_v1_auroc_diagnostic_correlations.csv"
    d3.to_csv(corr_path, index=False)
    print()
    print(f"Saved summary to: {out_path}")
    print(f"Saved correlation table to: {corr_path}")
    print()
    print("This script only reports. No frozen MAGD-Fraud module was modified, and test was never loaded.")


if __name__ == "__main__":
    main()
