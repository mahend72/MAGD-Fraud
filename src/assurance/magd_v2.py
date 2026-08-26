"""MAGD-Fraud v2 assurance risk score - additive, parallel to v1.

This module implements the v2 design agreed in review: it does NOT modify any v1
function (compute_magd_risk, compute_deployable_wrong_confident_risk, route_magd_cases,
adaptive threshold, or any routing logic). It is a self-contained, separately-invokable
scoring pipeline. Nothing here is wired into the production routing pipeline; v1 remains
the frozen, canonical implementation for all existing MAGD-Heuristic/Learned/Constrained/
ValidationTuned results.

Design summary (see conversation record for the full root-cause diagnosis this responds
to):

- MAGD_AR_v2 is a pure AI-reliance/failure-risk score over four normalized components
  (distance uncertainty, calibration risk, neighbourhood reliability, gated
  wrong-confident risk). It carries NO drift/business/capacity terms - those remain the
  adaptive-threshold layer's responsibility (unchanged, v1 function reused as-is).
- wrong_confident_risk_v2 gates independently-normalized unreliability evidence
  (calibration risk, neighbourhood reliability) by raw confidence, multiplicatively -
  confidence is never added, and distance uncertainty is never duplicated into this term
  (it already has its own top-level weight in MAGD_AR_v2).
- Every ECDF normalization (distance uncertainty, calibration risk, neighbourhood
  reliability) AND the calibration-risk mapping itself (bin edges/gaps) are fit strictly
  inside each CV fold's inner-training portion, then frozen and applied (never refit) to
  that fold's held-out portion. This is nested cross-validation done properly: no
  candidate is ever evaluated on labels that informed its own fitting.
- Weights are learned directly on the non-negative simplex (w_j >= 0, sum(w_j) = 1) via
  a constrained optimizer, not via unconstrained NNLS followed by post-hoc
  renormalization - the loss is evaluated only at points that are already valid simplex
  weights throughout the search.
- CV folds are stratified on (y_true, ai_wrong) jointly when every joint cell has at
  least n_splits members (checked at runtime), else on ai_wrong alone.
- ai_wrong is constructed from the base fraud model's predictions, which are fixed
  (fit once on train only, never refit per fold) - so ai_wrong is automatically valid
  and out-of-fold for every row in every fold, by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

from src.assurance.calibration import assign_calibration_risk, build_calibration_bins

V2_COMPONENT_KEYS = ["distance_uncertainty", "calibration_risk", "neighbor_error_rate", "wrong_confident_risk"]


# ---------------------------------------------------------------------------
# ECDF (percentile-rank) normalization - fit on one frame, applied to any frame.
# ---------------------------------------------------------------------------

@dataclass
class EcdfTransform:
    """A frozen empirical-CDF transform: sorted fit-time values, used for
    interpolated percentile-rank lookup. Must be fit on inner-train data only."""

    sorted_values: np.ndarray
    n_fit: int

    def transform(self, values: pd.Series | np.ndarray) -> np.ndarray:
        """Strict/left empirical percentile: r~(x) = P_fit(X < x).

        Using side="left" counts only fit-time values STRICTLY LESS than the query
        value. This matters specifically for tied minimum-risk clusters (e.g.
        calibration_risk's dominant well-calibrated bin, shared by the large majority
        of rows): under side="right" that entire tied cluster would be assigned the
        rank AFTER all of itself (pushed toward 1.0, i.e. "high risk" - wrong, since
        it is the low-risk majority). Under side="left", nothing is strictly less than
        the tied minimum, so it correctly maps to 0.
        """
        values = np.asarray(values, dtype=float)
        ranks = np.searchsorted(self.sorted_values, values, side="left")
        return np.clip(ranks / max(self.n_fit, 1), 0.0, 1.0)


def fit_ecdf(values: pd.Series | np.ndarray) -> EcdfTransform:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot fit an ECDF transform on an empty series.")
    return EcdfTransform(sorted_values=np.sort(arr), n_fit=arr.size)


# ---------------------------------------------------------------------------
# Fold-local calibration-risk mapping (reuses v1's calibration.py unmodified -
# the only difference from v1 is which rows are passed in as "validation_predictions").
# ---------------------------------------------------------------------------

def fit_calibration_mapping(inner_train_predictions: pd.DataFrame, *, n_bins: int = 10):
    """Fit calibration bins on inner_train_predictions only (must contain case_id,
    split, y_true, ai_score, ai_pred). Returns the frozen bins table; apply via
    assign_calibration_risk(frame, bins) - both functions are v1's, unmodified."""
    required = {"case_id", "split", "y_true", "ai_score", "ai_pred"}
    missing = required - set(inner_train_predictions.columns)
    if missing:
        raise ValueError(f"Inner-train predictions frame missing columns for calibration fitting: {sorted(missing)}")
    return build_calibration_bins(inner_train_predictions, n_bins=n_bins)


# ---------------------------------------------------------------------------
# Wrong-confident risk v2: confidence gates independently-normalized unreliability
# evidence. No additive confidence term, no distance-uncertainty duplication.
# ---------------------------------------------------------------------------

def compute_wrong_confident_risk_v2(
    frame: pd.DataFrame,
    *,
    calibration_risk_norm_col: str = "calibration_risk_norm",
    neighbor_error_norm_col: str = "neighbor_error_rate_norm",
    confidence_col: str = "numerical_confidence",
    beta_calibration: float = 0.5,
    beta_neighbor: float = 0.5,
) -> pd.Series:
    if beta_calibration < 0 or beta_neighbor < 0:
        raise ValueError("beta_calibration and beta_neighbor must be non-negative.")
    total = beta_calibration + beta_neighbor
    if total <= 0:
        raise ValueError("beta_calibration + beta_neighbor must be positive.")
    r_base = (beta_calibration * frame[calibration_risk_norm_col].astype(float) + beta_neighbor * frame[neighbor_error_norm_col].astype(float)) / total
    confidence = frame[confidence_col].astype(float)
    r_wc = (confidence * r_base).clip(0.0, 1.0)
    return r_wc


# ---------------------------------------------------------------------------
# MAGD_AR v2: pure failure-risk composite, no drift/business terms.
# ---------------------------------------------------------------------------

def compute_magd_ar_v2(
    frame: pd.DataFrame,
    weights: dict[str, float],
    *,
    distance_norm_col: str = "distance_uncertainty_norm",
    calibration_norm_col: str = "calibration_risk_norm",
    neighbor_norm_col: str = "neighbor_error_rate_norm",
    wrong_confident_col: str = "wrong_confident_risk_v2",
) -> pd.Series:
    for key in V2_COMPONENT_KEYS:
        if key not in weights:
            raise ValueError(f"Missing MAGD_AR_v2 weight for `{key}`.")
        if weights[key] < 0:
            raise ValueError(f"MAGD_AR_v2 weight for `{key}` must be non-negative, got {weights[key]}.")
    total_weight = sum(weights[key] for key in V2_COMPONENT_KEYS)
    if total_weight <= 0:
        raise ValueError("MAGD_AR_v2 weights must sum to a positive value.")

    raw = (
        weights["distance_uncertainty"] * frame[distance_norm_col].astype(float)
        + weights["calibration_risk"] * frame[calibration_norm_col].astype(float)
        + weights["neighbor_error_rate"] * frame[neighbor_norm_col].astype(float)
        + weights["wrong_confident_risk"] * frame[wrong_confident_col].astype(float)
    ) / total_weight
    return raw.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# SUPERSEDED (kept only for reference/tests - NOT called by run_nested_cv): simplex-
# constrained squared-error weight learning. Confirmed via direct validation-only
# analysis to systematically favor whichever normalized signal's mean is closest to
# ai_wrong's low base rate, rather than the signal with genuine ranking power - it is
# the wrong loss for a severely imbalanced binary target. Replaced below by
# fit_simplex_weights_ranking.
# ---------------------------------------------------------------------------

def fit_simplex_weights(features: np.ndarray, target: np.ndarray, *, keys: list[str]) -> dict[str, float]:
    """features: (n, k) array of the k normalized component signals (already in the
    order given by `keys`); target: (n,) binary ai_wrong array. Returns weights on the
    simplex minimizing squared error, found via an equality-constrained optimizer -
    every point the optimizer evaluates is itself already on the simplex."""
    n_features = features.shape[1]
    if n_features != len(keys):
        raise ValueError("features column count must match len(keys).")

    def objective(w: np.ndarray) -> float:
        score = features @ w
        return float(np.mean((score - target) ** 2))

    initial = np.full(n_features, 1.0 / n_features, dtype=float)
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bounds = [(0.0, 1.0)] * n_features

    result = minimize(objective, initial, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 200, "ftol": 1e-10})
    weights = np.clip(result.x, 0.0, None)
    total = weights.sum()
    if total <= 0:
        weights = initial
    else:
        weights = weights / total  # guard against tiny numerical constraint-violation drift
    return dict(zip(keys, weights.tolist(), strict=True))


# ---------------------------------------------------------------------------
# Simplex-constrained PAIRWISE LOGISTIC RANKING loss: w_j >= 0, sum(w_j) = 1. Ranks
# ai_wrong=1 cases above ai_wrong=0 cases, rather than regressing the (severely
# imbalanced) binary level - immune to the mean-matching failure mode above, since it
# only ever compares scores within positive/negative pairs, never against an absolute
# target level.
# ---------------------------------------------------------------------------

def _pairwise_ranking_loss_and_grad(
    w: np.ndarray, features_pos: np.ndarray, features_neg: np.ndarray
) -> tuple[float, np.ndarray]:
    scores_pos = features_pos @ w  # (n_pos,)
    scores_neg = features_neg @ w  # (n_neg,)
    diff = scores_pos[:, None] - scores_neg[None, :]  # (n_pos, n_neg): want this > 0
    # log(1 + exp(-diff)), computed via logaddexp(0, -diff) for numerical stability.
    loss_matrix = np.logaddexp(0.0, -diff)
    loss = float(loss_matrix.mean())

    sigmoid_neg_diff = 1.0 / (1.0 + np.exp(diff))  # d/d(diff) of the loss, elementwise
    # d(loss)/dw = mean_ij [ sigmoid(-diff_ij) * (-(features_pos_i - features_neg_j)) ]
    grad = (sigmoid_neg_diff[:, :, None] * (-(features_pos[:, None, :] - features_neg[None, :, :]))).mean(axis=(0, 1))
    return loss, grad


def fit_simplex_weights_ranking(
    features: np.ndarray,
    target: np.ndarray,
    *,
    keys: list[str],
    max_negatives: int = 20000,
    seed: int = 42,
) -> dict[str, float]:
    """Simplex-constrained (w>=0, sum(w)=1) pairwise logistic ranking loss:
    minimize mean over (i in positives, j in negatives) of log(1 + exp(-(s_i - s_j))),
    where s = features @ w. This directly optimizes AI-wrong-above-AI-correct ranking
    (an AUC surrogate), rather than regressing toward the ~1% positive base rate.

    Negatives are subsampled (fixed seed, deterministic given `target` and `seed`) to
    `max_negatives` for computational tractability when the majority class is large;
    all positives are always used. This subsampling happens only within whichever data
    is passed in (i.e. within an already-fold-local inner-train set) - it never reaches
    across the inner-train/held-out boundary and never touches test.
    """
    n_features = features.shape[1]
    if n_features != len(keys):
        raise ValueError("features column count must match len(keys).")
    target = np.asarray(target).astype(int)
    pos_mask = target == 1
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Pairwise ranking loss requires at least one positive (ai_wrong=1) and one negative (ai_wrong=0) case.")

    features_pos = features[pos_mask]
    features_neg_full = features[neg_mask]
    if n_neg > max_negatives:
        rng = np.random.default_rng(seed)
        sampled_idx = rng.choice(n_neg, size=max_negatives, replace=False)
        features_neg = features_neg_full[sampled_idx]
    else:
        features_neg = features_neg_full

    def objective_and_grad(w: np.ndarray):
        loss, grad = _pairwise_ranking_loss_and_grad(w, features_pos, features_neg)
        return loss, grad

    initial = np.full(n_features, 1.0 / n_features, dtype=float)
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bounds = [(0.0, 1.0)] * n_features

    result = minimize(
        lambda w: objective_and_grad(w)[0],
        initial,
        jac=lambda w: objective_and_grad(w)[1],
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-10},
    )
    weights = np.clip(result.x, 0.0, None)
    total = weights.sum()
    if total <= 0:
        weights = initial
    else:
        weights = weights / total
    return dict(zip(keys, weights.tolist(), strict=True))


# ---------------------------------------------------------------------------
# Fold-local feature construction: fits calibration mapping + ECDF normalization on
# inner-train only, then applies (never refits) to both inner-train and the target
# frame (which may be the held-out fold or, for the final refit, the full split).
# ---------------------------------------------------------------------------

@dataclass
class FoldArtifacts:
    calibration_bins: pd.DataFrame
    distance_ecdf: EcdfTransform
    calibration_ecdf: EcdfTransform
    neighbor_ecdf: EcdfTransform


def fit_fold_artifacts(inner_train: pd.DataFrame, *, n_calibration_bins: int = 10) -> FoldArtifacts:
    calibration_bins = fit_calibration_mapping(
        inner_train[["case_id", "split", "y_true", "ai_score", "ai_pred"]], n_bins=n_calibration_bins
    )
    inner_calibration = assign_calibration_risk(
        inner_train[["case_id", "split", "y_true", "ai_score", "ai_pred"]], calibration_bins
    )
    calibration_ecdf = fit_ecdf(inner_calibration["calibration_risk"])
    distance_ecdf = fit_ecdf(inner_train["distance_uncertainty"])
    neighbor_ecdf = fit_ecdf(inner_train["neighbor_error_rate"])
    return FoldArtifacts(
        calibration_bins=calibration_bins,
        distance_ecdf=distance_ecdf,
        calibration_ecdf=calibration_ecdf,
        neighbor_ecdf=neighbor_ecdf,
    )


def apply_fold_artifacts(frame: pd.DataFrame, artifacts: FoldArtifacts, *, beta_calibration: float = 0.5, beta_neighbor: float = 0.5) -> pd.DataFrame:
    """Applies a frozen FoldArtifacts (fit on some OTHER inner-train set) to `frame` -
    never refits anything. `frame` may be a held-out fold or a fresh split (val/test)."""
    working = frame.copy()
    calibration_assigned = assign_calibration_risk(
        working[["case_id", "split", "y_true", "ai_score", "ai_pred"]], artifacts.calibration_bins
    )
    working = working.merge(
        calibration_assigned[["case_id", "calibration_risk"]], on="case_id", how="left", suffixes=("", "_assigned")
    )
    if "calibration_risk_assigned" in working.columns:
        working["calibration_risk"] = working["calibration_risk_assigned"]
        working = working.drop(columns=["calibration_risk_assigned"])

    working["distance_uncertainty_norm"] = artifacts.distance_ecdf.transform(working["distance_uncertainty"])
    working["calibration_risk_norm"] = artifacts.calibration_ecdf.transform(working["calibration_risk"])
    working["neighbor_error_rate_norm"] = artifacts.neighbor_ecdf.transform(working["neighbor_error_rate"])
    working["wrong_confident_risk_v2"] = compute_wrong_confident_risk_v2(
        working, beta_calibration=beta_calibration, beta_neighbor=beta_neighbor
    )
    return working


# ---------------------------------------------------------------------------
# Nested CV orchestration.
# ---------------------------------------------------------------------------

ENRICHMENT_PERCENTILES = (20, 10, 5)
HIGH_CONFIDENCE_THRESHOLD = 0.80  # matches the convention used throughout v1 evaluation


@dataclass
class DiscriminationSummary:
    auroc: float
    pr_auc: float
    spearman_rho: float
    spearman_p: float
    ai_error_enrichment: dict[int, float]  # top-K% -> AI-error rate
    wc_error_enrichment: dict[int, float]  # top-K% -> wrong-confident-error rate


@dataclass
class FoldResult:
    fold: int
    n_inner_train: int
    n_held_out: int
    weights: dict[str, float]
    composite: DiscriminationSummary
    calibration_alone: DiscriminationSummary
    distance_alone: DiscriminationSummary


@dataclass
class NestedCvResult:
    fold_results: list[FoldResult]
    mean_auroc: float
    mean_pr_auc: float
    mean_calibration_alone_auroc: float
    mean_distance_alone_auroc: float
    final_artifacts: FoldArtifacts
    final_weights: dict[str, float]


def _stratification_labels(frame: pd.DataFrame, *, n_splits: int) -> np.ndarray:
    y = frame["y_true"].astype(int).to_numpy()
    wrong = frame["ai_wrong"].astype(int).to_numpy()
    joint = y * 2 + wrong  # 4 joint cells
    _, counts = np.unique(joint, return_counts=True)
    if counts.min() >= n_splits:
        return joint
    return wrong  # fall back to ai_wrong-only stratification if any joint cell is too small


def _discrimination_summary(scores: np.ndarray, y_wrong: np.ndarray, wc_error: np.ndarray) -> DiscriminationSummary:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from scipy.stats import spearmanr

    auroc = float(roc_auc_score(y_wrong, scores))
    pr_auc = float(average_precision_score(y_wrong, scores))
    rho, p = spearmanr(scores, y_wrong)
    order = np.argsort(-scores)
    ai_enrich: dict[int, float] = {}
    wc_enrich: dict[int, float] = {}
    n = len(scores)
    for pct in ENRICHMENT_PERCENTILES:
        n_top = max(1, int(round(n * pct / 100)))
        top_idx = order[:n_top]
        ai_enrich[pct] = float(y_wrong[top_idx].mean())
        wc_enrich[pct] = float(wc_error[top_idx].mean())
    return DiscriminationSummary(
        auroc=auroc, pr_auc=pr_auc, spearman_rho=float(rho), spearman_p=float(p),
        ai_error_enrichment=ai_enrich, wc_error_enrichment=wc_enrich,
    )


def run_nested_cv(
    development_core: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    beta_calibration: float = 0.5,
    beta_neighbor: float = 0.5,
    n_calibration_bins: int = 10,
    max_ranking_negatives: int = 20000,
) -> NestedCvResult:
    """development_core must contain: case_id, split, y_true, ai_pred, ai_score,
    numerical_confidence, distance_uncertainty, calibration_risk (raw, will be
    refit per fold - this column is only used to know the frame shape), neighbor_error_rate.
    Intended to be called with the VALIDATION split as development_core. Test must
    never be passed to this function."""
    required = {"case_id", "split", "y_true", "ai_pred", "ai_score", "numerical_confidence", "distance_uncertainty", "neighbor_error_rate"}
    missing = required - set(development_core.columns)
    if missing:
        raise ValueError(f"development_core missing required columns: {sorted(missing)}")

    core = development_core.copy().reset_index(drop=True)
    core["ai_wrong"] = (core["ai_pred"].astype(int) != core["y_true"].astype(int)).astype(int)
    core["wc_error"] = ((core["ai_wrong"] == 1) & (core["numerical_confidence"].astype(float) >= HIGH_CONFIDENCE_THRESHOLD)).astype(int)

    strat_labels = _stratification_labels(core, n_splits=n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    feature_cols = ["distance_uncertainty_norm", "calibration_risk_norm", "neighbor_error_rate_norm", "wrong_confident_risk_v2"]
    fold_results: list[FoldResult] = []
    for fold_idx, (inner_train_idx, held_out_idx) in enumerate(skf.split(core, strat_labels)):
        inner_train = core.iloc[inner_train_idx].reset_index(drop=True)
        held_out = core.iloc[held_out_idx].reset_index(drop=True)

        artifacts = fit_fold_artifacts(inner_train, n_calibration_bins=n_calibration_bins)
        inner_train_scored = apply_fold_artifacts(inner_train, artifacts, beta_calibration=beta_calibration, beta_neighbor=beta_neighbor)
        held_out_scored = apply_fold_artifacts(held_out, artifacts, beta_calibration=beta_calibration, beta_neighbor=beta_neighbor)

        X_inner = inner_train_scored[feature_cols].astype(float).to_numpy()
        y_inner = inner_train_scored["ai_wrong"].astype(int).to_numpy()
        weights = fit_simplex_weights_ranking(X_inner, y_inner, keys=V2_COMPONENT_KEYS, max_negatives=max_ranking_negatives, seed=seed)

        held_out_score = compute_magd_ar_v2(held_out_scored, weights).to_numpy()
        y_held_out = held_out_scored["ai_wrong"].astype(int).to_numpy()
        wc_held_out = held_out_scored["wc_error"].astype(int).to_numpy()

        composite = _discrimination_summary(held_out_score, y_held_out, wc_held_out)
        calibration_alone = _discrimination_summary(held_out_scored["calibration_risk_norm"].to_numpy(), y_held_out, wc_held_out)
        distance_alone = _discrimination_summary(held_out_scored["distance_uncertainty_norm"].to_numpy(), y_held_out, wc_held_out)

        fold_results.append(
            FoldResult(
                fold=fold_idx, n_inner_train=len(inner_train), n_held_out=len(held_out),
                weights=weights, composite=composite, calibration_alone=calibration_alone, distance_alone=distance_alone,
            )
        )

    mean_auroc = float(np.mean([f.composite.auroc for f in fold_results]))
    mean_pr_auc = float(np.mean([f.composite.pr_auc for f in fold_results]))
    mean_calibration_alone_auroc = float(np.mean([f.calibration_alone.auroc for f in fold_results]))
    mean_distance_alone_auroc = float(np.mean([f.distance_alone.auroc for f in fold_results]))

    # Final refit on the FULL development split (no held-out portion) - this is what
    # WOULD get frozen and applied once to test, once/if that step is authorized.
    final_artifacts = fit_fold_artifacts(core, n_calibration_bins=n_calibration_bins)
    full_scored = apply_fold_artifacts(core, final_artifacts, beta_calibration=beta_calibration, beta_neighbor=beta_neighbor)
    X_full = full_scored[feature_cols].astype(float).to_numpy()
    y_full = full_scored["ai_wrong"].astype(int).to_numpy()
    final_weights = fit_simplex_weights_ranking(X_full, y_full, keys=V2_COMPONENT_KEYS, max_negatives=max_ranking_negatives, seed=seed)

    return NestedCvResult(
        fold_results=fold_results,
        mean_auroc=mean_auroc,
        mean_pr_auc=mean_pr_auc,
        mean_calibration_alone_auroc=mean_calibration_alone_auroc,
        mean_distance_alone_auroc=mean_distance_alone_auroc,
        final_artifacts=final_artifacts,
        final_weights=final_weights,
    )


# ---------------------------------------------------------------------------
# FINALIZED v2 assurance-risk aggregation: the validated regularized logistic
# interaction model (nested-CV confirmed: consistent, material AUROC/PR-AUC gain over
# calibration-risk-alone in all 5 folds - see conversation record). This REPLACES
# compute_magd_ar_v2/fit_simplex_weights_ranking as the score the v2 pipeline actually
# uses; those functions are kept above only as tested, superseded reference points, not
# called by anything below this line. The model, its 7 features, and its hyperparameters
# are exactly and only what was evaluated in nested CV - nothing here may be changed,
# searched, or tuned without re-running that validation.
# ---------------------------------------------------------------------------

LOGISTIC_FEATURE_NAMES = [
    "distance_uncertainty_norm",
    "calibration_risk_norm",
    "neighbor_error_rate_norm",
    "wrong_confident_risk_v2",
    "calibration_x_distance",
    "calibration_x_neighbourhood",
    "confidence_x_calibration",
]

# Fixed, pre-specified hyperparameters - identical to what was nested-CV-validated.
# Not a search space; changing these requires re-validating on validation before any
# further test use.
LOGISTIC_MODEL_PARAMS: dict[str, object] = {
    "C": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "max_iter": 1000,
}


def build_logistic_features(scored_frame: pd.DataFrame) -> np.ndarray:
    """Exactly the 7 features nested-CV-validated: the 4 normalized assurance signals
    plus only the three pre-specified interactions (calibration x distance,
    calibration x neighbourhood, confidence x calibration). No other feature is
    computed or permitted here."""
    required = {"distance_uncertainty_norm", "calibration_risk_norm", "neighbor_error_rate_norm", "wrong_confident_risk_v2", "numerical_confidence"}
    missing = required - set(scored_frame.columns)
    if missing:
        raise ValueError(f"scored_frame missing columns required for build_logistic_features: {sorted(missing)}")
    dist = scored_frame["distance_uncertainty_norm"].astype(float).to_numpy()
    cal = scored_frame["calibration_risk_norm"].astype(float).to_numpy()
    nbr = scored_frame["neighbor_error_rate_norm"].astype(float).to_numpy()
    wc = scored_frame["wrong_confident_risk_v2"].astype(float).to_numpy()
    conf = scored_frame["numerical_confidence"].astype(float).to_numpy()
    return np.column_stack([dist, cal, nbr, wc, cal * dist, cal * nbr, conf * cal])


def fit_logistic_assurance_model(features: np.ndarray, target: np.ndarray):
    """Fits the fixed-hyperparameter L2 logistic assurance model. No hyperparameter
    here is searched or chosen based on any result - these are exactly the values
    nested-CV-validated before this function existed."""
    from sklearn.linear_model import LogisticRegression

    if features.shape[1] != len(LOGISTIC_FEATURE_NAMES):
        raise ValueError(f"Expected {len(LOGISTIC_FEATURE_NAMES)} features, got {features.shape[1]}.")
    model = LogisticRegression(**LOGISTIC_MODEL_PARAMS)
    model.fit(features, target)
    return model


def compute_assurance_risk_v2(scored_frame: pd.DataFrame, model) -> pd.Series:
    """Applies a FROZEN, already-fit logistic model to produce the v2 assurance-risk
    score. This is a learned AI-reliance/failure-risk score, not a calibrated
    probability of AI error - it is used only for ranking/thresholding, exactly as
    v1's magd_assurance_risk was. Never refits anything."""
    features = build_logistic_features(scored_frame)
    risk = model.predict_proba(features)[:, 1]
    return pd.Series(np.clip(risk, 0.0, 1.0), index=scored_frame.index, name="magd_assurance_risk")


@dataclass
class FinalV2Artifacts:
    """Everything needed to score and route any frame under the frozen, finalized v2
    pipeline: preprocessing (fit on the FULL validation split, not a CV fold), the
    fitted logistic model, and validation-quantile-derived routing thresholds."""

    fold_artifacts: FoldArtifacts
    model: object
    low_risk: float
    high_risk: float
    beta_calibration: float
    beta_neighbor: float


def fit_final_v2_pipeline(
    validation_core: pd.DataFrame,
    *,
    low_quantile: float = 0.60,
    high_quantile: float = 0.90,
    beta_calibration: float = 0.5,
    beta_neighbor: float = 0.5,
    n_calibration_bins: int = 10,
) -> FinalV2Artifacts:
    """Fits preprocessing, the logistic model, and routing thresholds on the FULL
    validation split (not a fold) - this is the one-time "freeze" step. Must be called
    with validation only; must never be called with test data."""
    required = {"case_id", "split", "y_true", "ai_pred", "ai_score", "numerical_confidence", "distance_uncertainty", "neighbor_error_rate"}
    missing = required - set(validation_core.columns)
    if missing:
        raise ValueError(f"validation_core missing required columns: {sorted(missing)}")

    from src.assurance.magd_risk import derive_validation_risk_thresholds  # v1, unmodified

    core = validation_core.copy().reset_index(drop=True)
    core["ai_wrong"] = (core["ai_pred"].astype(int) != core["y_true"].astype(int)).astype(int)

    fold_artifacts = fit_fold_artifacts(core, n_calibration_bins=n_calibration_bins)
    scored = apply_fold_artifacts(core, fold_artifacts, beta_calibration=beta_calibration, beta_neighbor=beta_neighbor)

    X = build_logistic_features(scored)
    y = scored["ai_wrong"].astype(int).to_numpy()
    model = fit_logistic_assurance_model(X, y)

    validation_risk = compute_assurance_risk_v2(scored, model)
    low_risk, high_risk = derive_validation_risk_thresholds(validation_risk, low_quantile=low_quantile, high_quantile=high_quantile)

    return FinalV2Artifacts(
        fold_artifacts=fold_artifacts, model=model, low_risk=low_risk, high_risk=high_risk,
        beta_calibration=beta_calibration, beta_neighbor=beta_neighbor,
    )


def apply_final_v2_pipeline(frame: pd.DataFrame, artifacts: FinalV2Artifacts) -> pd.DataFrame:
    """Applies a FROZEN FinalV2Artifacts to `frame` (validation or test) - transform
    only, never refits anything. Writes `magd_assurance_risk` and `risk_category`
    columns using v1's own add_risk_categories_and_actions (unmodified), so the result
    can be handed directly to v1's unmodified routing (route_magd_cases) without any
    change to the routing architecture."""
    from src.assurance.magd_risk import add_risk_categories_and_actions  # v1, unmodified

    scored = apply_fold_artifacts(frame, artifacts.fold_artifacts, beta_calibration=artifacts.beta_calibration, beta_neighbor=artifacts.beta_neighbor)
    risk = compute_assurance_risk_v2(scored, artifacts.model)
    scored = scored.drop(columns=["magd_assurance_risk"], errors="ignore")
    scored["magd_assurance_risk"] = risk
    scored = add_risk_categories_and_actions(scored, low_risk=artifacts.low_risk, high_risk=artifacts.high_risk)
    return scored
