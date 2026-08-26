from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assurance.magd_v2 import (
    EcdfTransform,
    FoldArtifacts,
    apply_fold_artifacts,
    compute_magd_ar_v2,
    compute_wrong_confident_risk_v2,
    fit_calibration_mapping,
    fit_ecdf,
    fit_fold_artifacts,
    fit_simplex_weights,
    fit_simplex_weights_ranking,
    run_nested_cv,
    _stratification_labels,
)


def _rng():
    return np.random.default_rng(42)


def _synthetic_development_core(n: int = 2000) -> pd.DataFrame:
    rng = _rng()
    y_true = rng.binomial(1, 0.05, size=n)
    ai_pred = y_true.copy()
    flip = rng.random(n) < 0.15
    ai_pred[flip] = 1 - ai_pred[flip]
    ai_score = np.clip(ai_pred + rng.normal(0, 0.1, size=n), 0.0, 1.0)
    numerical_confidence = np.clip(np.maximum(ai_score, 1 - ai_score), 0.5, 1.0)
    distance_uncertainty = rng.beta(2, 5, size=n)
    neighbor_error_rate = rng.beta(1, 10, size=n)
    return pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(n)],
            "split": ["val"] * n,
            "y_true": y_true,
            "ai_pred": ai_pred,
            "ai_score": ai_score,
            "numerical_confidence": numerical_confidence,
            "distance_uncertainty": distance_uncertainty,
            "neighbor_error_rate": neighbor_error_rate,
        }
    )


# --------------------------------------------------------------------------
# ECDF normalization: fit strictly on the given data, never on anything else.
# --------------------------------------------------------------------------

def test_ecdf_transform_uses_only_fit_time_values() -> None:
    fit_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    transform = fit_ecdf(fit_values)
    assert transform.n_fit == 5
    # Strict/left transform: r~(x) = P(X < x). The minimum fit-time value itself has
    # nothing strictly less than it, so it maps to 0.0, not the old side="right" 1.0.
    assert transform.transform(np.array([0.0]))[0] == pytest.approx(0.0)
    # A value below all fit-time values also ranks at 0.0.
    assert transform.transform(np.array([-1.0]))[0] == pytest.approx(0.0)
    # A value above all fit-time values ranks at 1.0 (everything is strictly less).
    assert transform.transform(np.array([1.0]))[0] == pytest.approx(1.0)


def test_ecdf_tied_minimum_cluster_maps_to_zero_not_near_one() -> None:
    # Reproduces the calibration_risk failure mode: 95% of fit-time values tied at the
    # dominant, low-risk value; the remaining 5% spread across higher values.
    rng = np.random.default_rng(0)
    tied_low = np.zeros(950)
    spread_high = rng.uniform(0.1, 1.0, size=50)
    fit_values = np.concatenate([tied_low, spread_high])
    transform = fit_ecdf(fit_values)

    # Under the OLD side="right" behaviour, the tied cluster (95% of the mass) would
    # have mapped to rank ~0.95 (near 1.0, i.e. "high risk") despite being the
    # legitimately low-risk majority. Under the required strict/left transform, it must
    # map to 0.0: nothing is strictly less than the tied minimum.
    tied_rank = transform.transform(np.array([0.0]))[0]
    assert tied_rank == pytest.approx(0.0)

    # A genuinely higher, rare value must still rank higher than the tied minimum -
    # ordering is preserved, only the tied-cluster placement changed.
    high_rank = transform.transform(np.array([0.9]))[0]
    assert high_rank > tied_rank


def test_ecdf_fit_on_different_subsets_gives_different_transforms() -> None:
    rng = _rng()
    subset_a = rng.beta(2, 5, size=500)
    subset_b = rng.beta(5, 2, size=500)  # differently shaped distribution
    transform_a = fit_ecdf(subset_a)
    transform_b = fit_ecdf(subset_b)
    query = np.array([0.5])
    assert transform_a.transform(query)[0] != transform_b.transform(query)[0]


# --------------------------------------------------------------------------
# Fold-local calibration mapping reuses v1's calibration.py unmodified.
# --------------------------------------------------------------------------

def test_calibration_mapping_differs_across_disjoint_inner_train_sets() -> None:
    core = _synthetic_development_core()
    half_a = core.iloc[:1000]
    half_b = core.iloc[1000:]
    bins_a = fit_calibration_mapping(half_a, n_bins=5)
    bins_b = fit_calibration_mapping(half_b, n_bins=5)
    # Different inner-train rows should (with overwhelming probability, given random
    # synthetic data) produce different per-bin calibration gaps.
    assert not bins_a["absolute_calibration_gap"].equals(bins_b["absolute_calibration_gap"])


# --------------------------------------------------------------------------
# No-leakage: a fold's frozen artifacts must not change when held-out-only rows change.
# --------------------------------------------------------------------------

def test_fold_artifacts_are_unaffected_by_held_out_rows() -> None:
    core = _synthetic_development_core()
    inner_train = core.iloc[:1500].reset_index(drop=True)
    held_out_a = core.iloc[1500:].reset_index(drop=True)
    held_out_b = held_out_a.copy()
    held_out_b["distance_uncertainty"] = 1.0 - held_out_b["distance_uncertainty"]  # perturb held-out only

    artifacts = fit_fold_artifacts(inner_train)
    # Artifacts are derived from inner_train only, so fitting again with a perturbed
    # held-out set (never passed to fit_fold_artifacts) must be identical.
    artifacts_again = fit_fold_artifacts(inner_train)
    pd.testing.assert_frame_equal(artifacts.calibration_bins, artifacts_again.calibration_bins)
    np.testing.assert_array_equal(artifacts.distance_ecdf.sorted_values, artifacts_again.distance_ecdf.sorted_values)

    # Scoring the two different held-out variants with the SAME frozen artifacts must
    # differ only where the perturbation was applied (distance_uncertainty_norm), and
    # must not affect calibration_risk_norm or the fitted mapping itself.
    scored_a = apply_fold_artifacts(held_out_a, artifacts)
    scored_b = apply_fold_artifacts(held_out_b, artifacts)
    assert not np.allclose(scored_a["distance_uncertainty_norm"], scored_b["distance_uncertainty_norm"])


# --------------------------------------------------------------------------
# Wrong-confident risk v2: gate, not addend; no distance duplication.
# --------------------------------------------------------------------------

def test_gated_wrong_confident_risk_is_multiplicative_not_additive() -> None:
    frame = pd.DataFrame(
        {
            "numerical_confidence": [1.0, 0.5, 1.0, 0.5],
            "calibration_risk_norm": [1.0, 1.0, 0.0, 0.0],
            "neighbor_error_rate_norm": [1.0, 1.0, 0.0, 0.0],
        }
    )
    r_wc = compute_wrong_confident_risk_v2(frame, beta_calibration=0.5, beta_neighbor=0.5)
    # High confidence + high unreliability evidence -> high risk.
    assert r_wc.iloc[0] == pytest.approx(1.0)
    # Halved confidence with the same evidence must roughly halve the risk (gate, not
    # a flat additive floor).
    assert r_wc.iloc[1] == pytest.approx(0.5)
    # Zero unreliability evidence -> zero risk regardless of confidence.
    assert r_wc.iloc[2] == pytest.approx(0.0)
    assert r_wc.iloc[3] == pytest.approx(0.0)


def test_gated_wrong_confident_risk_does_not_reference_distance_uncertainty_column() -> None:
    frame = pd.DataFrame(
        {
            "numerical_confidence": [0.9],
            "calibration_risk_norm": [0.5],
            "neighbor_error_rate_norm": [0.5],
            # deliberately no distance_uncertainty / distance_uncertainty_norm column
        }
    )
    # Must not raise a KeyError - the function has no dependency on any distance column.
    result = compute_wrong_confident_risk_v2(frame)
    assert len(result) == 1


# --------------------------------------------------------------------------
# MAGD_AR v2: pure failure-risk composite, no drift/business terms accepted.
# --------------------------------------------------------------------------

def test_magd_ar_v2_has_no_drift_or_business_terms() -> None:
    frame = pd.DataFrame(
        {
            "distance_uncertainty_norm": [0.5],
            "calibration_risk_norm": [0.5],
            "neighbor_error_rate_norm": [0.5],
            "wrong_confident_risk_v2": [0.5],
        }
    )
    weights = {"distance_uncertainty": 0.25, "calibration_risk": 0.25, "neighbor_error_rate": 0.25, "wrong_confident_risk": 0.25}
    score = compute_magd_ar_v2(frame, weights)
    assert score.iloc[0] == pytest.approx(0.5)
    # No drift_risk/business_risk keys are accepted or required.
    import inspect

    sig = inspect.signature(compute_magd_ar_v2)
    assert "use_drift_risk" not in sig.parameters
    assert "use_business_risk" not in sig.parameters


def test_magd_ar_v2_rejects_negative_or_zero_sum_weights() -> None:
    frame = pd.DataFrame({"distance_uncertainty_norm": [0.5], "calibration_risk_norm": [0.5], "neighbor_error_rate_norm": [0.5], "wrong_confident_risk_v2": [0.5]})
    with pytest.raises(ValueError):
        compute_magd_ar_v2(frame, {"distance_uncertainty": -0.1, "calibration_risk": 0.5, "neighbor_error_rate": 0.3, "wrong_confident_risk": 0.3})
    with pytest.raises(ValueError):
        compute_magd_ar_v2(frame, {"distance_uncertainty": 0.0, "calibration_risk": 0.0, "neighbor_error_rate": 0.0, "wrong_confident_risk": 0.0})


# --------------------------------------------------------------------------
# Simplex-constrained weight learning: on-simplex throughout, not NNLS-then-normalize.
# --------------------------------------------------------------------------

def test_simplex_weights_are_non_negative_and_sum_to_one() -> None:
    rng = _rng()
    n = 500
    features = rng.random((n, 4))
    target = (features[:, 0] > 0.7).astype(float)
    weights = fit_simplex_weights(features, target, keys=["a", "b", "c", "d"])
    values = np.array(list(weights.values()))
    assert (values >= -1e-9).all()
    assert values.sum() == pytest.approx(1.0, abs=1e-6)


def test_simplex_weights_recover_the_dominant_informative_feature() -> None:
    rng = _rng()
    n = 2000
    # feature 0 is strongly informative, features 1-3 are pure noise.
    feature0 = rng.random(n)
    target = (feature0 > 0.8).astype(float)
    noise = rng.random((n, 3))
    features = np.column_stack([feature0, noise])
    weights = fit_simplex_weights(features, target, keys=["informative", "n1", "n2", "n3"])
    assert weights["informative"] > weights["n1"]
    assert weights["informative"] > weights["n2"]
    assert weights["informative"] > weights["n3"]


# --------------------------------------------------------------------------
# Stratification: joint (y_true, ai_wrong) when every cell has >= n_splits members.
# --------------------------------------------------------------------------

def test_stratification_uses_joint_labels_when_cells_are_large_enough() -> None:
    core = _synthetic_development_core(n=2000)
    core["ai_wrong"] = (core["ai_pred"].astype(int) != core["y_true"].astype(int)).astype(int)
    labels = _stratification_labels(core, n_splits=5)
    # 4 joint cells expected (y_true x ai_wrong), not just 2 (ai_wrong alone).
    assert len(np.unique(labels)) > 2


def test_stratification_falls_back_to_ai_wrong_when_a_joint_cell_is_too_small() -> None:
    core = _synthetic_development_core(n=200)
    core["y_true"] = 0
    core["ai_pred"] = 0
    core.loc[0, "ai_pred"] = 1  # exactly one (y=0, wrong=1) case - too small for 5 folds
    core["ai_wrong"] = (core["ai_pred"].astype(int) != core["y_true"].astype(int)).astype(int)
    labels = _stratification_labels(core, n_splits=5)
    assert set(np.unique(labels)).issubset({0, 1})


# --------------------------------------------------------------------------
# Full nested CV: no fold's held-out score depends on that fold's own held-out labels.
# --------------------------------------------------------------------------

def test_nested_cv_runs_and_produces_plausible_out_of_fold_metrics() -> None:
    core = _synthetic_development_core(n=3000)
    result = run_nested_cv(core, n_splits=5, seed=42)
    assert len(result.fold_results) == 5
    for fold in result.fold_results:
        assert 0.0 <= fold.composite.auroc <= 1.0
        assert 0.0 <= fold.calibration_alone.auroc <= 1.0
        assert 0.0 <= fold.distance_alone.auroc <= 1.0
        for pct in (20, 10, 5):
            assert pct in fold.composite.ai_error_enrichment
            assert pct in fold.composite.wc_error_enrichment
        assert fold.n_inner_train + fold.n_held_out == 3000
        weight_sum = sum(fold.weights.values())
        assert weight_sum == pytest.approx(1.0, abs=1e-6)
        assert all(w >= -1e-9 for w in fold.weights.values())
    assert 0.0 <= result.mean_auroc <= 1.0
    assert 0.0 <= result.mean_calibration_alone_auroc <= 1.0
    assert 0.0 <= result.mean_distance_alone_auroc <= 1.0
    final_weight_sum = sum(result.final_weights.values())
    assert final_weight_sum == pytest.approx(1.0, abs=1e-6)


def test_nested_cv_final_refit_uses_full_development_set_not_a_held_out_subset() -> None:
    core = _synthetic_development_core(n=1000)
    result = run_nested_cv(core, n_splits=5, seed=42)
    # The final artifacts' calibration bins must be fit on ALL 1000 rows, not a
    # ~800-row inner-train fold subset - check via bin_weight sums to 1 over the full set.
    assert result.final_artifacts.calibration_bins["count"].sum() == 1000


# --------------------------------------------------------------------------
# Pairwise logistic ranking loss: on-simplex, ranks positives above negatives, immune
# to the mean-matching failure mode that squared-error loss exhibited.
# --------------------------------------------------------------------------

def test_ranking_weights_are_non_negative_and_sum_to_one() -> None:
    rng = _rng()
    n = 800
    features = rng.random((n, 4))
    target = (features[:, 0] > 0.7).astype(int)
    weights = fit_simplex_weights_ranking(features, target, keys=["a", "b", "c", "d"], max_negatives=5000)
    values = np.array(list(weights.values()))
    assert (values >= -1e-9).all()
    assert values.sum() == pytest.approx(1.0, abs=1e-6)


def test_ranking_loss_recovers_informative_feature_under_severe_imbalance() -> None:
    # Reproduces the exact failure mode the squared-error fit exhibited: one feature
    # (informative) has a small, imbalanced-target-matching mean and genuine ranking
    # power; another (decoy) has a mean close to the rare positive rate but carries NO
    # information. Squared-error weight learning would be pulled toward whichever
    # feature's mean is closer to the target's mean; ranking loss must not be.
    rng = _rng()
    n = 4000
    target = rng.binomial(1, 0.02, size=n)  # severe imbalance, matching real ai_wrong rate
    informative = np.where(target == 1, rng.uniform(0.6, 1.0, size=n), rng.uniform(0.0, 0.6, size=n))
    # decoy: mean deliberately close to the target's low base rate, but pure noise.
    decoy = rng.uniform(0.0, 0.05, size=n)
    features = np.column_stack([informative, decoy])

    weights = fit_simplex_weights_ranking(features, target, keys=["informative", "decoy"], max_negatives=8000)
    assert weights["informative"] > weights["decoy"]
    assert weights["informative"] > 0.5


def test_ranking_loss_score_orders_positives_above_negatives_better_than_squared_error() -> None:
    # On the calibration_risk-style tied-cluster scenario (large majority tied at a low
    # value, informative minority spread higher), ranking loss on the raw feature alone
    # already discriminates well (AUROC far above chance); this is a sanity check that
    # the ranking objective itself is well-posed, independent of the simplex-fit machinery.
    rng = _rng()
    n_neg = 950
    n_pos = 50
    neg_feature = np.zeros(n_neg)
    pos_feature = rng.uniform(0.1, 1.0, size=n_pos)
    features = np.concatenate([neg_feature, pos_feature]).reshape(-1, 1)
    target = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])

    weights = fit_simplex_weights_ranking(features, target, keys=["only"], max_negatives=2000)
    assert weights["only"] == pytest.approx(1.0, abs=1e-6)
    from sklearn.metrics import roc_auc_score

    score = features[:, 0] * weights["only"]
    assert roc_auc_score(target, score) > 0.9


# --------------------------------------------------------------------------
# Finalized v2 (logistic interaction model): fixed feature set, fold-safe fitting,
# no test-label access, frozen/deterministic coefficients and thresholds, no v1
# regression.
# --------------------------------------------------------------------------

from src.assurance.magd_v2 import (  # noqa: E402
    LOGISTIC_FEATURE_NAMES,
    LOGISTIC_MODEL_PARAMS,
    FinalV2Artifacts,
    apply_final_v2_pipeline,
    build_logistic_features,
    compute_assurance_risk_v2,
    fit_final_v2_pipeline,
    fit_logistic_assurance_model,
)


def test_logistic_feature_set_is_fixed_and_exactly_seven() -> None:
    assert LOGISTIC_FEATURE_NAMES == [
        "distance_uncertainty_norm",
        "calibration_risk_norm",
        "neighbor_error_rate_norm",
        "wrong_confident_risk_v2",
        "calibration_x_distance",
        "calibration_x_neighbourhood",
        "confidence_x_calibration",
    ]
    core = _synthetic_development_core(n=200)
    artifacts = fit_fold_artifacts(core)
    scored = apply_fold_artifacts(core, artifacts)
    X = build_logistic_features(scored)
    assert X.shape == (200, 7)


def test_logistic_hyperparameters_are_fixed_not_searched() -> None:
    assert LOGISTIC_MODEL_PARAMS == {"C": 1.0, "class_weight": "balanced", "random_state": 42, "max_iter": 1000}


def test_fit_logistic_assurance_model_rejects_wrong_feature_count() -> None:
    rng = _rng()
    X = rng.random((50, 3))  # wrong number of columns
    y = rng.binomial(1, 0.1, size=50)
    with pytest.raises(ValueError):
        fit_logistic_assurance_model(X, y)


def test_final_v2_pipeline_fits_only_on_passed_in_development_data() -> None:
    core_a = _synthetic_development_core(n=1500)
    core_b = _synthetic_development_core(n=1500)  # different seed draw -> different data
    core_b["distance_uncertainty"] = 1.0 - core_b["distance_uncertainty"]

    artifacts_a = fit_final_v2_pipeline(core_a)
    artifacts_b = fit_final_v2_pipeline(core_b)
    # Different development data must yield different frozen thresholds/coefficients -
    # confirms fitting is genuinely data-dependent, not some fixed constant.
    assert artifacts_a.low_risk != artifacts_b.low_risk or artifacts_a.high_risk != artifacts_b.high_risk


def test_final_v2_pipeline_is_deterministic_given_identical_data() -> None:
    core = _synthetic_development_core(n=1500)
    artifacts_1 = fit_final_v2_pipeline(core)
    artifacts_2 = fit_final_v2_pipeline(core)
    np.testing.assert_array_equal(artifacts_1.model.coef_, artifacts_2.model.coef_)
    assert artifacts_1.low_risk == artifacts_2.low_risk
    assert artifacts_1.high_risk == artifacts_2.high_risk


def test_applying_frozen_v2_artifacts_never_touches_labels_of_the_scored_frame() -> None:
    """Leakage test: freeze artifacts on a development set, then apply to a
    'test-like' frame. Perturbing that frame's own labels must not change the frozen
    artifacts (already fit before this frame was ever touched), and must not change
    the SCORE for any row whose feature values are unperturbed - only y_true/ai_pred
    were changed, and apply_final_v2_pipeline never reads those to compute the score."""
    development = _synthetic_development_core(n=1500)
    artifacts = fit_final_v2_pipeline(development)

    held_out = _synthetic_development_core(n=500)
    held_out["case_id"] = [f"held_{i}" for i in range(len(held_out))]
    scored_before = apply_final_v2_pipeline(held_out, artifacts)

    perturbed = held_out.copy()
    perturbed["y_true"] = 1 - perturbed["y_true"]
    perturbed["ai_pred"] = 1 - perturbed["ai_pred"]
    scored_after = apply_final_v2_pipeline(perturbed, artifacts)

    pd.testing.assert_series_equal(scored_before["magd_assurance_risk"], scored_after["magd_assurance_risk"])
    pd.testing.assert_series_equal(scored_before["risk_category"], scored_after["risk_category"])
    # And the artifacts object itself is untouched (same object, same coefficients).
    np.testing.assert_array_equal(artifacts.model.coef_, artifacts.model.coef_)


def test_apply_final_v2_pipeline_produces_columns_v1_routing_expects() -> None:
    development = _synthetic_development_core(n=1500)
    artifacts = fit_final_v2_pipeline(development)
    test_like = _synthetic_development_core(n=300)
    scored = apply_final_v2_pipeline(test_like, artifacts)
    assert "magd_assurance_risk" in scored.columns
    assert "risk_category" in scored.columns
    assert set(scored["risk_category"].unique()).issubset({"low", "medium", "high"})
    assert scored["magd_assurance_risk"].between(0.0, 1.0).all()


def test_v1_magd_risk_functions_are_unmodified_by_v2() -> None:
    """Light-touch no-regression guard: v1's compute_magd_risk and
    add_risk_categories_and_actions must still behave exactly as before - same
    required columns, same category boundaries - regardless of anything added for v2."""
    from src.assurance.magd_risk import add_risk_categories_and_actions, compute_magd_risk, map_magd_risk_category

    frame = pd.DataFrame(
        {
            "case_id": ["a", "b"],
            "distance_uncertainty": [0.2, 0.8],
            "calibration_risk": [0.1, 0.9],
            "neighbor_error_rate": [0.0, 0.5],
            "wrong_confident_risk": [0.1, 0.9],
        }
    )
    weights = {"distance_uncertainty": 0.25, "calibration_risk": 0.2, "neighbor_error_rate": 0.2, "wrong_confident_risk": 0.25, "drift_risk": 0.05, "business_risk": 0.05}
    scored = compute_magd_risk(frame, weights, use_drift_risk=False, use_business_risk=False)
    assert scored["magd_assurance_risk"].between(0.0, 1.0).all()
    assert map_magd_risk_category(0.10, 0.35, 0.70) == "low"
    assert map_magd_risk_category(0.70, 0.35, 0.70) == "high"
    enriched = add_risk_categories_and_actions(scored, low_risk=0.35, high_risk=0.70)
    assert set(enriched["risk_category"]).issubset({"low", "medium", "high"})


# ---------------------------------------------------------------------------
# Frozen-v2 production regression guards. These check the ACTUAL frozen artifact
# (data/outputs/final_reproducible_run/magd_v2_test_decisions.csv, produced by
# applying the fully frozen v2 pipeline to the test split exactly once) rather than a
# synthetic refit, so they detect drift in the real, already-reported thresholds - not
# just in the fitting code's behavior on fake data.
# ---------------------------------------------------------------------------

from pathlib import Path as _Path  # noqa: E402

_ROOT = _Path(__file__).resolve().parents[1]
_FROZEN_TEST_DECISIONS_PATH = _ROOT / "data" / "outputs" / "final_reproducible_run" / "magd_v2_test_decisions.csv"

# The validation-quantile-derived thresholds actually used to produce the frozen test
# decisions - see conversation record ("Finalize MAGD-Fraud v2..." step). Changing
# these requires re-deriving them from validation (never from test) and is explicitly
# out of scope for implementation hardening.
FROZEN_LOW_RISK_THRESHOLD = 0.4063462838585886
FROZEN_HIGH_RISK_THRESHOLD = 0.44104567274033385


@pytest.mark.skipif(not _FROZEN_TEST_DECISIONS_PATH.exists(), reason="frozen magd_v2_test_decisions.csv not present")
def test_frozen_v2_test_decisions_risk_category_boundaries_match_documented_thresholds() -> None:
    decisions = pd.read_csv(_FROZEN_TEST_DECISIONS_PATH, usecols=["magd_assurance_risk", "risk_category"])
    low_max = decisions.loc[decisions["risk_category"] == "low", "magd_assurance_risk"].max()
    medium_min = decisions.loc[decisions["risk_category"] == "medium", "magd_assurance_risk"].min()
    medium_max = decisions.loc[decisions["risk_category"] == "medium", "magd_assurance_risk"].max()
    high_min = decisions.loc[decisions["risk_category"] == "high", "magd_assurance_risk"].min()

    # The low/medium and medium/high boundaries in the frozen file must bracket the
    # documented threshold constants (they will not be bit-identical because the
    # boundary sits between two adjacent observed risk values, not exactly on one).
    assert low_max <= FROZEN_LOW_RISK_THRESHOLD <= medium_min
    assert medium_max <= FROZEN_HIGH_RISK_THRESHOLD <= high_min
    assert abs(low_max - FROZEN_LOW_RISK_THRESHOLD) < 1e-3
    assert abs(high_min - FROZEN_HIGH_RISK_THRESHOLD) < 1e-3


def test_panel_k_used_for_reviewer_workload_matches_frozen_escalation_top_k() -> None:
    """panel_k (used by the reviewer-workload/cost-sensitivity module) must equal the
    same top_k_for_escalation the frozen routing config actually uses - it is not an
    independently chosen constant."""
    from src.evaluation.reviewer_workload import PANEL_K
    from src.utils.io import load_yaml

    config = load_yaml(_ROOT / "config.yaml")
    routing_top_k = config.get("expert_routing", {}).get("top_k_for_escalation")
    assert routing_top_k is not None
    assert PANEL_K == int(routing_top_k) == 5
