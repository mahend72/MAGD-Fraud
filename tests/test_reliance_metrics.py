from __future__ import annotations

import math

import pandas as pd
import pytest

from src.evaluation.reliance_metrics import compute_reliance_metrics


def test_reliance_metrics_match_expected_rates() -> None:
    frame = pd.DataFrame(
        {
            "used_ai": [1, 1, 0, 0],
            "ai_correct": [1, 0, 0, 1],
            "wrong_confident_label_offline": [0, 1, 0, 0],
        }
    )
    metrics = compute_reliance_metrics(frame)

    assert math.isclose(metrics["correct_reliance"], 0.25)
    assert math.isclose(metrics["correct_rejection"], 0.25)
    assert math.isclose(metrics["overreliance"], 0.25)
    assert math.isclose(metrics["underreliance"], 0.25)
    assert math.isclose(metrics["wrong_confident_avoidance_rate"], 0.0)
    # Backward-compatible aliases must be byte-identical to the unconditional values -
    # nothing that reads `overreliance`/`correct_rejection` by name (e.g. MAGD-Learned/
    # Constrained's optimization objective) is affected by this change.
    assert metrics["correct_rejection_all_cases"] == metrics["correct_rejection"]
    assert metrics["overreliance_all_cases"] == metrics["overreliance"]


def test_conditional_correct_rejection_and_overreliance_sum_to_one_when_ai_errors_exist() -> None:
    frame = pd.DataFrame(
        {
            "used_ai": [1, 1, 0, 0, 1, 0],
            "ai_correct": [1, 0, 0, 1, 0, 0],  # AI-wrong rows: index 1 (used_ai=1), 2 (used_ai=0), 4 (used_ai=1), 5 (used_ai=0)
        }
    )
    metrics = compute_reliance_metrics(frame)
    total = metrics["correct_rejection_given_ai_wrong"] + metrics["overreliance_given_ai_wrong"]
    assert total == pytest.approx(1.0)
    # 4 AI-wrong rows: 2 used AI (overrelied), 2 did not (correctly rejected).
    assert metrics["overreliance_given_ai_wrong"] == pytest.approx(0.5)
    assert metrics["correct_rejection_given_ai_wrong"] == pytest.approx(0.5)


def test_conditional_metrics_are_zero_when_no_ai_errors_exist() -> None:
    frame = pd.DataFrame({"used_ai": [1, 0, 1], "ai_correct": [1, 1, 1]})
    metrics = compute_reliance_metrics(frame)
    assert metrics["correct_rejection_given_ai_wrong"] == 0.0
    assert metrics["overreliance_given_ai_wrong"] == 0.0


def test_conditional_metrics_differ_from_unconditional_ones_in_general() -> None:
    # A case where the two definitions give materially different numbers, to guard
    # against them accidentally being computed identically.
    frame = pd.DataFrame(
        {
            "used_ai": [1] * 9 + [0],
            "ai_correct": [1] * 8 + [0, 0],  # only 2 AI-wrong rows out of 10
        }
    )
    metrics = compute_reliance_metrics(frame)
    # Unconditional overreliance: 1 of 10 rows (used_ai & wrong) = 0.1
    assert metrics["overreliance"] == pytest.approx(0.1)
    # Conditional overreliance: of the 2 AI-wrong rows, 1 used AI anyway = 0.5
    assert metrics["overreliance_given_ai_wrong"] == pytest.approx(0.5)
    assert metrics["overreliance"] != metrics["overreliance_given_ai_wrong"]


def test_wca_raises_when_required_and_label_missing() -> None:
    frame = pd.DataFrame({"used_ai": [1, 0], "ai_correct": [1, 0]})
    with pytest.raises(ValueError, match="wrong_confident_label_offline"):
        compute_reliance_metrics(frame, require_wrong_confident_label=True)


def test_wca_does_not_raise_when_required_and_label_present() -> None:
    frame = pd.DataFrame({"used_ai": [1, 0], "ai_correct": [1, 0], "wrong_confident_label_offline": [0, 1]})
    metrics = compute_reliance_metrics(frame, require_wrong_confident_label=True)
    assert metrics["wrong_confident_avoidance_rate"] == pytest.approx(1.0)


def test_wca_default_still_silently_defaults_to_zero_for_backward_compatibility() -> None:
    """Existing callers (MAGD-Constrained's feasibility search, MAGD-Learned/
    Constrained weight learning, MAGD-ValidationTuned's budget search) never merge
    the offline label and must keep working exactly as before - changing their
    behaviour risks altering already-frozen v1 routing/threshold selection, which is
    out of scope for this hardening pass."""
    frame = pd.DataFrame({"used_ai": [1, 0], "ai_correct": [1, 0]})
    metrics = compute_reliance_metrics(frame)  # require_wrong_confident_label defaults to False
    assert metrics["wrong_confident_avoidance_rate"] == 0.0
