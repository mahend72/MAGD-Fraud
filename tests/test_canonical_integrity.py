from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.canonical_integrity import (
    CANONICAL_DECISION_LOG_PATHS,
    CANONICAL_HASHES_PATH,
    compute_all_canonical_hashes,
    compute_decision_summary_hash,
    load_canonical_hashes,
)

pytestmark = pytest.mark.skipif(
    not CANONICAL_HASHES_PATH.exists(),
    reason="canonical_decision_hashes.json not present - run scripts/freeze_canonical_decision_hashes.py once",
)


def test_compute_decision_summary_hash_is_order_independent() -> None:
    decisions = pd.DataFrame(
        {
            "case_id": ["c1", "c2", "c3"],
            "selected_route": ["AI", "Human Expert", "Escalate"],
            "selected_expert": [None, "expert_a", "expert_a|expert_b"],
            "final_prediction": [0, 1, 1],
        }
    )
    shuffled = decisions.sample(frac=1.0, random_state=0).reset_index(drop=True)
    assert compute_decision_summary_hash(decisions) == compute_decision_summary_hash(shuffled)


def test_compute_decision_summary_hash_changes_if_a_route_changes() -> None:
    decisions = pd.DataFrame(
        {"case_id": ["c1", "c2"], "selected_route": ["AI", "AI"], "selected_expert": [None, None], "final_prediction": [0, 1]}
    )
    changed = decisions.copy()
    changed.loc[0, "selected_route"] = "Human Expert"
    assert compute_decision_summary_hash(decisions) != compute_decision_summary_hash(changed)


def test_all_canonical_decision_logs_match_saved_hashes() -> None:
    """The core regression guard: recompute the hash of every frozen decision log
    under data/outputs/final_reproducible_run/ right now and compare against the saved
    canonical_decision_hashes.json. A mismatch means a decision log changed (model
    refit, routing-code change, or an overwritten/stale artifact) since the hashes were
    last frozen with scripts/freeze_canonical_decision_hashes.py - it must not pass
    silently."""
    saved = load_canonical_hashes()
    current = compute_all_canonical_hashes()
    assert current == saved, (
        "One or more frozen decision logs no longer match the saved canonical hash. "
        "If this change was authorized and intentional, re-run "
        "scripts/freeze_canonical_decision_hashes.py to re-baseline; otherwise this is "
        "an unauthorized change to MAGD-v2 (or a baseline's) predictions/routing."
    )


def test_canonical_hash_manifest_covers_all_five_main_methods() -> None:
    saved = load_canonical_hashes()
    assert set(saved) == set(CANONICAL_DECISION_LOG_PATHS)
