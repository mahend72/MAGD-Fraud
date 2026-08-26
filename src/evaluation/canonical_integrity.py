"""Canonical decision-hash integrity guard.

Computes a stable hash over the decision-relevant columns of each frozen decision log
in data/outputs/final_reproducible_run/. The saved hashes (canonical_decision_hashes.json,
in the same directory) let a regression test detect - after the fact, with no rerouting
or recomputation - whether any frozen decision log has silently changed (e.g. from an
accidental model refit, a routing-code change, or a stale/overwritten artifact), without
requiring the reviewer to diff multi-million-row CSVs by hand.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FINAL_RUN_DIR = ROOT / "data" / "outputs" / "final_reproducible_run"
CANONICAL_HASHES_PATH = FINAL_RUN_DIR / "canonical_decision_hashes.json"

# Columns common to every frozen decision log and sufficient to fully determine the
# case-level outcome that matters for downstream metrics: which route was taken, which
# expert(s) (if any), and what the final prediction was.
DECISION_HASH_COLUMNS = ["case_id", "selected_route", "selected_expert", "final_prediction"]

CANONICAL_DECISION_LOG_PATHS: dict[str, Path] = {
    "AI-only": FINAL_RUN_DIR / "decision_logs" / "ai_only_decisions.csv",
    "L2D-Standard": FINAL_RUN_DIR / "decision_logs" / "learning_to_defer_decisions.csv",
    "MAGD-v1-Heuristic": FINAL_RUN_DIR / "decision_logs" / "magd_heuristic_decisions.csv",
    "MAGD-v1-ValidationTuned": FINAL_RUN_DIR / "decision_logs" / "magd_validation_tuned_decisions.csv",
    "MAGD-v2": FINAL_RUN_DIR / "magd_v2_test_decisions.csv",
}


def compute_decision_summary_hash(decisions: pd.DataFrame, *, columns: list[str] = DECISION_HASH_COLUMNS) -> str:
    missing = set(columns) - set(decisions.columns)
    if missing:
        raise ValueError(f"decisions frame missing columns required for canonical hash: {sorted(missing)}")
    frame = decisions[columns].sort_values("case_id").reset_index(drop=True)
    frame = frame.fillna("NA").map(str)
    joined_rows = frame.agg("|".join, axis=1)
    payload = "\n".join(joined_rows.tolist())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_all_canonical_hashes(paths: dict[str, Path] = CANONICAL_DECISION_LOG_PATHS) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for method, path in paths.items():
        if not path.exists():
            continue
        decisions = pd.read_csv(path)
        hashes[method] = compute_decision_summary_hash(decisions)
    return hashes


def load_canonical_hashes(path: Path = CANONICAL_HASHES_PATH) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"No canonical decision-hash manifest at {path}. Run "
            "scripts/freeze_canonical_decision_hashes.py once against the frozen "
            "final_reproducible_run artifacts to create it."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def save_canonical_hashes(hashes: dict[str, str], path: Path = CANONICAL_HASHES_PATH) -> None:
    path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
