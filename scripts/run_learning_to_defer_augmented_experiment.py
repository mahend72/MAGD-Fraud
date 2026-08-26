"""Runs L2D+MAGD (augmented) as an explicitly-labeled OPTIONAL experiment.

This is NOT the official baseline - see docs/... for the baseline-independence audit.
The official, MAGD-independent baseline is L2D-Standard, produced by
scripts/run_learning_to_defer_baseline.py (feature_set="standard", the default).
This script exists only so the augmented variant can still be regenerated on demand,
written to its own clearly-suffixed output files
(learning_to_defer_augmented_decisions.csv / _metrics.csv), never the canonical name.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deferral.learning_to_defer_baseline import run_learning_to_defer_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the OPTIONAL L2D+MAGD (augmented) experiment - not the official baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_learning_to_defer_baseline(args.config, feature_set="augmented")
    print(f"Saved L2D+MAGD (augmented, OPTIONAL experiment) decisions: {len(artifacts.decisions)}")
    print(f"Saved outputs to: {artifacts.output_dir}")
    print("Reminder: this is NOT the official baseline. Use L2D-Standard (run_learning_to_defer_baseline.py) for baseline comparisons.")


if __name__ == "__main__":
    main()
