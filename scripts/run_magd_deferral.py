from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deferral.magd_deferral import run_magd_deferral


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAGD heuristic or learned deferral routing.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--mode", type=str, choices=["heuristic", "learned"], required=True, help="MAGD routing mode to run.")
    parser.add_argument("--policy-variant", type=str, default=None, help="Backward-compatible alias for the policy variant.")
    parser.add_argument("--output-stem", type=str, default=None, help="Optional output filename override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_magd_deferral(args.config, mode=args.mode, policy_variant=args.policy_variant, output_stem=args.output_stem)
    print(f"Saved MAGD {args.mode} decisions: {len(artifacts.decisions)}")
    print(f"Saved outputs to: {artifacts.output_dir}")


if __name__ == "__main__":
    main()
