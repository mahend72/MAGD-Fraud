from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deferral.expert_routing import run_expert_routing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAGD-Fraud expert-aware routing.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_expert_routing(args.config)
    print(f"Saved expert reliability rows: {len(artifacts.expert_reliability)}")
    print(f"Saved routing decision rows: {len(artifacts.routing_decisions)}")
    print(f"Saved outputs to: {artifacts.output_dir}")


if __name__ == "__main__":
    main()
