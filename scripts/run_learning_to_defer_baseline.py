from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deferral.learning_to_defer_baseline import run_learning_to_defer_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the learning-to-defer baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_learning_to_defer_baseline(args.config)
    print(f"Saved learning-to-defer decisions: {len(artifacts.decisions)}")
    print(f"Saved outputs to: {artifacts.output_dir}")


if __name__ == "__main__":
    main()
