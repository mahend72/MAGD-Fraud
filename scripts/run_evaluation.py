from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_all_methods import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complete Human-AI assurance evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_dir, method_count = run_evaluation(args.config)
    print(f"Saved outputs to: {final_dir}")
    print(f"Methods evaluated: {method_count}")


if __name__ == "__main__":
    main()
