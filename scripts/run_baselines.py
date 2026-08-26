from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deferral.baselines import run_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deferral baselines.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_baselines(args.config)
    print(f"Saved baseline outputs to: {artifacts.output_dir}")
    print(f"Saved metrics rows: {len(artifacts.baseline_metrics)}")


if __name__ == "__main__":
    main()
