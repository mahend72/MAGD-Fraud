from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deferral.adaptive_threshold import run_adaptive_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAGD-Fraud adaptive thresholding.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_adaptive_threshold(args.config)
    print(f"Saved adaptive threshold rows: {len(artifacts.thresholds)}")
    print(f"Saved outputs to: {artifacts.assurance_dir}")


if __name__ == "__main__":
    main()
