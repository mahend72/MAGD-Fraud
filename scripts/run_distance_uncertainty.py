from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assurance.distance_uncertainty import run_distance_uncertainty


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run distance-based uncertainty assurance for FiFAR.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_distance_uncertainty(args.config)
    print(f"Saved distance uncertainty rows: {len(artifacts.distance_frame)}")
    print(f"Saved threshold exploration rows: {len(artifacts.threshold_metrics)}")


if __name__ == "__main__":
    main()
