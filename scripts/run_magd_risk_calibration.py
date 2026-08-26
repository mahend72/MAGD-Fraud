from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.magd_risk_calibration import run_magd_risk_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline MAGD risk calibration analysis.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_magd_risk_calibration(args.config)
    print(f"Saved MAGD calibration rows: {len(artifacts.calibration_table)}")
    print(f"Saved outputs to: {artifacts.paper_tables_dir}")


if __name__ == "__main__":
    main()
