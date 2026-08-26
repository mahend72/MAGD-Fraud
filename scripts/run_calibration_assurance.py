from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assurance.calibration import run_calibration_assurance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run numerical confidence and calibration assurance.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_calibration_assurance(args.config)
    print(f"Validation ECE: {artifacts.validation_ece:.6f}")
    print(f"Saved assurance outputs to: {artifacts.assurance_dir}")
    print(f"Saved paper table to: {artifacts.paper_tables_dir / 'ai_assurance.csv'}")


if __name__ == "__main__":
    main()
