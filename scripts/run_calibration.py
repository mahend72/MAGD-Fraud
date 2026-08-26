from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.calibrate_model import run_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run calibration and confidence assurance checks.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_calibration(args.config)
    print(f"Calibration ECE: {artifacts.ece:.6f}")
    print(f"Saved assurance outputs to: {artifacts.assurance_dir}")
    print(f"Saved plots to: {artifacts.plots_dir}")


if __name__ == "__main__":
    main()
