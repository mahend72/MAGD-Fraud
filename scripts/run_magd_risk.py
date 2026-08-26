from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assurance.magd_risk import run_magd_risk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run composite MAGD assurance risk scoring.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_magd_risk(args.config)
    print(f"Saved MAGD risk rows: {len(artifacts.magd_frame)}")
    print(f"Saved outputs to: {artifacts.assurance_dir}")


if __name__ == "__main__":
    main()
