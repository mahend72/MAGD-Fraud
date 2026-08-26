from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deferral.magd_constrained import run_magd_constrained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run intervention-calibrated MAGD constrained routing.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_magd_constrained(args.config)
    print(f"Saved MAGD constrained calibrated decisions: {len(artifacts.calibrated_decisions)}")
    print(f"Saved outputs to: {artifacts.output_dir}")


if __name__ == "__main__":
    main()
