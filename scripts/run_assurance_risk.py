from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assurance.assurance_risk import run_assurance_risk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final assurance risk aggregation.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_assurance_risk(args.config)
    print(f"Saved assurance risk rows: {len(artifacts.assurance_frame)}")
    print(f"Saved outputs to: {artifacts.assurance_dir}")
    print(f"Saved plots to: {artifacts.plots_dir}")


if __name__ == "__main__":
    main()
