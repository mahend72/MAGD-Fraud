from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.train_model import run_model_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run base fraud model predictions.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_model_predictions(args.config)
    print(f"Model source: {artifacts.model_source}")
    print(f"Model name: {artifacts.model_name}")
    print(f"Saved outputs to: {artifacts.model_dir}")


if __name__ == "__main__":
    main()
