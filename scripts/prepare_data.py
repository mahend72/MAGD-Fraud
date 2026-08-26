from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import prepare_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FiFAR train/validation/test data.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = prepare_data(args.config)
    print(f"Processed data written to: {artifacts.processed_dir}")
    print(f"Model scores available: {artifacts.model_scores_available}")


if __name__ == "__main__":
    main()
