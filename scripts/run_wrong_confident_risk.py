from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assurance.wrong_confident_detector import run_wrong_confident_detection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deployable wrong-confident AI risk scoring.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_wrong_confident_detection(args.config)
    for key in ["precision", "recall", "f1", "top_k_capture_rate"]:
        print(f"{key}: {artifacts.metrics[key]}")


if __name__ == "__main__":
    main()
