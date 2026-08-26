from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.validate_splits import validate_data_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prepared train/validation/test splits.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = validate_data_splits(args.config)
    print(f"Validated splits from: {artifacts.split_data.processed_data_dir}")
    print(
        "Cases: "
        f"train={artifacts.statistics.train_cases}, "
        f"validation={artifacts.statistics.validation_cases}, "
        f"test={artifacts.statistics.test_cases}"
    )
    print(
        "Fraud prevalence: "
        f"train={artifacts.statistics.train_fraud_prevalence:.4f}, "
        f"validation={artifacts.statistics.validation_fraud_prevalence:.4f}, "
        f"test={artifacts.statistics.test_fraud_prevalence:.4f}"
    )
    print(f"Synthetic experts: {artifacts.statistics.n_experts}")
    print(f"Dataset summary written to: {artifacts.paper_tables_dir}")


if __name__ == "__main__":
    main()
