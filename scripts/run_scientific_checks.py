from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.scientific_checks import ScientificCheckError, run_scientific_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scientific guardrail checks and write structured outputs.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = run_scientific_checks(args.config)
    except ScientificCheckError as exc:
        print(str(exc))
        raise
    print(f"Scientific checks status: {payload['status']}")
    print(f"Saved JSON to: {Path(payload['outputs_root']) / 'final_metrics' / 'scientific_checks.json'}")


if __name__ == "__main__":
    main()
