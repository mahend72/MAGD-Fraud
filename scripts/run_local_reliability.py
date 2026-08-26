from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assurance.local_reliability import run_local_reliability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local neighbourhood reliability for FiFAR.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_local_reliability(args.config)
    print(f"Saved local reliability rows: {len(artifacts.local_reliability)}")
    print(f"Saved outputs to: {artifacts.assurance_dir}")


if __name__ == "__main__":
    main()
