from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.inspect_data import run_inspection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect FiFAR dataset files.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_inspection(args.config)


if __name__ == "__main__":
    main()
