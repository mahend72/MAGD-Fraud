from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.make_paper_tables import generate_paper_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MAGD paper-ready tables and manifest.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_dir, generated = generate_paper_tables(args.config)
    print(f"Saved MAGD paper tables to: {paper_dir}")
    print(f"Generated tables: {len(generated)}")


if __name__ == "__main__":
    main()
