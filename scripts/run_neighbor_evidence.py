from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assurance.explanation_neighbors import run_neighbor_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nearest-neighbor evidence generation for FiFAR.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_neighbor_evidence(args.config)
    print(f"Saved neighbor evidence rows: {len(artifacts.neighbor_evidence_long)}")
    print(f"Saved neighbor summary rows: {len(artifacts.neighbor_summary)}")


if __name__ == "__main__":
    main()
