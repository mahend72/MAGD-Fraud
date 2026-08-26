from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assurance.claim_evidence_matrix import run_claim_evidence_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the MAGD claim-evidence matrix.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_claim_evidence_matrix(args.config)
    print(f"Saved claim-evidence rows: {len(artifacts.matrix)}")
    print(f"Saved outputs to: {artifacts.output_dir}")


if __name__ == "__main__":
    main()
