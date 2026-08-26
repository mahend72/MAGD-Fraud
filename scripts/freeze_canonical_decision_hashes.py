"""One-time (or deliberately re-run) utility: computes and saves the canonical
decision-summary hashes for every frozen decision log in
data/outputs/final_reproducible_run/. Run this ONLY when the frozen artifacts
themselves are intentionally regenerated (e.g. after an authorized methodology
change) - running it after an accidental change would just re-baseline the guard
against the accident, defeating its purpose.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.canonical_integrity import (
    CANONICAL_HASHES_PATH,
    compute_all_canonical_hashes,
    save_canonical_hashes,
)


def main() -> None:
    hashes = compute_all_canonical_hashes()
    if not hashes:
        raise SystemExit("No canonical decision logs found under data/outputs/final_reproducible_run/ - nothing to hash.")
    save_canonical_hashes(hashes)
    print(f"Saved canonical decision hashes for {len(hashes)} methods to: {CANONICAL_HASHES_PATH}")
    for method, digest in hashes.items():
        print(f"  {method}: {digest}")


if __name__ == "__main__":
    main()
