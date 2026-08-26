"""Generates the reviewer-workload / cost-sensitivity report from the canonical,
already-frozen metrics audit at data/outputs/final_reproducible_run/final_canonical_metrics_audit.csv.

Pure arithmetic only: ReviewerWorkload = N_expert + panel_k * N_escalation,
L_total = L_fraud + c_R * ReviewerWorkload. No case is rerouted, no model is refit, no
value is manually typed - everything is read from the canonical audit CSV that Step
"metric-integrity audit" already produced and saved.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.reviewer_workload import (
    DEFAULT_REVIEWER_COST_RANGE,
    PANEL_K,
    break_even_reviewer_cost,
    cost_sensitivity_table,
    summary_from_canonical_metrics_row,
)

FINAL_RUN_DIR = ROOT / "data" / "outputs" / "final_reproducible_run"
CANONICAL_METRICS_PATH = FINAL_RUN_DIR / "final_canonical_metrics_audit.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the reviewer-cost sensitivity report from the canonical final-run metrics audit.")
    parser.add_argument("--metrics-path", type=str, default=str(CANONICAL_METRICS_PATH), help="Path to final_canonical_metrics_audit.csv")
    parser.add_argument("--output-dir", type=str, default=str(FINAL_RUN_DIR), help="Directory to write reviewer_cost_sensitivity.csv into")
    return parser.parse_args()


def generate_reviewer_cost_sensitivity(metrics_path: str | Path, output_dir: str | Path) -> tuple[pd.DataFrame, float | None]:
    metrics_path = Path(metrics_path)
    output_dir = Path(output_dir)
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Canonical metrics audit not found at {metrics_path}. "
            "Reviewer-cost sensitivity must be derived only from the canonical "
            "final_reproducible_run artifact - it will not fall back to any other source."
        )
    metrics = pd.read_csv(metrics_path)
    required_columns = {"method", "n", "expert_deferral_rate", "escalation_rate", "reviewer_workload_cases", "cost_sensitive_loss"}
    missing = required_columns - set(metrics.columns)
    if missing:
        raise ValueError(f"final_canonical_metrics_audit.csv is missing required columns for reviewer-cost sensitivity: {sorted(missing)}")

    summaries = [summary_from_canonical_metrics_row(row) for _, row in metrics.iterrows()]
    table = cost_sensitivity_table(summaries, DEFAULT_REVIEWER_COST_RANGE)

    magd_v2 = next((s for s in summaries if s.method == "MAGD-v2"), None)
    l2d_standard = next((s for s in summaries if s.method == "L2D-Standard"), None)
    break_even = None
    if magd_v2 is not None and l2d_standard is not None:
        break_even = break_even_reviewer_cost(magd_v2, l2d_standard)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "reviewer_cost_sensitivity.csv"
    table.to_csv(out_path, index=False)

    summary_path = output_dir / "reviewer_cost_sensitivity_summary.txt"
    lines = [
        f"panel_k = {PANEL_K}",
        f"reviewer cost range evaluated (c_R): {list(DEFAULT_REVIEWER_COST_RANGE)}",
        "",
    ]
    if magd_v2 is not None and l2d_standard is not None:
        lines.append(f"MAGD-v2: N_expert={magd_v2.n_expert}, N_escalation={magd_v2.n_escalation}, ReviewerWorkload={magd_v2.reviewer_workload}, L_fraud={magd_v2.fraud_loss:.3f}")
        lines.append(f"L2D-Standard: N_expert={l2d_standard.n_expert}, N_escalation={l2d_standard.n_escalation}, ReviewerWorkload={l2d_standard.reviewer_workload}, L_fraud={l2d_standard.fraud_loss:.3f}")
        if break_even is None:
            lines.append("Break-even reviewer cost: none (identical reviewer workload; no finite break-even c_R exists).")
        elif break_even < 0:
            lines.append(
                f"Analytic break-even c_R = {break_even:.6f} (negative). MAGD-v2 has both lower fraud "
                "loss and higher reviewer workload than L2D-Standard, so there is no positive reviewer "
                "cost at which L2D-Standard becomes cheaper than a MAGD-v2 that ALSO wins on fraud loss "
                "at c_R=0 - re-check sign/direction before reporting."
            )
        else:
            lines.append(f"Break-even reviewer cost c_R* = {break_even:.6f}")
            lines.append(f"  For c_R < {break_even:.6f}: MAGD-v2 has lower total cost.")
            lines.append(f"  For c_R > {break_even:.6f}: L2D-Standard has lower total cost.")
    else:
        lines.append("MAGD-v2 and/or L2D-Standard rows not found in canonical metrics audit; break-even not computed.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return table, break_even


def main() -> None:
    args = parse_args()
    table, break_even = generate_reviewer_cost_sensitivity(args.metrics_path, args.output_dir)
    print(table.to_string(index=False))
    if break_even is not None:
        print(f"\nBreak-even reviewer cost (MAGD-v2 vs L2D-Standard): c_R* = {break_even:.6f}")
    else:
        print("\nNo finite break-even reviewer cost between MAGD-v2 and L2D-Standard.")


if __name__ == "__main__":
    main()
