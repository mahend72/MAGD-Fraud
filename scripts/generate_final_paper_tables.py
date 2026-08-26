"""Generates the final paper-ready tables SOLELY from data/outputs/final_reproducible_run/.

This script performs presentation-only transforms (column renaming/reordering,
rounding for display) of values that are already computed and saved in the canonical
audit artifacts. It never computes a metric independently and never accepts a
manually-typed number - every value in the output tables traces back to exactly one
cell in final_canonical_metrics_audit.csv, final_canonical_statistics_audit.csv, or
reviewer_cost_sensitivity.csv. tests/test_final_paper_tables.py asserts this by
re-reading the canonical sources and comparing them against the generated tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_reviewer_cost_sensitivity import generate_reviewer_cost_sensitivity

FINAL_RUN_DIR = ROOT / "data" / "outputs" / "final_reproducible_run"
PAPER_TABLES_DIR = FINAL_RUN_DIR / "paper_tables"

CANONICAL_METRICS_PATH = FINAL_RUN_DIR / "final_canonical_metrics_audit.csv"
CANONICAL_STATISTICS_PATH = FINAL_RUN_DIR / "final_canonical_statistics_audit.csv"
REVIEWER_COST_PATH = FINAL_RUN_DIR / "reviewer_cost_sensitivity.csv"

MAIN_RESULTS_DISPLAY_COLUMNS = {
    "method": "Method",
    "n": "N",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "cost_sensitive_loss": "Cost-Sensitive Loss",
    "wca": "Wrong-Confident Avoidance",
    "correct_rejection": "Correct Rejection (all cases)",
    "overreliance": "Overreliance (all cases)",
    "ai_coverage": "AI Coverage",
    "expert_deferral_rate": "Expert Deferral Rate",
    "escalation_rate": "Escalation Rate",
    "reviewer_workload_cases": "Reviewer Workload (cases)",
}

STATISTICS_DISPLAY_COLUMNS = {
    "comparison": "Comparison",
    "n_paired": "N (paired)",
    "delta_f1": "ΔF1",
    "delta_f1_ci_low": "ΔF1 CI low",
    "delta_f1_ci_high": "ΔF1 CI high",
    "delta_cost": "ΔCost",
    "delta_cost_ci_low": "ΔCost CI low",
    "delta_cost_ci_high": "ΔCost CI high",
    "mcnemar_p": "McNemar p",
}


def _round_floats(frame: pd.DataFrame, ndigits: int = 4) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].round(ndigits)
    return out


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows]) + "\n"


def _load_required(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing canonical source for final paper tables: {description} at {path}. "
            "Final paper tables are generated ONLY from data/outputs/final_reproducible_run/ "
            "and will not fall back to any other location or a manually-typed value."
        )
    return pd.read_csv(path)


def generate_final_paper_tables() -> dict[str, pd.DataFrame]:
    metrics = _load_required(CANONICAL_METRICS_PATH, "final_canonical_metrics_audit.csv")
    statistics = _load_required(CANONICAL_STATISTICS_PATH, "final_canonical_statistics_audit.csv")
    if not REVIEWER_COST_PATH.exists():
        generate_reviewer_cost_sensitivity(CANONICAL_METRICS_PATH, FINAL_RUN_DIR)
    reviewer_cost = _load_required(REVIEWER_COST_PATH, "reviewer_cost_sensitivity.csv")

    missing_metric_cols = set(MAIN_RESULTS_DISPLAY_COLUMNS) - set(metrics.columns)
    if missing_metric_cols:
        raise ValueError(f"final_canonical_metrics_audit.csv missing expected columns: {sorted(missing_metric_cols)}")
    missing_stat_cols = set(STATISTICS_DISPLAY_COLUMNS) - set(statistics.columns)
    if missing_stat_cols:
        raise ValueError(f"final_canonical_statistics_audit.csv missing expected columns: {sorted(missing_stat_cols)}")

    main_results = _round_floats(metrics[list(MAIN_RESULTS_DISPLAY_COLUMNS)]).rename(columns=MAIN_RESULTS_DISPLAY_COLUMNS)
    statistical_comparison = _round_floats(statistics[list(STATISTICS_DISPLAY_COLUMNS)]).rename(columns=STATISTICS_DISPLAY_COLUMNS)
    reviewer_cost_table = _round_floats(reviewer_cost)

    PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "main_results": main_results,
        "statistical_comparison": statistical_comparison,
        "reviewer_cost_sensitivity": reviewer_cost_table,
    }
    for name, frame in tables.items():
        frame.to_csv(PAPER_TABLES_DIR / f"{name}.csv", index=False)
        (PAPER_TABLES_DIR / f"{name}.md").write_text(_markdown_table(frame), encoding="utf-8")
    return tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final paper tables from data/outputs/final_reproducible_run/ only.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    tables = generate_final_paper_tables()
    print(f"Saved final paper tables to: {PAPER_TABLES_DIR}")
    for name, frame in tables.items():
        print(f"  {name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
