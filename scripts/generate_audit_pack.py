from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assurance.claim_evidence_matrix import (
    REQUIRED_EVIDENCE_COLUMNS,
    build_auditability_table,
    build_claim_evidence_matrix,
    build_decision_audit_log,
    markdown_table,
)
from src.evaluation.audit_metrics import compute_audit_metrics
from src.utils.reporting import (
    PLOT_FILES,
    build_assurance_summary,
    build_audit_coverage,
    build_audit_report_markdown,
    copy_artifact,
    ensure_exists,
    resolve_audit_pack_dirs,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate audit pack artifacts and report.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config, outputs_root, audit_pack_dir = resolve_audit_pack_dirs(config_path)
    paper_dir = outputs_root / "paper_tables"
    paper_dir.mkdir(parents=True, exist_ok=True)

    threshold_src = ensure_exists(
        outputs_root / "assurance" / "threshold_exploration_distance.csv",
        description="threshold exploration output",
    )
    final_metrics_src = ensure_exists(
        outputs_root / "final_metrics" / "all_method_metrics.csv",
        description="final metrics output",
    )
    reliance_metrics_src = ensure_exists(
        outputs_root / "final_metrics" / "reliance_metrics.csv",
        description="reliance metrics output",
    )
    deferral_metrics_src = ensure_exists(
        outputs_root / "final_metrics" / "deferral_metrics.csv",
        description="deferral metrics output",
    )
    audit_metrics_src = ensure_exists(
        outputs_root / "final_metrics" / "audit_metrics.csv",
        description="audit metrics output",
    )

    fairness_src = outputs_root / "final_metrics" / "fairness_metrics.csv"
    fairness_available = fairness_src.exists()

    decision_logs = build_decision_audit_log(config_path)
    final_metrics = pd.read_csv(final_metrics_src)
    reliance_metrics = pd.read_csv(reliance_metrics_src)
    deferral_metrics = pd.read_csv(deferral_metrics_src)
    audit_metrics = pd.read_csv(audit_metrics_src)
    fairness_metrics = pd.read_csv(fairness_src) if fairness_available else pd.DataFrame()

    decision_logs.to_csv(audit_pack_dir / "decision_audit_log.csv", index=False)
    decision_logs.to_csv(audit_pack_dir / "decision_logs.csv", index=False)

    assurance_summary = build_assurance_summary(decision_logs)
    assurance_summary.to_csv(audit_pack_dir / "assurance_summary.csv", index=False)

    claim_matrix = build_claim_evidence_matrix(decision_logs)
    claim_matrix.to_csv(audit_pack_dir / "claim_evidence_matrix.csv", index=False)
    (audit_pack_dir / "claim_evidence_matrix.md").write_text(
        markdown_table(claim_matrix.round({"coverage_score": 4})),
        encoding="utf-8",
    )

    auditability = build_auditability_table(decision_logs, claim_matrix)
    auditability.to_csv(paper_dir / "auditability.csv", index=False)

    copy_artifact(threshold_src, audit_pack_dir / "threshold_exploration.csv")
    copy_artifact(final_metrics_src, audit_pack_dir / "final_metrics.csv")

    if fairness_available:
        fairness_metrics.to_csv(audit_pack_dir / "fairness_summary.csv", index=False)

    fresh_audit_metrics = compute_audit_metrics(decision_logs, REQUIRED_EVIDENCE_COLUMNS)
    method_name = str(decision_logs["method"].iloc[0]) if not decision_logs.empty else "unknown"
    audit_metrics = pd.concat([audit_metrics, pd.DataFrame([{"method": method_name, **fresh_audit_metrics}])], ignore_index=True)

    audit_coverage = build_audit_coverage(
        decision_logs=decision_logs,
        audit_metrics=audit_metrics,
        fairness_available=fairness_available,
        capacity_available=bool(config.get("dataset", {}).get("capacity_file")),
    )
    write_json(audit_coverage, audit_pack_dir / "audit_coverage.json")

    plots_root = outputs_root / "plots"
    for plot_name in PLOT_FILES:
        plot_path = plots_root / plot_name
        if plot_path.exists():
            copy_artifact(plot_path, audit_pack_dir / "plots" / plot_name)

    report_text = build_audit_report_markdown(
        config=config,
        final_metrics=final_metrics,
        reliance_metrics=reliance_metrics,
        deferral_metrics=deferral_metrics,
        audit_coverage=audit_coverage,
        fairness_available=fairness_available,
    )
    (audit_pack_dir / "audit_report.md").write_text(report_text, encoding="utf-8")

    print(f"Audit pack generated at: {audit_pack_dir}")


if __name__ == "__main__":
    main()
