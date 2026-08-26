from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.assurance.claim_evidence_matrix import build_auditability_table, build_decision_audit_log
from src.utils.io import load_yaml


EXPECTED_TABLES = [
    "dataset_summary",
    "ai_assurance",
    "baseline_comparison",
    "intervention_calibrated_results",
    "human_ai_metrics",
    "magd_risk_calibration",
    "ablation",
    "auditability",
    "statistical_comparison",
    "implementation_structure",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate canonical paper-ready summary tables and manifest.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def _resolve_roots(config_path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", config.get("experiment", {}).get("output_dir", "data/outputs")))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    processed_root = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_root.is_absolute():
        processed_root = (config_path.parent / processed_root).resolve()
    paper_dir = outputs_root / "paper_tables"
    paper_dir.mkdir(parents=True, exist_ok=True)
    return config, outputs_root, processed_root, paper_dir


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.fillna("NA")
    header = "| " + " | ".join(display.columns.astype(str)) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows]) + "\n"


def _for_export(frame: pd.DataFrame) -> pd.DataFrame:
    exported = frame.copy()
    for column in exported.columns:
        if pd.api.types.is_float_dtype(exported[column]):
            exported[column] = exported[column].round(4)
    return exported.fillna("NA")


def _save_table(frame: pd.DataFrame, paper_dir: Path, name: str) -> None:
    exported = _for_export(frame)
    exported.to_csv(paper_dir / f"{name}.csv", index=False)
    (paper_dir / f"{name}.md").write_text(_markdown_table(exported), encoding="utf-8")


def _load_required_csv(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return pd.read_csv(path)


def _load_or_placeholder(path: Path, *, description: str, placeholder_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if path.exists():
        return pd.read_csv(path), []
    return pd.DataFrame([{column: "NA" for column in placeholder_columns}]), [f"Missing optional source: {description} ({path.name})"]


def _infer_methods(frame: pd.DataFrame) -> list[str]:
    for column in ["method", "variant", "split", "stage", "item", "component"]:
        if column in frame.columns:
            values = frame[column].dropna().astype(str).tolist()
            return sorted(dict.fromkeys(values))
    return []


def _missing_value_summary(frame: pd.DataFrame) -> dict[str, int]:
    return {column: int(frame[column].isna().sum()) for column in frame.columns if int(frame[column].isna().sum()) > 0}


def _build_dataset_summary(config: dict[str, Any], outputs_root: Path, processed_root: Path) -> tuple[pd.DataFrame, list[Path], list[str]]:
    paper_path = outputs_root / "paper_tables" / "dataset_summary.csv"
    if paper_path.exists():
        return pd.read_csv(paper_path), [paper_path], []

    y_train = _load_required_csv(processed_root / "y_train.csv", "y_train split")
    y_val = _load_required_csv(processed_root / "y_val.csv", "y_val split")
    y_test = _load_required_csv(processed_root / "y_test.csv", "y_test split")
    label_col = y_train.columns[0]
    experts_path = processed_root / "expert_predictions.csv"
    capacity_enabled = bool(config.get("expert_routing", {}).get("capacity_enabled", False))
    n_experts = 0
    warnings: list[str] = []
    if experts_path.exists():
        expert_predictions = pd.read_csv(experts_path)
        n_experts = max(0, len(expert_predictions.columns) - 1)
    else:
        warnings.append("Expert predictions file missing while building dataset summary.")
    frame = pd.DataFrame(
        [
            {
                "total_cases": int(len(y_train) + len(y_val) + len(y_test)),
                "train_cases": int(len(y_train)),
                "validation_cases": int(len(y_val)),
                "test_cases": int(len(y_test)),
                "train_fraud_prevalence": float(pd.to_numeric(y_train[label_col], errors="coerce").mean()),
                "validation_fraud_prevalence": float(pd.to_numeric(y_val[label_col], errors="coerce").mean()),
                "test_fraud_prevalence": float(pd.to_numeric(y_test[label_col], errors="coerce").mean()),
                "number_of_synthetic_experts": int(n_experts),
                "sensitive_attribute_used": str(config.get("data", {}).get("sensitive_attribute", "NA")),
                "capacity_configured": "yes" if capacity_enabled else "no",
            }
        ]
    )
    return frame, [processed_root / "y_train.csv", processed_root / "y_val.csv", processed_root / "y_test.csv"], warnings


def _build_ai_assurance(outputs_root: Path) -> tuple[pd.DataFrame, list[Path], list[str]]:
    path = outputs_root / "paper_tables" / "ai_assurance.csv"
    frame, warnings = _load_or_placeholder(
        path,
        description="AI assurance table",
        placeholder_columns=["split", "pr_auc", "f1", "expected_calibration_error", "mean_numerical_confidence", "mean_calibration_risk"],
    )
    return frame, [path], warnings


def _build_baseline_comparison(outputs_root: Path) -> tuple[pd.DataFrame, list[Path], list[str]]:
    path = outputs_root / "paper_tables" / "baseline_comparison.csv"
    frame, warnings = _load_or_placeholder(
        path,
        description="baseline comparison table",
        placeholder_columns=["method", "precision", "recall", "f1", "cost_sensitive_loss", "ai_coverage", "expert_deferral_rate", "escalation_rate"],
    )
    return frame, [path], warnings


def _build_intervention_results(outputs_root: Path) -> tuple[pd.DataFrame, list[Path], list[str]]:
    path = outputs_root / "paper_tables" / "intervention_calibrated_results.csv"
    frame, warnings = _load_or_placeholder(
        path,
        description="intervention-calibrated results table",
        placeholder_columns=["stage", "phase", "fraud_loss", "overreliance", "wrong_confident_avoidance", "deferral_rate", "audit_coverage", "feasible"],
    )
    return frame, [path], warnings


def _build_human_ai_metrics(outputs_root: Path) -> tuple[pd.DataFrame, list[Path], list[str]]:
    path = outputs_root / "paper_tables" / "human_ai_metrics.csv"
    frame, warnings = _load_or_placeholder(
        path,
        description="human-AI metrics table",
        placeholder_columns=["method", "precision", "recall", "f1", "cost_sensitive_loss", "overreliance", "underreliance", "audit_coverage"],
    )
    return frame, [path], warnings


def _build_magd_risk_calibration(outputs_root: Path) -> tuple[pd.DataFrame, list[Path], list[str]]:
    path = outputs_root / "paper_tables" / "magd_risk_calibration.csv"
    frame, warnings = _load_or_placeholder(
        path,
        description="MAGD risk calibration table",
        placeholder_columns=["magd_risk_bin", "cases", "ai_error_rate", "wrong_confident_rate", "deferral_rate"],
    )
    return frame, [path], warnings


def _build_ablation(outputs_root: Path) -> tuple[pd.DataFrame, list[Path], list[str]]:
    path = outputs_root / "paper_tables" / "ablation.csv"
    frame, warnings = _load_or_placeholder(
        path,
        description="ablation table",
        placeholder_columns=["variant", "status", "signals_used", "f1", "cost_sensitive_loss", "overreliance", "audit_coverage"],
    )
    return frame, [path], warnings


def _build_auditability(config_path: Path, outputs_root: Path) -> tuple[pd.DataFrame, list[Path], list[str]]:
    path = outputs_root / "paper_tables" / "auditability.csv"
    if path.exists():
        return pd.read_csv(path), [path], []
    audit_log = build_decision_audit_log(config_path)
    claim_matrix_path = outputs_root / "audit_pack" / "claim_evidence_matrix.csv"
    claim_matrix = _load_required_csv(claim_matrix_path, "claim-evidence matrix")
    frame = build_auditability_table(audit_log, claim_matrix)
    return frame, [claim_matrix_path], ["Auditability table rebuilt from audit pack because canonical CSV was missing."]


def _build_statistical_comparison(outputs_root: Path) -> tuple[pd.DataFrame, list[Path], list[str]]:
    path = outputs_root / "paper_tables" / "statistical_comparison.csv"
    frame, warnings = _load_or_placeholder(
        path,
        description="statistical comparison table",
        placeholder_columns=["method_A", "method_B", "metric", "test", "statistic", "p_value", "ci_low", "ci_high", "interpretation"],
    )
    return frame, [path], warnings


def _build_implementation_structure() -> tuple[pd.DataFrame, list[Path], list[str]]:
    report_path = ROOT / "docs" / "implementation_coverage_report.md"
    if not report_path.exists():
        frame = pd.DataFrame([{"component": "NA", "status": "NA", "notes": "Implementation coverage report missing."}])
        return frame, [report_path], ["Implementation coverage report missing; implementation_structure generated as placeholder."]

    rows: list[dict[str, str]] = []
    pattern = re.compile(r"^\| (.+?) \| (IMPLEMENTED|PARTIALLY IMPLEMENTED|MISSING) \| (.+?) \|$")
    for line in report_path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            rows.append(
                {
                    "component": match.group(1).strip(),
                    "status": match.group(2).strip(),
                    "notes": match.group(3).strip(),
                }
            )
    if not rows:
        rows.append({"component": "NA", "status": "NA", "notes": "No implementation rows parsed from report."})
        warnings = ["Could not parse implementation coverage rows from implementation_coverage_report.md."]
    else:
        warnings = []
    return pd.DataFrame(rows), [report_path], warnings


def generate_paper_tables(config_path: str | Path) -> tuple[Path, dict[str, pd.DataFrame]]:
    resolved = Path(config_path).resolve()
    config, outputs_root, processed_root, paper_dir = _resolve_roots(resolved)

    builders = {
        "dataset_summary": lambda: _build_dataset_summary(config, outputs_root, processed_root),
        "ai_assurance": lambda: _build_ai_assurance(outputs_root),
        "baseline_comparison": lambda: _build_baseline_comparison(outputs_root),
        "intervention_calibrated_results": lambda: _build_intervention_results(outputs_root),
        "human_ai_metrics": lambda: _build_human_ai_metrics(outputs_root),
        "magd_risk_calibration": lambda: _build_magd_risk_calibration(outputs_root),
        "ablation": lambda: _build_ablation(outputs_root),
        "auditability": lambda: _build_auditability(resolved, outputs_root),
        "statistical_comparison": lambda: _build_statistical_comparison(outputs_root),
        "implementation_structure": _build_implementation_structure,
    }

    manifest_entries: list[dict[str, Any]] = []
    generated_frames: dict[str, pd.DataFrame] = {}
    timestamp = datetime.now(timezone.utc).isoformat()

    for name in EXPECTED_TABLES:
        frame, sources, warnings = builders[name]()
        if frame.empty:
            raise ValueError(f"Generated paper table `{name}` is empty.")
        generated_frames[name] = frame
        _save_table(frame, paper_dir, name)
        manifest_entries.append(
            {
                "table_name": name,
                "source_files": [str(source) for source in sources],
                "generated_timestamp": timestamp,
                "methods_included": _infer_methods(frame),
                "missing_values": _missing_value_summary(frame),
                "warnings": warnings,
            }
        )

    manifest_path = paper_dir / "table_manifest.json"
    manifest_path.write_text(json.dumps(manifest_entries, indent=2), encoding="utf-8")
    return paper_dir, generated_frames


def main() -> None:
    args = parse_args()
    paper_dir, generated = generate_paper_tables(args.config)
    print(f"Saved paper tables to: {paper_dir}")
    print(f"Generated tables: {len(generated)}")


if __name__ == "__main__":
    main()
