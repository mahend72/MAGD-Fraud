from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.statistical_tests import build_statistical_test_rows
from src.utils.io import load_yaml


COMPARISONS: list[tuple[str, str]] = [
    ("AI-only", "distance-threshold"),
    ("distance-threshold", "MAGD-Constrained initial"),
    ("MAGD-Constrained initial", "MAGD-Constrained intervention-calibrated"),
    ("learning-to-defer", "MAGD-Constrained intervention-calibrated"),
    ("distance-threshold", "MAGD-Constrained intervention-calibrated"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run statistical comparisons between Human-AI routing methods.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def _paths(config_path: Path) -> tuple[Path, Path, Path]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    final_dir = outputs_root / "final_metrics"
    paper_dir = outputs_root / "paper_tables"
    final_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    return outputs_root, final_dir, paper_dir


def _find_existing_path(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_method_paths(outputs_root: Path) -> dict[str, Path]:
    mapping = {
        "AI-only": outputs_root / "baselines" / "ai_only_decisions.csv",
        "distance-threshold": outputs_root / "baselines" / "distance_threshold_decisions.csv",
        "MAGD-Constrained initial": _find_existing_path(
            outputs_root / "assurance_deferral" / "magd_constrained_initial_decisions.csv",
            outputs_root / "ablations" / "ablation_decisions_full_magd_constrained_initial.csv",
        ),
        "MAGD-Constrained intervention-calibrated": _find_existing_path(
            outputs_root / "assurance_deferral" / "magd_constrained_calibrated_decisions.csv",
            outputs_root / "assurance_deferral" / "magd_constrained_decisions.csv",
            outputs_root / "ablations" / "ablation_decisions_full_magd_constrained_intervention_calibrated.csv",
        ),
        "learning-to-defer": _find_existing_path(
            outputs_root / "baselines" / "learning_to_defer_decisions.csv",
            outputs_root / "assurance_deferral" / "learning_to_defer_decisions.csv",
        ),
    }
    missing = [method for method, path in mapping.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required method outputs for statistical comparisons: "
            + ", ".join(missing)
        )
    return {method: path for method, path in mapping.items() if path is not None}


def _core_join_tables(outputs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    assurance = pd.read_csv(outputs_root / "assurance" / "wrong_confident_risk.csv")
    assurance["case_id"] = assurance["case_id"].astype(str)
    magd_path = outputs_root / "assurance" / "magd_risk.csv"
    magd = pd.read_csv(magd_path) if magd_path.exists() else pd.DataFrame(columns=["case_id", "magd_assurance_risk", "risk_category"])
    if "case_id" in magd.columns:
        magd["case_id"] = magd["case_id"].astype(str)
    return assurance, magd


def _normalize_log(
    frame: pd.DataFrame,
    method_name: str,
    assurance: pd.DataFrame,
    magd: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    working = frame.copy()
    working["case_id"] = working["case_id"].astype(str)
    if "selected_route" not in working.columns:
        source = working.get("decision_source", pd.Series(["AI"] * len(working), index=working.index)).astype(str)
        mapped = source.copy()
        mapped[source.str.contains("Expert")] = "Human Expert"
        mapped[source.str.contains("Escalate")] = "Escalate"
        mapped[~(source.str.contains("Expert") | source.str.contains("Escalate"))] = "AI"
        working["selected_route"] = mapped
    if "selected_expert" not in working.columns:
        working["selected_expert"] = working.get("assigned_expert", "")
    if "decision_reason" not in working.columns:
        working["decision_reason"] = working.get("decision_source", "")
    if "capacity_status" not in working.columns:
        working["capacity_status"] = "not_available"
    if "is_correct" not in working.columns and {"final_prediction", "y_true"} <= set(working.columns):
        working["is_correct"] = (working["final_prediction"].astype(int) == working["y_true"].astype(int)).astype(int)

    assurance_columns = [
        "case_id",
        "ai_score",
        "ai_pred",
        "numerical_confidence",
        "distance_confidence",
        "distance_uncertainty",
        "calibration_risk",
        "neighbor_error_rate",
        "wrong_confident_risk",
        "wrong_confident_label_offline",
    ]
    working = (
        working.merge(
            assurance[[column for column in assurance_columns if column in assurance.columns]],
            on="case_id",
            how="left",
            suffixes=("", "_assurance"),
        )
        .merge(magd[[column for column in ["case_id", "magd_assurance_risk", "risk_category"] if column in magd.columns]], on="case_id", how="left")
        .merge(metadata, on="case_id", how="left")
    )
    if "ai_pred" not in working.columns and "ai_pred_assurance" in working.columns:
        working["ai_pred"] = working["ai_pred_assurance"]
    if "ai_score" not in working.columns and "ai_score_assurance" in working.columns:
        working["ai_score"] = working["ai_score_assurance"]
    working["used_ai"] = working["selected_route"].astype(str).eq("AI")
    working["ai_correct"] = (working["ai_pred"].astype(int) == working["y_true"].astype(int)).astype(int)
    working["method"] = method_name
    return working.sort_values("case_id").reset_index(drop=True)


def run_statistical_tests(config_path: str | Path) -> tuple[pd.DataFrame, Path]:
    resolved_config_path = Path(config_path).resolve()
    config = load_yaml(resolved_config_path)
    outputs_root, final_dir, paper_dir = _paths(resolved_config_path)
    method_paths = _load_method_paths(outputs_root)
    assurance, magd = _core_join_tables(outputs_root)

    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = (resolved_config_path.parent / processed_dir).resolve()
    metadata = pd.read_csv(processed_dir / "test_metadata.csv")
    metadata["case_id"] = metadata["case_id"].astype(str)

    fp_cost = float(config.get("costs", {}).get("false_positive", 1.0))
    fn_cost = float(config.get("costs", {}).get("false_negative", 5.0))
    stats_cfg = config.get("statistics", {})
    n_bootstrap = int(stats_cfg.get("bootstrap_iterations", 1000))
    alpha = 1.0 - float(stats_cfg.get("confidence_level", 0.95))
    random_state = int(config.get("experiment", {}).get("seed", 42))

    normalized_logs: dict[str, pd.DataFrame] = {}
    for method_name, path in method_paths.items():
        normalized_logs[method_name] = _normalize_log(pd.read_csv(path), method_name, assurance, magd, metadata)

    rows: list[dict[str, float | str]] = []
    for method_a, method_b in COMPARISONS:
        if method_a not in normalized_logs or method_b not in normalized_logs:
            raise FileNotFoundError(f"Missing required method output for comparison: {method_a} vs {method_b}")
        frame_a = normalized_logs[method_a]
        frame_b = normalized_logs[method_b]
        if list(frame_a["case_id"]) != list(frame_b["case_id"]):
            raise ValueError(f"Case alignment mismatch between {method_a} and {method_b}.")
        rows.extend(
            build_statistical_test_rows(
                frame_a,
                frame_b,
                method_a=method_a,
                method_b=method_b,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                random_state=random_state,
            )
        )

    results = pd.DataFrame(rows).rename(
        columns={
            "test_name": "test",
            "confidence_interval_low": "ci_low",
            "confidence_interval_high": "ci_high",
        }
    )
    results.to_csv(final_dir / "statistical_comparisons.csv", index=False)
    results.to_csv(paper_dir / "statistical_comparison.csv", index=False)
    return results, final_dir


def main() -> None:
    args = parse_args()
    results, final_dir = run_statistical_tests(args.config)
    print(f"Saved statistical comparisons to: {final_dir / 'statistical_comparisons.csv'}")
    print(f"Rows written: {len(results)}")


if __name__ == "__main__":
    main()
