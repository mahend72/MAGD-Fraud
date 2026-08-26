from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.statistical_tests import (
    mcnemar_paired_correctness,
    paired_correctness,
    paired_bootstrap_interval,
    per_case_correct_rejection,
    per_case_cost_loss,
    per_case_overreliance,
    wilcoxon_signed_rank,
)
from src.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAGD-Fraud statistical tests.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def _paths(config_path: Path) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    final_dir = outputs_root / "final_metrics"
    final_dir.mkdir(parents=True, exist_ok=True)
    return outputs_root, final_dir


def _find_existing_path(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_method_logs(outputs_root: Path) -> dict[str, Path]:
    logs: dict[str, Path] = {}
    ai_only = outputs_root / "baselines" / "ai_only_decisions.csv"
    if ai_only.exists():
        logs["AI-only"] = ai_only

    distance_only = _find_existing_path(
        outputs_root / "ablations" / "ablation_decisions_distance_only.csv",
        outputs_root / "baselines" / "distance_threshold_decisions.csv",
    )
    if distance_only is not None:
        logs["distance-only"] = distance_only

    confidence_threshold = outputs_root / "baselines" / "numerical_threshold_decisions.csv"
    if confidence_threshold.exists():
        logs["confidence-threshold"] = confidence_threshold

    full_magd = _find_existing_path(
        outputs_root / "assurance_deferral" / "magd_fraud_decisions.csv",
        outputs_root / "assurance_deferral" / "magd_constrained_decisions.csv",
        outputs_root / "ablations" / "ablation_decisions_full_magd_constrained.csv",
        outputs_root / "ablations" / "ablation_decisions_full_magd_constrained_capacity_fairness.csv",
        outputs_root / "ablations" / "ablation_decisions_full_magd_with_capacity_and_fairness.csv",
    )
    if full_magd is not None:
        logs["MAGD-Constrained"] = full_magd

    heuristic = _find_existing_path(
        outputs_root / "assurance_deferral" / "magd_heuristic_decisions.csv",
        outputs_root / "ablations" / "ablation_decisions_full_magd_heuristic.csv",
    )
    if heuristic is not None:
        logs["MAGD-Heuristic"] = heuristic

    full_magd_without_capacity = outputs_root / "ablations" / "ablation_decisions_full_magd_without_capacity.csv"
    if full_magd_without_capacity.exists():
        logs["full MAGD without capacity"] = full_magd_without_capacity

    full_magd_with_capacity = _find_existing_path(
        outputs_root / "ablations" / "ablation_decisions_full_magd_constrained_capacity.csv",
        outputs_root / "ablations" / "ablation_decisions_full_magd_with_capacity.csv",
        outputs_root / "ablations" / "ablation_decisions_full_magd_constrained_capacity_fairness.csv",
        outputs_root / "ablations" / "ablation_decisions_full_magd_with_capacity_and_fairness.csv",
    )
    if full_magd_with_capacity is not None:
        logs["full MAGD with capacity"] = full_magd_with_capacity

    ltd = _find_existing_path(
        outputs_root / "baselines" / "learning_to_defer_decisions.csv",
        outputs_root / "assurance_deferral" / "learning_to_defer_decisions.csv",
        outputs_root / "ltd" / "learning_to_defer_decisions.csv",
    )
    if ltd is not None:
        logs["learning-to-defer baseline"] = ltd

    return logs


def _core_join_tables(outputs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    assurance = pd.read_csv(outputs_root / "assurance" / "wrong_confident_risk.csv")
    magd_path = outputs_root / "assurance" / "magd_risk.csv"
    magd = pd.read_csv(magd_path) if magd_path.exists() else pd.DataFrame(columns=["case_id", "magd_assurance_risk", "risk_category"])
    assurance["case_id"] = assurance["case_id"].astype(str)
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
    available_assurance_columns = [column for column in assurance_columns if column in assurance.columns]

    working = working.merge(
        assurance[available_assurance_columns],
        on="case_id",
        how="left",
        suffixes=("", "_assurance"),
    ).merge(
        magd[["case_id", "magd_assurance_risk", "risk_category"]],
        on="case_id",
        how="left",
    ).merge(
        metadata,
        on="case_id",
        how="left",
    )
    if "ai_pred" not in working.columns and "ai_pred_assurance" in working.columns:
        working["ai_pred"] = working["ai_pred_assurance"]
    if "ai_score" not in working.columns and "ai_score_assurance" in working.columns:
        working["ai_score"] = working["ai_score_assurance"]
    working["used_ai"] = working["selected_route"].astype(str).eq("AI")
    working["ai_correct"] = (working["ai_pred"].astype(int) == working["y_true"].astype(int)).astype(int)
    working["method"] = method_name
    return working.sort_values("case_id").reset_index(drop=True)


def per_case_wrong_confident_avoidance(frame: pd.DataFrame) -> np.ndarray:
    wrong_conf = frame.get("wrong_confident_label_offline", pd.Series([0] * len(frame), index=frame.index)).astype(int).to_numpy()
    uses_ai = frame["used_ai"].astype(bool).to_numpy()
    return ((wrong_conf == 1) & (~uses_ai)).astype(float)


def _bootstrap_wrong_confident_avoidance(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    n_bootstrap: int,
    alpha: float,
    random_state: int,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    values_a = per_case_wrong_confident_avoidance(frame_a)
    values_b = per_case_wrong_confident_avoidance(frame_b)
    n = len(values_a)
    samples = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        samples[idx] = float(values_a[sample_idx].mean() - values_b[sample_idx].mean())
    point = float(values_a.mean() - values_b.mean())
    return {
        "statistic": point,
        "confidence_interval_low": float(np.quantile(samples, alpha / 2.0)),
        "confidence_interval_high": float(np.quantile(samples, 1.0 - alpha / 2.0)),
    }


def _interpretation(
    *,
    method_a: str,
    method_b: str,
    metric: str,
    test_name: str,
    statistic: float,
    p_value: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> str:
    if test_name == "McNemar":
        if p_value is not None and p_value < 0.05:
            return f"Paired correctness differs significantly between {method_a} and {method_b}."
        return "No statistically significant paired correctness difference."
    if test_name == "Bootstrap":
        if ci_low is not None and ci_high is not None:
            if ci_low > 0:
                return f"{method_a} is higher on {metric}; CI excludes 0."
            if ci_high < 0:
                return f"{method_b} is higher on {metric}; CI excludes 0."
        return f"Bootstrap CI for {metric} crosses 0."
    if test_name == "Wilcoxon":
        if p_value is not None and p_value < 0.05:
            return f"Paired Wilcoxon suggests a difference in {metric}."
        return f"No statistically significant Wilcoxon difference in {metric}."
    return "Comparison computed."


def _row(
    *,
    method_a: str,
    method_b: str,
    metric: str,
    test: str,
    statistic: float,
    p_value: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> dict[str, float | str]:
    return {
        "method_A": method_a,
        "method_B": method_b,
        "metric": metric,
        "test": test,
        "statistic": float(statistic),
        "p_value": float(p_value) if p_value is not None and not math.isnan(p_value) else math.nan,
        "ci_low": float(ci_low) if ci_low is not None and not math.isnan(ci_low) else math.nan,
        "ci_high": float(ci_high) if ci_high is not None and not math.isnan(ci_high) else math.nan,
        "interpretation": _interpretation(
            method_a=method_a,
            method_b=method_b,
            metric=metric,
            test_name=test,
            statistic=statistic,
            p_value=p_value,
            ci_low=ci_low,
            ci_high=ci_high,
        ),
    }


def _build_rows_for_pair(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    fp_cost: float,
    fn_cost: float,
    n_bootstrap: int,
    alpha: float,
    random_state: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    mcnemar = mcnemar_paired_correctness(paired_correctness(frame_a), paired_correctness(frame_b))
    rows.append(
        _row(
            method_a=method_a,
            method_b=method_b,
            metric="correctness",
            test="McNemar",
            statistic=float(mcnemar["c"] - mcnemar["b"]),
            p_value=float(mcnemar["p_value"]),
        )
    )

    for metric in ["cost_sensitive_loss", "overreliance", "correct_rejection"]:
        boot = paired_bootstrap_interval(
            frame_a,
            frame_b,
            metric=metric,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            random_state=random_state,
        )
        rows.append(
            _row(
                method_a=method_a,
                method_b=method_b,
                metric=metric,
                test="Bootstrap",
                statistic=float(boot["statistic"]),
                ci_low=float(boot["confidence_interval_low"]),
                ci_high=float(boot["confidence_interval_high"]),
            )
        )

    wc_boot = _bootstrap_wrong_confident_avoidance(
        frame_a,
        frame_b,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )
    rows.append(
        _row(
            method_a=method_a,
            method_b=method_b,
            metric="wrong_confident_avoidance",
            test="Bootstrap",
            statistic=float(wc_boot["statistic"]),
            ci_low=float(wc_boot["confidence_interval_low"]),
            ci_high=float(wc_boot["confidence_interval_high"]),
        )
    )

    wilcoxon_specs = {
        "cost_sensitive_loss": (
            per_case_cost_loss(frame_a, fp_cost=fp_cost, fn_cost=fn_cost),
            per_case_cost_loss(frame_b, fp_cost=fp_cost, fn_cost=fn_cost),
        ),
        "overreliance": (per_case_overreliance(frame_a), per_case_overreliance(frame_b)),
        "correct_rejection": (per_case_correct_rejection(frame_a), per_case_correct_rejection(frame_b)),
        "wrong_confident_avoidance": (per_case_wrong_confident_avoidance(frame_a), per_case_wrong_confident_avoidance(frame_b)),
    }
    for metric, (values_a, values_b) in wilcoxon_specs.items():
        wilc = wilcoxon_signed_rank(values_a, values_b)
        rows.append(
            _row(
                method_a=method_a,
                method_b=method_b,
                metric=metric,
                test="Wilcoxon",
                statistic=float(wilc["statistic"]),
                p_value=float(wilc["p_value"]),
            )
        )

    return rows


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    outputs_root, final_dir = _paths(config_path)
    assurance, magd = _core_join_tables(outputs_root)

    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = (config_path.parent / processed_dir).resolve()
    metadata = pd.read_csv(processed_dir / "test_metadata.csv")
    metadata["case_id"] = metadata["case_id"].astype(str)

    fp_cost = float(config.get("magd", {}).get("costs", {}).get("false_positive", config.get("model", {}).get("false_positive_cost", 1.0)))
    fn_cost = float(config.get("magd", {}).get("costs", {}).get("false_negative", config.get("model", {}).get("false_negative_cost", 5.0)))
    stats_cfg = config.get("evaluation", {}).get("statistical_tests", {})
    n_bootstrap = int(stats_cfg.get("bootstrap_iterations", 2000))
    alpha = float(stats_cfg.get("alpha", 0.05))
    random_state = int(stats_cfg.get("random_state", 42))

    method_logs = _load_method_logs(outputs_root)
    normalized_logs: dict[str, pd.DataFrame] = {}
    for method_name, path in method_logs.items():
        frame = pd.read_csv(path)
        normalized_logs[method_name] = _normalize_log(frame, method_name, assurance, magd, metadata)

    comparisons = [
        ("AI-only", "distance-only"),
        ("distance-only", "MAGD-Heuristic"),
        ("distance-only", "MAGD-Constrained"),
        ("confidence-threshold", "MAGD-Constrained"),
        ("learning-to-defer baseline", "MAGD-Constrained"),
        ("full MAGD without capacity", "full MAGD with capacity"),
    ]

    rows: list[dict[str, float | str]] = []
    completed = 0
    for method_a, method_b in comparisons:
        if method_a not in normalized_logs or method_b not in normalized_logs:
            continue
        frame_a = normalized_logs[method_a]
        frame_b = normalized_logs[method_b]
        if list(frame_a["case_id"]) != list(frame_b["case_id"]):
            raise ValueError(f"Case alignment mismatch between {method_a} and {method_b}.")
        rows.extend(
            _build_rows_for_pair(
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
        completed += 1

    results = pd.DataFrame(rows)
    output_path = final_dir / "magd_statistical_tests.csv"
    results.to_csv(output_path, index=False)
    print(f"Saved MAGD statistical test results to: {output_path}")
    print(f"Comparisons evaluated: {completed}")


if __name__ == "__main__":
    main()
