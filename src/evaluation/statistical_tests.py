from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def paired_correctness(frame: pd.DataFrame) -> np.ndarray:
    return frame["is_correct"].astype(int).to_numpy()


def per_case_cost_loss(frame: pd.DataFrame, *, fp_cost: float, fn_cost: float) -> np.ndarray:
    y_true = frame["y_true"].astype(int).to_numpy()
    y_pred = frame["final_prediction"].astype(int).to_numpy()
    fp = ((y_true == 0) & (y_pred == 1)).astype(float)
    fn = ((y_true == 1) & (y_pred == 0)).astype(float)
    return fp_cost * fp + fn_cost * fn


def per_case_overreliance(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["used_ai"].astype(bool).to_numpy() & ~frame["ai_correct"].astype(bool).to_numpy()
    ).astype(float)


def per_case_correct_rejection(frame: pd.DataFrame) -> np.ndarray:
    return (
        ~frame["used_ai"].astype(bool).to_numpy() & ~frame["ai_correct"].astype(bool).to_numpy()
    ).astype(float)


def per_case_wrong_confident_avoidance(frame: pd.DataFrame) -> np.ndarray:
    wrong_conf = frame.get(
        "wrong_confident_label_offline",
        frame.get("offline_wrong_confident_label", pd.Series([0] * len(frame), index=frame.index)),
    ).astype(int).to_numpy()
    uses_ai = frame["used_ai"].astype(bool).to_numpy()
    return ((wrong_conf == 1) & (~uses_ai)).astype(float)


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def metric_difference(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    metric: str,
    fp_cost: float,
    fn_cost: float,
) -> float:
    if metric == "cost_sensitive_loss":
        return float(
            per_case_cost_loss(frame_a, fp_cost=fp_cost, fn_cost=fn_cost).sum()
            - per_case_cost_loss(frame_b, fp_cost=fp_cost, fn_cost=fn_cost).sum()
        )
    if metric == "f1":
        y_true = frame_a["y_true"].astype(int).to_numpy()
        return float(
            _f1_score(y_true, frame_a["final_prediction"].astype(int).to_numpy())
            - _f1_score(y_true, frame_b["final_prediction"].astype(int).to_numpy())
        )
    if metric == "overreliance":
        return float(per_case_overreliance(frame_a).mean() - per_case_overreliance(frame_b).mean())
    if metric == "correct_rejection":
        return float(
            per_case_correct_rejection(frame_a).mean() - per_case_correct_rejection(frame_b).mean()
        )
    if metric == "wrong_confident_avoidance":
        return float(
            per_case_wrong_confident_avoidance(frame_a).mean()
            - per_case_wrong_confident_avoidance(frame_b).mean()
        )
    raise ValueError(f"Unsupported metric for paired difference: {metric}")


def mcnemar_paired_correctness(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, float]:
    if correct_a.shape != correct_b.shape:
        raise ValueError("Correctness arrays must have the same shape for McNemar test.")

    b = int(np.sum((correct_a == 1) & (correct_b == 0)))
    c = int(np.sum((correct_a == 0) & (correct_b == 1)))
    discordant = b + c
    if discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": float(b), "c": float(c)}

    statistic = (abs(b - c) - 1.0) ** 2 / discordant
    p_value = float(binomtest(min(b, c), n=discordant, p=0.5, alternative="two-sided").pvalue)
    return {"statistic": float(statistic), "p_value": p_value, "b": float(b), "c": float(c)}


def paired_bootstrap_interval(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    metric: str,
    fp_cost: float,
    fn_cost: float,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, float]:
    if len(frame_a) != len(frame_b):
        raise ValueError("Paired bootstrap requires frames of equal length.")

    rng = np.random.default_rng(random_state)
    y_true = frame_a["y_true"].astype(int).to_numpy()
    pred_a = frame_a["final_prediction"].astype(int).to_numpy()
    pred_b = frame_b["final_prediction"].astype(int).to_numpy()
    over_a = per_case_overreliance(frame_a)
    over_b = per_case_overreliance(frame_b)
    rej_a = per_case_correct_rejection(frame_a)
    rej_b = per_case_correct_rejection(frame_b)
    wc_a = per_case_wrong_confident_avoidance(frame_a)
    wc_b = per_case_wrong_confident_avoidance(frame_b)
    cost_a = per_case_cost_loss(frame_a, fp_cost=fp_cost, fn_cost=fn_cost)
    cost_b = per_case_cost_loss(frame_b, fp_cost=fp_cost, fn_cost=fn_cost)

    n = len(frame_a)
    samples = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        if metric == "cost_sensitive_loss":
            samples[idx] = float(cost_a[sample_idx].sum() - cost_b[sample_idx].sum())
        elif metric == "f1":
            samples[idx] = float(
                _f1_score(y_true[sample_idx], pred_a[sample_idx])
                - _f1_score(y_true[sample_idx], pred_b[sample_idx])
            )
        elif metric == "overreliance":
            samples[idx] = float(over_a[sample_idx].mean() - over_b[sample_idx].mean())
        elif metric == "correct_rejection":
            samples[idx] = float(rej_a[sample_idx].mean() - rej_b[sample_idx].mean())
        elif metric == "wrong_confident_avoidance":
            samples[idx] = float(wc_a[sample_idx].mean() - wc_b[sample_idx].mean())
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")

    point_estimate = metric_difference(frame_a, frame_b, metric=metric, fp_cost=fp_cost, fn_cost=fn_cost)
    lower = float(np.quantile(samples, alpha / 2.0))
    upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return {
        "statistic": float(point_estimate),
        "confidence_interval_low": lower,
        "confidence_interval_high": upper,
    }


def wilcoxon_signed_rank(values_a: np.ndarray, values_b: np.ndarray) -> dict[str, float]:
    if values_a.shape != values_b.shape:
        raise ValueError("Wilcoxon test requires paired arrays of the same shape.")

    diffs = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    if np.allclose(diffs, 0.0):
        return {"statistic": 0.0, "p_value": 1.0}

    result = wilcoxon(values_a, values_b, zero_method="zsplit", alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def interpretation_for_result(
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
    if test_name == "mcnemar":
        if p_value is not None and p_value < 0.05:
            direction = "higher" if statistic > 0 else "different"
            return f"Paired correctness differs significantly; {method_a} vs {method_b} is {direction} under McNemar."
        return "No statistically significant paired correctness difference under McNemar."

    if ci_low is not None and ci_high is not None and not (math.isnan(ci_low) or math.isnan(ci_high)):
        if ci_low > 0:
            return f"{method_a} has higher {metric} than {method_b}; bootstrap CI excludes 0."
        if ci_high < 0:
            return f"{method_a} has lower {metric} than {method_b}; bootstrap CI excludes 0."
        return f"Bootstrap CI for {metric} difference crosses 0."

    if p_value is not None:
        if p_value < 0.05:
            return f"Paired Wilcoxon test suggests a difference in {metric} between {method_a} and {method_b}."
        return f"No statistically significant Wilcoxon difference in {metric}."

    return "Comparison computed."


def build_statistical_test_rows(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    fp_cost: float,
    fn_cost: float,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    mcnemar = mcnemar_paired_correctness(paired_correctness(frame_a), paired_correctness(frame_b))
    rows.append(
        {
            "method_A": method_a,
            "method_B": method_b,
            "metric": "correctness",
            "test_name": "mcnemar",
            "statistic": mcnemar["statistic"],
            "p_value": mcnemar["p_value"],
            "confidence_interval_low": math.nan,
            "confidence_interval_high": math.nan,
            "interpretation": interpretation_for_result(
                method_a=method_a,
                method_b=method_b,
                metric="correctness",
                test_name="mcnemar",
                statistic=mcnemar["c"] - mcnemar["b"],
                p_value=mcnemar["p_value"],
            ),
        }
    )

    bootstrap_metrics = ["cost_sensitive_loss", "f1", "overreliance", "correct_rejection", "wrong_confident_avoidance"]
    wilcoxon_arrays: dict[str, Callable[[], tuple[np.ndarray, np.ndarray]]] = {
        "cost_sensitive_loss": lambda: (
            per_case_cost_loss(frame_a, fp_cost=fp_cost, fn_cost=fn_cost),
            per_case_cost_loss(frame_b, fp_cost=fp_cost, fn_cost=fn_cost),
        ),
        "overreliance": lambda: (per_case_overreliance(frame_a), per_case_overreliance(frame_b)),
        "correct_rejection": lambda: (per_case_correct_rejection(frame_a), per_case_correct_rejection(frame_b)),
        "wrong_confident_avoidance": lambda: (
            per_case_wrong_confident_avoidance(frame_a),
            per_case_wrong_confident_avoidance(frame_b),
        ),
    }

    for metric in bootstrap_metrics:
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
            {
                "method_A": method_a,
                "method_B": method_b,
                "metric": metric,
                "test_name": "paired_bootstrap",
                "statistic": boot["statistic"],
                "p_value": math.nan,
                "confidence_interval_low": boot["confidence_interval_low"],
                "confidence_interval_high": boot["confidence_interval_high"],
                "interpretation": interpretation_for_result(
                    method_a=method_a,
                    method_b=method_b,
                    metric=metric,
                    test_name="paired_bootstrap",
                    statistic=boot["statistic"],
                    ci_low=boot["confidence_interval_low"],
                    ci_high=boot["confidence_interval_high"],
                ),
            }
        )
        if metric in wilcoxon_arrays:
            values_a, values_b = wilcoxon_arrays[metric]()
            wilc = wilcoxon_signed_rank(values_a, values_b)
            rows.append(
                {
                    "method_A": method_a,
                    "method_B": method_b,
                    "metric": metric,
                    "test_name": "wilcoxon_signed_rank",
                    "statistic": wilc["statistic"],
                    "p_value": wilc["p_value"],
                    "confidence_interval_low": math.nan,
                    "confidence_interval_high": math.nan,
                    "interpretation": interpretation_for_result(
                        method_a=method_a,
                        method_b=method_b,
                        metric=metric,
                        test_name="wilcoxon_signed_rank",
                        statistic=wilc["statistic"],
                        p_value=wilc["p_value"],
                    ),
                }
            )

    return rows
