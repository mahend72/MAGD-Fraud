from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.models.predict import pr_auc_score_manual


def compute_fraud_metrics(
    frame: pd.DataFrame,
    *,
    score_column: str | None = None,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
) -> dict[str, float]:
    y_true = frame["y_true"].astype(int).to_numpy()
    y_pred = frame["final_prediction"].astype(int).to_numpy()

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    cost_sensitive_loss = fp_cost * fp + fn_cost * fn

    pr_auc = math.nan
    if score_column and score_column in frame.columns:
        scores = pd.to_numeric(frame[score_column], errors="coerce")
        if scores.notna().any():
            valid = scores.notna()
            pr_auc = pr_auc_score_manual(y_true[valid.to_numpy()], scores[valid].to_numpy(dtype=float))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "cost_sensitive_loss": float(cost_sensitive_loss),
    }
