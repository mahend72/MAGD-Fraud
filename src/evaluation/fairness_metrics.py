from __future__ import annotations

import json

import pandas as pd


def compute_group_error_rates(
    frame: pd.DataFrame,
    *,
    sensitive_column: str,
    y_true_column: str,
    y_pred_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if sensitive_column not in frame.columns:
        return pd.DataFrame(
            columns=[
                "sensitive_column",
                "group",
                "false_positive_rate",
                "false_negative_rate",
                "group_cost",
            ]
        )

    for group_value, group in frame.groupby(sensitive_column, dropna=False):
        y_true = group[y_true_column].astype(int)
        y_pred = group[y_pred_column].astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        rows.append(
            {
                "sensitive_column": sensitive_column,
                "group": str(group_value),
                "false_positive_rate": float(fpr),
                "false_negative_rate": float(fnr),
            }
        )
    return pd.DataFrame(rows)


def summarize_disparity(group_rates: pd.DataFrame) -> dict[str, float | str]:
    if group_rates.empty:
        return {
            "groupwise_false_positive_rate": 0.0,
            "groupwise_false_negative_rate": 0.0,
            "false_positive_rate_disparity": 0.0,
            "false_negative_rate_disparity": 0.0,
            "bias_risk": 0.0,
            "disparity_summary": "",
        }

    fpr_max = float(group_rates["false_positive_rate"].max())
    fnr_max = float(group_rates["false_negative_rate"].max())
    fpr_disp = float(group_rates["false_positive_rate"].max() - group_rates["false_positive_rate"].min())
    fnr_disp = float(group_rates["false_negative_rate"].max() - group_rates["false_negative_rate"].min())
    summary = {
        "groupwise_false_positive_rate": fpr_max,
        "groupwise_false_negative_rate": fnr_max,
        "false_positive_rate_disparity": fpr_disp,
        "false_negative_rate_disparity": fnr_disp,
        "bias_risk": float(max(fpr_disp, fnr_disp)),
        "disparity_summary": json.dumps(
            {
                "groups": group_rates[["group", "false_positive_rate", "false_negative_rate"]].to_dict(orient="records"),
                "fpr_disparity": fpr_disp,
                "fnr_disparity": fnr_disp,
            },
            sort_keys=True,
        ),
    }
    return summary


def compute_fairness_metrics(
    frame: pd.DataFrame,
    sensitive_columns: list[str],
    *,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    if not sensitive_columns:
        return pd.DataFrame(columns=["sensitive_column", "group", "false_positive_rate", "false_negative_rate", "group_cost"])

    for sensitive in sensitive_columns:
        if sensitive not in frame.columns:
            continue
        per_group: list[dict[str, float | str]] = []
        rates = compute_group_error_rates(
            frame,
            sensitive_column=sensitive,
            y_true_column="y_true",
            y_pred_column="final_prediction",
        )
        for _, rate_row in rates.iterrows():
            group_value = rate_row["group"]
            group = frame.loc[frame[sensitive].astype(str) == str(group_value)]
            y_true = group["y_true"].astype(int)
            y_pred = group["final_prediction"].astype(int)
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            fpr = float(rate_row["false_positive_rate"])
            fnr = float(rate_row["false_negative_rate"])
            group_cost = fp_cost * fp + fn_cost * fn
            per_group.append(
                {
                    "sensitive_column": sensitive,
                    "group": str(group_value),
                    "false_positive_rate": fpr,
                    "false_negative_rate": fnr,
                    "group_cost": float(group_cost),
                }
            )

        if per_group:
            group_df = pd.DataFrame(per_group)
            disparity = summarize_disparity(group_df)
            group_df["false_positive_rate_disparity"] = float(disparity["false_positive_rate_disparity"])
            group_df["false_negative_rate_disparity"] = float(disparity["false_negative_rate_disparity"])
            rows.extend(group_df.to_dict(orient="records"))

    return pd.DataFrame(rows)
