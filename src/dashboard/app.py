from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml


st.set_page_config(page_title="HAAF FiFAR Dashboard", layout="wide")

PROTOTYPE_NOTICE = "This dashboard is a research prototype and not a validated operational fraud-review tool."
CASE_REVIEW_FIELDS = [
    "case_id",
    "ai_score",
    "ai_pred",
    "numerical_confidence",
    "calibration_risk",
    "distance_uncertainty",
    "neighbor_error_rate",
    "wrong_confident_risk",
    "magd_assurance_risk",
    "risk_category",
    "selected_route",
    "selected_expert",
    "final_prediction",
    "decision_reason",
    "method",
]
EVALUATION_ONLY_FIELDS = ["y_true", "is_correct", "ai_correct", "wrong_confident_label_offline"]
DISPLAY_NAME_MAP = {
    "case_id": "Case ID",
    "ai_score": "AI Score",
    "ai_pred": "AI Prediction",
    "numerical_confidence": "Numerical Confidence",
    "calibration_risk": "Calibration Risk",
    "distance_uncertainty": "Distance Uncertainty",
    "distance_confidence": "Distance Confidence",
    "neighbor_error_rate": "Neighbor Error Rate",
    "neighbor_fraud_rate": "Neighbor Fraud Rate",
    "neighbor_ai_agreement": "Neighbor AI Agreement",
    "mean_neighbor_distance": "Mean Neighbor Distance",
    "wrong_confident_risk": "Wrong-Confident Risk",
    "confidence_disagreement": "Confidence Disagreement",
    "magd_assurance_risk": "MAGD Risk",
    "risk_category": "Risk Category",
    "selected_route": "Selected Route",
    "selected_expert": "Selected Expert",
    "final_prediction": "Final Prediction",
    "decision_reason": "Decision Reason",
    "method": "Method",
    "y_true": "Offline Label",
    "is_correct": "Decision Correct",
    "ai_correct": "AI Correct",
}


def _resolve_paths(config_path: str | Path | None = None) -> tuple[Path, dict[str, Any], Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    resolved_config = Path(config_path).resolve() if config_path is not None else project_root / "config.yaml"
    config_root = resolved_config.parent
    config = load_yaml(resolved_config)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", config.get("experiment", {}).get("output_dir", "data/outputs")))
    if not outputs_root.is_absolute():
        outputs_root = (config_root / outputs_root).resolve()
    processed_root = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    if not processed_root.is_absolute():
        processed_root = (config_root / processed_root).resolve()
    return project_root, config, outputs_root, processed_root


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "case_id" in frame.columns:
        frame["case_id"] = frame["case_id"].astype(str)
    return frame


def _normalize_case_columns(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    if "offline_wrong_confident_label" in working.columns and "wrong_confident_label_offline" not in working.columns:
        working = working.rename(columns={"offline_wrong_confident_label": "wrong_confident_label_offline"})
    if "selected_expert" not in working.columns:
        working["selected_expert"] = ""
    working["selected_expert"] = working["selected_expert"].fillna("")
    if "decision_reason" not in working.columns:
        working["decision_reason"] = ""
    if "method" not in working.columns:
        working["method"] = "unknown"
    return working


def _friendly_missing_message(name: str, path: Path) -> str:
    return f"{name} is not available yet. Expected file: {path.name}"


@st.cache_data(show_spinner=False)
def load_dashboard_data(config_path: str | Path | None = None) -> dict[str, Any]:
    project_root, config, outputs_root, processed_root = _resolve_paths(config_path)

    file_map = {
        "decision_audit_log": outputs_root / "audit_pack" / "decision_audit_log.csv",
        "wrong_confident_risk": outputs_root / "assurance" / "wrong_confident_risk.csv",
        "local_reliability": outputs_root / "assurance" / "local_reliability.csv",
        "magd_risk": outputs_root / "assurance" / "magd_risk.csv",
        "expert_reliability": outputs_root / "assurance_deferral" / "expert_reliability.csv",
        "summary_metrics": outputs_root / "final_metrics" / "all_method_metrics.csv",
        "audit_metrics": outputs_root / "final_metrics" / "audit_metrics.csv",
        "claim_evidence_matrix": outputs_root / "audit_pack" / "claim_evidence_matrix.csv",
        "test_metadata": processed_root / "test_metadata.csv",
    }

    frames: dict[str, pd.DataFrame] = {}
    file_status: dict[str, dict[str, Any]] = {}
    for name, path in file_map.items():
        frame = _read_csv_or_empty(path)
        if not frame.empty:
            frame = _normalize_case_columns(frame)
        frames[name] = frame
        file_status[name] = {
            "path": str(path),
            "present": path.exists(),
            "message": "" if path.exists() else _friendly_missing_message(name.replace("_", " ").title(), path),
        }

    return {
        "project_root": project_root,
        "config": config,
        "outputs_root": outputs_root,
        "processed_root": processed_root,
        "frames": frames,
        "file_status": file_status,
    }


def _merge_case_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    audit_log = frames.get("decision_audit_log", pd.DataFrame()).copy()
    if audit_log.empty:
        base = frames.get("test_metadata", pd.DataFrame()).copy()
        if base.empty:
            return pd.DataFrame()
        if "case_id" in base.columns:
            base["case_id"] = base["case_id"].astype(str)
        audit_log = base

    if "case_id" in audit_log.columns:
        audit_log["case_id"] = audit_log["case_id"].astype(str)

    for source_name, extra_columns in [
        ("wrong_confident_risk", ["case_id", "numerical_confidence", "distance_confidence", "distance_uncertainty", "calibration_risk", "neighbor_error_rate", "confidence_disagreement", "wrong_confident_risk", "wrong_confident_label_offline"]),
        ("local_reliability", ["case_id", "neighbor_error_rate", "neighbor_fraud_rate", "neighbor_ai_agreement", "mean_neighbor_distance", "knn_k"]),
        ("magd_risk", ["case_id", "magd_assurance_risk", "risk_category", "drift_risk", "business_risk", "drift_available", "business_available"]),
        ("test_metadata", ["case_id"]),
    ]:
        source = frames.get(source_name, pd.DataFrame())
        if source.empty or "case_id" not in source.columns:
            continue
        keep = [column for column in extra_columns if column in source.columns]
        if len(keep) <= 1:
            continue
        audit_log = audit_log.merge(source[keep].drop_duplicates(subset=["case_id"]), on="case_id", how="left", suffixes=("", f"_{source_name}"))
        for column in keep:
            suffixed = f"{column}_{source_name}"
            if suffixed in audit_log.columns and column in audit_log.columns:
                audit_log[column] = audit_log[column].where(audit_log[column].notna(), audit_log[suffixed])
                audit_log = audit_log.drop(columns=[suffixed])
            elif suffixed in audit_log.columns:
                audit_log = audit_log.rename(columns={suffixed: column})

    for column in CASE_REVIEW_FIELDS:
        if column not in audit_log.columns:
            audit_log[column] = pd.NA
    return audit_log.drop_duplicates(subset=["case_id"]).reset_index(drop=True)


def build_dashboard_case_table(data: dict[str, Any], evaluation_mode: bool = False) -> pd.DataFrame:
    case_frame = _merge_case_frame(data["frames"])
    if case_frame.empty:
        return case_frame
    visible_columns = CASE_REVIEW_FIELDS.copy()
    optional_columns = [
        "distance_confidence",
        "neighbor_fraud_rate",
        "neighbor_ai_agreement",
        "mean_neighbor_distance",
        "confidence_disagreement",
        "capacity_status",
        "fairness_risk",
        "capacity_pressure",
        "drift_risk",
        "business_risk",
    ]
    visible_columns.extend([column for column in optional_columns if column in case_frame.columns])
    if evaluation_mode:
        visible_columns.extend([column for column in EVALUATION_ONLY_FIELDS if column in case_frame.columns])
    ordered = [column for column in visible_columns if column in case_frame.columns]
    return case_frame[ordered].copy()


def dashboard_missing_messages(data: dict[str, Any]) -> list[str]:
    return [status["message"] for status in data["file_status"].values() if status["message"]]


def _format_field_name(name: str) -> str:
    return DISPLAY_NAME_MAP.get(name, name.replace("_", " ").title())


def _case_record_for_display(case_row: pd.Series, evaluation_mode: bool = False) -> pd.DataFrame:
    fields = CASE_REVIEW_FIELDS + [
        field
        for field in [
            "distance_confidence",
            "neighbor_fraud_rate",
            "neighbor_ai_agreement",
            "mean_neighbor_distance",
            "confidence_disagreement",
            "capacity_status",
            "fairness_risk",
            "capacity_pressure",
            "drift_risk",
            "business_risk",
        ]
        if field in case_row.index
    ]
    if evaluation_mode:
        fields += [field for field in EVALUATION_ONLY_FIELDS if field in case_row.index]
    return pd.DataFrame({"Field": [_format_field_name(field) for field in fields], "Value": [case_row.get(field) for field in fields]})


def _metric_value(row: pd.Series, column: str) -> str:
    value = row.get(column)
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _friendly_notice(messages: list[str]) -> None:
    for message in messages:
        st.info(message)


def _select_case(case_frame: pd.DataFrame) -> pd.Series | None:
    if case_frame.empty or "case_id" not in case_frame.columns:
        st.info("No case-level outputs are available yet. Run the MAGD routing and audit-pack stages first.")
        return None
    case_ids = case_frame["case_id"].astype(str).tolist()
    selected_case = st.selectbox("Select case_id", case_ids)
    return case_frame.loc[case_frame["case_id"].astype(str) == selected_case].iloc[0]


def _case_review_tab(case_frame: pd.DataFrame, evaluation_mode: bool) -> None:
    row = _select_case(case_frame)
    if row is None:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI Score", _metric_value(row, "ai_score"))
    c2.metric("AI Prediction", _metric_value(row, "ai_pred"))
    c3.metric("MAGD Risk", _metric_value(row, "magd_assurance_risk"))
    c4.metric("Selected Route", _metric_value(row, "selected_route"))
    st.dataframe(_case_record_for_display(row, evaluation_mode=evaluation_mode), use_container_width=True, hide_index=True)


def _assurance_evidence_tab(case_frame: pd.DataFrame) -> None:
    row = _select_case(case_frame)
    if row is None:
        return
    evidence_fields = [
        "numerical_confidence",
        "calibration_risk",
        "distance_uncertainty",
        "neighbor_error_rate",
        "wrong_confident_risk",
        "magd_assurance_risk",
        "risk_category",
    ]
    st.dataframe(
        pd.DataFrame({"Evidence": [_format_field_name(field) for field in evidence_fields], "Value": [row.get(field) for field in evidence_fields]}),
        use_container_width=True,
        hide_index=True,
    )


def _routing_decision_tab(case_frame: pd.DataFrame) -> None:
    row = _select_case(case_frame)
    if row is None:
        return
    fields = ["selected_route", "selected_expert", "final_prediction", "decision_reason", "method", "capacity_status"]
    st.dataframe(
        pd.DataFrame({"Routing Field": [_format_field_name(field) for field in fields if field in row.index], "Value": [row.get(field) for field in fields if field in row.index]}),
        use_container_width=True,
        hide_index=True,
    )


def _wrong_confident_tab(case_frame: pd.DataFrame, evaluation_mode: bool) -> None:
    row = _select_case(case_frame)
    if row is None:
        return
    fields = ["numerical_confidence", "distance_confidence", "distance_uncertainty", "calibration_risk", "neighbor_error_rate", "confidence_disagreement", "wrong_confident_risk"]
    if evaluation_mode and "wrong_confident_label_offline" in row.index:
        fields.append("wrong_confident_label_offline")
    st.dataframe(
        pd.DataFrame({"Field": [_format_field_name(field) for field in fields if field in row.index], "Value": [row.get(field) for field in fields if field in row.index]}),
        use_container_width=True,
        hide_index=True,
    )


def _local_reliability_tab(case_frame: pd.DataFrame) -> None:
    row = _select_case(case_frame)
    if row is None:
        return
    fields = ["neighbor_error_rate", "neighbor_fraud_rate", "neighbor_ai_agreement", "mean_neighbor_distance", "distance_uncertainty"]
    st.dataframe(
        pd.DataFrame({"Field": [_format_field_name(field) for field in fields if field in row.index], "Value": [row.get(field) for field in fields if field in row.index]}),
        use_container_width=True,
        hide_index=True,
    )


def _expert_selection_tab(data: dict[str, Any], case_frame: pd.DataFrame) -> None:
    row = _select_case(case_frame)
    if row is None:
        return
    expert_frame = data["frames"].get("expert_reliability", pd.DataFrame())
    st.subheader("Selected Expert for This Case")
    st.write(f"Selected expert(s): `{row.get('selected_expert', 'NA')}`")
    st.write(f"Decision reason: `{row.get('decision_reason', 'NA')}`")
    if expert_frame.empty:
        st.info(_friendly_missing_message("Expert reliability", Path(data["file_status"]["expert_reliability"]["path"])))
        return
    keep = [column for column in ["expert", "accuracy", "false_positive_rate", "false_negative_rate", "cost_sensitive_loss", "bias_risk", "capacity_enabled", "expected_cost"] if column in expert_frame.columns]
    st.subheader("Expert Reliability Table")
    st.dataframe(expert_frame[keep], use_container_width=True)


def _audit_log_tab(data: dict[str, Any], case_frame: pd.DataFrame, evaluation_mode: bool) -> None:
    row = _select_case(case_frame)
    if row is None:
        return
    st.dataframe(_case_record_for_display(row, evaluation_mode=evaluation_mode), use_container_width=True, hide_index=True)
    claim_matrix = data["frames"].get("claim_evidence_matrix", pd.DataFrame())
    if claim_matrix.empty:
        st.info(_friendly_missing_message("Claim evidence matrix", Path(data["file_status"]["claim_evidence_matrix"]["path"])))
    else:
        st.subheader("Claim-Evidence Coverage")
        st.dataframe(claim_matrix, use_container_width=True)


def _summary_metrics_tab(data: dict[str, Any]) -> None:
    metrics = data["frames"].get("summary_metrics", pd.DataFrame())
    audit = data["frames"].get("audit_metrics", pd.DataFrame())
    if metrics.empty:
        st.info(_friendly_missing_message("Summary metrics", Path(data["file_status"]["summary_metrics"]["path"])))
        return
    st.subheader("All Method Metrics")
    st.dataframe(metrics, use_container_width=True)
    if not audit.empty:
        st.subheader("Audit Metrics")
        st.dataframe(audit, use_container_width=True)


def main() -> None:
    data = load_dashboard_data()
    config = data["config"]
    st.title("HAAF FiFAR MAGD-Fraud Review Dashboard")
    st.warning(PROTOTYPE_NOTICE)
    evaluation_mode = st.sidebar.checkbox("Enable evaluation mode", value=False, help="Show offline evaluation labels and correctness fields.")
    st.sidebar.caption("Evaluation mode is off by default so offline test labels are not shown during case review.")
    _friendly_notice(dashboard_missing_messages(data))

    case_frame = build_dashboard_case_table(data, evaluation_mode=evaluation_mode)

    tabs = st.tabs(
        [
            "Case Review",
            "Assurance Evidence",
            "Routing Decision",
            "Wrong-Confident Risk",
            "Local Neighbour Reliability",
            "Expert Selection",
            "Audit Log",
            "Summary Metrics",
        ]
    )

    with tabs[0]:
        _case_review_tab(case_frame, evaluation_mode=evaluation_mode)
    with tabs[1]:
        _assurance_evidence_tab(case_frame)
    with tabs[2]:
        _routing_decision_tab(case_frame)
    with tabs[3]:
        _wrong_confident_tab(case_frame, evaluation_mode=evaluation_mode)
    with tabs[4]:
        _local_reliability_tab(case_frame)
    with tabs[5]:
        _expert_selection_tab(data, case_frame)
    with tabs[6]:
        _audit_log_tab(data, case_frame, evaluation_mode=evaluation_mode)
    with tabs[7]:
        _summary_metrics_tab(data)


if __name__ == "__main__":
    main()
