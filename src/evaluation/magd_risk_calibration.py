from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.io import load_yaml


@dataclass
class MagdRiskCalibrationArtifacts:
    calibration_table: pd.DataFrame
    paper_tables_dir: Path


def _resolve_output_dirs(config_path: Path) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    assurance_dir = outputs_root / "assurance"
    paper_tables_dir = outputs_root / "paper_tables"
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    return assurance_dir, paper_tables_dir


def _load_required_csv(path: Path, *, required_columns: set[str], friendly_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {friendly_name} at {path}.")
    frame = pd.read_csv(path)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{friendly_name} is missing required columns: {sorted(missing)}")
    return frame


def _resolve_decision_frame(assurance_dir: Path) -> pd.DataFrame | None:
    candidates = [
        assurance_dir.parent / "deferral" / "magd_deferral_decisions.csv",
        assurance_dir.parent / "final_metrics" / "magd_decisions.csv",
        assurance_dir.parent / "deferral" / "assurance_deferral_decisions.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            if {"case_id", "selected_route"}.issubset(frame.columns):
                return frame
    return None


def build_magd_risk_calibration_table(config_path: str | Path) -> pd.DataFrame:
    resolved_config_path = Path(config_path).resolve()
    assurance_dir, _ = _resolve_output_dirs(resolved_config_path)

    magd = _load_required_csv(
        assurance_dir / "magd_risk.csv",
        required_columns={"case_id", "split", "magd_assurance_risk"},
        friendly_name="MAGD risk outputs",
    )
    wrong_conf = _load_required_csv(
        assurance_dir / "wrong_confident_risk.csv",
        required_columns={"case_id", "split", "wrong_confident_label_offline"},
        friendly_name="wrong-confident risk outputs",
    )

    y_true_frame = None
    for candidate_name in ["calibration_risk.csv", "distance_uncertainty.csv"]:
        candidate_path = assurance_dir / candidate_name
        if candidate_path.exists():
            candidate = pd.read_csv(candidate_path)
            if {"case_id", "split", "y_true", "ai_pred"}.issubset(candidate.columns):
                y_true_frame = candidate[["case_id", "split", "y_true", "ai_pred"]].drop_duplicates(
                    subset=["case_id", "split"],
                    keep="first",
                )
                break
    if y_true_frame is None:
        raise ValueError("Missing offline labels (`y_true`, `ai_pred`) required for MAGD risk calibration analysis.")

    working = (
        magd.loc[magd["split"].astype(str) == "test", ["case_id", "split", "magd_assurance_risk"]]
        .merge(
            wrong_conf[["case_id", "split", "wrong_confident_label_offline"]],
            on=["case_id", "split"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            y_true_frame,
            on=["case_id", "split"],
            how="left",
            validate="one_to_one",
        )
    )
    working["wrong_confident_label_offline"] = (
        pd.to_numeric(working["wrong_confident_label_offline"], errors="coerce").fillna(0).astype(int)
    )
    working["ai_error"] = (
        pd.to_numeric(working["ai_pred"], errors="coerce").fillna(0).astype(int)
        != pd.to_numeric(working["y_true"], errors="coerce").fillna(0).astype(int)
    ).astype(int)

    decision_frame = _resolve_decision_frame(assurance_dir)
    if decision_frame is not None:
        decision_frame = decision_frame[["case_id", "selected_route"]].drop_duplicates(subset=["case_id"], keep="first")
        working = working.merge(decision_frame, on="case_id", how="left")
        working["deferral_indicator"] = working["selected_route"].astype(str).isin(["Human Expert", "Escalate"]).astype(float)
    else:
        working["deferral_indicator"] = pd.NA

    bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
    bin_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    working["magd_risk_bin"] = pd.cut(
        working["magd_assurance_risk"].astype(float),
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=False,
    )

    rows: list[dict[str, object]] = []
    for label in bin_labels:
        bucket = working.loc[working["magd_risk_bin"].astype(str) == label]
        deferral_rate: object
        if bucket.empty or bucket["deferral_indicator"].isna().all():
            deferral_rate = ""
        else:
            deferral_rate = float(pd.to_numeric(bucket["deferral_indicator"], errors="coerce").dropna().mean())
        rows.append(
            {
                "magd_risk_bin": label,
                "cases": int(len(bucket)),
                "ai_error_rate": float(bucket["ai_error"].mean()) if len(bucket) else 0.0,
                "wrong_confident_rate": float(bucket["wrong_confident_label_offline"].mean()) if len(bucket) else 0.0,
                "deferral_rate": deferral_rate,
            }
        )

    return pd.DataFrame(rows)


def run_magd_risk_calibration(config_path: str | Path) -> MagdRiskCalibrationArtifacts:
    resolved_config_path = Path(config_path).resolve()
    _, paper_tables_dir = _resolve_output_dirs(resolved_config_path)
    calibration_table = build_magd_risk_calibration_table(resolved_config_path)
    calibration_table.to_csv(paper_tables_dir / "magd_risk_calibration_fixed_bins.csv", index=False)
    return MagdRiskCalibrationArtifacts(calibration_table=calibration_table, paper_tables_dir=paper_tables_dir)
