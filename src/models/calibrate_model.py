from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.assurance.calibration import run_calibration_assurance
from src.utils.io import load_yaml


@dataclass
class CalibrationRunArtifacts:
    assurance_dir: Path
    plots_dir: Path
    calibration_report: pd.DataFrame
    assurance_base: pd.DataFrame
    ece: float


def _resolve_output_dirs(config_path: Path) -> tuple[Path, Path, Path]:
    config = load_yaml(config_path)
    outputs_root = Path(config.get("paths", {}).get("outputs_dir", "data/outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()

    model_dir = outputs_root / "model"
    assurance_dir = outputs_root / "assurance"
    plots_dir = outputs_root / "plots"
    assurance_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, assurance_dir, plots_dir


def _calibration_bins(config: dict) -> int:
    return int(config.get("assurance", {}).get("calibration_bins", 10))


def _load_test_predictions(model_dir: Path) -> pd.DataFrame:
    test_predictions_path = model_dir / "test_predictions.csv"
    if not test_predictions_path.exists():
        raise FileNotFoundError(
            f"Missing model predictions at {test_predictions_path}. "
            "Run `python scripts/train_or_load_model.py --config config.yaml` first."
        )
    frame = pd.read_csv(test_predictions_path)
    required_columns = {"case_id", "y_true", "ai_score", "ai_pred"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"Model predictions file is missing required columns: {sorted(missing)}"
        )
    return frame


def _plot_reliability_diagram(calibration_table: pd.DataFrame, plot_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to generate the reliability diagram. "
            "Install dependencies from requirements.txt and rerun calibration."
        ) from exc

    populated = calibration_table.dropna(subset=["average_confidence", "observed_accuracy"]).copy()

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    if not populated.empty:
        plt.plot(
            populated["average_confidence"],
            populated["observed_accuracy"],
            marker="o",
            linewidth=2,
            label="Observed",
        )
    plt.xlabel("Average confidence")
    plt.ylabel("Observed accuracy")
    plt.title("Reliability Diagram")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()


def run_calibration(config_path: str | Path) -> CalibrationRunArtifacts:
    resolved_config_path = Path(config_path).resolve()
    model_dir, assurance_dir, plots_dir = _resolve_output_dirs(resolved_config_path)
    artifacts = run_calibration_assurance(resolved_config_path)
    calibration_report = pd.read_csv(assurance_dir / "calibration_report.csv")
    assurance_base = pd.read_csv(assurance_dir / "test_assurance_base.csv")
    _plot_reliability_diagram(calibration_report, plots_dir / "reliability_diagram.png")

    return CalibrationRunArtifacts(
        assurance_dir=assurance_dir,
        plots_dir=plots_dir,
        calibration_report=calibration_report,
        assurance_base=assurance_base,
        ece=artifacts.validation_ece,
    )
