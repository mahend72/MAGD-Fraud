from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PIPELINE_STEPS: list[tuple[str, str]] = [
    ("Prepare data", "prepare_data.py"),
    ("Train or load model predictions", "train_or_load_model.py"),
    ("Compute calibration and numerical confidence", "run_calibration.py"),
    ("Compute distance-based uncertainty", "run_distance_uncertainty.py"),
    ("Compute nearest-neighbor evidence", "run_neighbor_evidence.py"),
    ("Compute local neighbourhood reliability", "run_local_reliability.py"),
    ("Compute wrong-confident risk", "run_wrong_confident_detection.py"),
    ("Compute assurance risk", "run_assurance_risk.py"),
    ("Run deferral baselines", "run_baselines.py"),
    ("Run assurance-guided deferral", "run_assurance_deferral.py"),
    ("Evaluate all methods", "evaluate_all_methods.py"),
    ("Run statistical tests", "run_statistical_tests.py"),
    ("Generate audit pack", "generate_audit_pack.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full HAAF FiFAR experiment pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_config(project_root: Path, config_arg: str) -> Path:
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = (project_root / config_arg).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return config_path


def _ensure_scripts_exist(project_root: Path) -> None:
    scripts_dir = project_root / "scripts"
    missing = [script_name for _, script_name in PIPELINE_STEPS if not (scripts_dir / script_name).exists()]
    if missing:
        raise FileNotFoundError(
            "One or more required pipeline scripts are missing: "
            + ", ".join(str(scripts_dir / name) for name in missing)
        )


def _run_step(
    *,
    project_root: Path,
    config_path: Path,
    step_number: int,
    step_name: str,
    script_name: str,
) -> None:
    script_path = project_root / "scripts" / script_name
    command = [sys.executable, str(script_path), "--config", str(config_path)]
    print(f"[{step_number}/{len(PIPELINE_STEPS)}] {step_name}")
    print(f"  Command: {' '.join(command)}")
    try:
        subprocess.run(command, cwd=project_root, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Pipeline stopped at step {step_number} ({step_name}). "
            f"Script failed with exit code {exc.returncode}: {script_path}"
        ) from exc


def main() -> None:
    args = parse_args()
    project_root = _project_root()
    config_path = _resolve_config(project_root, args.config)
    _ensure_scripts_exist(project_root)

    outputs_dir = project_root / "data" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("Starting full HAAF FiFAR experiment pipeline")
    print(f"Project root: {project_root}")
    print(f"Config: {config_path}")
    print(f"Outputs directory: {outputs_dir}")

    for step_number, (step_name, script_name) in enumerate(PIPELINE_STEPS, start=1):
        _run_step(
            project_root=project_root,
            config_path=config_path,
            step_number=step_number,
            step_name=step_name,
            script_name=script_name,
        )

    print("Full experiment pipeline completed successfully.")
    print(f"Artifacts are available under: {outputs_dir}")


if __name__ == "__main__":
    main()
