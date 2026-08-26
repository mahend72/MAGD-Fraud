from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_full_magd_experiment as full_magd


ROOT = Path(__file__).resolve().parents[1]


def _write_temp_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text((ROOT / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    return config_path


def test_full_magd_experiment_script_imports() -> None:
    assert hasattr(full_magd, "run_full_magd_experiment")
    assert callable(full_magd.run_full_magd_experiment)
    for stage in full_magd._planned_stages():
        assert stage.runner_name in full_magd.__dict__


def test_full_magd_experiment_dry_run_skips_execution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = _write_temp_config(tmp_path)
    payload = full_magd.run_full_magd_experiment(config_path, project_root=tmp_path, dry_run=True)

    captured = capsys.readouterr().out
    assert payload["status"] == "dry_run"
    assert len(payload["planned_stages"]) == full_magd.STAGE_COUNT
    assert f"[1/{full_magd.STAGE_COUNT}] validate config" in captured
    final_stage = full_magd._planned_stages()[-1]
    assert f"[{final_stage.number}/{full_magd.STAGE_COUNT}] write final run summary" in captured
    assert "run MAGD-Fraud validation-tuned" in captured
    assert not (tmp_path / "data" / "outputs" / "run_summary.json").exists()


def test_full_magd_experiment_stops_clearly_when_required_data_missing(tmp_path: Path) -> None:
    config_path = _write_temp_config(tmp_path)

    with pytest.raises(RuntimeError, match=r"Pipeline stopped at step 2 \(validate data splits\)"):
        full_magd.run_full_magd_experiment(config_path, project_root=tmp_path)


def test_full_magd_experiment_logs_each_stage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = _write_temp_config(tmp_path)
    full_magd.run_full_magd_experiment(config_path, project_root=tmp_path, dry_run=True)

    captured = capsys.readouterr().out
    for stage in full_magd._planned_stages():
        assert f"[{stage.number}/{full_magd.STAGE_COUNT}] {stage.label}" in captured
