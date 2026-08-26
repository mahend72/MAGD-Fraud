"""Freeze the final experiment setup used to produce the fresh MAGD-Fraud results.

This script does not change any methodology, config value, or code path. It only
snapshots the *current* config.yaml plus content hashes of the raw and processed
dataset files into a single, versioned manifest, so every downstream table/figure can
be traced back to a specific, pinned setup. Run again only if the config or data
genuinely changes; do not re-run this to "pick" a different setup after seeing results.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml  # noqa: E402


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "sha256": _sha256_file(path),
        "size_bytes": stat.st_size,
    }


def build_manifest(config_path: Path) -> dict[str, object]:
    raw_config_path = config_path.resolve()
    config = load_yaml(raw_config_path)

    processed_dir = Path(config["paths"]["processed_data_dir"])
    if not processed_dir.is_absolute():
        processed_dir = (raw_config_path.parent / processed_dir).resolve()

    raw_dataset_files = {
        "train_file": config["dataset"].get("train_file"),
        "test_file": config["dataset"].get("test_file"),
        "expert_predictions_file": config["dataset"].get("expert_predictions_file"),
        "historical_expert_predictions_file": config["dataset"].get("historical_expert_predictions_file"),
        "capacity_file": config["dataset"].get("capacity_file"),
    }
    raw_hashes: dict[str, object] = {}
    combined = hashlib.sha256()
    for name, rel_path in sorted(raw_dataset_files.items()):
        if rel_path is None:
            raw_hashes[name] = {"path": None, "exists": False}
            continue
        path = Path(rel_path)
        if not path.is_absolute():
            path = (raw_config_path.parent / path).resolve()
        record = _file_record(path)
        raw_hashes[name] = record
        if record.get("exists"):
            combined.update(record["sha256"].encode("utf-8"))

    processed_files = [
        "X_train.csv", "y_train.csv", "train_metadata.csv",
        "X_val.csv", "y_val.csv", "val_metadata.csv",
        "X_test.csv", "y_test.csv", "test_metadata.csv",
        "expert_predictions.csv", "capacity.csv",
    ]
    processed_hashes: dict[str, object] = {}
    for name in processed_files:
        record = _file_record(processed_dir / name)
        processed_hashes[name] = record
        if record.get("exists"):
            combined.update(record["sha256"].encode("utf-8"))

    magd = config["magd"]
    manifest = {
        "manifest_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/freeze_experiment_setup.py (no methodology/config change; snapshot only)",
        "config_path": str(raw_config_path),
        "config_snapshot": config,
        "dataset_provenance": {
            "raw_source_files": raw_hashes,
            "processed_split_files": processed_hashes,
            "combined_dataset_sha256": combined.hexdigest(),
        },
        "split": {
            "train_size": config["split"]["train_size"],
            "val_size": config["split"]["val_size"],
            "test_size": config["split"]["test_size"],
            "random_state": config["split"]["random_state"],
            "stratify": config["split"]["stratify"],
        },
        "seeds": {
            "experiment_seed": config["experiment"]["seed"],
            "split_random_state": config["split"]["random_state"],
            "model_xgboost_random_state": config.get("model", {}).get("xgboost", {}).get("random_state"),
            "model_random_forest_random_state": config.get("model", {}).get("random_forest", {}).get("random_state"),
            "baselines_random_state": config["baselines"]["random_state"],
            "bootstrap_seed": config["experiment"]["seed"],
        },
        "costs": {
            "false_positive": config["costs"]["false_positive"],
            "false_negative": config["costs"]["false_negative"],
            "human_review": config["costs"]["human_review"],
            "escalation": config["costs"]["escalation"],
        },
        "magd_settings": {
            "mode": magd["mode"],
            "risk_weights_config_default": magd["risk_weights"],
            "thresholds": magd["thresholds"],
            "adaptive_threshold": magd["adaptive_threshold"],
            "constrained_policy": magd["constrained_policy"],
            "validation_tuned": magd["validation_tuned"],
            "expert_routing": magd["expert_routing"],
            "costs": magd["costs"],
        },
        "review_budgets": {
            "validation_tuned_deferral_budgets": magd["validation_tuned"]["deferral_budgets"],
            "intervention_constraints_deferral_range": [
                config["intervention_constraints"]["min_deferral_rate"],
                config["intervention_constraints"]["max_deferral_rate"],
            ],
        },
        "panel_k": {
            "escalation_top_k_experts": magd["expert_routing"]["top_k_for_escalation"],
            "note": "top-k reliable experts used for majority-vote escalation (Section 3.6/3.7 of the method section)",
            "neighbourhood_k_distinct_from_panel_k": config["assurance"]["neighbor_top_k"],
            "neighbourhood_k_note": "kNN neighbourhood-reliability k (local_reliability.py) - a different k from the escalation panel size above",
        },
        "validation_selection_rules": {
            "magd_risk_thresholds": {
                "rule": "low_risk/high_risk are the low_quantile/high_quantile of magd_assurance_risk computed on the VALIDATION split only, frozen once, then applied unchanged to train/val/test.",
                "mode": magd["thresholds"]["mode"],
                "low_quantile": magd["thresholds"]["low_quantile"],
                "high_quantile": magd["thresholds"]["high_quantile"],
                "implementation": "src/assurance/magd_risk.py::resolve_magd_risk_thresholds, derive_validation_risk_thresholds",
            },
            "magd_learned_weights": {
                "rule": "SLSQP (fallback: coarse grid) minimizing cost_sensitive_loss + lambda_overreliance*overreliance, evaluated only on a validation-split policy frame.",
                "implementation": "src/deferral/magd_policy.py::learn_policy_variant, evaluate_policy_weights",
            },
            "magd_constrained_weights": {
                "rule": "Same search as magd_learned_weights, with added capacity/fairness/audit-gap penalty terms and a hard correct_rejection floor; evaluated on validation only.",
                "implementation": "src/deferral/magd_policy.py::learn_policy_variant(variant='constrained')",
            },
            "magd_constrained_thresholds": {
                "rule": "SLSQP (fallback: grid derived from the 30/50/70th and 75/90/97th percentiles of the validation magd_assurance_risk distribution) minimizing cost + constraint penalties, subject to overreliance/WCA/deferral-rate/audit-coverage constraints; evaluated on validation only. feasible=False is reported explicitly when no candidate satisfies all constraints.",
                "implementation": "src/deferral/magd_constrained.py::_optimize_policy, _candidate_grids",
            },
            "magd_validation_tuned_policy": {
                "rule": "Grid search over {config-default, equal-weight, and a systematic multi-evidence weight grid} x {deferral budgets [1%,2%,5%,10%,20%]}, minimizing a cost/deferral/recall/F1 objective, evaluated on validation only. The single selected (weights, budget, threshold) triple is frozen and applied to test exactly once.",
                "objective": magd["validation_tuned"]["objective"],
                "implementation": "src/deferral/magd_validation_tuned.py::tune_validation_policy, _weight_candidates",
            },
            "test_label_usage": "Test labels are used only for final offline metrics/statistical tests after every threshold/weight/budget selection above is frozen. Enforced by src/utils/scientific_checks.py::deployable_y_true_guard and verified by dedicated leakage tests (test_calibration.py, test_local_reliability.py, test_magd_deferral.py, test_magd_constrained.py, test_magd_validation_tuned.py).",
        },
    }
    return manifest


def _read_json(path: Path) -> object | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    import csv

    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def attach_resolved_runtime_values(manifest: dict[str, object], outputs_root: Path) -> None:
    """Attach the ACTUAL values this run resolved (not just config defaults/fallbacks) -
    e.g. validation-quantile-derived thresholds are computed at runtime and only live in
    magd_risk_thresholds.json, not in config.yaml itself. This section is what should be
    cited as "the final experiment setup" for this specific completed run.
    """
    thresholds = _read_json(outputs_root / "assurance" / "magd_risk_thresholds.json")
    weights_rows = _read_csv_rows(outputs_root / "magd_policy" / "learned_weights.csv")
    weights_by_variant = {row["variant"]: row for row in weights_rows} if weights_rows else None
    validation_tuned = _read_json(outputs_root / "magd_policy" / "validation_tuned_policy_config.json")
    constrained_config = _read_json(outputs_root / "magd_policy" / "constrained_policy_config.json")
    constrained_diagnostics = _read_json(outputs_root / "magd_policy" / "constrained_policy_diagnostics.json")

    manifest["resolved_runtime_values"] = {
        "note": "Values actually produced by validation-only selection for this run, as opposed to config.yaml's static fallback defaults shown in magd_settings above.",
        "magd_risk_thresholds": thresholds,
        "heuristic_weights": weights_by_variant.get("heuristic") if weights_by_variant else None,
        "learned_weights": weights_by_variant.get("learned") if weights_by_variant else None,
        "constrained_weight_search_weights": weights_by_variant.get("constrained") if weights_by_variant else None,
        "validation_tuned_selected_policy": (
            {k: validation_tuned[k] for k in ["budget", "threshold", "weights", "selection_metric", "objective"]}
            if validation_tuned
            else None
        ),
        "constrained_selected_params": constrained_config.get("selected_params") if constrained_config else None,
        "constrained_feasible_found": constrained_diagnostics.get("feasible_found") if constrained_diagnostics else None,
    }


def main() -> None:
    config_path = ROOT / "config.yaml"
    manifest = build_manifest(config_path)
    outputs_root = Path(manifest["config_snapshot"]["paths"]["outputs_dir"])
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    attach_resolved_runtime_values(manifest, outputs_root)
    out_dir = ROOT / "data" / "outputs" / "final_reproducible_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "frozen_experiment_setup.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"combined_dataset_sha256 = {manifest['dataset_provenance']['combined_dataset_sha256']}")


if __name__ == "__main__":
    main()
