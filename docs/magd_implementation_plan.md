# MAGD-Fraud Implementation Plan

## 1. Existing files to reuse

- `src/data/load_fifar.py`
  - Canonical FiFAR data loading, column resolution, expert predictions, capacity inputs.
- `src/models/calibrate_model.py`
  - Existing calibration run wrapper and output conventions.
- `src/assurance/numerical_confidence.py`
  - Numerical confidence, calibration table, ECE, and case-level `calibration_risk`.
- `src/assurance/distance_uncertainty.py`
  - Distance-based confidence/uncertainty computation and threshold exploration pattern.
- `src/assurance/explanation_neighbors.py`
  - Nearest-neighbor retrieval and per-case neighbor summary generation.
- `src/assurance/wrong_confident_detector.py`
  - Existing multi-signal wrong-confident risk scaffold.
- `src/assurance/assurance_risk.py`
  - Current risk aggregation pattern, risk categories, and action mapping.
- `src/deferral/capacity_assignment.py`
  - Capacity-aware expert allocation utilities.
- `src/deferral/baselines.py`
  - Baseline evaluation/logging structure and expert availability handling.
- `src/deferral/assurance_deferral.py`
  - Current assurance-guided routing logic, expert reliability estimation, and decision log format.
- `src/evaluation/fraud_metrics.py`
  - Main fraud task metrics.
- `src/evaluation/reliance_metrics.py`
  - Human/AI reliance behavior metrics.
- `src/evaluation/deferral_metrics.py`
  - AI coverage, human deferral, escalation, oracle gap, and capacity metrics.
- `src/evaluation/audit_metrics.py`
  - Audit coverage and evidence completeness scoring.
- `src/utils/reporting.py`
  - Audit-pack summaries, markdown report generation, and artifact copy helpers.
- `scripts/run_calibration.py`
- `scripts/run_distance_uncertainty.py`
- `scripts/run_neighbor_evidence.py`
- `scripts/run_wrong_confident_detection.py`
- `scripts/run_assurance_risk.py`
- `scripts/run_baselines.py`
- `scripts/run_assurance_deferral.py`
- `scripts/evaluate_all_methods.py`
- `scripts/generate_audit_pack.py`
- `scripts/run_full_experiment.py`
- Existing tests:
  - `tests/test_calibration.py`
  - `tests/test_distance_uncertainty.py`
  - `tests/test_neighbor_evidence.py`
  - `tests/test_wrong_confident_detector.py`
  - `tests/test_assurance_risk.py`
  - `tests/test_audit_metrics.py`

## 2. Existing files to modify

- `config.yaml`
  - Add MAGD-Fraud-specific config block for signal weights, evidence fusion, routing thresholds, and output naming.
- `src/assurance/wrong_confident_detector.py`
  - Extend from current weighted wrong-confident score to MAGD-Fraud evidence fusion features.
- `src/assurance/assurance_risk.py`
  - Replace or augment the current final risk combiner with MAGD-Fraud multi-evidence assurance aggregation.
- `src/deferral/assurance_deferral.py`
  - Update routing policy so decisions are driven by MAGD-Fraud risk and evidence-aware expert escalation logic.
- `src/deferral/baselines.py`
  - Keep current baselines, but register MAGD-Fraud as the new main method alongside them.
- `scripts/run_assurance_risk.py`
  - Point to the MAGD-Fraud risk builder.
- `scripts/run_assurance_deferral.py`
  - Point to the MAGD-Fraud router.
- `scripts/evaluate_all_methods.py`
  - Add MAGD-Fraud outputs/method name to the evaluation registry.
- `scripts/generate_audit_pack.py`
  - Include MAGD-Fraud decision logs and MAGD-specific evidence columns in the audit pack.
- `scripts/run_full_experiment.py`
  - Rename or reorder steps if MAGD-Fraud introduces a dedicated aggregation stage.
- `src/utils/reporting.py`
  - Update narrative text, assurance summary columns, and audit-report wording from generic assurance-guided deferral to MAGD-Fraud.
- `README.md`
- `ARCHITECTURE.md`
- `WORKFLOW.md`
  - Update documentation to reflect MAGD-Fraud as the primary algorithm.

## 3. New files to create

- `src/assurance/magd_fraud.py`
  - Main MAGD-Fraud evidence fusion module that combines calibration, distance, neighbor, wrong-confident, and optional business/expert signals into a unified per-case score.
- `scripts/run_magd_fraud.py`
  - Optional dedicated entrypoint if MAGD-Fraud is separated from the current `run_assurance_risk.py`.
- `tests/test_magd_fraud.py`
  - Unit tests for MAGD-Fraud scoring, bounds, evidence handling, and category assignment.
- `tests/test_magd_deferral.py`
  - Unit tests for MAGD-Fraud routing behavior, expert selection, and escalation decisions.

## 4. Expected outputs

- New or updated assurance-stage artifact:
  - `data/outputs/assurance/magd_fraud_risk.csv`
- New or updated deferral-stage artifacts:
  - `data/outputs/assurance_deferral/magd_fraud_decisions.csv`
  - `data/outputs/assurance_deferral/magd_fraud_metrics.csv`
- Evaluation outputs updated to include MAGD-Fraud as a named method:
  - `data/outputs/final_metrics/all_method_metrics.csv`
  - `data/outputs/final_metrics/deferral_metrics.csv`
  - `data/outputs/final_metrics/audit_metrics.csv`
  - `data/outputs/final_metrics/reliance_metrics.csv`
- Audit-pack outputs updated to summarize MAGD-Fraud evidence and routing:
  - `data/outputs/audit_pack/assurance_summary.csv`
  - `data/outputs/audit_pack/final_metrics.csv`
  - `data/outputs/audit_pack/decision_logs.csv`
  - `data/outputs/audit_pack/audit_report.md`

## 5. Any missing dependencies

- No obvious new Python dependency is required for a first MAGD-Fraud extension.
- The current stack already covers the needed pieces:
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `PyYAML`
- If MAGD-Fraud later adds a more advanced fusion model or explainability layer, that may justify extra dependencies, but nothing in the current inspection makes that necessary yet.
