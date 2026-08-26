# Implementation Coverage Report

## Scope

This report compares the current `haaf_fifar` repository against the intended paper pipeline for Human-AI assurance in FiFAR-style fraud alert review.

Status labels:

- `IMPLEMENTED`: code path exists, is wired into the current pipeline, and produces the expected output artifacts.
- `PARTIALLY IMPLEMENTED`: some plumbing exists, but the feature is incomplete, offline-only, optional, or inconsistent in a way that matters for the paper claims.
- `MISSING`: no concrete implementation path was found.

## Overall Coverage

| Component | Status | Notes |
| --- | --- | --- |
| 1. Load and preprocess FiFAR-style train/val/test data | IMPLEMENTED | `src/data/load_fifar.py`, `src/data/preprocess.py`, `scripts/prepare_data.py` produce processed splits and dataset summaries. |
| 2. Reuse existing AI scores or train a classifier | IMPLEMENTED | `src/models/train_model.py`, `scripts/train_or_load_model.py` load benchmark scores or train a classifier and write prediction artifacts. |
| 3a. Numerical confidence | IMPLEMENTED | `src/assurance/numerical_confidence.py` and the calibration stage write `numerical_confidence.csv`. |
| 3b. Calibration risk | IMPLEMENTED | `src/models/calibrate_model.py`, `scripts/run_calibration.py` write calibration reports and risk artifacts. |
| 3c. Distance uncertainty | IMPLEMENTED | `src/assurance/distance_uncertainty.py`, `scripts/run_distance_uncertainty.py` write deployable distance-confidence outputs. |
| 3d. Local neighbour error rate | IMPLEMENTED | `src/assurance/local_reliability.py`, `scripts/run_local_reliability.py` write local reliability outputs. |
| 3e. Wrong-confident AI risk | IMPLEMENTED | `src/assurance/wrong_confident_detector.py`, `scripts/run_wrong_confident_detection.py` write `wrong_confident_risk.csv`. |
| 3f. Optional drift risk | PARTIALLY IMPLEMENTED | MAGD can ingest `drift_risk`, but there is still no dedicated drift-risk producer module or standalone script. |
| 3g. Optional business risk | PARTIALLY IMPLEMENTED | MAGD can ingest `business_risk`, but there is still no dedicated producer module or standalone script. |
| 4. Combine signals into MAGD assurance risk | IMPLEMENTED | `src/assurance/magd_risk.py`, `scripts/run_magd_risk.py` produce `magd_risk.csv`. |
| 5. Route cases to AI, expert, or escalation | IMPLEMENTED | `src/deferral/magd_deferral.py`, `src/deferral/expert_routing.py`, `scripts/run_magd_deferral.py` produce routed decisions. |
| 5a. Expected-cost routing | IMPLEMENTED | AI, expert, and escalation expected-cost logic exists and is used in routing. |
| 5b. Fairness-aware routing | IMPLEMENTED | Fairness penalties and fairness-aware expert ranking are wired into policy learning and routing. |
| 5c. Capacity-aware routing | IMPLEMENTED | Capacity state, penalties, and capacity-aware routing outputs are present. |
| 5d. Intervention constraints | PARTIALLY IMPLEMENTED | Constrained MAGD optimizes overreliance, capacity, fairness, and audit penalties and enforces some bounds, but there is not a separate general intervention-constraint engine. |
| 6a. AI-only baseline | IMPLEMENTED | `src/deferral/baselines.py` writes `baselines/ai_only_decisions.csv`. |
| 6b. Best expert baseline | IMPLEMENTED | `src/deferral/baselines.py` writes `best_expert_decisions.csv`. |
| 6c. Random expert baseline | IMPLEMENTED | `src/deferral/baselines.py` writes `random_expert_decisions.csv`. |
| 6d. Confidence-threshold baseline | IMPLEMENTED | `src/deferral/baselines.py` writes `numerical_threshold_decisions.csv`. |
| 6e. Distance-threshold baseline | IMPLEMENTED | `src/deferral/baselines.py` writes `distance_threshold_decisions.csv`. |
| 6f. Learning-to-defer baseline | PARTIALLY IMPLEMENTED | The baseline runs and writes outputs, but it is offline-only and relies on `y_true` to build rejector labels. |
| 6g. Oracle upper bound | IMPLEMENTED | `src/deferral/baselines.py` writes `oracle_upper_bound_decisions.csv` as an offline reference only. |
| 7a. Precision / recall / F1 / PR-AUC | IMPLEMENTED | Reported in `src/evaluation/fraud_metrics.py` and `scripts/evaluate_all_methods.py`. |
| 7b. Cost-sensitive loss | IMPLEMENTED | Reported in `all_method_metrics.csv`, `baseline_comparison.csv`, and `human_ai_metrics.csv`. |
| 7c. Overreliance | IMPLEMENTED | Reported in `src/evaluation/reliance_metrics.py` and downstream tables. |
| 7d. Underreliance | IMPLEMENTED | Reported in `src/evaluation/reliance_metrics.py` and downstream tables. |
| 7e. Correct rejection | IMPLEMENTED | Reported in `src/evaluation/reliance_metrics.py` and downstream tables. |
| 7f. Wrong-confident avoidance | IMPLEMENTED | Reported in `src/evaluation/reliance_metrics.py` and downstream tables. |
| 7g. AI coverage | IMPLEMENTED | Reported in `src/evaluation/deferral_metrics.py` and downstream tables. |
| 7h. Expert deferral rate | IMPLEMENTED | Reported in `src/evaluation/deferral_metrics.py` and downstream tables. |
| 7i. Escalation rate | IMPLEMENTED | Reported in `src/evaluation/deferral_metrics.py` and downstream tables. |
| 7j. Audit coverage | IMPLEMENTED | Reported in `src/evaluation/audit_metrics.py` and the audit pack. |
| 8. MAGD ablations | IMPLEMENTED | `scripts/run_magd_ablations.py` writes ablation decisions, metrics, and plots. |
| 9. MAGD risk calibration analysis | IMPLEMENTED | `src/evaluation/magd_risk_calibration.py`, `scripts/run_magd_risk_calibration.py`, and the full runner produce `magd_risk_calibration.csv`. |
| 10. Statistical comparisons | IMPLEMENTED | `src/evaluation/statistical_tests.py`, `scripts/run_statistical_tests.py`, and `scripts/run_magd_statistical_tests.py` produce statistical outputs. |
| 11. Paper-ready tables | IMPLEMENTED | `scripts/make_paper_tables.py` and `scripts/make_magd_paper_tables.py` produce paper tables and manifests. |
| 12. Audit and claim-evidence outputs | IMPLEMENTED | `scripts/generate_audit_pack.py` and `scripts/generate_claim_evidence_matrix.py` produce the audit pack. |
| 13. Strict no-test-label-leakage in deployable routing | PARTIALLY IMPLEMENTED | Deployable MAGD routing is guarded, but the repository still contains offline artifacts and baselines that legitimately use `y_true` for evaluation or learning. |

## Final File Map

### Core implementation files

- `src/data/load_fifar.py`
  - config normalization, dataset resolution, processed split loading
- `src/data/preprocess.py`
  - split preparation and processed artifacts
- `src/models/train_model.py`
  - model loading/training and prediction artifacts
- `src/models/calibrate_model.py`
  - calibration and numerical confidence artifacts
- `src/assurance/numerical_confidence.py`
  - confidence and calibration-table utilities
- `src/assurance/distance_uncertainty.py`
  - PCA-space distance uncertainty
- `src/assurance/local_reliability.py`
  - neighbour reliability outputs
- `src/assurance/wrong_confident_detector.py`
  - wrong-confident risk outputs
- `src/assurance/magd_risk.py`
  - deployable MAGD risk computation
- `src/evaluation/magd_risk_calibration.py`
  - MAGD-risk calibration analysis
- `src/deferral/baselines.py`
  - AI-only, expert, threshold, and oracle baselines
- `src/deferral/learning_to_defer_baseline.py`
  - offline learning-to-defer baseline
- `src/deferral/expert_routing.py`
  - expert reliability and routing utilities
- `src/deferral/magd_deferral.py`
  - heuristic / learned MAGD deferral
- `src/deferral/magd_constrained.py`
  - intervention-calibrated MAGD-Constrained policy
- `src/deferral/magd_policy.py`
  - policy learning and diagnostics
- `src/evaluation/fraud_metrics.py`
  - precision, recall, F1, PR-AUC, cost loss
- `src/evaluation/reliance_metrics.py`
  - correct reliance, overreliance, correct rejection, wrong-confident avoidance
- `src/evaluation/deferral_metrics.py`
  - AI coverage, deferral, escalation, oracle gap
- `src/evaluation/audit_metrics.py`
  - audit coverage
- `src/evaluation/fairness_metrics.py`
  - fairness summaries when sensitive data are available
- `src/evaluation/statistical_tests.py`
  - paired statistical comparisons
- `src/utils/scientific_checks.py`
  - scientific guardrail checks and report generation
- `src/utils/reporting.py`
  - audit-pack and report helpers
- `scripts/run_full_magd_experiment.py`
  - full end-to-end MAGD pipeline and run summary writer
- `scripts/make_paper_tables.py`
  - canonical paper tables and manifest
- `scripts/generate_audit_pack.py`
  - audit-pack generation
- `scripts/run_statistical_tests.py`
  - paired comparisons

### Documentation files

- `README.md`
- `docs/magd_method_summary.md`
- `docs/reproducibility.md`
- `docs/results_table_guide.md`
- `docs/limitations.md`
- `docs/scientific_guardrails_report.md`
- `docs/magd_architecture_section.tex`
- `docs/magd_facct_method_experimental_setup.tex`
- `docs/magd_paper_architecture.md`
- `docs/implementation_coverage_report.md`

## Final Output Map

### Model and assurance outputs

- `data/outputs/model/`
  - `train_predictions.csv`
  - `val_predictions.csv`
  - `test_predictions.csv`
  - `model_metrics.csv`
  - `model_manifest.pkl`
- `data/outputs/assurance/`
  - `numerical_confidence.csv`
  - `calibration_risk.csv`
  - `distance_uncertainty.csv`
  - `local_reliability.csv`
  - `wrong_confident_risk.csv`
  - `magd_risk.csv`
  - `assurance_risk.csv`
  - `adaptive_thresholds.csv`
  - `threshold_exploration_distance.csv`
  - `calibration_report.csv`
  - `calibration_bins.csv`

### Routing and baseline outputs

- `data/outputs/baselines/`
  - `ai_only_decisions.csv`
  - `best_expert_decisions.csv`
  - `random_expert_decisions.csv`
  - `numerical_threshold_decisions.csv`
  - `distance_threshold_decisions.csv`
  - `learning_to_defer_decisions.csv`
  - `learning_to_defer_metrics.csv`
  - `oracle_upper_bound_decisions.csv`
  - `baseline_metrics.csv`
- `data/outputs/assurance_deferral/`
  - `magd_heuristic_decisions.csv`
  - `magd_learned_decisions.csv`
  - `magd_constrained_initial_decisions.csv`
  - `magd_constrained_decisions.csv`
  - `magd_constrained_calibrated_decisions.csv`
  - `expert_reliability.csv`
  - `magd_metrics.csv`
- `data/outputs/magd_policy/`
  - `learned_weights.csv`
  - `optimization_diagnostics.json`
  - `constrained_policy_config.json`
  - `constrained_policy_diagnostics.json`

### Evaluation and paper outputs

- `data/outputs/final_metrics/`
  - `all_method_metrics.csv`
  - `reliance_metrics.csv`
  - `deferral_metrics.csv`
  - `audit_metrics.csv`
  - `assurance_metrics.csv`
  - `fairness_metrics.csv`
  - `statistical_tests.csv`
  - `statistical_comparisons.csv`
  - `magd_statistical_tests.csv`
  - `scientific_checks.json`
- `data/outputs/paper_tables/`
  - `dataset_summary.csv`
  - `ai_assurance.csv`
  - `baseline_comparison.csv`
  - `human_ai_metrics.csv`
  - `ablation.csv`
  - `auditability.csv`
  - `intervention_calibrated_results.csv`
  - `magd_risk_calibration.csv`
  - `statistical_comparison.csv`
  - `implementation_structure.csv`
  - `table_manifest.json`
- `data/outputs/audit_pack/`
  - `decision_logs.csv`
  - `decision_audit_log.csv`
  - `assurance_summary.csv`
  - `claim_evidence_matrix.csv`
  - `claim_evidence_matrix.md`
  - `audit_coverage.json`
  - `audit_report.md`
  - copied metrics and plots
- `data/outputs/plots/`
  - calibration, threshold, reliability, MAGD-risk, and ablation plots
- `data/outputs/run_summary.json`
- `data/outputs/run_summary.md`

## Final Test Status

- `pytest` passed: `128 passed`
- Full MAGD runner tests passed: `tests/test_full_magd_experiment.py`
- Scientific guardrail tests passed: `tests/test_scientific_checks.py`

## Final Scientific Check Status

- `python scripts/run_scientific_checks.py --config config.yaml` passed
- `data/outputs/final_metrics/scientific_checks.json` reports `status: passed`
- `docs/scientific_guardrails_report.md` reports zero critical failures

## Remaining Limitations

- FiFAR experts are synthetic benchmark experts, not real human reviewers.
- The dashboard is a prototype, not a validated operational interface.
- Optional capacity, fairness, drift, and business signals still depend on data availability, and drift/business still lack dedicated producer modules.
- Human-subject validation has not been established.
- Operational deployment has not been established.
- The learning-to-defer baseline remains offline-only and uses labels to construct rejector targets.
- The repository still contains offline evaluation artifacts that intentionally include `y_true`; deployable routing is guarded, but offline artifacts must not be treated as runtime inputs.

## Manuscript Claims That Are Supported

- MAGD-Fraud is implemented as a multi-evidence assurance-guided deferral pipeline for FiFAR-style fraud review.
- Deployable routing does not use test `y_true`.
- The pipeline supports heuristic, learned, and constrained MAGD variants.
- The pipeline produces baselines, ablations, evaluation metrics, statistical tests, audit-pack outputs, paper tables, scientific checks, and run summaries.
- Cost-sensitive loss, overreliance, correct rejection, wrong-confident avoidance, and audit coverage are reported.
- Optional capacity/fairness/drift/business signals are treated as conditional inputs and are logged when unavailable.
- The repository is explicit that the FiFAR experts are synthetic and that the dashboard is a prototype.

## Manuscript Claims That Are Not Supported

- Any claim of human-subject validation.
- Any claim of operational deployment readiness.
- Any claim that the dashboard is a validated production interface.
- Any claim that the oracle baseline is deployable.
- Any claim that the learning-to-defer baseline is deployable without label use.
- Any claim that drift and business risk are fully realized producer signals in the same sense as the core assurance signals.
- Any claim that MAGD-Fraud dominates every baseline on every metric.

