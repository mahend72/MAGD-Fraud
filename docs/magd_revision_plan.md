# MAGD-Fraud Revision Plan

## 1. Existing Files To Reuse

- Data loading and preprocessing
  - `src/data/load_fifar.py`
  - `src/data/preprocess.py`
  - `scripts/prepare_data.py`
- Model training and prediction loading
  - `src/models/train_model.py`
  - `src/models/calibrate_model.py`
  - `src/models/embeddings.py`
  - `scripts/train_or_load_model.py`
  - `scripts/run_calibration.py`
- Distance uncertainty and neighbour evidence
  - `src/assurance/distance_uncertainty.py`
  - `src/assurance/explanation_neighbors.py`
  - `scripts/run_distance_uncertainty.py`
  - `scripts/run_neighbor_evidence.py`
- Existing assurance and routing layers
  - `src/assurance/local_reliability.py`
  - `src/assurance/wrong_confident_detector.py`
  - `src/assurance/magd_risk.py`
  - `src/deferral/adaptive_threshold.py`
  - `src/deferral/expert_routing.py`
  - `src/deferral/magd_deferral.py`
  - `scripts/run_magd_risk.py`
  - `scripts/run_adaptive_threshold.py`
  - `scripts/run_expert_routing.py`
  - `scripts/run_magd_deferral.py`
- Baselines and evaluation
  - `src/deferral/baselines.py`
  - `scripts/run_baselines.py`
  - `scripts/evaluate_all_methods.py`
  - `src/evaluation/*.py`
  - `scripts/run_magd_statistical_tests.py`
- Audit pack and dashboard
  - `scripts/generate_audit_pack.py`
  - `src/utils/reporting.py`
  - `src/dashboard/app.py`
- Existing tests
  - `tests/test_magd_config.py`
  - `tests/test_local_reliability.py`
  - `tests/test_wrong_confident_detector.py`
  - `tests/test_magd_risk.py`
  - `tests/test_adaptive_threshold.py`
  - `tests/test_expert_routing.py`
  - `tests/test_magd_deferral.py`
  - `tests/test_assurance_metrics.py`

## 2. Existing Files To Modify

- `config.yaml`
- `src/utils/io.py`
- `src/assurance/local_reliability.py`
- `src/assurance/wrong_confident_detector.py`
- `src/assurance/magd_risk.py`
- `src/deferral/magd_deferral.py`
- `scripts/evaluate_all_methods.py`
- `scripts/run_magd_ablations.py`
- `scripts/run_magd_statistical_tests.py`
- `scripts/make_magd_paper_tables.py`
- `scripts/generate_audit_pack.py`
- `scripts/run_full_magd_experiment.py`
- `src/utils/scientific_checks.py`
- `README.md`

## 3. New Files To Add

- `src/deferral/magd_policy.py`
- `src/deferral/learning_to_defer_baseline.py`
- `src/assurance/claim_evidence_matrix.py`
- `scripts/run_magd_policy.py`
- `scripts/run_learning_to_defer_baseline.py`
- `scripts/generate_claim_evidence_matrix.py`
- `docs/magd_revision_plan.md`

## 4. Expected Outputs

- Core MAGD assurance
  - `data/outputs/assurance/local_reliability.csv`
  - `data/outputs/assurance/wrong_confident_risk.csv`
  - `data/outputs/assurance/magd_risk.csv`
  - `data/outputs/assurance/adaptive_thresholds.csv`
- MAGD policy learning
  - `data/outputs/magd_policy/learned_weights.csv`
  - `data/outputs/magd_policy/optimization_diagnostics.json`
- Routing
  - `data/outputs/assurance_deferral/expert_reliability.csv`
  - `data/outputs/assurance_deferral/magd_fraud_decisions.csv`
  - `data/outputs/assurance_deferral/magd_fraud_metrics.csv`
- Stronger baselines
  - `data/outputs/baselines/learning_to_defer_decisions.csv`
  - `data/outputs/baselines/learning_to_defer_metrics.csv`
- Ablations and comparisons
  - `data/outputs/ablations/ablation_metrics.csv`
  - `data/outputs/ablations/ablation_decisions_{variant}.csv`
  - `data/outputs/final_metrics/assurance_metrics.csv`
  - `data/outputs/final_metrics/magd_statistical_tests.csv`
- Assurance case and paper outputs
  - `data/outputs/audit_pack/claim_evidence_matrix.csv`
  - `data/outputs/audit_pack/claim_evidence_matrix.md`
  - `data/outputs/paper_tables/*.csv`
  - `data/outputs/paper_tables/*.md`

## 5. How MAGD-Fraud Extends The Current Distance-Uncertainty Pipeline

- The previous pipeline used distance uncertainty as the main deployable uncertainty signal.
- MAGD-Fraud keeps distance uncertainty but adds:
  - calibration risk
  - local neighbour error rate from similar historical cases
  - wrong-confident AI risk
  - adaptive thresholding
  - expert reliability, fairness, and capacity-aware routing
- MAGD-Fraud adds three policy variants:
  - heuristic
  - learned
  - constrained
- Weight learning uses validation data only.
- Test labels are reserved for final reporting and offline evaluation.

## 6. Commands For Testing

```bash
pytest
```

Focused checks:

```bash
pytest tests/test_magd_config.py
pytest tests/test_local_reliability.py
pytest tests/test_wrong_confident_detector.py
pytest tests/test_magd_risk.py
pytest tests/test_adaptive_threshold.py
pytest tests/test_expert_routing.py
pytest tests/test_magd_deferral.py
pytest tests/test_assurance_metrics.py
```

Pipeline commands:

```bash
python scripts/run_magd_policy.py --config config.yaml
python scripts/run_learning_to_defer_baseline.py --config config.yaml
python scripts/run_magd_ablations.py --config config.yaml
python scripts/run_magd_statistical_tests.py --config config.yaml
python scripts/generate_claim_evidence_matrix.py --config config.yaml
python scripts/make_magd_paper_tables.py --config config.yaml
python scripts/run_full_magd_experiment.py --config config.yaml
```
