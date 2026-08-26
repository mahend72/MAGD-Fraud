# Full MAGD-Fraud Run Summary

- Status: `passed_with_warnings`
- Scientific checks passed: `True`
- Outputs root: `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs`

## Dataset Summary

| Field | Value |
| --- | --- |
| total_cases | 494793 |
| train_cases | 278627 |
| validation_cases | 119323 |
| test_cases | 96843 |
| train_fraud_prevalence | 0.010207194564776565 |
| validation_fraud_prevalence | 0.01182504630289215 |
| test_fraud_prevalence | 0.014745515938167962 |
| number_of_synthetic_experts | 51 |
| sensitive_attribute_used | customer_age |
| capacity_configured | no |

## Methods Run

- AI-only
- best expert only
- random expert
- numerical threshold
- distance threshold
- oracle upper bound
- learning-to-defer baseline
- MAGD-Heuristic
- MAGD-Learned
- MAGD-Fraud-ValidationTuned
- MAGD-Constrained

## Metrics Files

- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/final_metrics/all_method_metrics.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/final_metrics/assurance_metrics.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/final_metrics/audit_metrics.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/final_metrics/deferral_metrics.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/final_metrics/fairness_metrics.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/final_metrics/per_case_predictions.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/final_metrics/reliance_metrics.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/final_metrics/scientific_checks.json`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/final_metrics/statistical_comparisons.csv`

## Paper Tables

- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/ablation.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/ablation.md`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/ai_assurance.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/ai_assurance.md`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/artifact_table_map.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/auditability.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/auditability.md`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/baseline_comparison.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/baseline_comparison.md`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/budget_matched_results.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/constraint_sensitivity.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/dataset_summary.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/dataset_summary.md`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/expert_reliability_summary.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/human_ai_metrics.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/human_ai_metrics.md`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/implementation_structure.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/implementation_structure.md`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/intervention_calibrated_results.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/intervention_calibrated_results.md`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/magd_risk_calibration.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/magd_risk_calibration.md`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/magd_risk_calibration_fixed_bins.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/magd_validation_tuned_budget_results.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/statistical_comparison.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/statistical_comparison.md`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/statistical_tests.csv`
- `/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/wrong_confident_detection.csv`

## Warnings

- Capacity is not configured; validation will continue.

## Missing Optional Components

- None

## Stage Log

| Step | Stage | Status | Message |
| --- | --- | --- | --- |
| 1 | validate config | completed | dict |
| 2 | validate data splits | completed | paper_tables_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables, summary_table=1 rows |
| 3 | run/load base model predictions | completed | metrics=3 rows |
| 4 | compute numerical confidence and calibration risk | completed | assurance_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/assurance, plots_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/plots |
| 5 | compute distance uncertainty | completed | distance_frame=494793 rows |
| 6 | compute local neighbour reliability | completed | assurance_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/assurance, local_reliability=216166 rows |
| 7 | compute wrong-confident AI risk | completed | metrics=dict, risk_frame=216166 rows |
| 8 | compute MAGD assurance risk | completed | assurance_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/assurance, plots_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/plots |
| 9 | compute MAGD risk calibration | completed | paper_tables_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables, calibration_table=5 rows |
| 10 | estimate expert reliability | completed | output_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/assurance_deferral, expert_reliability=50 rows, routing_decisions=0 rows |
| 11 | run baselines | completed | output_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/baselines, baseline_metrics=6 rows |
| 12 | run learning-to-defer baseline | completed | output_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/baselines, decisions=96843 rows, metrics=1 rows |
| 13 | run MAGD-Heuristic | completed | output_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/assurance_deferral, decisions=96843 rows, metrics=1 rows |
| 14 | run MAGD-Learned | completed | output_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/assurance_deferral, decisions=96843 rows, metrics=1 rows |
| 15 | run MAGD-Fraud validation-tuned | completed | output_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/assurance_deferral, decisions=96843 rows |
| 16 | run MAGD-Constrained initial | completed | output_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/assurance_deferral |
| 17 | run MAGD-Constrained intervention-calibrated | completed | dict |
| 18 | run ablations | completed | ablation_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/ablations, metrics=10 rows |
| 19 | evaluate all methods | completed | /mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/final_metrics; 11 |
| 20 | run budget-matched deferral analysis | completed | 20 rows; /mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/budget_matched_results.csv |
| 21 | run required MAGD risk calibration | completed | 4 rows; /mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/magd_risk_calibration.csv |
| 22 | run constraint sensitivity analysis | completed | 4 rows; /mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/constraint_sensitivity.csv |
| 23 | run paired statistical tests for paper | completed | 2 rows; /mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables/statistical_tests.csv |
| 24 | run statistical tests | completed | 50 rows; /mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/final_metrics |
| 25 | generate audit pack | completed | PosixPath |
| 26 | generate claim-evidence matrix | completed | output_dir=/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/audit_pack, matrix=7 rows |
| 27 | generate paper-ready tables | completed | /mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/data/outputs/paper_tables; ablation, ai_assurance, auditability, baseline_comparison, dataset_summary, human_ai_metrics, implementation_structure, intervention_calibrated_results, magd_risk_calibration, statistical_comparison |
| 28 | run scientific checks | completed | status=passed, critical_failures=0, warnings=0 |
