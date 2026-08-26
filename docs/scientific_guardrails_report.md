# Scientific Guardrails Report

- Status: `passed`
- Critical failures: `0`
- Optional warnings: `0`

## Checks

| Check | Severity | Passed | Message |
| --- | --- | --- | --- |
| deployable_y_true_guard | critical | yes | Deployable routing and deployable risk code do not reference `y_true` inside decision logic. |
| oracle_upper_bound_only | critical | yes | Oracle is restricted to the upper-bound baseline only. |
| accuracy_not_headline_metric | critical | yes | Accuracy is not treated as the headline metric. |
| cost_sensitive_loss_reported | critical | yes | `cost_sensitive_loss` is reported in `all_method_metrics.csv`. |
| overreliance_reported | critical | yes | `overreliance` is reported in `reliance_metrics.csv`. |
| correct_rejection_reported | critical | yes | `correct_rejection` is reported in `reliance_metrics.csv`. |
| wrong_confident_avoidance_reported | critical | yes | `wrong_confident_avoidance_rate` is reported in `reliance_metrics.csv`. |
| audit_coverage_reported | critical | yes | `audit_coverage` is reported in `audit_metrics.csv`. |
| synthetic_experts_labelled | critical | yes | README labels FiFAR experts as synthetic. |
| dashboard_labelled_prototype | critical | yes | README and dashboard label the interface as a research prototype. |
| optional_signals_logged_when_missing | warning | yes | Optional capacity/fairness/drift/business signals are explicitly logged as unavailable when missing. |
| intervention_calibrated_diagnostics_present | critical | yes | Intervention-calibrated MAGD-Constrained diagnostics are present. |
| required_budget_matched_results_present | critical | yes | budget_matched_results.csv is present, non-empty, and schema-valid. |
| required_magd_risk_calibration_present | critical | yes | magd_risk_calibration.csv is present, non-empty, and schema-valid. |
| required_constraint_sensitivity_present | critical | yes | constraint_sensitivity.csv is present, non-empty, and schema-valid. |
| required_statistical_tests_present | critical | yes | statistical_tests.csv is present, non-empty, and schema-valid. |
| required_magd_risk_calibration_thresholds_present | critical | yes | magd_risk_calibration_thresholds.json is present and non-empty. |
| required_constraint_sensitivity_diagnostics_present | critical | yes | constraint_sensitivity_diagnostics.json is present and non-empty. |
| required_artifact_table_map_present | critical | yes | artifact_table_map.csv is present and non-empty. |
| paper_table_placeholder_strings_absent | critical | yes | No paper-table CSV contains placeholder string `--`. |
