| component | status | notes |
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
