# HAAF FiFAR

HAAF FiFAR is a research prototype for Human-AI assurance in financial fraud alert review. The main end-to-end method is MAGD-Fraud, short for Multi-evidence Assurance-Guided Deferral. It combines numerical confidence, calibration risk, distance-based uncertainty, local neighbourhood reliability, wrong-confident AI risk, adaptive thresholding, and expert-aware routing.

FiFAR experts in this repository are synthetic benchmark experts, not real human reviewers. The dashboard is also a prototype interface rather than evidence of validated operational use.

## What MAGD-Fraud Is

MAGD-Fraud is a multi-evidence routing layer built on top of a base fraud predictor. It uses deployable assurance signals to decide whether a case should stay with AI, go to a specific expert, or escalate to a top-expert panel. The method is designed to reduce harmful overreliance while keeping routing auditable and test-label leakage out of deployable decisions.

## Research Scope

The repository studies whether multi-evidence assurance signals improve fraud alert review compared with simpler confidence-threshold and distance-threshold baselines. The scope is computational and benchmark-based. It covers:

- FiFAR-style tabular fraud data
- AI score loading or model training
- assurance signals and MAGD risk scoring
- heuristic, learned, and constrained routing
- baseline comparison and statistical testing
- audit-pack and claim-evidence outputs
- paper-ready tables and run summaries

## Repository Structure

- `src/`
  - data loading, preprocessing, assurance signals, deferral logic, evaluation, reporting, and utilities
- `scripts/`
  - runnable entry points for the data, assurance, routing, and paper pipelines
- `data/`
  - raw inputs, processed splits, and generated outputs
- `docs/`
  - paper-facing summaries, reproducibility notes, limitations, and table guides
- `tests/`
  - unit and integration tests for the pipeline

## Installation

```bash
cd haaf_fifar
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main runtime dependencies:

- `pandas`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `streamlit`
- `pytest`

## How To Run The Full Experiment

```bash
python scripts/run_full_magd_experiment.py --config config.yaml
```

That pipeline runs the full MAGD-Fraud sequence end to end:

1. validate config
2. validate data splits
3. run/load base model predictions
4. compute numerical confidence and calibration risk
5. compute distance uncertainty
6. compute local neighbour reliability
7. compute wrong-confident AI risk
8. compute MAGD assurance risk
9. compute MAGD risk calibration
10. estimate expert reliability
11. run baselines
12. run learning-to-defer baseline
13. run MAGD-Heuristic
14. run MAGD-Learned
15. run MAGD-Fraud validation-tuned
16. run MAGD-Constrained initial
17. verify MAGD-Constrained intervention-calibrated outputs
18. run ablations
19. evaluate all methods
20. run budget-matched deferral analysis
21. run required MAGD risk calibration
22. run constraint sensitivity analysis
23. run paired statistical tests for paper
24. run statistical tests
25. generate audit pack
26. generate claim-evidence matrix
27. generate paper-ready tables
28. run scientific checks
29. write final run summary

The runner writes:

- `data/outputs/run_summary.json`
- `data/outputs/run_summary.md`
- `data/outputs/final_metrics/scientific_checks.json`
- `docs/scientific_guardrails_report.md`

## Outputs Generated

Common generated output directories:

- `data/outputs/model/`
- `data/outputs/assurance/`
- `data/outputs/baselines/`
- `data/outputs/assurance_deferral/`
- `data/outputs/magd_policy/`
- `data/outputs/ablations/`
- `data/outputs/final_metrics/`
- `data/outputs/paper_tables/`
- `data/outputs/audit_pack/`
- `data/outputs/plots/`

Useful summary files:

- `data/outputs/final_metrics/all_method_metrics.csv`
- `data/outputs/final_metrics/reliance_metrics.csv`
- `data/outputs/final_metrics/audit_metrics.csv`
- `data/outputs/final_metrics/statistical_comparisons.csv`
- `data/outputs/final_metrics/per_case_predictions.csv`
- `data/outputs/paper_tables/budget_matched_results.csv`
- `data/outputs/paper_tables/magd_risk_calibration.csv`
- `data/outputs/paper_tables/constraint_sensitivity.csv`
- `data/outputs/paper_tables/statistical_tests.csv`
- `data/outputs/paper_tables/artifact_table_map.csv`
- `data/outputs/paper_tables/table_manifest.json`
- `data/outputs/run_summary.json`
- `data/outputs/run_summary.md`

## How To Interpret Paper Tables

The paper tables are generated from the same artifacts used by the experiment runner. The short version is:

- dataset and setting tables describe the benchmark and its splits
- assurance tables describe the model-assurance signals
- baseline comparison tables compare MAGD-Fraud against threshold and oracle baselines
- reliance tables focus on overreliance, correct rejection, and wrong-confident avoidance
- ablation tables show what each assurance signal contributes
- auditability tables summarize how well routing decisions can be reconstructed
- statistical comparison tables report paired tests across methods

Additional paper evaluation artifacts are written by the full MAGD-Fraud runner:

- `budget_matched_results.csv`: budget-matched deferral table/figure comparing MAGD-Fraud, learning-to-defer, distance threshold, and confidence threshold at review budgets 0.01, 0.02, 0.05, 0.10, and 0.20.
- `magd_risk_calibration.csv`: MAGD risk calibration table/figure using held-out test quartiles; `magd_risk_calibration_thresholds.json` records the bin thresholds.
- `constraint_sensitivity.csv`: constraint sensitivity table/figure for strict, moderate, relaxed, and high-review settings; `data/outputs/magd_policy/constraint_sensitivity_diagnostics.json` records feasibility details.
- `statistical_tests.csv`: statistical comparison table using paired bootstrap intervals and McNemar tests over test-case predictions.
- `artifact_table_map.csv`: map from each paper table or figure to its source CSV and description.

For a table-by-table map from label to output file and source script, see [docs/results_table_guide.md](docs/results_table_guide.md).

## No-Leakage Policy

Deployable routing functions do not use test `y_true`. Test labels are reserved for offline evaluation, paper tables, and scientific checks. Validation labels may be used for policy learning, but only inside the learning stages that are explicitly offline and non-deployable. Any method that relies on `y_true` for a deployable decision is treated as a violation.

## Synthetic Expert Limitation

The FiFAR experts are synthetic benchmark experts. They are useful for testing routing and audit logic, but they do not establish how real reviewers would behave in production. This repository therefore supports computational assurance analysis, not a claim of human-subject evidence.

## Dashboard Prototype Limitation

The dashboard is a prototype interface for inspection and demonstration. It is not evidence of validated operational deployment, and it should not be treated as a completed production tool.

## Citation / Acknowledgement Placeholder

If you use this repository in a paper or internal report, cite the project or acknowledge the implementation once the final citation text is available. Placeholder:

> `MAGD-Fraud / HAAF FiFAR implementation, Human-AI assurance pipeline for financial fraud alert review, repository citation to be added.`

## Dataset

FiFAR (Financial Fraud Alert Review) is a public synthetic benchmark for learning-to-defer research; it is **not redistributed in this repository**. Obtain it from its original public source and cite the FiFAR paper/dataset directly.

> TODO: insert the official FiFAR paper/dataset citation and DOI/URL here (do not substitute an unverified link).

`data/`, `data/raw/`, `data/processed/`, and `data/ICAIF_KAGGLE/` are excluded from version control via `.gitignore` because they hold the downloaded benchmark, generated splits, and large per-case model/assurance outputs. After downloading FiFAR, place it locally using one of the layouts below so the pipeline can find it; only small, curated reproducibility artifacts under `data/outputs/` (final metrics, paper tables, policy configs, figure-generation data) are tracked in Git.

## Expected FiFAR Data Layout

Place FiFAR files under `data/raw/` if you want to use the generic inspector, or point `config.yaml` directly to known train/test files.

Example raw layout:

```text
haaf_fifar/
  data/
    raw/
      fifar_cases.csv
      fifar_expert_predictions.csv
      fifar_capacity.csv
```

Example direct-file layout:

```text
haaf_fifar/
  data/
    ICAIF_KAGGLE/
      testbed/
        train/
          small__regular/
            train.csv
        test/
          test.csv
          test_expert_pred.csv
      experts/
        train_predictions.csv
```

Supported file types for inspection:

- `.csv`
- `.tsv`
- `.parquet`
- `.xlsx`
- `.xls`
- `.json`

No synthetic data is created for the main experiment.

## `config.yaml` Guide

`config.yaml` controls paths, canonical column mappings, split logic, assurance thresholds, costs, and routing weights.

Important sections:

- `paths`
  - where raw, processed, and output files live
- `dataset`
  - `main_file` or predefined `train_file` and `test_file`
  - expert and capacity table paths
- `columns`
  - `case_id`, label columns, model score column, expert prediction column, sensitive attributes
- `split`
  - train/validation/test fractions and stratification
- `model`
  - operating threshold and cost-sensitive loss weights
- `assurance`
  - calibration bins, distance thresholds, wrong-confident settings, assurance-risk weights
- `magd`
  - evidence toggles, MAGD weights, adaptive-threshold settings, expert-routing penalties, and MAGD cost parameters
- `baselines`
  - threshold defaults for baseline deferral methods
- `assurance_deferral`
  - escalation expert count and routing penalty weights

The project does not assume fixed FiFAR column names. Update the mappings in `config.yaml` after inspection.

## Core Concepts

### Numerical Confidence

For binary fraud prediction:

`numerical_confidence = max(p, 1 - p)`

This captures how certain the model appears under its own score distribution. It is used for calibration analysis, threshold baselines, and wrong-confident risk.

### Distance-Based Uncertainty

Tabular features are projected into a PCA embedding space. Fraud and non-fraud class centroids are computed from training data. A test case gets higher distance confidence when it lies closer to the centroid of its predicted class. Distance uncertainty is:

`distance_uncertainty = 1 - distance_confidence`

This signal is useful, but centroid-only uncertainty is limited for fraud because historical fraud patterns are often multimodal and locally irregular.

### MAGD-Fraud

MAGD-Fraud extends distance uncertainty instead of replacing it. The method keeps distance uncertainty as one signal, then adds:

- calibration risk
- local neighbour error rate from similar historical cases
- wrong-confident AI risk
- optional business, fairness, drift, and capacity signals

The deployable MAGD assurance score is a weighted combination of these signals. It is then combined with an adaptive threshold and expert-aware expected-cost routing.

At a high level:

- low MAGD risk with low AI cost: use AI
- medium MAGD risk or lower expert cost: defer to the best available expert
- high MAGD risk or high wrong-confident risk: escalate via majority vote among top reliable experts

### Assurance Risk

Assurance risk combines the main model-assurance signals:

- calibration risk
- distance uncertainty
- neighbor error rate
- wrong-confident risk
- optional business risk

The normalized score is mapped into:

- `low`
- `medium`
- `high`

These categories drive the assurance-guided routing policy.

### Baselines

The project compares the main method against:

- AI-only
- best expert only
- random expert
- numerical confidence threshold
- distance confidence threshold
- learning-to-defer baseline
- oracle upper bound

### Metrics

The evaluation suite includes:

- fraud metrics: precision, recall, F1, PR-AUC, ROC-AUC, false positives, false negatives, false positive rate, false negative rate, cost-sensitive loss
- reliance metrics: correct reliance, correct rejection, overreliance, underreliance, wrong-confident avoidance
- deferral metrics: AI coverage, human deferral rate, escalation rate, deferral precision, deferral recall, capacity violation rate, oracle gap
- fairness metrics: group-wise FPR/FNR/cost and disparities when sensitive attributes exist
- audit metrics: audit coverage, complete logs, missing evidence rate, missing rationale rate

## Additional Commands

### Inspect Data

```bash
python scripts/inspect_fifar.py --config config.yaml
```

### Prepare Data

```bash
python scripts/prepare_data.py --config config.yaml
```

### Run Tests

```bash
pytest
```

All experiment outputs are written to `data/outputs/`.

### Launch Dashboard

```bash
streamlit run src/dashboard/app.py
```

The dashboard includes:

- Overview
- Case Review
- MAGD Case Review
- Threshold Explorer
- MAGD Risk Explorer
- Results
- Ablation Results
- Audit Evidence
- Audit Logs

The dashboard is a research prototype for inspecting computational outputs. It is not evidence of human-subject validation.

## Outputs

Important outputs include:

- `data/processed/`
- `data/outputs/model/`
- `data/outputs/assurance/`
- `data/outputs/baselines/`
- `data/outputs/assurance_deferral/`
- `data/outputs/ablations/`
- `data/outputs/final_metrics/`
- `data/outputs/audit_pack/`
- `data/outputs/paper_tables/`

Key MAGD-Fraud artifacts include:

- `data/outputs/assurance/local_reliability.csv`
- `data/outputs/assurance/wrong_confident_risk.csv`
- `data/outputs/assurance/magd_risk.csv`
- `data/outputs/assurance/adaptive_thresholds.csv`
- `data/outputs/assurance_deferral/expert_reliability.csv`
- `data/outputs/assurance_deferral/magd_routing_decisions.csv`
- `data/outputs/assurance_deferral/magd_fraud_decisions.csv`
- `data/outputs/baselines/learning_to_defer_decisions.csv`
- `data/outputs/magd_policy/learned_weights.csv`
- `data/outputs/ablations/ablation_metrics.csv`
- `data/outputs/final_metrics/magd_statistical_tests.csv`
- `data/outputs/audit_pack/claim_evidence_matrix.csv`
- `data/outputs/paper_tables/table_*.csv`

Interpretation guide:

- lower `cost_sensitive_loss` is better
- lower `overreliance` is better
- higher `correct_rejection` is better
- higher `wrong_confident_avoidance` is better
- higher `audit_coverage` is better
- ablation outputs show whether full MAGD improves over distance-only uncertainty

## Architecture And Workflow

High-level system design:

- [ARCHITECTURE.md](./ARCHITECTURE.md)

Step-by-step execution guide:

- [WORKFLOW.md](./WORKFLOW.md)

## Testing

Run the test suite with:

```bash
pytest
```

Coverage includes:

- expected calibration error
- distance confidence range
- assurance risk range
- reliance metric calculations
- cost-sensitive loss
- audit coverage

## Research Notes

## Scientific Limitations

- Distance uncertainty still depends on the quality of the embedding and can miss structure that is not well represented in the chosen projection.
- Local neighbourhood reliability is historical and retrospective. If the data distribution shifts, neighbour-based evidence can become stale.
- Wrong-confident risk is deployable because it avoids test labels at decision time, but its offline validation still depends on historical truth labels.
- Expert reliability, fairness, and capacity estimates are only as good as the available historical expert data. Sparse expert histories make routing noisier.
- Capacity-aware comparisons may collapse to the unconstrained setting when `dataset.capacity_file` is not configured.
- Business risk and drift risk are optional in the current implementation. If they are not provided, MAGD-Fraud falls back to a narrower evidence set.
- This repository is a research prototype. Several outputs, including audit-pack generation and some evaluation tables, still include legacy assurance-guided artifacts alongside MAGD-Fraud outputs for comparison.

- If model scores already exist in FiFAR, the pipeline reuses them.
- If scores are missing, the model stage can train XGBoost with RandomForest fallback.
- Capacity-aware routing is supported, but if no capacity table is configured the experiment runs unconstrained and logs that limitation.
- Oracle results are reported only as an upper bound, never as a deployable method.
