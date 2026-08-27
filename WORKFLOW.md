# MAGD-Fraud Workflow

**Learning AI Reliance Risk for Assurance-Guided Human–AI Routing in Financial Fraud Review**

## Purpose

This document describes the **canonical experimental workflow used by the submitted MAGD-Fraud paper**.

The current paper does **not** use the older `run_full_experiment.py` pipeline as the primary entry point. The canonical end-to-end runner is:

```bash
python scripts/run_full_magd_experiment.py --config config.yaml
```

MAGD-Fraud separates:

```text
Base AI prediction
        ↓
Assurance evidence
        ↓
Learned AI-reliance risk
        ↓
Human–AI routing
        ↓
Evaluation + reviewer workload + audit evidence
```

The workflow below follows that structure.

---

# 1. Environment Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
```

Run the test suite before starting a full experiment:

```bash
pytest -q
```

---

# 2. Obtain the FiFAR Benchmark

The FiFAR benchmark is **not redistributed** through this repository.

Public source:

```text
https://doi.org/10.6084/m9.figshare.28351172
```

Place the downloaded benchmark files in the locations expected by `config.yaml`.

Large benchmark files, processed feature matrices, embeddings, and per-case decision logs are intentionally excluded from Git.

---

# 3. Inspect the Dataset

If the FiFAR schema or file layout needs to be checked:

```bash
python scripts/inspect_fifar.py --config config.yaml
```

Typical outputs include:

- file and column summaries;
- candidate label fields;
- model-score fields;
- expert-prediction fields;
- missingness diagnostics.

This step is diagnostic and does not change the scientific protocol.

---

# 4. Prepare Train / Development / Test Data

Run:

```bash
python scripts/prepare_data.py --config config.yaml
```

The submitted-paper evaluation uses:

| Split | Cases | Fraud prevalence |
|---|---:|---:|
| Train | 278,627 | 1.02% |
| Validation / development | 119,323 | 1.18% |
| Benchmark test | 96,843 | 1.47% |

Typical generated files include:

```text
data/processed/X_train.csv
data/processed/X_val.csv
data/processed/X_test.csv

data/processed/y_train.csv
data/processed/y_val.csv
data/processed/y_test.csv
```

The pipeline also prepares metadata and expert-related tables required by routing and evaluation.

---

# 5. Load the Frozen Base Fraud Predictor

Run:

```bash
python scripts/train_or_load_model.py --config config.yaml
```

For the submitted paper, the canonical experiment **reuses the existing FiFAR benchmark fraud scores** rather than training the fallback XGBoost / RandomForest model path.

The base model produces:

```text
AI score
AI prediction
```

with decision threshold:

```text
0.5
```

These outputs remain frozen for the downstream assurance experiment.

---

# 6. Compute Core Assurance Evidence

MAGD-Fraud constructs multiple forms of case-level evidence.

## 6.1 Calibration Risk

Run:

```bash
python scripts/run_calibration.py --config config.yaml
```

Purpose:

- fit calibration mapping on permitted development data;
- compute case-level calibration risk;
- generate calibration diagnostics.

The submitted-paper configuration uses:

```text
10 calibration bins
```

---

## 6.2 Distance Uncertainty

Run:

```bash
python scripts/run_distance_uncertainty.py --config config.yaml
```

Purpose:

- generate PCA-based representations;
- estimate representation-space distance;
- convert distance into uncertainty.

Submitted-paper setting:

```text
PCA dimensions = 20
```

The resulting signal is used as **representation unfamiliarity evidence**.

---

## 6.3 Local Neighbourhood Reliability

Run:

```bash
python scripts/run_neighbor_evidence.py --config config.yaml
```

Purpose:

- identify similar historical/training cases;
- estimate local AI error behaviour.

Submitted-paper setting:

```text
k = 25 nearest neighbours
```

The key case-level signal is the local neighbourhood error/reliability estimate.

---

## 6.4 Revised Wrong-Confident Risk

The final paper uses the **revised confidence-gated wrong-confident representation**, not the earlier additive confidence formulation.

The relevant implementation is in:

```text
src/assurance/magd_v2.py
src/assurance/wrong_confident_detector.py
```

Conceptually:

```text
independent unreliability evidence
              ×
         AI confidence
              ↓
     wrong-confident risk
```

High confidence is therefore used as a **gate/context variable**, not as additive unreliability evidence.

---

# 7. Five-Fold Out-of-Fold Development Evaluation

The final MAGD-Fraud assurance representation is developed using five-fold out-of-fold evaluation on the validation/development split.

For each development fold:

```text
development split
      ↓
inner-development data
      ↓
fit calibration mapping
      ↓
fit fold-local percentile / normalization transforms
      ↓
transform inner-development + held-out fold
      ↓
construct fixed 7-term assurance representation
      ↓
fit regularized assurance model
      ↓
score held-out fold only
      ↓
record AUROC / PR-AUC / enrichment
```

This is the **primary evidence for model development** in the submitted paper.

The benchmark test split is not used to fit:

- normalization transforms;
- assurance coefficients;
- interaction structure;
- routing thresholds.

---

# 8. Final MAGD-Fraud Assurance Model

The submitted paper uses a fixed seven-term assurance representation:

```text
1. normalized distance uncertainty
2. normalized calibration risk
3. normalized neighbourhood risk
4. revised wrong-confident risk
5. calibration × distance
6. calibration × neighbourhood
7. confidence × calibration
```

The final assurance model is:

```text
LogisticRegression
penalty = L2
C = 1.0
class_weight = balanced
random_state = 42
```

The model output is treated as an **AI-reliance risk ranking score**, not as a calibrated probability of AI error.

After five-fold development evaluation, the fixed model specification is refitted on the complete development split.

---

# 9. Freeze Validation-Derived Risk Thresholds

After the full-development refit, derive the routing thresholds from the development-side MAGD-Fraud score distribution.

Frozen submitted-paper thresholds:

```text
low-risk threshold  = 0.4063
high-risk threshold = 0.4410
```

Risk states:

```text
low:
    MAGD_AR < 0.4063

medium:
    0.4063 <= MAGD_AR < 0.4410

high:
    MAGD_AR >= 0.4410
```

No benchmark-test labels are used to select these thresholds.

---

# 10. Run Human–AI Routing

The routing layer is separate from the assurance scorer.

Supported routes:

```text
AI
single expert
multi-expert escalation
```

The routing decision combines:

- frozen MAGD-Fraud assurance state;
- expected-cost comparison;
- available expert information;
- panel escalation logic.

Submitted-paper panel size:

```text
5 experts
```

Important empirical result:

```text
AI retention       84.84%
single expert       0.00%
escalation         15.16%
```

The final benchmark policy therefore behaves as an **AI-versus-panel router**, even though single-expert review remains an implemented architectural option.

---

# 11. Run Baselines and Developmental Comparators

Run:

```bash
python scripts/run_baselines.py --config config.yaml
```

and the dedicated Learning-to-Defer runner used by the repository.

The evaluation includes:

- AI-only;
- numerical-confidence threshold;
- distance-based threshold;
- best-expert reference;
- random-expert reference;
- independent Learning-to-Defer (`L2D-Standard`);
- oracle upper bound;
- developmental additive MAGD variants.

`L2D-Standard` must remain independent of the final MAGD aggregate.

The earlier weighted/additive MAGD variants are retained as **developmental comparators**, not as the proposed final assurance model.

---

# 12. Run the Canonical Full Experiment

The preferred way to reproduce the paper-facing experiment is:

```bash
python scripts/run_full_magd_experiment.py --config config.yaml
```

The runner executes the complete research pipeline, including:

1. configuration validation;
2. split validation;
3. base-model prediction loading;
4. assurance-signal computation;
5. MAGD assurance processing;
6. expert-reliability estimation;
7. baseline evaluation;
8. Learning-to-Defer evaluation;
9. MAGD developmental variants;
10. MAGD-Fraud evaluation;
11. ablations;
12. predictive evaluation;
13. assurance discrimination;
14. reviewer-workload analysis;
15. statistical comparisons;
16. audit artifact generation;
17. paper-table generation;
18. scientific checks;
19. reproducibility summary generation.

The exact internal stage count may evolve with the repository, so the runner itself is the source of truth for execution ordering.

---

# 13. Evaluate All Methods

Method-level evaluation covers several dimensions.

## Predictive metrics

- precision;
- recall;
- F1;
- cost-sensitive classification loss;
- PR-AUC where appropriate.

## Assurance discrimination

- AUROC for AI error;
- PR-AUC;
- rank association;
- top-k enrichment;
- AI-error capture.

## Reliance metrics

- wrong-confident avoidance;
- correct rejection;
- overreliance;
- AI coverage.

## Routing metrics

- expert-deferral rate;
- escalation rate;
- intervention rate.

## Reviewer workload

Reviewer workload counts individual review actions:

```text
reviewer workload
=
single-expert cases
+
panel_size × escalated cases
```

For the final MAGD-Fraud benchmark run:

```text
escalation rate   = 15.16%
panel size        = 5
reviewer workload = 75.81%
```

---

# 14. Run Statistical Analysis

Use the statistical-analysis scripts provided under `scripts/`.

The submitted paper reports paired comparisons using:

- paired bootstrap;
- confidence intervals for F1 differences;
- confidence intervals for cost differences;
- McNemar tests.

The canonical statistical outputs are written under:

```text
data/outputs/paper_tables/
data/outputs/final_reproducible_run/
```

---

# 15. Run Assurance Ablations

The submitted paper includes a progressive assurance ablation:

```text
Calibration only
      ↓
+ normalized distance + neighbourhood
      ↓
+ revised wrong-confident evidence
      ↓
+ fixed interactions
```

The reported development AUROC progression is:

```text
0.6466
→ 0.6729
→ 0.6826
→ 0.6837
```

Use the dedicated MAGD-v2 ablation script:

```bash
python scripts/run_magd_v2_ablation.py --config config.yaml
```

where supported by the current repository revision.

Developmental additive ablation artifacts are retained separately for historical comparison.

---

# 16. Generate Audit Evidence

Run:

```bash
python scripts/generate_audit_pack.py --config config.yaml
```

The audit layer records the evidence chain:

```text
AI prediction
      ↓
assurance signals
      ↓
AI-reliance score
      ↓
risk category
      ↓
selected route
      ↓
final decision
```

Typical outputs include:

- case-level evidence logs;
- route metadata;
- policy configuration;
- rationale fields;
- audit completeness summaries.

Audit completeness indicates computational traceability; it is not evidence of validated usability by real fraud investigators.

---

# 17. Generate Paper Tables

Paper-facing summaries are produced under:

```text
data/outputs/paper_tables/
```

Relevant scripts include the repository's paper-table generation utilities and final reproducibility scripts.

Important compact artifacts include:

- main method metrics;
- reliance metrics;
- reviewer-workload summaries;
- assurance discrimination;
- error-capture tables;
- ablations;
- statistical comparisons;
- policy/configuration summaries.

Large per-case decision files are intentionally not tracked in Git.

---

# 18. Generate Paper Figures

The repository contains dedicated scripts for the submitted-paper figures, including:

```text
scripts/make_ai_error_capture_curve_figure.py
scripts/make_calibration_interaction_figures.py
scripts/make_f1_vs_workload_figure.py
scripts/make_reviewer_cost_sensitivity_figure.py
```

Figures should be generated from the **frozen reproducibility artifacts**, not from ad hoc intermediate outputs.

Tracked figures are stored under:

```text
figures/
```

---

# 19. Run Scientific Checks

Run the repository's scientific-assurance checks after the experiment.

These checks are intended to verify properties such as:

- deployable routing does not consume test ground truth;
- oracle outputs remain non-deployable;
- baseline independence;
- expected artifacts exist;
- configuration/results are internally consistent;
- synthetic experts and prototype status are correctly disclosed.

Use the repository's scientific-check runner where available.

---

# 20. Freeze / Verify Reproducibility Artifacts

The repository includes utilities for freezing and verifying the experiment state, including scripts such as:

```text
scripts/freeze_experiment_setup.py
scripts/freeze_canonical_decision_hashes.py
```

The frozen reproducibility area is:

```text
data/outputs/final_reproducible_run/
```

Typical compact artifacts include:

```text
run_metadata.json
config_snapshot.yaml
canonical_decision_hashes.json
final metrics
paper tables
figure-generation data
statistical summaries
```

Large row-level decision logs may be excluded from Git and regenerated locally.

---

# 21. Launch the Prototype Dashboard

Optional:

```bash
streamlit run src/dashboard/app.py
```

The dashboard is a **research prototype** for inspecting outputs.

It should not be described as a deployed fraud-review interface.

---

# 22. Main Output Structure

```text
data/outputs/
├── model/
├── assurance/
├── baselines/
├── assurance_deferral/
├── magd_policy/
├── ablations/
├── final_metrics/
├── paper_tables/
├── audit_pack/
├── plots/
└── final_reproducible_run/
```

The most important paper-facing outputs are the compact files under:

```text
data/outputs/final_metrics/
data/outputs/paper_tables/
data/outputs/final_reproducible_run/
```

---

# 23. Canonical Paper Results

The submitted-paper benchmark operating point is:

```text
MAGD-Fraud

precision             0.9414
recall                0.2927
F1                    0.4466
cost-sensitive loss   1011.48

AI coverage           84.84%
single-expert review   0.00%
escalation            15.16%

reviewer workload     75.81%
```

Assurance discrimination:

```text
5-fold OOF development:
AUROC  = 0.6837
PR-AUC = 0.1269

Frozen benchmark test:
AUROC  = 0.6912
PR-AUC = 0.1460
```

AI-error concentration:

```text
Top 5% highest-risk cases:
AI-error rate       = 12.49%
AI errors captured  = 42.55%
enrichment           = 8.51×
```

These values should be regenerated from the frozen reproducibility artifacts rather than manually edited into output files.

---

# 24. Legacy Workflow

The older command:

```bash
python scripts/run_full_experiment.py --config config.yaml
```

belongs to the earlier assurance-deferral workflow.

It remains useful for understanding the historical pipeline, but it is **not the canonical entry point for reproducing the submitted MAGD-Fraud paper**.

Use:

```bash
python scripts/run_full_magd_experiment.py --config config.yaml
```

for the current paper-aligned experiment.

---

# 25. Workflow Summary

The submitted-paper workflow can be summarized as:

```text
1. Obtain FiFAR
2. Prepare train / development / benchmark-test splits
3. Load frozen base-model scores
4. Compute calibration evidence
5. Compute distance uncertainty
6. Compute neighbourhood reliability
7. Construct revised wrong-confident evidence
8. Fit fold-local assurance transformations
9. Run five-fold OOF assurance evaluation
10. Refit the fixed 7-term MAGD-Fraud assurance model
11. Freeze development-derived routing thresholds
12. Apply expert-aware Human–AI routing
13. Evaluate baselines and developmental comparators
14. Measure predictive, assurance, reliance and workload outcomes
15. Run ablation and statistical analysis
16. Generate audit evidence
17. Run scientific/reproducibility checks
18. Generate paper tables and figures
19. Freeze final reproducibility artifacts
```

That is the canonical MAGD-Fraud research workflow.
