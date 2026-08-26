# Architecture

## Purpose

HAAF FiFAR is a research prototype for assurance-guided fraud alert review. It connects:

- `Level 3 model assurance`
  - confidence
  - calibration
  - distance-based uncertainty
  - neighbor evidence
  - wrong-but-confident risk
- `Level 4 human oversight`
  - AI handling
  - expert deferral
  - escalation

The system evaluates whether assurance evidence improves routing decisions compared with standard baselines.

## High-Level Flow

```text
FiFAR / fraud-review data
        |
        v
  Data loading + preprocessing
        |
        v
  AI score source
    - existing model scores
    - or trained classifier
        |
        v
  Assurance signals
    - numerical confidence
    - calibration risk
    - distance confidence / uncertainty
    - neighbor evidence
    - wrong-confident risk
        |
        v
  Final assurance risk
        |
        v
  Routing policy
    - AI
    - Human Expert
    - Escalate
        |
        v
  Evaluation
    - fraud metrics
    - reliance metrics
    - deferral metrics
    - fairness metrics
    - audit metrics
    - statistical tests
        |
        v
  Audit pack + dashboard + paper tables
```

## Main Components

### 1. Data Layer

Location:

- `src/data/`

Responsibilities:

- inspect unknown FiFAR schema
- load main case table
- load expert predictions
- load capacity table
- preprocess tabular features
- preserve `case_id`, `batch_id`, and sensitive attributes
- create train / validation / test splits

Key files:

- `src/data/inspect_data.py`
- `src/data/load_fifar.py`
- `src/data/preprocess.py`

## 2. Model Layer

Location:

- `src/models/`

Responsibilities:

- reuse existing model scores when available
- otherwise train a fraud classifier
- evaluate model outputs using fraud-oriented metrics
- generate PCA embeddings for uncertainty and example evidence

Key files:

- `src/models/train_model.py`
- `src/models/predict.py`
- `src/models/calibrate_model.py`
- `src/models/embeddings.py`

Default model behavior:

- primary: `XGBoost`
- fallback: `RandomForestClassifier`

## 3. Assurance Layer

Location:

- `src/assurance/`

Responsibilities:

- convert model probabilities into numerical confidence
- compute calibration error and case-level calibration risk
- compute distance-based confidence and uncertainty in PCA space
- retrieve nearest-neighbor evidence
- detect wrong-but-confident AI risk
- combine signals into a final assurance risk score

Key files:

- `src/assurance/numerical_confidence.py`
- `src/assurance/distance_uncertainty.py`
- `src/assurance/explanation_neighbors.py`
- `src/assurance/wrong_confident_detector.py`
- `src/assurance/assurance_risk.py`

### Assurance Signals

1. `numerical_confidence`
   - `max(p, 1 - p)` for binary fraud prediction

2. `calibration_risk`
   - local error from the calibration bin containing the case

3. `distance_confidence`
   - proximity of the case embedding to the predicted class centroid

4. `distance_uncertainty`
   - `1 - distance_confidence`

5. `neighbor_error_rate`
   - fraction of nearest training neighbors where AI was wrong

6. `wrong_confident_risk`
   - deployable risk score for dangerous confident-but-likely-wrong AI behavior

7. `assurance_risk`
   - normalized weighted combination of the above signals plus optional business risk

## 4. Deferral Layer

Location:

- `src/deferral/`

Responsibilities:

- run standard deferral baselines
- estimate expert reliability
- compare AI expected cost vs expert expected cost
- assign cases to AI, human expert, or escalation
- respect capacity constraints when available

Key files:

- `src/deferral/baselines.py`
- `src/deferral/assurance_deferral.py`
- `src/deferral/capacity_assignment.py`

### Baselines

- AI-only
- best expert only
- random expert
- numerical confidence threshold
- distance confidence threshold
- oracle upper bound

### Main Method: Assurance-Guided Deferral

Decision logic:

- low assurance risk:
  - use AI if estimated AI cost is lower than expert cost
- medium assurance risk:
  - defer to best available expert
- high assurance risk:
  - escalate via majority vote among top-k experts

## 5. Evaluation Layer

Location:

- `src/evaluation/`

Responsibilities:

- compare methods on performance and oversight behavior
- measure reliance and overreliance
- evaluate deferral and audit quality
- run paired statistical comparisons

Key files:

- `src/evaluation/fraud_metrics.py`
- `src/evaluation/reliance_metrics.py`
- `src/evaluation/deferral_metrics.py`
- `src/evaluation/fairness_metrics.py`
- `src/evaluation/audit_metrics.py`
- `src/evaluation/statistical_tests.py`

## 6. Reporting Layer

Location:

- `scripts/`
- `src/utils/reporting.py`
- `src/dashboard/`

Responsibilities:

- generate end-to-end experiment outputs
- generate audit pack
- generate paper tables
- provide interactive dashboard for demonstration

Key files:

- `scripts/run_full_experiment.py`
- `scripts/generate_audit_pack.py`
- `scripts/make_paper_tables.py`
- `src/dashboard/app.py`

## Output Structure

```text
data/outputs/
  model/
  assurance/
  baselines/
  assurance_deferral/
  final_metrics/
  audit_pack/
  paper_tables/
  plots/
```

## Working Model

The current working implementation on the ICAIF-style dataset uses:

- existing fraud model scores from the dataset
- calibration analysis on saved model predictions
- PCA embeddings for tabular distance uncertainty
- nearest-neighbor evidence from training cases
- assurance-guided routing for final decisions

You can inspect the working outputs here:

- `data/outputs/final_metrics/all_method_metrics.csv`
- `data/outputs/assurance_deferral/assurance_guided_decisions.csv`
- `data/outputs/audit_pack/audit_report.md`

