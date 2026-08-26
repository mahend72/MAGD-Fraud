# MAGD-Fraud Paper Architecture

## Title

**MAGD-Fraud: Multi-evidence Assurance-Guided Deferral for Human-AI Financial Fraud Review**

## Purpose of This Document

This document gives a paper-facing description of the implemented MAGD-Fraud architecture in the `haaf_fifar` repository. It is written to support the method, system, and experimental setup sections of a research paper. The description follows the current codebase closely and separates:

- the base fraud prediction model
- deployable assurance signals
- the MAGD-Fraud assurance score
- adaptive thresholding
- expert-aware routing
- learned and constrained policy variants
- offline evaluation and audit outputs

The document is intentionally honest about what is deployable and what is used only for offline analysis.

## 1. Problem Setting

MAGD-Fraud addresses **financial fraud alert review** as a **Human-AI deferral problem**. For each case \(x_i\), the system must decide whether the case should be handled by:

1. the AI model
2. a selected synthetic fraud expert
3. escalation / senior review via a panel of top-ranked experts

The goal is not only predictive performance, but **assurance-aware routing** under asymmetric fraud costs, limited expert capacity, and potential fairness constraints.

The repository uses the **FiFAR benchmark setting**, where expert panels are **synthetic experts**, not real human subjects.

## 2. High-Level Architecture

MAGD-Fraud is a layered architecture:

1. **Predictive layer**
   - A base fraud classifier produces `ai_score` and `ai_pred`.

2. **Assurance layer**
   - Numerical confidence
   - Calibration risk
   - Distance-based uncertainty
   - Local neighbourhood reliability
   - Wrong-confident AI risk
   - Composite MAGD assurance risk

3. **Decision layer**
   - Adaptive thresholding
   - Expert reliability and fairness estimation
   - Expert-capacity-aware routing
   - Escalation by majority vote among top reliable experts

4. **Evaluation and audit layer**
   - Human-AI assurance metrics
   - Statistical tests
   - Claim-evidence matrix
   - Audit pack
   - Paper tables

## 3. End-to-End Data Flow

The main pipeline is orchestrated by:

- [scripts/run_full_magd_experiment.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/scripts/run_full_magd_experiment.py:1)

The implemented flow is:

1. Load and preprocess FiFAR data.
2. Train a fraud model or load existing AI scores.
3. Produce split-level predictions for train, validation, and test.
4. Compute deployable assurance signals from model scores and historical data.
5. Learn MAGD policy weights on validation data only.
6. Route test cases using deployable signals only.
7. Evaluate performance and assurance metrics offline.
8. Generate paper tables, audit outputs, and statistical tests.

The repository preserves a strict split discipline:

- **training split**: fraud model fitting and historical neighbour statistics
- **validation split**: policy learning for MAGD-Learned and MAGD-Constrained
- **test split**: final reporting only

## 4. Base Fraud Prediction Model

Primary implementation:

- [src/models/train_model.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/models/train_model.py:1)
- [src/models/predict.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/models/predict.py:1)

### 4.1 Input

The model consumes tabular fraud-review features prepared from FiFAR.

Processed data artifacts:

- `data/processed/X_train.csv`
- `data/processed/X_val.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_val.csv`
- `data/processed/y_test.csv`

### 4.2 Model Class

The current implementation supports:

- **XGBoost classifier** as the preferred model
- **RandomForestClassifier** as fallback if XGBoost is unavailable

The classifier outputs:

- `ai_score`: estimated fraud probability
- `ai_pred`: binary prediction under the configured threshold

### 4.3 Cost Sensitivity

The predictive layer is evaluated under asymmetric fraud costs:

- false positive cost
- false negative cost

These costs are propagated through both model evaluation and downstream routing.

### 4.4 Prediction Artifacts

The model stage writes:

- `data/outputs/model/train_predictions.csv`
- `data/outputs/model/val_predictions.csv`
- `data/outputs/model/test_predictions.csv`
- `data/outputs/model/model_metrics.csv`

## 5. Assurance Signal Architecture

MAGD-Fraud does not rely on a single uncertainty measure. It constructs a **multi-evidence assurance representation** for each case.

### 5.1 Numerical Confidence

Implementation:

- [src/assurance/numerical_confidence.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/assurance/numerical_confidence.py:1)

For a binary fraud score \(p_i\):

\[
\text{numerical\_confidence}_i = \max(p_i, 1 - p_i)
\]

This captures the model’s own apparent certainty.

### 5.2 Calibration Risk

Implementation:

- [src/assurance/numerical_confidence.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/assurance/numerical_confidence.py:1)

Scores are binned into confidence intervals. For each bin:

- average model confidence is compared against observed accuracy
- the absolute calibration gap is computed

For each case, `calibration_risk` is assigned from the bin-level absolute gap.

Expected Calibration Error (ECE) is also computed offline for reporting.

### 5.3 Distance-Based Uncertainty

Implementation:

- [src/assurance/distance_uncertainty.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/assurance/distance_uncertainty.py:1)
- [src/models/embeddings.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/models/embeddings.py:1)

The system projects tabular features into a PCA embedding space.

Training examples are used to compute:

- centroid of the non-fraud class
- centroid of the fraud class

For each query case:

- compute Euclidean distance to both centroids
- select the distance to the centroid of the predicted class
- normalize to form `distance_confidence`

\[
\text{distance\_uncertainty}_i = 1 - \text{distance\_confidence}_i
\]

This is the baseline uncertainty signal that MAGD-Fraud extends.

### 5.4 Local Neighbourhood Reliability

Implementation:

- [src/assurance/local_reliability.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/assurance/local_reliability.py:1)

This module addresses a key weakness of centroid-only distance uncertainty: fraud patterns are often rare, overlapping, and multimodal.

For each query case, MAGD-Fraud finds the `k` nearest historical training cases in embedding space and computes:

- `neighbor_error_rate`
- `neighbor_fraud_rate`
- `neighbor_ai_agreement`
- `mean_neighbor_distance`

Deployable design constraint:

- only **historical training labels** and **historical AI correctness** are used
- **test `y_true` is not used**

This converts local historical reliability into a deployable assurance feature.

### 5.5 Wrong-Confident AI Risk

Implementation:

- [src/assurance/wrong_confident_detector.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/assurance/wrong_confident_detector.py:1)

This module estimates whether the AI appears confident while being likely to fail.

Deployable score inputs:

- `numerical_confidence`
- `distance_uncertainty`
- `calibration_risk`
- `neighbor_error_rate`
- `confidence_disagreement = | numerical_confidence - distance_confidence |`

The deployable score is:

\[
\text{WCR}_i =
\frac{
w_1 c_i +
w_2 u_i +
w_3 r_i +
w_4 n_i +
w_5 d_i
}{
\sum_j w_j
}
\]

where:

- \(c_i\) is numerical confidence
- \(u_i\) is distance uncertainty
- \(r_i\) is calibration risk
- \(n_i\) is neighbour error rate
- \(d_i\) is confidence disagreement

The result is clipped to \([0,1]\).

Offline only:

- `wrong_confident_label_offline` uses `y_true`
- detection precision / recall / F1 / top-k capture are computed only for evaluation

## 6. MAGD Assurance Risk

Implementation:

- [src/assurance/magd_risk.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/assurance/magd_risk.py:1)

MAGD-Fraud aggregates multiple assurance signals into a single deployable risk score:

\[
\text{MAGD\_AR}_i =
\frac{
w_1 \cdot \text{distance\_uncertainty}_i +
w_2 \cdot \text{calibration\_risk}_i +
w_3 \cdot \text{neighbor\_error\_rate}_i +
w_4 \cdot \text{wrong\_confident\_risk}_i +
w_5 \cdot \text{drift\_risk}_i +
w_6 \cdot \text{business\_risk}_i
}{
\sum_{j=1}^{6} w_j
}
\]

Optional signals:

- `drift_risk`
- `business_risk`

If optional signals are unavailable:

- they are set to `0`
- availability flags are recorded
- a warning is logged

### 6.1 Risk Categories

The score is mapped into:

- `low`
- `medium`
- `high`

using the configured `low_risk` and `high_risk` thresholds.

### 6.2 Recommended Actions

The composite risk is also mapped to a suggested action:

- `low` -> `AI`
- `medium` -> `Human Expert`
- `high` -> `Escalate`

This stage writes:

- `data/outputs/assurance/magd_risk.csv`
- `data/outputs/plots/magd_risk_distribution.png`

## 7. MAGD Policy Variants

Implementation:

- [src/deferral/magd_policy.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/deferral/magd_policy.py:1)
- [scripts/run_magd_policy.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/scripts/run_magd_policy.py:1)

This is the main methodological novelty beyond static uncertainty deferral.

MAGD-Fraud implements three policy variants.

### 7.1 MAGD-Heuristic

Uses the configured MAGD weights directly.

This is the simplest multi-evidence policy and acts as a structured extension of distance-only uncertainty.

### 7.2 MAGD-Learned

Learns evidence weights on the validation split by minimizing:

\[
\text{FraudLoss} + \lambda_{\text{overreliance}} \cdot \text{OverReliance}
\]

This variant learns how much each assurance signal should matter rather than using fixed values only.

### 7.3 MAGD-Constrained

Learns weights on the validation split while penalizing additional operational risks:

\[
\text{FraudLoss}
 + \lambda_{\text{overreliance}} \cdot \text{OverReliance}
 + \lambda_{\text{capacity}} \cdot \text{CapacityViolation}
 + \lambda_{\text{fairness}} \cdot \text{FairnessPenalty}
 + \lambda_{\text{audit}} \cdot \text{AuditGap}
\]

subject to:

\[
\text{CorrectRejection} \ge \tau_{\text{CR}}
\]

\[
\text{AuditCoverage} \ge \tau_{\text{audit}}
\]

### 7.4 Optimization Strategy

The implementation:

- learns on **validation data only**
- uses **sampled validation / historical subsets** for runtime practicality
- uses **SLSQP** when available
- falls back to a bounded coarse search if optimization is unavailable or fails

### 7.5 Policy Outputs

The policy-learning stage writes:

- `data/outputs/magd_policy/learned_weights.csv`
- `data/outputs/magd_policy/optimization_diagnostics.json`

## 8. Adaptive Thresholding

Implementation:

- [src/deferral/adaptive_threshold.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/deferral/adaptive_threshold.py:1)

Rather than using a single global threshold, MAGD-Fraud computes a per-case threshold:

\[
\tau_i =
\tau_0 +
\alpha \cdot \text{business\_risk}_i +
\beta \cdot \text{fairness\_risk}_i +
\gamma \cdot \text{capacity\_pressure}_i
\]

where:

- \(\tau_0\) is the base threshold
- business, fairness, and capacity terms are optional context signals

The threshold is clipped to \([0,1]\).

Missing optional signals are set to `0` and logged.

This stage writes:

- `data/outputs/assurance/adaptive_thresholds.csv`

## 9. Expert Reliability, Fairness, and Capacity Layer

Implementation:

- [src/deferral/expert_routing.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/deferral/expert_routing.py:1)
- [src/deferral/capacity_assignment.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/deferral/capacity_assignment.py:1)

For each synthetic expert, the system estimates:

- accuracy
- false positive rate
- false negative rate
- cost-sensitive loss
- group-wise false positive rate if sensitive attributes exist
- group-wise false negative rate if sensitive attributes exist
- bias risk
- remaining capacity

### 9.1 Expert Expected Cost

For expert \(j\) on case \(i\):

\[
L_{\text{expert},j}(i) =
\text{FP}_{\text{cost}} \cdot \text{FPR}_j +
\text{FN}_{\text{cost}} \cdot \text{FNR}_j +
\lambda_f \cdot \text{BiasRisk}_j +
\lambda_c \cdot \text{CapacityPenalty}_j
\]

### 9.2 AI Expected Cost

\[
L_{\text{AI}}(i) =
\text{FP}_{\text{cost}} \cdot \widehat{FP\_risk}_i +
\text{FN}_{\text{cost}} \cdot \widehat{FN\_risk}_i +
\text{MAGD\_AR}_i
\]

The AI expected cost is driven by score-dependent risk plus assurance risk.

### 9.3 Escalation

If escalation is triggered, the implementation uses:

- majority vote among the top-\(k\) most reliable available experts

Deployable constraint:

- **oracle is not used for routing**
- oracle remains an upper-bound baseline only

This stage writes:

- `data/outputs/assurance_deferral/expert_reliability.csv`
- `data/outputs/assurance_deferral/magd_routing_decisions.csv`

## 10. Final MAGD-Fraud Routing Policy

Implementation:

- [src/deferral/magd_deferral.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/deferral/magd_deferral.py:1)
- [scripts/run_magd_deferral.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/scripts/run_magd_deferral.py:1)

For each test case, the final policy loads:

- `ai_score`
- `ai_pred`
- `numerical_confidence`
- `distance_confidence`
- `distance_uncertainty`
- `calibration_risk`
- local neighbour reliability
- `wrong_confident_risk`
- `magd_assurance_risk`
- adaptive threshold
- expert reliability summaries

### 10.1 Routing Logic

At a high level:

1. Compute AI expected cost.
2. Compute best available expert expected cost.
3. If MAGD risk is high or wrong-confident risk is high, escalate.
4. Else if MAGD risk is low, threshold passes, and AI cost is lower, use AI.
5. Otherwise defer to the best available expert.

### 10.2 Human-Readable Decision Reasons

The implementation records reasons such as:

- `low_risk_ai_allowed`
- `medium_risk_defer_to_best_expert`
- `high_wrong_confident_risk_escalate`
- `similar_cases_often_misclassified`
- `confidence_not_reliable`
- `capacity_limit_use_next_best_expert`
- `fairness_risk_avoid_biased_expert`

### 10.3 Outputs

The final deferral stage writes:

- `data/outputs/assurance_deferral/magd_fraud_decisions.csv`
- `data/outputs/assurance_deferral/magd_fraud_metrics.csv`

## 11. Learning-to-Defer Baseline

Implementation:

- [src/deferral/learning_to_defer_baseline.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/deferral/learning_to_defer_baseline.py:1)

This baseline trains a rejector to choose between AI and human review using:

- `ai_score`
- `numerical_confidence`
- `distance_uncertainty`
- `calibration_risk`
- `neighbor_error_rate`
- `wrong_confident_risk`

It serves as a stronger algorithmic baseline than threshold rules alone.

## 12. Deployable vs Offline Components

This separation is critical for the paper.

### 12.1 Deployable Components

Deployable routing may use:

- model score and binary prediction
- numerical confidence
- calibration risk
- distance uncertainty
- local historical neighbour reliability
- wrong-confident deployable risk
- MAGD risk
- adaptive threshold context
- expert reliability estimates learned from historical expert behavior

### 12.2 Offline-Only Components

Offline evaluation may use:

- `y_true` on validation for policy learning objectives
- `y_true` on test for final metrics only
- wrong-confident offline label
- correctness-based evaluation metrics
- claim-evidence and audit analyses

### 12.3 Leakage Guardrail

The codebase includes scientific checks in:

- [src/utils/scientific_checks.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/utils/scientific_checks.py:1)

These checks enforce that deployable routing logic does not depend on `y_true`.

## 13. Evaluation Architecture

Implemented evaluation modules:

- [src/evaluation/fraud_metrics.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/evaluation/fraud_metrics.py:1)
- [src/evaluation/reliance_metrics.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/evaluation/reliance_metrics.py:1)
- [src/evaluation/assurance_metrics.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/evaluation/assurance_metrics.py:1)
- [src/evaluation/deferral_metrics.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/evaluation/deferral_metrics.py:1)
- [src/evaluation/fairness_metrics.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/evaluation/fairness_metrics.py:1)
- [src/evaluation/audit_metrics.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/evaluation/audit_metrics.py:1)

Key paper metrics include:

- precision
- recall
- F1
- PR-AUC
- ECE
- cost-sensitive loss
- overreliance
- underreliance
- correct rejection
- wrong-confident avoidance
- AI coverage
- expert deferral rate
- escalation rate
- capacity violation rate
- audit coverage

## 14. Audit and Assurance Case Architecture

Implemented audit modules:

- [src/assurance/claim_evidence_matrix.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/assurance/claim_evidence_matrix.py:1)
- [scripts/generate_claim_evidence_matrix.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/scripts/generate_claim_evidence_matrix.py:1)
- [scripts/generate_audit_pack.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/scripts/generate_audit_pack.py:1)

MAGD-Fraud is framed not only as a predictive system but as an **assurance case**:

- confidence evidence
- uncertainty evidence
- local similarity evidence
- wrong-confident risk evidence
- routing evidence
- expert reliability evidence
- decision rationale
- audit completeness

## 15. File-Level Architecture Map

### Data

- [src/data/load_fifar.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/data/load_fifar.py:1)

### Predictive model

- [src/models/train_model.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/models/train_model.py:1)
- [src/models/predict.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/models/predict.py:1)
- [src/models/embeddings.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/models/embeddings.py:1)

### Assurance signals

- [src/assurance/numerical_confidence.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/assurance/numerical_confidence.py:1)
- [src/assurance/distance_uncertainty.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/assurance/distance_uncertainty.py:1)
- [src/assurance/local_reliability.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/assurance/local_reliability.py:1)
- [src/assurance/wrong_confident_detector.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/assurance/wrong_confident_detector.py:1)
- [src/assurance/magd_risk.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/assurance/magd_risk.py:1)

### Decision policy

- [src/deferral/magd_policy.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/deferral/magd_policy.py:1)
- [src/deferral/adaptive_threshold.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/deferral/adaptive_threshold.py:1)
- [src/deferral/expert_routing.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/deferral/expert_routing.py:1)
- [src/deferral/magd_deferral.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/deferral/magd_deferral.py:1)
- [src/deferral/learning_to_defer_baseline.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/deferral/learning_to_defer_baseline.py:1)
- [src/deferral/baselines.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/deferral/baselines.py:1)

### Evaluation and reporting

- [src/evaluation/assurance_metrics.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/evaluation/assurance_metrics.py:1)
- [src/evaluation/statistical_tests.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/src/evaluation/statistical_tests.py:1)
- [scripts/run_magd_ablations.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/scripts/run_magd_ablations.py:1)
- [scripts/run_magd_statistical_tests.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/scripts/run_magd_statistical_tests.py:1)
- [scripts/make_magd_paper_tables.py](/mnt/c/Users/kumar2_m/Human-AI-Assurance/haaf_fifar/scripts/make_magd_paper_tables.py:1)

## 16. Suggested Paper Figure

The architecture can be visualized as a left-to-right pipeline:

1. **FiFAR Case Features**
2. **Fraud Predictor**
3. **Assurance Signal Bank**
   - numerical confidence
   - calibration risk
   - distance uncertainty
   - neighbour reliability
   - wrong-confident risk
4. **MAGD Assurance Aggregator**
5. **Adaptive Threshold + Policy Layer**
6. **Expert Reliability / Capacity Layer**
7. **Routing Decision**
   - AI
   - selected expert
   - escalation panel
8. **Audit and Evaluation Outputs**

Suggested caption:

“MAGD-Fraud extends centroid-based distance uncertainty into a multi-evidence assurance-guided routing architecture for financial fraud review. The system combines deployable model-assurance signals with expert reliability, fairness, and capacity information to decide when to rely on AI, defer to a synthetic expert, or escalate.”

## 17. Scientific Scope and Limitations

The paper should state these limits explicitly.

1. FiFAR experts in this repository are synthetic experts, not observed human reviewers.
2. The dashboard is a prototype interface, not evidence of human-subject validation.
3. The learned and constrained MAGD policies are computational policy learners on benchmark data, not deployed institutional decision systems.
4. Business risk, drift risk, fairness risk, and capacity pressure may be partially unavailable depending on dataset configuration.
5. The current distance representation is PCA-based rather than task-specific representation learning.
6. Escalation quality depends on historical synthetic expert behavior and capacity assumptions.

## 18. Reproducibility Commands

Main paper pipeline:

```bash
python scripts/run_full_magd_experiment.py --config config.yaml
```

Key component commands:

```bash
python scripts/run_magd_policy.py --config config.yaml
python scripts/run_magd_deferral.py --config config.yaml
python scripts/run_learning_to_defer_baseline.py --config config.yaml
python scripts/run_magd_ablations.py --config config.yaml
python scripts/run_magd_statistical_tests.py --config config.yaml
python scripts/make_magd_paper_tables.py --config config.yaml
```

Verification:

```bash
pytest
```

## 19. Short Paper Summary

MAGD-Fraud is best described as a **multi-evidence assurance-guided deferral architecture** for fraud review. Its main contribution is not a new fraud classifier alone, but a structured decision system that:

- estimates multiple deployable assurance signals
- composes them into a case-level assurance risk
- adapts routing thresholds to operational context
- reasons about expert reliability, bias, and capacity
- supports AI use, expert deferral, and escalation
- provides evaluation and audit outputs aligned with Human-AI assurance

That framing is more accurate and stronger for a paper than describing the method as only “distance uncertainty plus thresholds.”
