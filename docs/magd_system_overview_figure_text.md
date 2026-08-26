# MAGD-Fraud System Overview Figure Text

## Figure Title

**System Overview of MAGD-Fraud**

## Short Figure Caption

MAGD-Fraud extends distance-based uncertainty into a multi-evidence Human-AI assurance architecture for financial fraud review. A base fraud model produces a score and binary prediction, which are transformed into deployable assurance signals including numerical confidence, calibration risk, distance uncertainty, local neighbourhood reliability, and wrong-confident AI risk. These signals are aggregated into a case-level MAGD assurance risk and combined with adaptive thresholds, expert reliability, fairness, and capacity information to route each case to AI, a selected synthetic expert, or escalation.

## Long Figure Caption

Figure X illustrates the MAGD-Fraud pipeline. A fraud prediction model first outputs an AI score and prediction for each alert. An assurance layer then computes multiple deployable evidence signals: score-based confidence, calibration risk, embedding-based distance uncertainty, local historical neighbour reliability, and wrong-confident AI risk. These signals are combined into a normalized MAGD assurance risk score. A decision layer then uses this risk together with an adaptive threshold and expert-level reliability, fairness, and capacity estimates to choose among three routes: use AI, defer to the best available synthetic expert, or escalate to a majority vote among top reliable experts. Offline evaluation and audit modules compute assurance metrics, statistical tests, paper tables, and claim-evidence artifacts. Ground truth is used only for model training, validation-time policy learning, and final evaluation, but not for deployable routing decisions.

## Suggested Figure Layout

Left-to-right blocks:

1. `FiFAR Case Features`
2. `Fraud Predictor`
   - outputs: `ai_score`, `ai_pred`
3. `Assurance Signal Bank`
   - numerical confidence
   - calibration risk
   - distance uncertainty
   - local neighbourhood reliability
   - wrong-confident risk
4. `MAGD Assurance Aggregator`
   - weighted risk composition
   - low / medium / high risk category
5. `Adaptive Threshold + Policy Layer`
   - heuristic / learned / constrained policy
6. `Expert Reliability Layer`
   - expert accuracy
   - FPR / FNR
   - bias risk
   - capacity
7. `Routing Output`
   - AI
   - selected expert
   - escalation panel
8. `Audit and Evaluation Outputs`
   - assurance metrics
   - ablations
   - statistical tests
   - claim-evidence matrix

## In-Figure Text

### Main Title

`MAGD-Fraud: Multi-evidence Assurance-Guided Deferral`

### Sub-labels

- `Predictive Layer`
- `Assurance Layer`
- `Decision Layer`
- `Audit Layer`

### Arrow Labels

- `score, prediction`
- `deployable assurance evidence`
- `case-level assurance risk`
- `cost- and risk-aware routing`
- `auditable decision outputs`

## One-Sentence Camera-Ready Caption

MAGD-Fraud routes fraud alerts using multiple deployable assurance signals rather than distance uncertainty alone, enabling case-level selection among AI, a synthetic expert, and escalation while preserving auditability and split-safe evaluation.
