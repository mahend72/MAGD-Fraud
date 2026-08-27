# MAGD-Fraud Architecture

**Learning AI Reliance Risk for Assurance-Guided Human–AI Routing in Financial Fraud Review**

## Purpose

MAGD-Fraud is a research architecture for **AI assurance in Human–AI decision routing**. It is not a new fraud classifier. The base fraud model produces an AI score and prediction; MAGD-Fraud operates above that model to estimate **case-level AI-reliance risk** from multiple assurance signals and to use that evidence when deciding whether the case should:

- remain with the AI;
- be assigned to a single expert; or
- be escalated to a multi-expert panel.

Financial fraud review is the experimental domain. The broader research question is whether explicit, multi-evidence assurance can improve decisions about **when AI reliance is justified**.

> **Research scope:** the architecture is evaluated offline on the FiFAR benchmark with synthetic experts. It should not be interpreted as a production fraud system or as a validated human-subject deployment.

---

## Core Design Principle

MAGD-Fraud separates three questions that are often collapsed into one:

1. **Prediction** — what does the base AI model predict?
2. **Assurance** — how risky is it to rely on that prediction for this case?
3. **Action** — should the case stay with AI, go to one expert, or escalate?

The central architecture is therefore:

```text
AI prediction
      ↓
Assurance evidence
      ↓
Learned AI-reliance risk
      ↓
Human–AI routing
      ↓
Final decision + audit evidence
```

This separation allows assurance quality to be evaluated independently of routing behaviour.

---

## High-Level Architecture

```text
┌──────────────────────────────┐
│   FiFAR fraud-review case    │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│      Base fraud predictor    │
│   AI score + AI prediction   │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────────────────────┐
│            Assurance evidence                │
│                                              │
│  • Calibration risk                          │
│  • Distance uncertainty                      │
│  • Local neighbourhood reliability           │
│  • Confidence-gated wrong-confident risk     │
│  • AI confidence as contextual/gating input  │
└──────────────────┬───────────────────────────┘
                   ↓
┌──────────────────────────────────────────────┐
│        Fold-local signal processing          │
│                                              │
│  • calibration mapping fitted on             │
│    development data only                     │
│  • empirical percentile transforms fitted   │
│    inside the development fold               │
└──────────────────┬───────────────────────────┘
                   ↓
┌──────────────────────────────────────────────┐
│      MAGD-Fraud assurance representation     │
│                                              │
│  Fixed seven-term feature representation     │
│  + L2-regularized logistic model             │
│                                              │
│  Output: MAGD_AR AI-reliance risk score      │
└──────────────────┬───────────────────────────┘
                   ↓
┌──────────────────────────────────────────────┐
│      Validation-derived risk states          │
│                                              │
│  low      : MAGD_AR < τ_low                  │
│  medium   : τ_low ≤ MAGD_AR < τ_high         │
│  high     : MAGD_AR ≥ τ_high                 │
│                                              │
│  τ_low  = 0.4063                             │
│  τ_high = 0.4410                             │
└──────────────────┬───────────────────────────┘
                   ↓
┌──────────────────────────────────────────────┐
│         Human–AI routing layer               │
│                                              │
│  assurance state + expected-cost logic       │
└───────────┬───────────────┬──────────────────┘
            ↓               ↓
       Retain AI       Single expert
            \               /
             \             /
              └──────┬────┘
                     ↓
             Escalation panel
               (top-k = 5)
                     ↓
┌──────────────────────────────────────────────┐
│        Final reviewed fraud decision         │
└──────────────────┬───────────────────────────┘
                   ↓
┌──────────────────────────────────────────────┐
│          Audit and evaluation layer          │
│                                              │
│  • assurance evidence                        │
│  • risk state                                │
│  • selected route                            │
│  • decision source / rationale               │
│  • predictive and reliance metrics           │
│  • reviewer workload                         │
│  • statistical comparisons                   │
└──────────────────────────────────────────────┘
```

---

# 1. Data Layer

**Location**

- `src/data/`
- `config.yaml`

**Responsibilities**

- load FiFAR fraud-review data;
- load expert predictions and expert-history information;
- preserve case identifiers and relevant metadata;
- create or validate train/development/test partitions;
- generate the tabular inputs required by the base model and assurance modules;
- prevent accidental leakage of test labels into deployable routing logic.

The submitted-paper evaluation uses:

| Split | Cases | Fraud prevalence |
|---|---:|---:|
| Train | 278,627 | 1.02% |
| Validation / development | 119,323 | 1.18% |
| Benchmark test | 96,843 | 1.47% |

Large benchmark files and generated per-case data are intentionally excluded from Git and must be obtained/generated locally.

---

# 2. Base Model Layer

**Location**

- `src/models/`

**Purpose**

The base model provides the initial fraud prediction. MAGD-Fraud does **not** replace this model.

For the submitted-paper configuration, the experiment reuses the pretrained FiFAR/ICAIF benchmark fraud scores from a LightGBM classifier rather than training the fallback models defined elsewhere in the repository.

**Frozen paper setting**

- model family: pretrained `LGBMClassifier`;
- benchmark scores reused through the existing-score path;
- decision threshold: `0.5`.

The repository also contains fallback model-training code for other experimental workflows, but these fallbacks are not the model used to produce the submitted-paper results.

---

# 3. Representation and Local-Evidence Layer

**Location**

- `src/models/embeddings.py`
- `src/assurance/distance_uncertainty.py`
- `src/assurance/local_reliability.py`
- related assurance utilities

**Purpose**

This layer produces evidence about whether the current case lies in a region where continued AI reliance appears well supported.

The submitted-paper configuration uses:

- PCA representation dimensions: `20`;
- neighbourhood size: `k = 25`;
- calibration bins: `10`.

The representation is used for both distance-based uncertainty and local-neighbour evidence.

---

# 4. Assurance Evidence Layer

**Location**

- `src/assurance/calibration.py`
- `src/assurance/distance_uncertainty.py`
- `src/assurance/local_reliability.py`
- `src/assurance/wrong_confident_detector.py`
- `src/assurance/magd_v2.py`

MAGD-Fraud uses complementary forms of evidence rather than relying on confidence alone.

## 4.1 Calibration Risk

Measures whether the base model's confidence is historically consistent with empirical correctness.

**Interpretation:** high calibration risk indicates that the stated probability may be unreliable in the relevant confidence region.

---

## 4.2 Distance Uncertainty

Measures representation-space unfamiliarity relative to training structure.

**Interpretation:** a case far from familiar training regions receives greater uncertainty.

This is a proxy for unfamiliarity rather than a formal out-of-distribution guarantee.

---

## 4.3 Local Neighbourhood Reliability

Measures the historical error behaviour of the AI among similar training cases.

**Interpretation:** if the model frequently failed on nearby cases, continued reliance is less well supported.

---

## 4.4 Wrong-Confident Risk

The final paper does not treat high confidence itself as additive evidence of unreliability.

Instead, confidence acts as a **gate** on independent unreliability evidence:

```text
unreliability evidence
        ×
AI confidence
        ↓
wrong-confident risk
```

The revised representation is designed to identify cases where the model remains confident despite other evidence that weakens reliance.

---

# 5. Fold-Local Assurance Processing

The final assurance model was developed using five-fold out-of-fold evaluation on the development split.

For each fold:

1. define the inner-development subset;
2. fit calibration mappings on inner-development data only;
3. fit signal percentile/normalization transforms on inner-development data only;
4. transform both inner-development and held-out fold using the frozen transforms;
5. construct the fixed assurance representation;
6. fit the assurance model using `AI_wrong` as the development target;
7. score only the held-out fold;
8. record AUROC, PR-AUC, and related discrimination measures.

This ensures that the assurance representation evaluated on a development fold was not fitted using that fold's own outcome labels.

After the five-fold evaluation, the same fixed architecture is refitted using the complete development split and frozen before benchmark evaluation.

---

# 6. Final MAGD-Fraud Assurance Representation

**Location**

- `src/assurance/magd_v2.py`

The proposed MAGD-Fraud model uses a fixed seven-term representation:

1. normalized distance uncertainty;
2. normalized calibration risk;
3. normalized neighbourhood risk;
4. revised wrong-confident risk;
5. calibration × distance;
6. calibration × neighbourhood;
7. confidence × calibration.

These features are aggregated using a fixed L2-regularized logistic model.

## Frozen model settings

| Component | Setting |
|---|---|
| Model | Logistic regression |
| Penalty | L2 |
| `C` | `1.0` |
| Class weighting | `balanced` |
| Random seed | `42` |
| Feature count | `7` |

The final full-development refit produced the following coefficient pattern:

| Feature | Coefficient |
|---|---:|
| Distance uncertainty | +0.163 |
| Calibration risk | +6.036 |
| Neighbourhood reliability | +11.468 |
| Wrong-confident risk | −22.675 |
| Calibration × distance | +0.098 |
| Calibration × neighbourhood | −2.000 |
| Confidence × calibration | +8.059 |
| Intercept | −0.500 |

Individual coefficients should **not** be interpreted causally or in isolation because the model contains correlated main effects and interaction terms.

The resulting `MAGD_AR` output is interpreted as an **AI-reliance risk ranking score**, not as a calibrated probability that the base model is wrong.

---

# 7. Risk-State Layer

After the final assurance scorer is fitted on the complete development split, two routing thresholds are derived from the development risk distribution:

```text
low risk:
    MAGD_AR < τ_low

medium risk:
    τ_low ≤ MAGD_AR < τ_high

high risk:
    MAGD_AR ≥ τ_high
```

**Frozen submitted-paper thresholds**

```text
τ_low  = 0.4063
τ_high = 0.4410
```

These correspond to the 60th and 90th percentiles of the full-development MAGD-Fraud score distribution.

The thresholds are frozen before the benchmark-test routing evaluation.

---

# 8. Human–AI Routing Layer

**Location**

- `src/deferral/`
- expert-routing and MAGD routing utilities

The routing layer is deliberately separate from the assurance model.

Its role is to convert assurance evidence into an operational action.

## Supported routes

### A. Retain AI

The original AI prediction is retained when the case remains suitable for automated handling under the frozen routing rule.

### B. Single-Expert Review

A selected expert can be used when the expert-aware expected-cost rule favours human review.

### C. Multi-Expert Escalation

High assurance-risk cases can be escalated to a panel.

**Submitted-paper panel size**

```text
k = 5
```

The panel decision is produced using the repository's expert-ranking / aggregation procedure.

## Important empirical limitation

Although the architecture supports all three destinations, the final benchmark policy produced:

```text
AI retention       84.84%
single expert       0.00%
escalation         15.16%
```

The submitted experiment therefore provides stronger empirical evidence for **assurance-based AI retention versus escalation** than for the single-expert routing branch.

The single-expert path remains an architectural capability but was not exercised by the final frozen policy.

---

# 9. Developmental Assurance Comparators

The repository retains earlier additive MAGD configurations for scientific comparison.

## MAGD-Additive

Combines assurance signals through a non-negative weighted sum.

This developmental formulation is retained because it demonstrates an important negative result: multiple individually meaningful signals do not automatically produce useful assurance when their scale, orientation, redundancy, and conditional relationships are ignored.

## MAGD-Additive-Tuned

Uses validation-selected additive weights / operating settings.

## MAGD-Fraud

The proposed method replaces direct additive aggregation with the fixed seven-term learned assurance representation while retaining the surrounding routing and audit architecture.

These older variants should be described as **developmental comparators**, not as the canonical final architecture.

---

# 10. Baseline Layer

**Location**

- `src/deferral/baselines.py`
- `src/deferral/learning_to_defer_baseline.py`

The evaluation includes:

- AI-only;
- numerical-confidence threshold;
- distance-based threshold;
- best-expert reference;
- random-expert reference;
- independent Learning-to-Defer (`L2D-Standard`);
- oracle upper bound;
- developmental additive MAGD variants.

`L2D-Standard` is kept independent of the final MAGD risk aggregate so that the main L2D comparator does not consume MAGD-derived risk as a feature.

The repository does not claim empirical state-of-the-art superiority over every recent Human–AI deferral architecture.

---

# 11. Evaluation Layer

**Location**

- `src/evaluation/`

MAGD-Fraud is evaluated along several dimensions rather than by classification performance alone.

## Predictive utility

- precision;
- recall;
- F1;
- cost-sensitive classification loss;
- PR-AUC where applicable.

## Assurance discrimination

- AUROC for `AI_wrong`;
- PR-AUC;
- rank association;
- top-k AI-error enrichment;
- AI-error capture.

## Reliance behaviour

- wrong-confident avoidance;
- correct rejection;
- overreliance;
- AI coverage.

## Routing behaviour

- expert-deferral rate;
- escalation rate;
- total intervention rate.

## Reviewer workload

Panel escalation is counted using reviewer actions rather than case count alone:

```text
reviewer workload
=
single-expert cases
+
panel_size × escalated cases
```

With `panel_size = 5`, the final 15.16% escalation rate corresponds to reviewer workload equal to 75.81% of the benchmark-test size.

## Statistical robustness

The paper uses paired resampling and paired correctness analysis, including:

- 1,000 paired bootstrap resamples;
- confidence intervals for F1 differences;
- confidence intervals for cost differences;
- McNemar tests.

---

# 12. Audit Layer

MAGD-Fraud preserves a case-level evidence chain:

```text
AI prediction
      ↓
assurance evidence
      ↓
MAGD_AR
      ↓
risk category
      ↓
selected route
      ↓
final decision
```

The audit record can include:

- AI score and prediction;
- assurance-signal values;
- learned assurance score;
- routing thresholds / risk category;
- selected route;
- expert/panel metadata;
- final decision source;
- routing rationale;
- experiment/configuration provenance.

The paper reports complete expected audit-field coverage for the benchmark run.

This should be interpreted as **computational traceability**, not as proof that the audit information is sufficient for real fraud investigators or institutional governance.

---

# 13. Scientific Leakage Boundary

Deployable routing may use:

- AI scores and predictions;
- development-derived assurance transforms;
- distance uncertainty;
- neighbourhood evidence derived from historical/training data;
- revised wrong-confident risk;
- expert reliability derived from historical expert behaviour;
- frozen risk thresholds;
- routing costs and supported operational metadata.

Deployable routing must not use:

- benchmark-test ground-truth labels;
- oracle correctness;
- future expert outcomes;
- offline diagnostic wrong-confident labels.

Ground truth is used only where scientifically appropriate for training/development evaluation and final offline reporting.

---

# 14. Research Pipeline

The canonical experimental workflow is:

```text
FiFAR data
   ↓
Base AI prediction
   ↓
Assurance signal computation
   ↓
Fold-local development processing
   ↓
7-term assurance representation
   ↓
MAGD-Fraud model
   ↓
Validation-derived risk thresholds
   ↓
Frozen Human–AI routing
   ↓
Baselines and developmental comparators
   ↓
Predictive / assurance / reliance evaluation
   ↓
Reviewer-workload analysis
   ↓
Statistical tests
   ↓
Audit and reproducibility artifacts
   ↓
Paper tables and figures
```

The main experiment runner is:

```bash
python scripts/run_full_magd_experiment.py --config config.yaml
```

The repository also contains targeted scripts for assurance diagnostics, ablation, figure generation, statistical analysis, and frozen-result verification.

---

# 15. Output Structure

Compact, paper-facing reproducibility outputs are organized under:

```text
data/outputs/
├── final_metrics/
├── paper_tables/
├── magd_policy/
├── audit_pack/
├── plots/
└── final_reproducible_run/
```

Large files such as:

- raw FiFAR benchmark data;
- processed feature matrices;
- embeddings;
- row-level decision logs;
- archived experimental snapshots;

are intentionally excluded from Git and regenerated locally.

---

# 16. Submitted-Paper Operating Point

The final benchmark evaluation reported in the submitted paper is:

```text
MAGD-Fraud
  precision             0.9414
  recall                0.2927
  F1                    0.4466
  cost-sensitive loss   1011.48

  AI coverage           84.84%
  expert review          0.00%
  escalation            15.16%

  panel size             5
  reviewer workload     75.81%
```

The architecture should therefore be understood as a **high-assurance / high-review operating point**, not as a uniformly review-efficient replacement for Learning-to-Defer.

---

# 17. Architectural Scope and Limitations

The implemented architecture is broader than what is empirically exercised by the submitted FiFAR experiment.

In particular:

- the single-expert route is implemented but unused by the final frozen policy;
- experts are synthetic benchmark experts;
- audit completeness does not establish real-user audit usability;
- optional fairness, capacity, drift, and business-context mechanisms are architectural extensions and are not all fully exercised in the submitted benchmark;
- assurance signals are empirical proxies, not formal guarantees;
- the final MAGD score is a ranking score rather than a calibrated error probability;
- independent temporal and external validation remain future work.

These distinctions should be preserved when describing the system in papers, documentation, or presentations.

---

# 18. Architectural Summary

MAGD-Fraud can be summarized in five stages:

1. **Predict** — obtain the frozen base model's fraud score and decision.
2. **Assure** — compute complementary case-level reliability evidence.
3. **Learn reliance risk** — normalize and combine that evidence using the fixed seven-term assurance representation.
4. **Route** — translate the frozen risk state into AI retention, expert review, or panel escalation.
5. **Audit and evaluate** — preserve the evidence chain and measure predictive utility, reliance behaviour, reviewer workload, and statistical robustness.

The main contribution is therefore not a new fraud classifier or a new multi-expert routing algorithm in isolation. It is the explicit connection:

```text
model-level assurance evidence
        ↓
learned AI-reliance risk
        ↓
Human–AI oversight action
        ↓
auditable decision evidence
```

That is the architectural core of MAGD-Fraud.
