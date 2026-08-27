# MAGD-Fraud

**Learning AI Reliance Risk for Assurance-Guided Human–AI Routing in Financial Fraud Review**

[![MAGD-FRAUD CI](https://github.com/mahend72/MAGD-Fraud/actions/workflows/ci.yml/badge.svg)](https://github.com/mahend72/MAGD-Fraud/actions/workflows/ci.yml)

MAGD-Fraud is a research framework for studying **when continued reliance on an AI prediction is justified and when human intervention is warranted**. Financial fraud review is used as the experimental domain, but the primary research problem is broader: **AI assurance for Human–AI decision routing**.

Rather than treating model confidence as sufficient evidence of reliability, MAGD-Fraud constructs a case-level **AI-reliance risk** representation from multiple complementary assurance signals and uses this evidence to support AI retention, expert review, or multi-expert escalation.

> **Research scope.** MAGD-Fraud is an offline research prototype evaluated on the FiFAR benchmark with synthetic experts. It is not a production fraud-detection or financial decision system.

---

## Research Motivation

High-stakes AI systems need to answer more than *what does the model predict?* They must also ask *is there sufficient evidence to rely on that prediction for this case?*

MAGD-Fraud therefore separates:

```text
Case
  ↓
Base AI prediction
  ↓
Assurance evidence
  ↓
Learned AI-reliance risk
  ↓
Human–AI routing
  ├── Retain AI
  ├── Single-expert review
  └── Multi-expert escalation
  ↓
Final decision + audit evidence
```

The central contribution is an explicit **multi-evidence assurance layer** between AI prediction and routing.

---

## Final Assurance Representation

The submitted paper uses:

- calibration risk;
- representation-space distance uncertainty;
- local neighbourhood reliability;
- confidence-gated wrong-confident evidence;
- pre-specified interaction terms.

The final 7-term assurance representation is:

1. normalized distance uncertainty;
2. normalized calibration risk;
3. normalized neighbourhood risk;
4. revised wrong-confident risk;
5. calibration × distance;
6. calibration × neighbourhood;
7. confidence × calibration.

A fixed L2-regularized logistic model is used with:

| Component | Setting |
|---|---|
| Regularization `C` | `1.0` |
| Class weighting | `balanced` |
| Random seed | `42` |
| Low-risk threshold | `0.4063` |
| High-risk threshold | `0.4410` |
| Escalation panel size | `5` |

The output is treated as a **ranking score for AI-reliance risk**, not a calibrated probability of AI error.

---

## Assurance Signals

| Signal | What it measures | Assurance role |
|---|---|---|
| Calibration risk | Whether confidence matches historical correctness | Detects poorly calibrated regions |
| Distance uncertainty | Distance from familiar training structure | Measures case unfamiliarity |
| Neighbourhood reliability | Historical AI error among similar cases | Measures local reliability |
| Wrong-confident risk | High confidence gated by independent unreliability evidence | Targets dangerous confident errors |
| AI confidence | Model certainty | Used as context/gating evidence |

A key methodological finding is that simply adding assurance signals is not sufficient. The developmental additive formulation produced near-random failure discrimination. The final model improves this by handling signal scale, orientation, redundancy, and conditional relationships.

---

## Evaluation Protocol

MAGD-Fraud is evaluated on the **FiFAR** financial fraud review benchmark.

| Split | Cases | Fraud prevalence |
|---|---:|---:|
| Train | 278,627 | 1.02% |
| Validation / development | 119,323 | 1.18% |
| Benchmark test | 96,843 | 1.47% |

The assurance model is developed with **five-fold out-of-fold evaluation**. Fold-local preprocessing and signal transformations are fitted only on the corresponding inner-development data.

The benchmark test split had previously been inspected during diagnosis of the earlier additive formulation. Therefore, the paper treats the five-fold out-of-fold development evaluation as the primary model-development evidence and the frozen benchmark-test result as supporting post-development evidence.

---

## Main Results

### Assurance discrimination

| Evaluation | AUROC | PR-AUC |
|---|---:|---:|
| Calibration risk alone | 0.6466 | 0.0748 |
| MAGD-Fraud, five-fold OOF development | **0.6837** | **0.1269** |
| MAGD-Fraud, frozen benchmark test | **0.6912** | **0.1460** |

### AI-error concentration

| Risk fraction reviewed | AI errors captured | AI-error rate | Enrichment |
|---|---:|---:|---:|
| Top 1% | 21.45% | 31.51% | 21.46× |
| Top 5% | **42.55%** | **12.49%** | **8.51×** |
| Top 10% | 46.13% | 6.77% | 4.61× |
| Top 20% | 48.24% | 3.54% | 2.41× |

The highest-risk 5% of cases contain 42.55% of all AI errors.

---

## Routing Results

| Method | Precision | Recall | F1 | Cost | AI coverage | Expert | Escalation | Reviewer workload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AI-only | 0.5288 | 0.0385 | 0.0718 | 1375.79 | 100.00% | 0.00% | 0.00% | 0.00% |
| L2D-Standard | 0.6986 | 0.2451 | 0.3629 | 1086.61 | 96.25% | 3.75% | 0.00% | 3.75% |
| MAGD-Additive | 0.7608 | 0.1359 | 0.2305 | 1237.48 | 80.43% | 0.00% | 19.57% | 97.83% |
| MAGD-Additive-Tuned | 0.4966 | 0.0518 | 0.0938 | 1358.28 | 98.05% | 1.95% | 0.00% | 1.95% |
| **MAGD-Fraud** | **0.9414** | **0.2927** | **0.4466** | **1011.48** | 84.84% | 0.00% | 15.16% | 75.81% |

MAGD-Fraud achieves the strongest F1 and lowest cost-sensitive classification loss among the evaluated methods, but at substantially higher reviewer workload.

The final benchmark policy effectively behaves as an **AI-versus-panel router**: 84.84% of cases remain with AI, 0% use single-expert review, and 15.16% are escalated to a five-expert panel.

---

## Reliance-Oriented Evaluation

| Method | Wrong-confident avoidance | Correct rejection | Overreliance |
|---|---:|---:|---:|
| AI-only | 0.0000 | 0.0000 | 0.0147 |
| L2D-Standard | 0.1470 | 0.0062 | 0.0085 |
| MAGD-Additive | 0.2179 | 0.0030 | 0.0117 |
| MAGD-Additive-Tuned | 0.0156 | 0.0004 | 0.0143 |
| **MAGD-Fraud** | **0.2200** | **0.0070** | **0.0077** |

These metrics distinguish predictive performance from the narrower question of whether the system appropriately avoids unsafe AI reliance.

---

## Ablation Findings

| Assurance representation | Development AUROC |
|---|---:|
| Calibration risk alone | 0.6466 |
| + normalized distance and neighbourhood evidence | 0.6729 |
| + revised wrong-confident evidence | 0.6826 |
| + pre-specified interactions | **0.6837** |

Most of the gain comes from properly representing complementary assurance evidence; interaction terms provide a smaller additional improvement.

---

## Statistical Robustness

| Comparison | ΔF1 | ΔCost | McNemar |
|---|---:|---:|---:|
| MAGD-Fraud vs AI-only | +0.3748 | −364.31 | p < 0.001 |
| MAGD-Fraud vs L2D-Standard | +0.0837 | −75.13 | p < 0.001 |
| MAGD-Fraud vs MAGD-Additive | +0.2160 | −226.00 | p < 0.001 |

Paired bootstrap confidence intervals for F1 and cost exclude zero in the favourable direction for these comparisons.

---

## Repository Structure

```text
MAGD-Fraud/
├── src/
├── scripts/
├── tests/
├── docs/
├── figures/
├── data/
│   └── outputs/
├── config.yaml
├── requirements.txt
├── ARCHITECTURE.md
├── WORKFLOW.md
└── README.md
```

Large raw benchmark files, processed datasets, embeddings, per-case decision logs, and archived experiment runs are intentionally excluded from Git.

---

## Installation

```bash
git clone https://github.com/mahend72/MAGD-Fraud.git
cd MAGD-Fraud

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
.venv\Scripts\activate
```

---

## Dataset

The FiFAR benchmark is **not redistributed** through this repository.

FiFAR is publicly available from:

**https://doi.org/10.6084/m9.figshare.28351172**

After downloading the benchmark, place the files according to the paths expected by `config.yaml` and `WORKFLOW.md`.

---

## Reproducing the Experiments

Full experiment:

```bash
python scripts/run_full_magd_experiment.py --config config.yaml
```

Run tests:

```bash
pytest -q
```

Key output locations:

```text
data/outputs/final_metrics/
data/outputs/paper_tables/
data/outputs/magd_policy/
data/outputs/audit_pack/
data/outputs/final_reproducible_run/
```

---

## Baselines

The repository includes:

- AI-only;
- numerical-confidence threshold;
- distance-based threshold;
- best-expert and random-expert references;
- independent Learning-to-Defer (`L2D-Standard`);
- oracle upper-bound analysis;
- developmental additive MAGD variants.

The repository does **not** claim state-of-the-art performance over every recent deferral architecture.

---

## Reproducibility and Scientific Assurance

The pipeline includes:

- frozen configuration and random seeds;
- train/development/test separation;
- fold-local preprocessing;
- validation-derived thresholds;
- no test-label use in deployable routing;
- independent L2D baseline features;
- canonical manifests and configuration snapshots;
- deterministic experiment checks;
- paired statistical comparisons;
- case-level audit evidence;
- scientific guardrail tests.

---

## Continuous Integration and Research Assurance

CI is treated as a **research-quality gate**, not as a production deployment pipeline.

```text
Push / Pull Request
        ↓
Environment setup
        ↓
Dependency installation
        ↓
Syntax/import checks
        ↓
Unit and integration tests
        ↓
Scientific invariant tests
        ↓
Pass / fail
```

The full FiFAR experiment is not run in standard CI because the benchmark data and large generated artifacts are intentionally excluded from the repository.

No production Continuous Deployment claim is made.

---

## Auditability

MAGD-Fraud preserves an evidence chain:

```text
AI prediction
→ assurance evidence
→ AI-reliance score
→ routing decision
→ final decision
```

The audit layer records assurance signals, routing metadata, selected route, decision rationale, and policy configuration. This demonstrates computational traceability, not validated usability by real fraud auditors.

---

## Limitations

- FiFAR uses synthetic benchmark experts, not real fraud investigators.
- Evaluation is offline and benchmark-based.
- The study uses one fraud-review domain.
- The final policy does not exercise single-expert routing on the benchmark test split.
- The assurance score is a ranking score, not a calibrated probability of AI failure.
- Reviewer workload is high because escalation uses a five-expert panel.
- Operational cost parameters require institution-specific validation.
- The benchmark test split had been inspected during earlier development diagnosis.
- Independent temporal and external validation remain necessary.
- Audit completeness does not establish real-world governance usability.

---

## Responsible Use

MAGD-Fraud is a **research prototype for AI assurance and Human–AI routing**. It should not be used as a production financial decision system without independent validation, monitoring, governance, human oversight, workload validation, and domain-specific risk assessment.

---

## Code and Data Availability

**Code:**  
https://github.com/mahend72/MAGD-Fraud

**FiFAR benchmark:**  
https://doi.org/10.6084/m9.figshare.28351172

---

## Citation

The MAGD-Fraud manuscript is currently under submission.

```bibtex
@misc{magdfraud2026,
  title  = {MAGD-Fraud: Learning AI Reliance Risk for Assurance-Guided Human--AI Routing in Financial Fraud Review},
  author = {Kumar, Mahender},
  year   = {2026},
  note   = {Research software and manuscript under submission},
  url    = {https://github.com/mahend72/MAGD-Fraud}
}
```

Update this entry after publication with the final venue and DOI.

---

## Author

**Mahender Kumar**  
GitHub: https://github.com/mahend72
