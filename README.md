# MAGD-Fraud

**Multi-evidence Assurance-Guided Deferral for Human–AI Fraud Review**

MAGD-Fraud is a research codebase for studying whether explicit, case-level assurance
evidence — beyond a model's raw output probability — can guide principled decisions
about when an AI fraud-screening prediction should be relied upon, deferred to a human
expert, or escalated for panel review. Fraud alert review is used as the experimental
domain; the object of study is the general Human–AI assurance and deferral problem, not
a production fraud-detection product.

The repository is published on GitHub as **MAGD-Fraud**. Internally, several
long-standing identifiers predate that name — the project directory, the `config.yaml`
project name (`haaf_fifar`), and the Streamlit dashboard title still read "HAAF FiFAR."
Both names refer to the same codebase; this README uses MAGD-Fraud throughout.

> FiFAR experts used in this repository are **synthetic experts** from a public
> benchmark panel, not real human reviewers, and the dashboard is a **research
> prototype**, not a validated operational interface. See [Limitations](#limitations).

## Overview

A fraud-screening model outputs a score. Standard practice treats that score's own
confidence (e.g. `max(p, 1 - p)`) as the signal for deciding whether to trust it. That
signal is convenient but incomplete: a model can be confidently wrong on cases that lie
far from its training distribution, in regions where it historically errs, or in classes
of cases where its own calibration is poor — and raw confidence does not distinguish
these situations from genuine certainty.

MAGD-Fraud studies an alternative: attach **case-level assurance evidence** to each
prediction — signals about calibration, embedding-space distance, local neighbourhood
error history, and confident-but-likely-wrong risk — and use that evidence, rather than
confidence alone, to route the case. The pipeline sits between model prediction and
final decision:

```
AI prediction  →  assurance evidence  →  assurance/risk aggregation  →  routing  →  AI / expert / escalation  →  evaluation and audit evidence
```

Each stage produces artifacts that are retained for offline evaluation and audit,
rather than being consumed and discarded.

## Research Motivation

The research question is whether multi-evidence, case-level assurance signals produce
better-calibrated reliance decisions in a Human–AI review workflow than simpler
single-signal baselines (a confidence threshold or a distance threshold), while keeping
the routing decision itself auditable and free of test-label leakage.

MAGD-Fraud is framed primarily as a contribution to **AI assurance and Human–AI
decision research**; the FiFAR fraud-alert-review benchmark is the experimental
substrate used to test the idea computationally, not a claim about a deployed fraud
system. The scope is intentionally computational and benchmark-based: it does not
involve real human reviewers, live deployment, or field validation (see
[Limitations](#limitations)).

## MAGD-Fraud Architecture

No system-architecture figure is currently tracked in `figures/` (the tracked figures
are result plots — see [Key Research Findings](#key-research-findings)), so the flow is
shown here as implemented in `src/` and `scripts/run_full_magd_experiment.py`:

```
Input case (FiFAR-style tabular fraud data)
        │
        ▼
Base AI model  (existing FiFAR model scores, or trained XGBoost / RandomForest fallback)
        │
        ▼
Prediction + probability
        │
        ▼
Assurance evidence
 ├── numerical confidence        (src/assurance/numerical_confidence.py)
 ├── calibration risk            (src/assurance/calibration.py)
 ├── distance uncertainty        (src/assurance/distance_uncertainty.py)
 ├── local neighbourhood reliability (src/assurance/local_reliability.py, explanation_neighbors.py)
 └── wrong-confident risk        (src/assurance/wrong_confident_detector.py)
        │
        ▼
MAGD assurance risk   (src/assurance/magd_risk.py, magd_v2.py — weighted combination + adaptive threshold)
        │
        ▼
Assurance-guided routing policy   (src/deferral/magd_deferral.py, magd_policy.py,
                                    magd_validation_tuned.py, magd_constrained.py)
 ├── AI
 ├── Human expert  (expert-aware expected-cost routing, src/deferral/expert_routing.py)
 └── Escalation    (majority vote among top-k reliable experts)
        │
        ▼
Evaluation + audit evidence   (src/evaluation/, scripts/generate_audit_pack.py,
                                scripts/generate_claim_evidence_matrix.py)
```

For a component-by-component description of each layer, see [ARCHITECTURE.md](./ARCHITECTURE.md).
For the pseudocode form of the routing algorithm, see
[docs/magd_algorithm1_pseudocode.md](docs/magd_algorithm1_pseudocode.md).

## Assurance Signals

| Signal | What it measures | Assurance interpretation |
| --- | --- | --- |
| `numerical_confidence` | `max(p, 1 - p)` of the base model's output probability | The model's own self-reported certainty. Used as a baseline signal and threshold-baseline input — not, by itself, treated as sufficient evidence for reliance. |
| `calibration_risk` | Local calibration error of the bin containing the case (`src/assurance/calibration.py`) | Whether the model's stated confidence historically matches its observed accuracy in that confidence range. |
| `distance_uncertainty` | `1 - distance_confidence`, where `distance_confidence` is proximity to the predicted class's PCA-embedding centroid (`src/assurance/distance_uncertainty.py`) | How atypical the case is relative to the training distribution. Centroid-based, so it is a coarse proxy — fraud patterns are often multimodal, which limits this signal on its own. |
| `neighbor_error_rate` | Fraction of the case's k-nearest training neighbours (embedding space) on which the AI was historically wrong (`src/assurance/local_reliability.py`) | Local, retrospective reliability evidence — whether the model tends to err on similar historical cases. Degrades under distribution shift. |
| `wrong_confident_risk` | Deployable risk score for high-confidence-but-likely-wrong behaviour, combining confidence, distance uncertainty, calibration risk, neighbour error rate, and confidence disagreement (`src/assurance/wrong_confident_detector.py`) | Targets the specific failure mode of overreliance: confident AI outputs that are disproportionately likely to be wrong. |
| `magd_assurance_risk` | Weighted combination of the active signals above (weights in `config.yaml: magd.weights`), optionally including drift/business risk if configured | The aggregate MAGD-Fraud risk score used by the canonical routing policy. This is **not** a single learned probability of error — it is a configured, weighted aggregate of the signals above. |

A related, earlier aggregate — `assurance_risk` (`src/assurance/assurance_risk.py`) —
is also present in the codebase and used by a legacy `assurance_deferral` routing path
referenced in `ARCHITECTURE.md`. The canonical aggregate for the current pipeline is
`magd_assurance_risk`, computed in `src/assurance/magd_risk.py`.

No claim is made in this repository that any individual signal, by itself, is a strong
discriminator of AI error — see [Ablation and Assurance Evidence](#ablation-and-assurance-evidence).

## Assurance-Guided Routing

Given `magd_assurance_risk` and an adaptive threshold, MAGD-Fraud chooses one of three
routes for each case, using expert-aware expected-cost comparison:

- **low** assurance risk (below the low threshold) and AI's expected cost is no higher
  than the best available expert's → **use AI**
- **medium** assurance risk, or the expert's expected cost is lower → **defer to the
  best available expert** (subject to fairness/capacity constraints where configured)
- **high** assurance risk, or high wrong-confident risk → **escalate**: majority vote
  among the top-`k` most reliable available experts

Thresholds (`magd.thresholds.low_risk` / `high_risk`) and the adaptive-threshold
adjustment (`magd.adaptive_threshold`, folding in business/fairness/capacity pressure)
are configured in `config.yaml`, not hardcoded.

The repository implements several routing variants, all built on the same assurance
signals, distinguished by how weights/thresholds are set:

| Variant | Module | How it is configured | Role |
| --- | --- | --- | --- |
| MAGD-Heuristic | `src/deferral/magd_deferral.py` | Fixed weights from `config.yaml: magd.weights` | Reference heuristic variant |
| MAGD-Learned | `src/deferral/magd_policy.py` | Weights selected by grid search over candidate weight combinations on validation data | Experimental variant |
| MAGD-Fraud (validation-tuned) | `src/deferral/magd_validation_tuned.py` | Thresholds tuned on validation data across a set of review-budget targets, objective = cost/recall/F1 blend | Experimental variant |
| MAGD-Constrained | `src/deferral/magd_constrained.py` | Constrained optimisation (SLSQP) enforcing minimum correct-rejection and audit-coverage constraints, with an intervention-calibrated follow-up stage | **Current canonical default** (`config.yaml: magd.mode: "constrained"`) |

Weight/threshold selection for the learned and constrained variants uses **validation
data only**; test labels are never used inside deployable routing logic (see
[Scientific Guardrails](#scientific-guardrails)).

## Research Pipeline

The canonical, end-to-end pipeline is:

```bash
python scripts/run_full_magd_experiment.py --config config.yaml
```

This is a single 29-stage runner (`scripts/run_full_magd_experiment.py`) that executes,
in order: config/data-split validation → base model predictions → the five assurance
signals above → MAGD assurance risk and its calibration → expert reliability estimation
→ baselines → learning-to-defer baseline → the four MAGD routing variants → ablations →
full method evaluation → budget-matched deferral analysis → risk-calibration and
constraint-sensitivity analysis → paired statistical tests → audit-pack and
claim-evidence-matrix generation → paper-table generation → scientific guardrail checks
→ a final run summary. Conceptually:

```
Dataset → preprocessing → AI predictions → assurance evidence → MAGD score →
routing → baselines → evaluation → statistical analysis → audit/reproducibility artifacts
```

An older, shorter runner, `scripts/run_full_experiment.py` (documented in
[WORKFLOW.md](./WORKFLOW.md)), predates the MAGD routing variants, ablations, and
statistical-testing stages; treat it as a legacy step-by-step walkthrough of the earlier
pipeline, not the canonical entry point.

## Repository Structure

```
MAGD-Fraud/
├── src/            # data loading/preprocessing, assurance signals, deferral/routing, evaluation, dashboard, utils
├── scripts/        # runnable entry points: data prep, per-signal runners, full pipeline, figures, audit pack
├── tests/          # unit and integration tests (synthetic fixtures — no FiFAR data required)
├── docs/           # reproducibility, limitations, results-table guide, manuscript drafts, experiment manifests
├── figures/        # tracked result figures (PNG/PDF) generated from frozen run artifacts
├── data/           # raw/processed inputs (not tracked) and data/outputs/ (tracked reproducibility artifacts)
├── config.yaml     # paths, column mappings, split logic, assurance/MAGD weights and thresholds, costs
├── requirements.txt
├── ARCHITECTURE.md # component-by-component system design
├── WORKFLOW.md     # legacy step-by-step execution guide
└── README.md
```

## Installation

```bash
git clone https://github.com/mahend72/MAGD-Fraud.git
cd MAGD-Fraud
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Runtime dependencies (`requirements.txt`): `PyYAML`, `pandas`, `pyarrow`, `openpyxl`,
`xgboost`, `scikit-learn`, `matplotlib`, `streamlit`, `pytest`.

## Dataset

MAGD-Fraud is evaluated on **FiFAR** (Financial Fraud Alert Review), a public synthetic
benchmark for learning-to-defer research. **The benchmark is not redistributed in this
repository.** Obtain it from its original public source and cite the FiFAR paper/dataset
directly:

> TODO: insert the official FiFAR paper/dataset citation and DOI/URL here — not
> substituted with an unverified link in this pass, per repository policy.

`data/`, `data/raw/`, `data/processed/`, and `data/ICAIF_KAGGLE/` are excluded from
version control (`.gitignore`) because they hold the downloaded benchmark, generated
splits, and large per-case model/assurance outputs. Only small, curated reproducibility
artifacts under `data/outputs/` — final metrics, paper tables, policy configs, and
figure-generation data — are tracked in Git.

After downloading FiFAR, place it locally using one of the layouts the loader and
`config.yaml` expect:

```text
MAGD-Fraud/
  data/
    raw/
      fifar_cases.csv
      fifar_expert_predictions.csv
      fifar_capacity.csv
```

or, matching the current `config.yaml` (`dataset.train_file` / `test_file` /
`expert_predictions_file` / `historical_expert_predictions_file`):

```text
MAGD-Fraud/
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

No synthetic data is generated in place of the benchmark; the pipeline expects real
FiFAR files at one of these locations.

## Reproducing the Experiments

```bash
# 1. environment setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. dataset preparation (after placing FiFAR under data/, see Dataset above)
python scripts/inspect_fifar.py --config config.yaml   # optional: inspect an unmapped schema
python scripts/prepare_data.py --config config.yaml    # writes data/processed/{X,y}_{train,val,test}.csv

# 3. full canonical experiment
python scripts/run_full_magd_experiment.py --config config.yaml

# 4. tests (no FiFAR data required)
pytest -q

# 5. dashboard (optional, prototype interface)
streamlit run src/dashboard/app.py
```

The full-experiment runner writes its outputs under `data/outputs/`, notably
`data/outputs/run_summary.{json,md}`, `data/outputs/final_metrics/`,
`data/outputs/paper_tables/`, and `data/outputs/audit_pack/`. Camera-ready figures
under `figures/` are generated separately from a frozen, hash-verified snapshot in
`data/outputs/final_reproducible_run/` via dedicated scripts
(`scripts/make_ai_error_capture_curve_figure.py`,
`scripts/make_calibration_interaction_figures.py`,
`scripts/make_f1_vs_workload_figure.py`,
`scripts/make_reviewer_cost_sensitivity_figure.py`); see
`scripts/freeze_experiment_setup.py` and `scripts/freeze_canonical_decision_hashes.py`
for how that snapshot is produced. For the full table-to-file-to-script mapping, see
[docs/results_table_guide.md](docs/results_table_guide.md).

## Baselines

Implemented in `src/deferral/baselines.py` and `src/deferral/learning_to_defer_baseline.py`:

| Baseline | Deployable? | Notes |
| --- | --- | --- |
| AI-only | Yes | Base model prediction used for every case. |
| Best expert only | Yes | Historically best-performing single expert, always used. |
| Random expert | Yes | A random expert is assigned per case. |
| Numerical-confidence threshold | Yes | Defers when `numerical_confidence` is below `baselines.numerical_conf_threshold`. |
| Distance-confidence threshold | Yes | Defers when `distance_confidence` is below `baselines.distance_conf_threshold`. |
| Learning-to-defer (L2D-Standard) | Yes | Trained on the base score plus the four independently-deployable assurance signals (confidence, distance uncertainty, calibration risk, neighbour error rate, wrong-confident risk) — explicitly *not* on `magd_assurance_risk`, so it is an independent comparator. |
| Learning-to-defer + MAGD (augmented) | Experimental only | Adds `magd_assurance_risk` as a feature (`scripts/run_learning_to_defer_augmented_experiment.py`). Because this makes L2D dependent on MAGD's own output, it is reported only as an explicitly-labelled augmented variant, never as the independent L2D baseline. |
| Oracle upper bound | **No** | Uses ground truth to select the best possible outcome per case. Reported only as a non-deployable upper-bound reference (enforced by a scientific guardrail check, see below). |

No baseline here is described as state-of-the-art; they are reference points for
interpreting MAGD-Fraud's routing behaviour.

## Evaluation

Evaluation is split across `src/evaluation/*.py` into model-quality metrics and
Human–AI routing metrics, computed per method and written to `data/outputs/final_metrics/`:

**Predictive performance** (`fraud_metrics.py`): precision, recall, F1, PR-AUC, false
positives/negatives, false-positive/negative rate, cost-sensitive loss
(`config.yaml: costs`). Accuracy and ROC-AUC are **not** computed or reported — a
scientific guardrail explicitly checks that accuracy is not treated as a headline metric.

**Reliance / routing behaviour** (`reliance_metrics.py`, `deferral_metrics.py`,
`reviewer_workload.py`): correct reliance, correct rejection, overreliance,
underreliance, wrong-confident avoidance rate, AI coverage, expert-deferral rate,
escalation rate, deferral precision/recall, capacity-violation rate, oracle gap, and
reviewer workload (case counts).

**Fairness** (`fairness_metrics.py`): group-wise false-positive/false-negative rate and
disparities, computed when `columns.sensitive_attributes` is configured and available.

**Auditability** (`audit_metrics.py`): audit coverage, evidence completeness, complete
decision-log rate, missing-evidence rate, missing-rationale rate.

**Statistical significance** (`statistical_tests.py`): paired bootstrap confidence
intervals, McNemar's test on paired correctness, and Wilcoxon signed-rank tests across
methods on the test set.

## Key Research Findings

The repository tracks two distinct sets of result artifacts, and their headline numbers
for the same methods currently differ:

- `data/outputs/paper_tables/` — generated end-to-end by the canonical
  `run_full_magd_experiment.py` run (manuscript Tables 1–7, per
  [docs/results_table_guide.md](docs/results_table_guide.md)).
- `data/outputs/final_reproducible_run/` — a separately frozen, hash-verified snapshot
  (`scripts/freeze_experiment_setup.py`, `scripts/freeze_canonical_decision_hashes.py`)
  used as the source of truth for the figures under `figures/` and for
  `scripts/generate_final_paper_tables.py`.

Because these two tracked artifact sets do not currently agree on method-level numbers
for the same routing variants, this README does not reproduce specific headline metrics
— doing so risks presenting a stale or inconsistent number as "the" result. Readers
wanting the current numbers should:

1. Regenerate both via the commands in [Reproducing the Experiments](#reproducing-the-experiments), or
2. Read the tracked CSVs directly — `data/outputs/paper_tables/*.csv` and
   `data/outputs/final_reproducible_run/paper_tables/*.csv` — alongside
   `data/outputs/paper_tables/table_manifest.json` and
   [docs/results_table_guide.md](docs/results_table_guide.md), which map every table
   and figure to its generating script.

This discrepancy is a real, open item for this repository's maintainers to reconcile
before the two artifact sets are cited together in a manuscript; see the final
recommendation communicated alongside this README revision.

## Ablation and Assurance Evidence

`data/outputs/paper_tables/ablation.csv` (generated by `scripts/run_magd_ablations.py`)
reports signal-level ablations — `distance_only`, `distance_plus_calibration`,
`distance_plus_neighbor_error`, `distance_plus_wrong_confident`, and the full
heuristic/learned/constrained MAGD variants — evaluated with the same metric set as the
main comparison.

In the currently tracked run, several of these ablation variants (including some
multi-signal combinations) produce routing decisions that are numerically identical or
very close to the AI-only baseline's operating point. This is disclosed rather than
smoothed over: it indicates that, under the current configuration, adding some
individual signals does not by itself change routing behaviour enough to move the
aggregate metrics — a signal can be conceptually well-motivated (e.g. calibration risk,
neighbour error rate) while still contributing weakly to standalone discrimination once
combined through the configured thresholds and weights. Readers should inspect
`data/outputs/paper_tables/ablation.csv` directly and treat any per-signal contribution
claim as conditional on the current threshold/weight configuration in `config.yaml`,
not as a general property of the signal.

## Reproducibility

- **Frozen configuration**: `config.yaml` fixes seeds (`experiment.seed: 42`,
  `split.random_state: 42`, per-model `random_state`), split fractions
  (70/15/15, stratified), and all assurance/MAGD weights and thresholds.
- **Train/validation/test separation**: training data fits the model and historical
  neighbour statistics; validation data supports MAGD policy learning and threshold
  tuning; test data is reserved for offline evaluation and scientific checks
  (`docs/reproducibility.md`).
- **No test-label leakage in deployable routing**: `experiment.test_labels_allowed_for_routing: false`
  in `config.yaml`; enforced in code and by a scientific guardrail (below).
- **Frozen, hash-verified snapshot**: `scripts/freeze_experiment_setup.py` and
  `scripts/freeze_canonical_decision_hashes.py` write a point-in-time snapshot to
  `data/outputs/final_reproducible_run/`, including `canonical_decision_hashes.json`,
  used to detect unauthorized drift in routing decisions (`tests/test_canonical_integrity.py`).
- **Provenance and run metadata**: each full run writes `data/outputs/run_summary.json`,
  `run_summary.md`, and (for the frozen snapshot) `run_metadata.json` and
  `config_snapshot.yaml`.
- **Tracked vs. regenerated artifacts**: small, curated tables, policy configs, and
  figure-generation data under `data/outputs/` are tracked in Git; large per-case
  decision logs, embeddings, and raw score dumps are excluded (`.gitignore`) and must be
  regenerated locally by rerunning the pipeline.

## Scientific Guardrails

`src/utils/scientific_checks.py` (invoked by `scripts/run_scientific_checks.py`, and
exercised against synthetic fixtures in `tests/test_scientific_checks.py`) implements
automated, code-level checks, including:

- **`deployable_y_true_guard`** — inspects the *source code* of the deployable routing
  functions (`route_magd_cases`, `route_cases`, and their cost-computation helpers) to
  confirm they do not reference `y_true` in decision logic.
- **`oracle_upper_bound_only`** — confirms the oracle baseline is confined to its own
  non-deployable artifact and never selected inside a deployable decision log.
- **`accuracy_not_headline_metric`** — confirms accuracy is not reported as a first-class
  metric and is not described as the primary metric in the README.
- **`synthetic_experts_labelled`** / **`dashboard_labelled_prototype`** — confirm the
  README and dashboard correctly label FiFAR experts as synthetic and the dashboard as a
  prototype.
- **`optional_signal_status`** — confirms optional drift/business/fairness/capacity
  signals are explicitly logged as unavailable, rather than silently defaulted, when
  not configured.
- A set of **required-artifact checks** confirming the paper-facing tables (budget-matched
  results, MAGD risk calibration, constraint sensitivity, statistical tests, the
  artifact-table map) are present, non-empty, and free of placeholder values.

These checks read the tracked `data/outputs/` artifacts and the README/source text
directly — they do not require the FiFAR dataset. As of this audit, running
`python scripts/run_scientific_checks.py --config config.yaml` against a fully
populated local `data/outputs/` tree reports `status: passed` with 0 critical failures
(`docs/scientific_guardrails_report.md`); the script also depends on at least one
gitignored, locally-regenerated artifact (`oracle_upper_bound_decisions.csv`), so it is
not run in CI against a fresh checkout — see
[Continuous Integration and Research Assurance](#continuous-integration-and-research-assurance).

## Testing

```bash
pytest -q
```

As of this audit (2026-08-27), the suite comprises **197 tests**. In a fully populated
local development environment (i.e. after running the full pipeline at least once, so
`data/outputs/` is complete), all 197 pass in roughly 4 minutes.

On a **fresh checkout** — no locally generated `data/outputs/` beyond what Git tracks —
195 tests pass, 1 is correctly skipped
(`tests/test_magd_v2.py`, guarded by `skipif` on a missing frozen artifact), and **1
test fails**: `tests/test_canonical_integrity.py::test_all_canonical_decision_logs_match_saved_hashes`.
That test recomputes hashes over frozen per-case decision logs
(`data/outputs/final_reproducible_run/decision_logs/`,
`magd_v2_test_decisions.csv`) that are intentionally excluded from Git, while the
hash manifest it compares against (`canonical_decision_hashes.json`) *is* tracked —
so the check has no `skipif` guard for the missing-decision-logs case and fails outright
on a clean clone. This is a real, verified gap rather than a hypothetical one; it is
deselected in CI (below) and worth fixing (adding a `skipif` on the missing decision
logs) so the suite is green on a fresh clone without special-casing it externally.

The rest of the suite (all using synthetic, in-test fixtures — no FiFAR data) covers:
data loading/splitting and leakage checks, each assurance signal (calibration, distance
uncertainty, local reliability, wrong-confident detection, MAGD risk), routing/deferral
logic across all MAGD variants and baselines, evaluation metrics, statistical tests,
audit-pack and claim-evidence-matrix generation, config schema validation, and the
scientific guardrail checks themselves.

## Continuous Integration and Research Assurance

There was no CI configuration in this repository prior to this change. This revision
adds `.github/workflows/ci.yml`, framed as **continuous integration for scientific and
research assurance**, not deployment: it re-checks code health and scientific
invariants on every push and pull request to `main`, and does **not** build, package, or
deploy anything (there is nothing in this repository to deploy).

The workflow:

```
Push / Pull Request
        ↓
Checkout + Python 3.11 setup
        ↓
pip install -r requirements.txt
        ↓
Static import/syntax check (python -m compileall)
        ↓
pytest -q  (unit tests, scientific invariant tests, config-schema validation —
            197 tests, all on synthetic fixtures; one data-dependent regression
            test deselected, see Testing above)
        ↓
Pass / fail
```

It deliberately does **not** run `scripts/run_full_magd_experiment.py` or
`scripts/run_scientific_checks.py`: the former requires the (intentionally
unredistributed) FiFAR dataset, and the latter, on inspection, also requires at least
one gitignored, locally-regenerated artifact (`oracle_upper_bound_decisions.csv`), so
neither can run correctly against a bare checkout. If the maintainer later commits a
guardrail check (or a data-free fixture path for the existing one) that is genuinely
independent of regenerated per-case artifacts, it can be added as a further CI step.

## Limitations

- FiFAR experts are **synthetic experts** from a benchmark panel, not real human
  reviewers; routing/audit behaviour observed here does not establish how real analysts
  would behave operationally.
- The Streamlit dashboard is a **research prototype** for inspecting computational
  outputs, not a validated operational tool.
- The evaluation is entirely **offline and benchmark-based**, on a single application
  domain (financial fraud alert review); no live or field deployment has been attempted
  or evaluated in this repository.
- Assurance signals (calibration risk, distance uncertainty, neighbour error rate,
  wrong-confident risk) are **proxies**, not formal guarantees of error detection — see
  [Ablation and Assurance Evidence](#ablation-and-assurance-evidence) for their measured,
  sometimes weak, standalone discrimination.
- Distance uncertainty depends on the quality of its PCA embedding and can miss
  structure the projection does not represent well.
- Local neighbourhood reliability is retrospective; it can go stale under distribution
  shift.
- Optional signals (drift risk, business risk, capacity, fairness) depend on data or
  configuration that may not be present; when absent, they are logged as unavailable
  rather than silently defaulted (verified by a scientific guardrail), but their absence
  narrows the evidence set actually used.
- Routing thresholds and weights in `config.yaml` were set for this benchmark and
  configuration; they should not be assumed to generalise unchanged to other
  institutions, datasets, or cost structures — see `docs/limitations.md` for further
  discussion.
- Operational cost assumptions (`config.yaml: costs`) are illustrative parameters for
  this benchmark, not validated real-world review costs.
- The oracle baseline is an unreachable upper bound, not a deployable method.

## Ethics and Responsible Use

MAGD-Fraud is a **research prototype** for studying Human–AI assurance and deferral. It
should not be treated as a production financial-fraud decision system. Any operational
use would require independent validation, governance, monitoring, human oversight, and
a domain-specific risk assessment — none of which is established or claimed by this
repository.

## Code and Data Availability

**Code**: https://github.com/mahend72/MAGD-Fraud

**Data**: FiFAR (Financial Fraud Alert Review), a public benchmark for learning-to-defer
research. The benchmark is **not redistributed in this repository** — obtain it from its
original public source and cite it directly (see [Dataset](#dataset) for the
placeholder citation slot pending the verified DOI/URL).

## Citation

A citation for the MAGD-Fraud method/paper is not yet available. Use the placeholder
below and update it once a DOI or venue citation exists:

```bibtex
@misc{magdfraud2026,
  title  = {MAGD-Fraud: Multi-evidence Assurance-Guided Deferral for Human--AI Fraud Review},
  author = {Kumar, Mahender},
  year   = {2026},
  note   = {Repository citation to be updated upon publication. DOI/venue not yet assigned.},
  url    = {https://github.com/mahend72/MAGD-Fraud}
}
```

## Author

Mahender Kumar
GitHub: https://github.com/mahend72
