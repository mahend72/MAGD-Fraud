# Reproducibility

This repository is designed to be rerun end to end from the checked-in config and data layout.

## Primary Run Command

```bash
python scripts/run_full_magd_experiment.py --config config.yaml
```

That runner validates configuration, checks data splits, builds model and assurance artifacts, runs routing variants, generates tables and audit outputs, runs scientific guardrails, and writes the final run summary.

## What To Expect

Successful runs produce:

- `data/outputs/run_summary.json`
- `data/outputs/run_summary.md`
- `data/outputs/final_metrics/scientific_checks.json`
- `docs/scientific_guardrails_report.md`

They also refresh the model, assurance, routing, ablation, evaluation, audit, and paper-table artifacts under `data/outputs/`.

## Data Discipline

The pipeline uses strict split discipline:

- training data supports model fitting and historical neighbour statistics
- validation data supports policy learning and constrained MAGD tuning
- test data is reserved for offline reporting and scientific checks

Deployable routing functions do not use test `y_true`.

## What To Check After A Run

Recommended verification points:

- `data/outputs/final_metrics/all_method_metrics.csv`
- `data/outputs/final_metrics/reliance_metrics.csv`
- `data/outputs/final_metrics/audit_metrics.csv`
- `data/outputs/final_metrics/statistical_comparisons.csv`
- `data/outputs/paper_tables/table_manifest.json`
- `data/outputs/audit_pack/audit_report.md`
- `data/outputs/run_summary.json`

## Environment Notes

The repository expects a Python environment with the dependencies in `requirements.txt`. The codebase uses `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `streamlit`, and `pytest` in the main workflow.

## Testing

Run the repository tests with:

```bash
pytest
```

The test suite includes coverage for the full MAGD runner, scientific checks, audit-pack generation, statistical tests, and the main assurance and deferral stages.
