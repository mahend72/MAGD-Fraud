# Workflow

## End-to-End Execution

Run the complete experiment with:

```bash
python scripts/run_full_experiment.py --config config.yaml
```

This executes the following sequence.

## 1. Inspect Data

```bash
python scripts/inspect_fifar.py --config config.yaml
```

Use this when the FiFAR schema is not yet mapped.

Outputs:

- console summary of files, columns, missingness, candidate label and score fields

## 2. Prepare Data

```bash
python scripts/prepare_data.py --config config.yaml
```

Outputs:

- `data/processed/X_train.csv`
- `data/processed/X_val.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_val.csv`
- `data/processed/y_test.csv`
- metadata, expert, and capacity tables

## 3. Train Or Load Model

```bash
python scripts/train_or_load_model.py --config config.yaml
```

Behavior:

- if FiFAR already contains model scores, reuse them
- otherwise train the fraud model

Outputs:

- model predictions for train / validation / test
- model metrics

## 4. Run Calibration

```bash
python scripts/run_calibration.py --config config.yaml
```

Outputs:

- calibration report
- test assurance base
- reliability diagram

## 5. Run Distance Uncertainty

```bash
python scripts/run_distance_uncertainty.py --config config.yaml
```

Outputs:

- distance uncertainty per case
- threshold exploration over distance confidence
- threshold plots

## 6. Run Neighbor Evidence

```bash
python scripts/run_neighbor_evidence.py --config config.yaml
```

Outputs:

- long neighbor evidence table
- per-case neighbor summary

## 7. Run Wrong-Confident Detection

```bash
python scripts/run_wrong_confident_detection.py --config config.yaml
```

Outputs:

- wrong-confident risk per case

## 8. Run Assurance Risk

```bash
python scripts/run_assurance_risk.py --config config.yaml
```

Outputs:

- assurance risk score
- risk category
- recommended action

## 9. Run Baselines

```bash
python scripts/run_baselines.py --config config.yaml
```

Outputs:

- decision logs for each baseline
- baseline metrics

## 10. Run Assurance-Guided Deferral

```bash
python scripts/run_assurance_deferral.py --config config.yaml
```

Outputs:

- final assurance-guided decision log
- method-level metrics

## 11. Evaluate All Methods

```bash
python scripts/evaluate_all_methods.py --config config.yaml
```

Outputs:

- fraud metrics
- reliance metrics
- deferral metrics
- fairness metrics
- audit metrics

## 12. Run Statistical Tests

```bash
python scripts/run_statistical_tests.py --config config.yaml
```

Outputs:

- paired method comparison table

## 13. Generate Audit Pack

```bash
python scripts/generate_audit_pack.py --config config.yaml
```

Outputs:

- audit decision log
- assurance summary
- audit coverage JSON
- plots
- audit report

## 14. Generate Paper Tables

```bash
python scripts/make_paper_tables.py --config config.yaml
```

Outputs:

- `data/outputs/paper_tables/*.csv`

## 15. Launch Dashboard

```bash
streamlit run src/dashboard/app.py
```

Dashboard pages:

- Overview
- Case Review
- Threshold Explorer
- Results
- Audit Logs

