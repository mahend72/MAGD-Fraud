# MAGD-Fraud Method Summary

MAGD-Fraud is the repository’s end-to-end Human-AI routing method for FiFAR-style fraud alert review. It is a multi-evidence assurance layer placed on top of a base fraud predictor.

## Method in One Paragraph

For each case, MAGD-Fraud combines numerical confidence, calibration risk, distance-based uncertainty, local neighbourhood reliability, and wrong-confident AI risk into a deployable assurance score. That score is then combined with adaptive thresholds and expert-aware expected-cost routing to decide whether the case should remain with AI, go to a specific expert, or escalate to a panel. Optional drift, business, fairness, and capacity signals are included only when the data support them, and they are logged as unavailable when they do not.

## Core Signals

- Numerical confidence: `max(p, 1 - p)`
- Calibration risk: absolute calibration gap from held-out confidence bins
- Distance uncertainty: inverse distance confidence from PCA-space centroid proximity
- Local neighbour reliability: historical neighbor error rate and related local statistics
- Wrong-confident AI risk: a weighted combination of confidence disagreement and local risk signals
- MAGD assurance risk: weighted aggregation of the deployable evidence signals

## Routing Logic

The routing layer is designed to be deployable and auditable.

- Low assurance risk with lower AI cost favors AI.
- Medium assurance risk or lower expert cost favors deferral to the best available expert.
- High assurance risk or high wrong-confident risk favors escalation.

The constrained MAGD variant adds validation-learned penalties for overreliance, capacity, fairness, and audit gaps, with test labels reserved for offline evaluation only.

## What MAGD-Fraud Is Not

- It is not just distance uncertainty.
- It is not a human-subject validation study.
- It is not an operational deployment.
- It is not an oracle policy in deployable routing.

## Relevant Scripts

- `scripts/run_full_magd_experiment.py`
- `scripts/run_magd_policy.py`
- `scripts/run_magd_constrained.py`
- `scripts/run_magd_deferral.py`
- `scripts/run_magd_ablations.py`
- `scripts/run_scientific_checks.py`
