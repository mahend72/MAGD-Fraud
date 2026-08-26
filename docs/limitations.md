# Limitations

MAGD-Fraud is implemented as a computational assurance and routing study on the FiFAR benchmark. The current repository should be read with the following limitations in mind.

## Explicit Limitations

- FiFAR experts are synthetic benchmark experts, not real human reviewers.
- The dashboard is a prototype and not a validated production interface.
- Optional capacity, fairness, drift, and business signals depend on data availability and are logged as unavailable when missing.
- Human-subject validation has not been established in this repository.
- Operational deployment has not been established in this repository.

## Additional Practical Limits

- The main experiment depends on the configured data layout and available processed splits.
- The strongest claims that can be made here are benchmark and audit claims, not field-deployment claims.
- The oracle baseline is included only as an upper bound reference and is not deployable.
- Some outputs are offline diagnostics and should not be confused with runtime decisions.

## Consequence For Interpretation

The correct interpretation of the repository is that it demonstrates a deployable-style assurance and routing pipeline for fraud review, not that it proves live human performance or production readiness.
