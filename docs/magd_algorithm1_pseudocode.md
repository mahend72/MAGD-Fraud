# Algorithm 1: MAGD-Fraud Pseudocode

## Algorithm 1. MAGD-Fraud Routing

**Inputs**

- training data \(D_{train}\)
- validation data \(D_{val}\)
- test data \(D_{test}\)
- historical synthetic expert predictions \(E\)
- optional capacity table \(C\)
- policy mode \(m \in \{\texttt{heuristic}, \texttt{learned}, \texttt{constrained}\}\)
- assurance weights \(w\)
- thresholds \(\tau_{low}, \tau_{high}, \tau_0\)

**Outputs**

- decision log \(R\)
- method metrics \(M\)

```text
Algorithm 1 MAGD-Fraud

1: Train fraud predictor on D_train or load benchmark AI scores
2: Compute train, validation, and test AI predictions and scores
3: Compute calibration table on held-out predictions
4: Build embedding space from training features

5: for each case x in {D_val, D_test} do
6:     compute numerical_confidence(x)
7:     compute calibration_risk(x)
8:     compute distance_uncertainty(x)
9:     retrieve k nearest historical training cases of x
10:    compute neighbor_error_rate(x)
11:    compute neighbor_fraud_rate(x)
12:    compute neighbor_ai_agreement(x)
13:    compute wrong_confident_risk(x)
14: end for

15: if m = learned or m = constrained then
16:    learn assurance weights on D_val only
17:    if constrained then
18:        optimize FraudLoss
19:        + lambda_overreliance * OverReliance
20:        + lambda_capacity * CapacityViolation
21:        + lambda_fairness * FairnessPenalty
22:        + lambda_audit * AuditGap
23:        subject to CorrectRejection >= min_correct_rejection
24:        subject to AuditCoverage >= min_audit_coverage
25:    end if
26: end if

27: estimate expert reliability statistics from historical expert data E
28: estimate expert bias risk from group-wise disparities when available
29: initialize remaining capacity from C when available

30: for each test case x_i in D_test do
31:    compute MAGD_AR_i from active assurance signals
32:    compute adaptive threshold
33:        tau_i = tau_0
34:              + beta_1 * business_risk_i
35:              + beta_2 * fairness_risk_i
36:              + beta_3 * capacity_pressure_i
37:    clip tau_i to [0, 1]

38:    compute AI expected cost L_AI(i)
39:    compute expected cost L_expert,j(i) for each available expert j
40:    choose best available expert j*

41:    if MAGD_AR_i >= tau_high or wrong_confident_risk_i is high then
42:        route x_i to Escalate
43:        collect votes from top-k reliable available experts
44:        final prediction = majority vote
45:        record reason = high-risk escalation
46:    else if MAGD_AR_i < tau_low and MAGD_AR_i < tau_i
47:            and L_AI(i) <= L_expert,j*(i) then
48:        route x_i to AI
49:        final prediction = AI prediction
50:        record reason = low-risk AI use
51:    else
52:        if expert j* is available under fairness and capacity constraints then
53:            route x_i to Human Expert
54:            final prediction = prediction of expert j*
55:            record reason = defer to best expert
56:        else
57:            use next feasible expert or AI fallback
58:            record capacity/fairness rationale
59:        end if
60:    end if

61:    append full audit evidence for x_i to decision log R
62: end for

63: evaluate R on D_test labels offline
64: compute fraud, reliance, assurance, fairness, and audit metrics
65: return R, M
```

## Short Paper Version

If you need a tighter camera-ready version:

```text
Algorithm 1 MAGD-Fraud
Input: trained predictor, historical data, expert panel, optional capacity table
Output: routing decisions and audit log

1: Compute deployable assurance signals for each case
2: Learn MAGD weights on validation data if using learned/constrained mode
3: Estimate expert reliability, bias, and remaining capacity
4: for each test case do
5:     Compute MAGD assurance risk
6:     Compute adaptive threshold and expected AI / expert costs
7:     if high MAGD risk or high wrong-confident risk then
8:         escalate via majority vote among top-k reliable experts
9:     else if low MAGD risk and AI expected cost is smallest then
10:        use AI
11:    else
12:        defer to best feasible expert
13:    end if
14:    record route, selected expert(s), evidence, and reason
15: end for
```
