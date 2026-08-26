# MAGD-Fraud Method Section Draft

## 3. Method

We propose **MAGD-Fraud** (Multi-evidence Assurance-Guided Deferral), a Human-AI routing framework for financial fraud alert review. The central motivation is that **distance-to-centroid uncertainty alone is too weak for fraud review**, where positive cases are rare, heterogeneous, and often locally irregular. MAGD-Fraud therefore replaces single-signal deferral with a **multi-evidence assurance representation** that combines model confidence, calibration, embedding-space uncertainty, local historical reliability, and wrong-confident AI risk. The resulting assurance score is then used by a routing policy that chooses among AI, a selected synthetic expert, or escalation.

Our setting is a three-way decision problem. For each alert case \(x_i\), the system observes a model score \(p_i \in [0,1]\), a binary AI prediction \(\hat{y}^{AI}_i\), and a collection of deployable assurance signals derived from training history and model outputs. The objective is to minimize task-level fraud loss under asymmetric false-positive and false-negative costs while also reducing harmful overreliance on wrong AI predictions and preserving auditable routing behavior.

### 3.1 Base Fraud Predictor

The predictive layer produces a fraud probability \(p_i\) and binary prediction \(\hat{y}^{AI}_i\). In the current implementation, the predictor is either:

- an existing benchmark AI score if FiFAR provides one, or
- a learned classifier, with XGBoost as the preferred model and Random Forest as fallback.

The AI prediction is obtained by thresholding the score at the configured operating point:

\[
\hat{y}^{AI}_i = \mathbb{1}[p_i \ge \tau_{model}]
\]

We retain this predictive layer largely unchanged. The main contribution of MAGD-Fraud is not a new fraud classifier, but a new **assurance-guided routing layer** built on top of the classifier.

### 3.2 Deployable Assurance Signals

MAGD-Fraud constructs a set of deployable assurance signals for each case.

**Numerical confidence.**
For binary fraud prediction, we define:

\[
c_i = \max(p_i, 1 - p_i)
\]

This captures how certain the model appears under its own score distribution.

**Calibration risk.**
Model scores are partitioned into confidence bins on held-out data, and we compute the absolute gap between average confidence and observed accuracy in each bin. Each case inherits the gap of its bin:

\[
r^{cal}_i = |\text{conf}_{bin(i)} - \text{acc}_{bin(i)}|
\]

This term captures whether the model’s apparent confidence is historically trustworthy.

**Distance-based uncertainty.**
We project tabular features into a PCA embedding space and compute class centroids from training data. Let \(z_i\) be the embedding of case \(i\), and let \(\mu_{\hat{y}^{AI}_i}\) be the centroid of the predicted class. The distance-to-predicted-centroid is:

\[
d_i = \| z_i - \mu_{\hat{y}^{AI}_i} \|_2
\]

Distance confidence is obtained by normalized inversion, and uncertainty is:

\[
u_i = 1 - \text{distance\_confidence}_i
\]

This recovers the core distance-uncertainty baseline used by prior versions of the repository.

**Local neighbourhood reliability.**
Because centroid-only uncertainty assumes relatively clean global structure, we augment it with local evidence. For each test case, we retrieve the \(k\) nearest historical training cases in embedding space and compute:

- `neighbor_error_rate`
- `neighbor_fraud_rate`
- `neighbor_ai_agreement`
- `mean_neighbor_distance`

The most important signal is local historical model unreliability:

\[
r^{nbr}_i = \frac{1}{k} \sum_{j \in \mathcal{N}_k(i)} \mathbb{1}[\hat{y}^{AI}_j \ne y_j]
\]

This signal is deployable because it uses only **historical training labels and historical AI correctness**, not test labels.

**Wrong-confident AI risk.**
We explicitly model the risk that the AI appears confident but may be wrong. Let:

\[
\Delta_i = |c_i - \text{distance\_confidence}_i|
\]

denote the disagreement between score-based and distance-based confidence. We define deployable wrong-confident risk as:

\[
r^{wc}_i =
\frac{
\alpha_1 c_i +
\alpha_2 u_i +
\alpha_3 r^{cal}_i +
\alpha_4 r^{nbr}_i +
\alpha_5 \Delta_i
}{
\sum_{m=1}^{5}\alpha_m
}
\]

with the result clipped to \([0,1]\).

This design intentionally avoids using \(y_i\) at deployment time. Ground truth is used only offline to evaluate whether the score captures wrong-but-confident AI failures.

### 3.3 MAGD Assurance Risk

The core assurance score combines multiple signals into a single case-level deployable quantity:

\[
\text{MAGD\_AR}_i =
\frac{
w_1 u_i +
w_2 r^{cal}_i +
w_3 r^{nbr}_i +
w_4 r^{wc}_i +
w_5 r^{drift}_i +
w_6 r^{biz}_i
}{
\sum_{m=1}^{6} w_m
}
\]

where:

- \(u_i\) is distance uncertainty
- \(r^{cal}_i\) is calibration risk
- \(r^{nbr}_i\) is neighbour error rate
- \(r^{wc}_i\) is wrong-confident risk
- \(r^{drift}_i\) is optional drift risk
- \(r^{biz}_i\) is optional business risk

Unavailable optional signals are set to zero and explicitly marked as unavailable in the output artifacts.

We then discretize the continuous assurance score into three risk regions:

- **low** if \(\text{MAGD\_AR}_i < \tau_{low}\)
- **medium** if \(\tau_{low} \le \text{MAGD\_AR}_i < \tau_{high}\)
- **high** if \(\text{MAGD\_AR}_i \ge \tau_{high}\)

These categories are used only as routing abstractions; the continuous score is still retained for expected-cost reasoning.

### 3.4 Adaptive Thresholding

Single global thresholds are often too crude for fraud review. MAGD-Fraud therefore computes a case-specific threshold:

\[
\tau_i =
\tau_0 +
\beta_1 r^{biz}_i +
\beta_2 r^{fair}_i +
\beta_3 r^{cap}_i
\]

where:

- \(\tau_0\) is a base threshold
- \(r^{biz}_i\) is optional business risk
- \(r^{fair}_i\) is optional fairness risk
- \(r^{cap}_i\) is capacity pressure

The threshold is clipped to \([0,1]\). Missing optional signals are replaced by zero and logged.

This mechanism lets the system become more conservative for cases with higher operational importance or higher routing pressure.

### 3.5 Expert Reliability, Fairness, and Capacity

MAGD-Fraud assumes a panel of synthetic experts provided by the FiFAR environment. For each expert \(j\), we estimate historical:

- accuracy
- false positive rate
- false negative rate
- cost-sensitive loss
- group-wise false positive and false negative rates, when sensitive attributes exist
- bias risk
- remaining capacity

The expected cost of routing case \(i\) to expert \(j\) is:

\[
L_{\text{expert},j}(i) =
\lambda_{FP} \cdot \text{FPR}_j +
\lambda_{FN} \cdot \text{FNR}_j +
\lambda_{fair} \cdot \text{BiasRisk}_j +
\lambda_{cap} \cdot \text{CapacityPenalty}_j
\]

The AI expected cost is approximated as:

\[
L_{AI}(i) =
\lambda_{FP} \cdot \widehat{FP\_risk}_i +
\lambda_{FN} \cdot \widehat{FN\_risk}_i +
\text{MAGD\_AR}_i
\]

This formulation intentionally mixes predictive and assurance terms: the AI can have a low raw fraud score but still be expensive to trust if its assurance signals are poor.

### 3.6 Routing Policy

MAGD-Fraud routes each case using both assurance risk and expected cost.

**AI route.**
Use AI when:

1. the assurance risk is low,
2. the adaptive threshold is satisfied, and
3. AI expected cost is lower than the best available expert alternative.

**Expert route.**
Defer to a selected expert when:

1. assurance risk is medium, or
2. expert expected cost is lower than AI expected cost.

**Escalation route.**
Escalate when:

1. the assurance risk is high, or
2. wrong-confident risk is high.

For escalation, MAGD-Fraud uses **majority vote among the top-\(k\) most reliable available experts**. Oracle is not used in deployable routing.

### 3.7 Heuristic, Learned, and Constrained MAGD Variants

We implement three policy variants.

**MAGD-Heuristic.**
This variant uses the configured signal weights directly. It is the cleanest extension of distance-only uncertainty because it preserves interpretable signal composition.

**MAGD-Learned.**
This variant learns signal weights on the validation split by minimizing:

\[
\mathcal{L}_{learned} =
\text{FraudLoss} +
\lambda_{over} \cdot \text{OverReliance}
\]

This formulation explicitly discourages assigning AI to cases where the AI is likely to be wrong.

**MAGD-Constrained.**
This variant augments the learned objective with operational and assurance constraints:

\[
\mathcal{L}_{constrained} =
\text{FraudLoss}
+ \lambda_{over} \cdot \text{OverReliance}
+ \lambda_{cap} \cdot \text{CapacityViolation}
+ \lambda_{fair} \cdot \text{FairnessPenalty}
+ \lambda_{audit} \cdot \text{AuditGap}
\]

subject to:

\[
\text{CorrectRejection} \ge \tau_{CR}
\]

\[
\text{AuditCoverage} \ge \tau_{audit}
\]

Weights are learned on validation data only. Test labels are reserved for final reporting.

### 3.8 Leakage Boundary

The most important methodological guardrail is that **no deployable routing function uses test ground truth**. Specifically:

- local reliability uses historical training correctness only
- wrong-confident deployable risk excludes test \(y_i\)
- MAGD risk is computed from deployable signals only
- final routing uses deployable signals and historical expert statistics only

Ground truth is used only for:

- training the predictive model
- validation-time policy learning
- offline evaluation on the test split

This separation is enforced both by design and by explicit scientific checks in the repository.

### 3.9 Why MAGD-Fraud Is More Than Distance Uncertainty

Distance-to-centroid uncertainty is useful but incomplete in fraud review. It captures whether a case lies far from the global geometry of the predicted class, but it does not capture:

- whether model confidence is historically calibrated
- whether similar historical cases were often misclassified
- whether different confidence mechanisms disagree
- whether certain experts are less biased or more reliable
- whether capacity or fairness considerations should change routing decisions

MAGD-Fraud adds these missing dimensions while retaining deployability. In that sense, it should be understood as a **multi-evidence assurance-guided routing architecture**, not simply as a stronger uncertainty threshold.

### 3.10 Outputs

The method writes the following core artifacts:

- `data/outputs/assurance/local_reliability.csv`
- `data/outputs/assurance/wrong_confident_risk.csv`
- `data/outputs/assurance/magd_risk.csv`
- `data/outputs/assurance/adaptive_thresholds.csv`
- `data/outputs/magd_policy/learned_weights.csv`
- `data/outputs/magd_policy/optimization_diagnostics.json`
- `data/outputs/assurance_deferral/expert_reliability.csv`
- `data/outputs/assurance_deferral/magd_fraud_decisions.csv`
- `data/outputs/assurance_deferral/magd_fraud_metrics.csv`

These outputs support both empirical evaluation and audit-oriented analysis.
