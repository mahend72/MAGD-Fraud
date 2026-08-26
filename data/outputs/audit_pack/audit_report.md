# Audit Report: haaf_fifar

## Project Aim
Generate an assurance-style record showing how Level 3 model assurance signals informed Level 4 human oversight decisions for financial fraud alert review.

## Dataset Used
- Train data: `data/ICAIF_KAGGLE/testbed/train/small__regular/train.csv`
- Test data: `data/ICAIF_KAGGLE/testbed/test/test.csv`
- Expert predictions: `data/ICAIF_KAGGLE/testbed/test/test_expert_pred.csv`
- Capacity table: `not configured`

## Baselines
- AI-only
- best expert only
- random expert
- numerical threshold
- distance threshold
- oracle upper bound

## Assurance-Guided Method
The main method combines calibration risk, distance uncertainty, neighbor error evidence, wrong-confident risk, and optional business risk into an assurance risk score. The selected route is AI for low risk when AI expected cost is lower, Human Expert for medium risk, and Escalate for high risk.

## Key Metrics
| method | precision | recall | f1 | cost_sensitive_loss | ai_coverage | human_deferral_rate | escalation_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AI-only | 0.5288461538461539 | 0.0385154061624649 | 0.0718015665796344 | 1375.793 | 1.0 | 0.0 | 0.0 |
| best expert only | 0.3328495403967102 | 0.4817927170868347 | 0.3937052932761087 | 818.6030000000001 | 0.0 | 1.0 | 0.0 |
| random expert | 0.1774800188058298 | 0.5287114845938375 | 0.2657514959521295 | 872.443 | 0.0 | 1.0 | 0.0 |
| numerical threshold | 0.5288461538461539 | 0.0385154061624649 | 0.0718015665796344 | 1375.793 | 1.0 | 0.0 | 0.0 |
| distance threshold | 0.5288461538461539 | 0.0385154061624649 | 0.0718015665796344 | 1375.793 | 1.0 | 0.0 | 0.0 |
| oracle upper bound | 1.0 | 1.0 | 1.0 | 0.0 | 0.985316440011152 | 0.0146835599888479 | 0.0 |
| learning-to-defer baseline | 0.6982248520710059 | 0.2478991596638655 | 0.365891472868217 | 1082.721 | 0.961453073531386 | 0.0385469264686141 | 0.0 |
| MAGD-Heuristic | 0.7607843137254902 | 0.1358543417366946 | 0.2305407011289364 | 1237.477 | 0.8043431120473343 | 0.0 | 0.1956568879526656 |
| MAGD-Learned | 0.639751552795031 | 0.0721288515406162 | 0.1296412838263058 | 1328.306 | 0.9096475739082844 | 0.0 | 0.0903524260917154 |
| MAGD-Fraud-ValidationTuned | 0.4966442953020134 | 0.0518207282913165 | 0.0938490805326569 | 1358.275 | 0.9805148539388496 | 0.0194851460611505 | 0.0 |
| MAGD-Constrained | 0.639751552795031 | 0.0721288515406162 | 0.1296412838263058 | 1328.306 | 0.9096475739082844 | 0.0 | 0.0903524260917154 |

## Reliance Metrics
| method | correct_reliance | correct_rejection | overreliance | underreliance | wrong_confident_avoidance_rate |
| --- | --- | --- | --- | --- | --- |
| AI-only | 0.985316440011152 | 0.0 | 0.0146835599888479 | 0.0 | 0.0 |
| best expert only | 0.0 | 0.0146835599888479 | 0.0 | 0.985316440011152 | 1.0 |
| random expert | 0.0 | 0.0146835599888479 | 0.0 | 0.985316440011152 | 1.0 |
| numerical threshold | 0.985316440011152 | 0.0 | 0.0146835599888479 | 0.0 | 0.0 |
| distance threshold | 0.985316440011152 | 0.0 | 0.0146835599888479 | 0.0 | 0.0 |
| oracle upper bound | 0.985316440011152 | 0.0146835599888479 | 0.0 | 0.0 | 1.0 |
| learning-to-defer baseline | 0.9531199983478412 | 0.0063504848053034 | 0.0083330751835445 | 0.0321964416633107 | 0.159541188738269 |
| MAGD-Heuristic | 0.792664415600508 | 0.0030048635420216 | 0.0116786964468263 | 0.192652024410644 | 0.2179353493222106 |
| MAGD-Learned | 0.8960069390663239 | 0.0010429251468872 | 0.0136406348419607 | 0.0893095009448282 | 0.0802919708029197 |
| MAGD-Fraud-ValidationTuned | 0.9662546596036884 | 0.0004233656536868 | 0.014260194335161 | 0.0190617804074636 | 0.0156412930135557 |
| MAGD-Constrained | 0.8960069390663239 | 0.0010429251468872 | 0.0136406348419607 | 0.0893095009448282 | 0.0802919708029197 |

## Deferral Metrics
| method | ai_coverage | human_deferral_rate | escalation_rate | deferral_precision | deferral_recall | capacity_violation_rate | oracle_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AI-only | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0146835599888479 |
| best expert only | 0.0 | 1.0 | 0.0 | 0.0146835599888479 | 1.0 | 0.0 | 0.0218807761015251 |
| random expert | 0.0 | 1.0 | 0.0 | 0.0146835599888479 | 1.0 | 0.0 | 0.0430800367605299 |
| numerical threshold | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0146835599888479 |
| distance threshold | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0146835599888479 |
| oracle upper bound | 0.985316440011152 | 0.0146835599888479 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| learning-to-defer baseline | 0.961453073531386 | 0.0385469264686141 | 0.0 | 0.1647468523975355 | 0.4324894514767932 | 0.0 | 0.0126699916359468 |
| MAGD-Heuristic | 0.8043431120473343 | 0.0 | 0.1956568879526656 | 0.0153578214059531 | 0.2046413502109704 | 0.0 | 0.0133721590615738 |
| MAGD-Learned | 0.9096475739082844 | 0.0 | 0.0903524260917154 | 0.0115428571428571 | 0.0710267229254571 | 0.0 | 0.0142808463182677 |
| MAGD-Fraud-ValidationTuned | 0.9805148539388496 | 0.0194851460611505 | 0.0 | 0.021727609962904 | 0.0288326300984528 | 0.0 | 0.0147558419297213 |
| MAGD-Constrained | 0.9096475739082844 | 0.0 | 0.0903524260917154 | 0.0115428571428571 | 0.0710267229254571 | 0.0 | 0.0142808463182677 |

## Audit Coverage
- Audit coverage: `1.0000`
- Complete decision logs rate: `1.0000`
- Missing evidence rate: `0.0000`
- Missing rationale rate: `0.0000`
- Total decisions audited: `96843`

## Limitations
- No capacity table was configured, so capacity-sensitive evaluation is unconstrained.
- Oracle upper bound is reported for reference only and is not deployable.
