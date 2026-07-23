# FairJob Stage2 Result and GPT Pro Discussion Brief

Date: 2026-07-23

## 0. Executive Summary

This document is self-contained and intended for a new GPT Pro conversation
with no memory of the project.

The FairJob-to-FuxiCTR engineering pipeline and the Stage2 experimental
protocol were implemented successfully. All 27 preregistered full-data
training jobs completed, checkpoints and predictions are aligned, and the
bounded independent representation audit is reproducible.

The primary Stage2 scientific hypothesis did not pass validation:

> Multi-layer adversarial pressure plus structured path gates would contain
> the protected behavioral proxy across the DCNv2 representation graph,
> without moving it to another layer/path or collapsing click prediction.

The method reduced proxy decodability on one DNN path, but increased it across
most cross layers, joint paths, and the raw-user-disjoint evaluation. This is
representation migration/amplification, not graph-wide containment.

The correct conclusion is:

> The code and experimental chain succeeded; the proposed containment
> hypothesis failed its preregistered scientific gate.

This is not a training failure and not fairness-by-collapse. Click prediction
quality remained healthy. The failure concerns the representation mechanism.

## 1. Project Context

Repository:

```text
local:  D:\code\JobFairness
server: /root/autodl-tmp/code/JobFairness
branch: codex/stage1-2
```

External server storage:

```text
dataset:  /root/autodl-tmp/datasets/FairJob
artifacts: /root/autodl-tmp/workdirs/JobFairness/stage2
```

The project integrates FairJob into FuxiCTR while preserving these boundaries:

- FairJob reference code is comparison material, not a runtime dependency.
- FuxiCTR core and `model_zoo` source are not modified.
- `protected_attribute` is excluded from proxy-excluded model inputs.
- Raw `protected_attribute` is retained only for training-time adversaries and
  evaluation.
- Raw `user_id` is excluded from the Stage1.1/Stage2 pre-ranking model input.
- Raw metadata, rather than tokenizer IDs, is used for fairness and
  user-disjoint evaluation.
- FairJob uses the official sequential split: first 80% train candidate pool,
  final 20% test.
- The available protected attribute is treated as a behavioral proxy, not a
  verified demographic attribute.
- Results are descriptive, not causal.
- Position-corrected evaluation remains
  `unavailable_missing_external_propensity`.

## 2. Why Stage2 Was Proposed

Stage1.2 found that earlier local interventions did not produce stable clean
leakage reduction:

- global suppression: redistributed;
- selective suppression: redistributed;
- adversarial intervention: redistributed and collapsed in 3/3 seeds;
- DP regularization: leakage not reduced;
- matched-random suppression: inconclusive due probe-family disagreement.

DP was already near the current measurement floor. Group-blind calibration
explained the apparent NLLH advantage of suppression methods. No stable Pareto
winner emerged.

Stage2 therefore moved the research question from final-layer suppression to
graph-level containment:

> Can protected-proxy decodability be reduced across DCNv2 layers and paths
> without migration and without prediction collapse?

## 3. Frozen Stage2 Method

Backbone:

```text
DCNv2, parallel cross/DNN structure
proxy-excluded, pre-ranking, no raw user_id input
seeds: 2019, 2020, 2021
```

The primary `multi_layer_path_gate_full` method uses:

- train-only adversaries on:
  - `cross_layer_0`
  - `cross_layer_1`
  - `cross_layer_2`
  - `parallel_dnn_path_output`
  - `fusion_pre_logit`
- a joint adversary on `cross_dnn_outputs`;
- structured path gates with a frozen suppression budget;
- smooth-max graph risk;
- frozen same-seed baseline teacher;
- normalized centered-logit preservation;
- pairwise ranking preservation;
- anti-collapse constraints.

The proxy is not needed at inference.

The full P2 matrix contains:

1. baseline;
2. primary multi-layer/path-gate method;
3. multi-layer without gates;
4. full method without joint adversary;
5. structured-random gate;
6. matched-capacity regularization;
7. layer-risk-sum control;
8. final-only path-gate control;
9. final-only no-gate control.

## 4. Engineering Status

### P1 bounded smoke

Status: engineering pass, not scientific evidence.

- GPU forward graph passed.
- Baseline checkpoint initialization passed.
- Multi-node and joint adversary losses were finite.
- Learned and random gates used the same suppression budget.
- Prediction and representation-export probabilities matched.
- The initial raw centered-logit MSE instability was found and corrected using
  teacher-standardized SmoothL1 before P2 was frozen.

### P2 full-data training

All 27 jobs completed:

```text
3 seeds x 9 methods = 27 runs
training commit: 7574ed45fcacc41dba046e10f962de8472ca500f
test rows per run: 214446
```

Artifact checks:

- 27 complete run manifests;
- 27 checkpoints;
- 27 aligned prediction files;
- 27 metric and hparameter files;
- every dependent method uses the exact same-seed baseline checkpoint;
- no NaN, OOM, traceback, or runtime-error markers.

### P3 representation audit

Because the server had only about 22 GB free after P2, representations were
reconstructed one checkpoint at a time and retained in memory. Only compact
JSON probe results were persisted.

The bounded pair used:

- seed-2019 baseline;
- seed-2019 primary method;
- 18 node/joint targets;
- row-disjoint sample: 50,000 rows per train/valid/test split;
- raw-user-disjoint sample:
  - train: 29,939 rows;
  - valid: 9,881 rows;
  - test: 10,217 rows;
- linear logistic probe;
- matched-capacity nonlinear MLP probe;
- independent bounded ExtraTrees probe;
- validation-only capacity and orientation selection;
- held-out test read after selection.

Checkpoint reconstruction reproduced the P2 predictions with maximum absolute
errors of approximately `1e-16`.

## 5. P2 Outcome Results

Three-seed means are shown below. Deltas are paired against the same-seed
baseline before averaging.

| Method | AUC | dAUC | NLLH | dNLLH | Brier | dBrier |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.764927 | +0.000000 | 0.042093 | +0.000000 | 0.006948 | +0.000000 |
| multi_layer_path_gate_full | 0.768455 | +0.003528 | 0.042001 | -0.000091 | 0.006925 | -0.000023 |
| multi_layer_no_gate | 0.766078 | +0.001151 | 0.041610 | -0.000482 | 0.006948 | +0.000001 |
| full_without_joint_adversary | 0.766639 | +0.001713 | 0.040792 | -0.001300 | 0.006917 | -0.000031 |
| structured_random_gate | 0.764712 | -0.000214 | 0.043172 | +0.001079 | 0.007068 | +0.000120 |
| matched_capacity_regularization | 0.772117 | +0.007190 | 0.040662 | -0.001431 | 0.006932 | -0.000016 |
| layer_risk_sum | 0.600764 | -0.164163 | 0.414900 | +0.372807 | 0.017633 | +0.010685 |
| final_only_path_gate | 0.559141 | -0.205785 | 0.410927 | +0.368834 | 0.018109 | +0.011162 |
| final_only_no_gate | 0.549212 | -0.215714 | 0.753097 | +0.711004 | 0.026395 | +0.019448 |

Interpretation:

- the primary method did not collapse;
- its AUC improved in all three seeds;
- matched-capacity regularization has stronger outcome metrics than the
  primary method;
- three intentionally restrictive controls collapsed;
- outcome quality alone cannot establish containment.

## 6. Frozen Containment Gate

For each node/path and protocol:

1. select each probe family's capacity and orientation using validation only;
2. evaluate the selected probe once on test;
3. take the maximum held-out AUC across the frozen probe families;
4. subtract the same-seed baseline maximum AUC.

Primary requirements:

- every target node must have a negative delta in all three seeds;
- mean target delta must be at most `-0.005`;
- joint paths must not have a material increase;
- no material leakage migration to non-target/sentinel nodes;
- outcome quality and calibration must remain within frozen tolerances;
- controls must not explain the result.

The all-seed negative-direction rule is irreversible: one nonnegative target
delta in one seed makes full and partial pass impossible.

## 7. P3 Seed-2019 Results

### Row-disjoint target nodes

| Target | Baseline max AUC | Primary max AUC | Delta |
|---|---:|---:|---:|
| cross_layer_0 | 0.672472 | 0.701167 | +0.028695 |
| cross_layer_1 | 0.685083 | 0.694357 | +0.009274 |
| cross_layer_2 | 0.685876 | 0.697374 | +0.011499 |
| parallel_dnn_path_output | 0.658539 | 0.632413 | -0.026126 |
| fusion_pre_logit | 0.698080 | 0.702644 | +0.004563 |

Four of five target nodes fail the required negative direction. Three cross
nodes show material amplification above `+0.005`.

### Raw-user-disjoint target nodes

| Target | Baseline max AUC | Primary max AUC | Delta |
|---|---:|---:|---:|
| cross_layer_0 | 0.563942 | 0.591582 | +0.027640 |
| cross_layer_1 | 0.569084 | 0.598257 | +0.029173 |
| cross_layer_2 | 0.573928 | 0.597033 | +0.023105 |
| parallel_dnn_path_output | 0.556614 | 0.557916 | +0.001303 |
| fusion_pre_logit | 0.563015 | 0.592621 | +0.029607 |

All five targets fail the negative-direction requirement. Four show material
amplification.

### Joint paths

| Joint path | Row-disjoint delta | User-disjoint delta |
|---|---:|---:|
| cross_dnn_outputs | +0.004563 | +0.029607 |
| cross_block_increments | +0.004272 | +0.024630 |
| branch_outputs_with_fusion | -0.001546 | +0.027707 |

The raw-user-disjoint joint paths all show material amplification.
`embedding_flat` also increases by:

```text
row-disjoint:  +0.018416
user-disjoint: +0.025836
```

This is evidence of migration/amplification outside the one successfully
contained DNN path.

Some independent tree probes decrease on selected nodes. These values cannot
be selected after observing test results because the frozen protocol uses the
maximum across all registered families. The nonlinear/linear probes expose
the remaining decodability.

## 8. Why This Is Not Prediction Collapse

Seed-2019 outcome metrics:

| Metric | Baseline | Primary | Delta |
|---|---:|---:|---:|
| AUC | 0.767565 | 0.769272 | +0.001707 |
| NLLH | 0.042120 | 0.041366 | -0.000754 |
| Brier | 0.007073 | 0.006948 | -0.000125 |
| DP | 0.000043 | 0.000763 | +0.000720 |
| U | 0.009993 | 0.010095 | +0.000102 |
| U_TILDE | 0.012309 | 0.012102 | -0.000207 |

AUC, NLLH, and Brier remain healthy. Therefore the representation finding
cannot be dismissed as a constant-score or destroyed-ranking artifact.

## 9. Formal Decision

Machine-readable decision:

```text
decision: early_fail
remaining_checkpoint_probes: not_approved
```

Machine result:

```text
/root/autodl-tmp/workdirs/JobFairness/stage2/reports/p3_seed2019_early_stop.json
```

Final decision implementation/report commit:

```text
0cc94639cdfba1546f257af3e0f14abcf181cfcf
```

The remaining 25 P3 checkpoint probes were not run. This is a preregistered
early stop, not missing evidence needed for the primary method decision:
seeds 2020 and 2021 cannot change the failed all-seed direction condition.

Precise project status:

- engineering implementation: complete and validated;
- P2 full training matrix: complete;
- primary scientific method decision: complete, negative;
- exhaustive representation characterization of every control: incomplete by
  design because it cannot change the primary decision;
- five-seed expansion: not approved;
- DeepFM cross-backbone expansion: not approved.

## 10. What Must Not Be Claimed

Do not claim:

- successful graph-wide protected-information removal;
- successful fairness mitigation;
- causal effects on demographic fairness;
- strict FairJob paper reproduction;
- position-corrected utility;
- that one favorable probe family overrides the registered maximum-family
  result;
- that additional weight tuning is a valid continuation of the same pilot.

The defensible finding is narrower:

> Multi-node adversarial pressure and path gating can preserve CTR quality and
> reduce proxy decodability on one path while redistributing or amplifying
> decodability across cross and joint representations, including unseen raw
> users.

## 11. Questions For GPT Pro

Please review the evidence and advise on the following decisions:

1. Is the leakage-migration/amplification result sufficiently novel and
   rigorous to retain as a negative mechanism study, or should it remain an
   internal failure analysis?
2. Should the next research stage focus on measuring representation
   redistribution rather than proposing another fairness intervention?
3. Is a new hypothesis based on conditional/path-specific information
   decomposition justified, or would that be an unprincipled repair after a
   failed pilot?
4. Is there scientific value in running a limited subset of the remaining
   controls' P3 probes purely for mechanism attribution, despite no ability to
   change the primary gate?
5. Should the stronger matched-capacity outcome result be investigated as
   evidence that generic regularization, rather than containment, explains the
   CTR improvement?
6. Does the independently maximized frozen probe-family AUC remain the correct
   conservative endpoint, or should a future protocol require a common probe
   family/capacity across paired models?
7. What minimum additional evidence would be required before proposing a new
   intervention direction suitable for a target journal?

Any continuation should be defined as a new preregistered hypothesis. It
should not relax the current thresholds, retune the failed method using test
results, or relabel local path reduction as graph-wide containment.

## 12. Source Documents

Relevant repository documents:

```text
docs/fairjob/stage1_2/stage1_2_decision_report.md
docs/fairjob/stage2/stage2_review_opinion.md
docs/fairjob/stage2/representation_graph_audit.md
docs/fairjob/stage2/probe_protocol_audit.md
docs/fairjob/stage2/p1_bounded_smoke_report.md
docs/fairjob/stage2/p2_training_audit.md
docs/fairjob/stage2/p3_seed2019_early_stop_audit.md
docs/fairjob/stage2/training_runbook.md
```
