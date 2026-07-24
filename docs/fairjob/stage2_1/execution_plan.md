# Stage2.1 Execution Plan

Status: `planning_pending_protocol_freeze`

Source protocol:

```text
docs/fairjob/stage2_1/Stage2.1_FairJob_Representation_Graph_Proxy_Decodability_Redistribution_Audit.md
SHA256: 25F317F09D5EB5740743885A2F68BC7E21556CA8B33AF36E926E65C6286C428B
```

## 1. Interpretation

Stage2.1 is accepted as a new measurement/mechanism audit proposal, not as a
repair or continuation of the failed Stage2-A containment method.

The following conclusions remain frozen:

- Stage2-A engineering and full training matrix passed.
- Stage2-A graph-wide containment hypothesis failed.
- No Stage2-A weight retuning or threshold relaxation is allowed.
- No new CTR or fairness-intervention training is approved.
- Remaining checkpoint probes may only answer the new redistribution and
  mechanism questions.
- `protected_attribute` remains a behavioral proxy rather than a verified
  demographic attribute.
- `position_corrected` remains
  `unavailable_missing_external_propensity`.

Stage2.1 may establish an internal cross-training-seed result. Reusing the
same FairJob test population does not constitute independent dataset
confirmation.

## 2. Implementation Strategy

The proposed file list is a responsibility map, not a requirement to duplicate
working code. Development should reuse:

```text
fuxictr_ext/fairjob/stage2_checkpoint_probe.py
fuxictr_ext/fairjob/run_stage2_probe_matrix.py
fuxictr_ext/fairjob/joint_path_probe.py
fuxictr_ext/fairjob/representation_graph.py
fuxictr_ext/fairjob/stage2_containment_audit.py
configs/fairjob/stage2_representation_graph.yaml
configs/fairjob/stage2_probe_protocol.yaml
```

New Stage2.1 code should be limited to behavior not already supported:

- frozen Stage2.1 protocol and provenance manifest;
- family-specific and fixed-capacity paired endpoints;
- held-out cross-entropy;
- parent-conditioned and joint predictive gain;
- redistribution count, mass, index, and protocol-consistency metrics;
- tiered checkpoint scheduling and decision reports.

Checkpoint representations remain in memory for one model at a time. Only
compact JSON results are persisted.

## 3. Execution Order

### S21-M0: Closure And Protocol Freeze

1. Generate an immutable Stage2-A closure manifest.
2. Record:
   - training commit
     `7574ed45fcacc41dba046e10f962de8472ca500f`;
   - decision commit
     `0cc94639cdfba1546f257af3e0f14abcf181cfcf`;
   - all 27 P2 run-manifest hashes;
   - row-disjoint and raw-user-disjoint sample hashes;
   - representation-graph and probe-protocol hashes;
   - seed roles: 2019 discovery, 2020/2021 internal confirmation.
3. Freeze the `0.005` material-effect threshold.
4. Freeze the primary nodes, joint targets, parent-child comparisons, probe
   families, fixed capacities, orientation rule, retry rule, and report
   vocabulary.
5. Create the external server workdir:

```text
/root/autodl-tmp/workdirs/JobFairness/stage2_1/
```

Exit condition: no seed-2020/2021 representation result has been read under
the new Stage2.1 endpoint before the protocol hash is recorded.

### S21-M1: Audit-Code Preparation

Implement and test:

1. `configs/fairjob/stage2_1_audit_protocol.yaml`
2. redistribution metrics:
   - reduced/stable/amplified;
   - reduction and migration count;
   - reduction and migration mass;
   - redistribution index;
   - row/user protocol direction agreement.
3. probe output extensions:
   - max-family AUC;
   - family-specific paired AUC;
   - one frozen fixed-capacity sensitivity per family;
   - held-out cross-entropy.
4. conditional probes for the frozen pairs:
   - `cross_layer_0 | embedding_flat`;
   - `cross_layer_1 | cross_layer_0`;
   - `cross_layer_2 | cross_layer_1`;
   - `parallel_dnn_path_output | embedding_flat`;
   - `fusion_pre_logit | [cross_layer_2, parallel_dnn_path_output]`.
5. joint predictive gain for cross and DNN branch outputs.
6. a foreground, resumable Tier-1 scheduler.
7. unit tests for graph targets, split hashes, paired capacities, conditional
   metrics, and deterministic decisions.

The existing Stage2 P3 JSON files and early-stop decision must not be modified.

### S21-M2: Readiness Validation

Before long audit execution:

1. Run all local FairJob tests.
2. Synchronize one exact commit to GitHub and the server.
3. Verify the server repository and external workdir are isolated.
4. Run a dry-run command expansion for the four Tier-1 checkpoints.
5. Run one bounded reconstruction/probe smoke that does not change frozen
   samples or inspect an unregistered target.
6. Confirm prediction reconstruction remains within `1e-6`.
7. Confirm no large representation arrays are persisted.

Exit condition: code, protocol hash, split hashes, and output schema are frozen.

### S21-M3: Tier-1 Cross-Seed Replication

This is the first long-running stage and should be launched manually in the
server console:

```text
baseline_seed2020
multi_layer_path_gate_full_seed2020
baseline_seed2021
multi_layer_path_gate_full_seed2021
```

For each checkpoint:

- reconstruct all 18 frozen node/joint targets;
- run row-disjoint and raw-user-disjoint protocols;
- run linear, MLP, and ExtraTrees probes;
- report max-family, family-specific, and fixed-capacity results;
- report conditional and joint predictive gain;
- preserve validation-only selection and one-time test evaluation.

Decision:

- `replicated`: the frozen H2.1-A conditions all hold;
- `partially_replicated`: direction recurs but one preregistered material or
  protocol condition fails;
- `not_replicated`: cross/joint amplification or the local-path decrease does
  not reproduce.

If `not_replicated`, skip component expansion and close as an internal negative
result.

### S21-M4: Tier-2 Component Attribution

Run only when M3 is at least `partially_replicated`.

Audit seed 2019 for:

```text
multi_layer_no_gate
full_without_joint_adversary
structured_random_gate
matched_capacity_regularization
```

Use same-seed baseline deltas and primary-control contrasts to assess:

- multi-layer adversarial pressure;
- learned gate versus equal-budget random gating;
- incremental role of the joint adversary;
- generic regularization/capacity explanation.

Do not include the three collapsed methods in the healthy-redistribution
analysis. Extend at most one informative control to seeds 2020/2021, using the
frozen selection rule.

### S21-M5: Measurement And Outcome Attribution

1. Produce max-family, family-specific, and fixed-capacity disagreement maps.
2. Report conditional predictive gain without calling it mutual information.
3. Report joint predictive gain without calling it PID synergy.
4. Add paired intervals:
   - raw-user cluster bootstrap as primary;
   - row bootstrap as supplementary where useful.
5. Compare decodability patterns with existing P2 outcome metrics and training
   dynamics.
6. Test whether matched-capacity regularization is a sufficient alternative
   explanation for CTR improvement.

No new model training is needed for this milestone.

### S21-M6: Final Decision

Select exactly one:

1. `measurement_mechanism_paper_candidate`
2. `new_intervention_hypothesis_requires_new_stage`
3. `broader_measurement_expansion_requires_second_setting`
4. `rigorous_internal_negative_result`

A paper candidate requires cross-seed and cross-protocol replication plus at
least partial component attribution. It does not yet establish generality
beyond FairJob/DCNv2. A second backbone, dataset, or controlled semi-synthetic
setting remains necessary for a strong journal claim.

## 4. Stop And Approval Points

Stop for user/scientific review when:

- the protocol or frozen hashes cannot be reproduced;
- server checkpoint prediction reconstruction differs from P2;
- seeds 2020/2021 do not replicate the discovery pattern;
- probe families give incompatible directions under the fixed-capacity check;
- conditional gain is unstable and no component contrast explains the result;
- server storage would require persistent full representation exports;
- a new dataset, backbone, long training run, or logging-propensity source is
  proposed.

Explicit approval is required before:

- the Tier-1 long checkpoint audit;
- any Tier-2 control audit;
- extending a control to seeds 2020/2021;
- introducing a second setting or new intervention.

## 5. Immediate Next Work

The next implementation batch should stop before long server execution:

1. complete S21-M0 closure and protocol freeze;
2. implement S21-M1 metrics and conditional probes;
3. add focused tests and the foreground Tier-1 scheduler;
4. complete S21-M2 local/server dry-run validation;
5. provide the exact Tier-1 server-console command for approval.

No dataset download, decompression, checkpoint retraining, or new fairness
method is required for this batch.
