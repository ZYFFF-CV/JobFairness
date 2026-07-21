# Stage1.1 Execution Plan

Last updated: 2026-07-20
Baseline commit: `6cdff7f01175318cea88af1b1b4cec53950aa84f`
Scope: scientific protocol calibration, deep-model diagnosis, fairness baselines,
and the Stage2 direction decision.

## 1. Objective

Stage1.1 must determine whether modern CTR interaction models amplify information
about FairJob's behavioral protected proxy beyond what is already predictable
from their inputs, where that amplification occurs, and whether it is associated
with stable calibration, disparity, and ranking-utility changes.

The phase does not assume that proxy inclusion is harmful, does not equate the
proxy with verified demographic gender, and does not select a final mitigation
method before the diagnostic evidence is available.

## 2. Non-Negotiable Constraints

- Preserve the sequential FairJob split: first 80% train candidate rows and final
  20% test rows. The frozen test set remains 214,446 rows.
- Use the same raw `test_meta.csv` and `row_id` alignment for every model.
- Keep legacy experiment IDs where required by code, but report `unaware` as
  `proxy-excluded` and `unfair` as `proxy-included`.
- Treat `protected_attribute` as a behavioral protected proxy, not a verified
  demographic ground-truth attribute.
- Select hyperparameters only from train/validation data. Never select models or
  fairness strength from test AUC, test DP, or test utility.
- Run formal training on the server. Local execution is limited to development,
  compilation, unit tests, synthetic fixtures, and lightweight smoke checks.
- Keep source code identical locally and remotely by running every server job
  from an explicit Git commit.
- Do not write data, checkpoints, representations, or predictions into the source
  checkout.
- Do not modify sibling repositories or shared datasets on the server.

## 3. Canonical Runtime and Artifact Layout

Git is the source of truth for code. Stage1.1 development should use a dedicated
branch created from the baseline commit, with both local and server worktrees
checked out at the same commit before any training job starts.

Server layout:

```text
/root/autodl-tmp/code/JobFairness
/root/autodl-tmp/datasets/FairJob
/root/autodl-tmp/workdirs/JobFairness/stage1_1/
  checkpoints/
  predictions/
  hparams/
  logs/
  reports/
  representations/
  probes/
  bootstrap/
  runtime_config/
```

The server Python environment is the canonical research runtime. Current anchor
runs use Python 3.12, scikit-learn 1.4.2, XGBoost 2.1.4, FuxiCTR 2.3.9, and
PyTorch 2.3.0+cu121. Local results from a different sklearn version are retained
as migration evidence, not mixed into formal Stage1.1 tables.

Every formal run must record:

```text
git_commit
environment_versions
model
backbone
protocol
proxy_regime
fairness_method
seed
feature_ablation
selection_protocol
config_hash
prediction_hash
```

## 4. Execution Order and Gates

| Milestone | Main work | Server training | Exit gate |
|---|---|---:|---|
| M0 | Freeze baseline and runtime | No | Reproducible anchor manifest |
| M1 | Task, terminology, and metric protocols | No | Legacy metrics unchanged; extended tests pass |
| M2 | DeepFM/DCNv2 support and representation export | Smoke then full | Both backbones produce aligned predictions and representations |
| M3 | Proxy leakage and interaction diagnosis | Yes | Stable leakage/amplification evidence or documented negative result |
| M4 | Logging, calibration, and selection diagnostics | Yes | Protocol-dependent effects quantified |
| M5 | Minimal fairness baselines and Pareto comparison | Yes | Fair comparison under matched budgets |
| M6 | Multi-seed statistics and Stage2 decision | Yes | Decision report selects one of five directions |

No later milestone starts merely because code exists. Each milestone must satisfy
its exit gate and preserve prior prediction/evaluator contracts.

## 5. M0 - Baseline Freeze

### Tasks

1. Create a Stage1.1 branch from commit `6cdff7f`.
2. Create the project-specific Stage1.1 server workdir tree.
3. Record the server environment, CUDA/GPU information, data manifest hash, and
   baseline commit in a machine-readable run manifest.
4. Re-evaluate the four current full predictions and preserve the resulting
   integration table as the Stage1.1 anchor.
5. Mark the server results as canonical because XGB TargetEncoder behavior differs
   between local sklearn 1.3.2 and server sklearn 1.4.2.

### Exit Gate

- Local, GitHub, and server resolve to the same commit.
- The source worktree is clean on both machines.
- Four anchor predictions pass 214,446-row alignment.
- No runtime artifact is stored in Git.

## 6. M1 - Protocol and Metric Foundation

### 6.1 Task Protocols

Add `configs/fairjob/task_protocols.yaml` with two explicit protocols:

- `pre_ranking`: primary research protocol. Exclude `rank`, `displayrandom`, and
  one-off `impression_id` from model inputs; retain them as evaluator/grouping
  metadata.
- `post_display`: auxiliary response-modeling protocol. Permit `rank` and
  `displayrandom` as inputs and label all conclusions as post-display estimates.

Audit `impression_id` cardinality, overlap, candidate-set size, and deployment
meaning before allowing it as a feature in any protocol.

### 6.2 Extended Metrics

Implement and test:

- `DP_signed`, `DP_abs`, group means, group counts, and prediction ratio;
- overall and group-wise NLLH, AUC, Brier score, and ECE;
- worst-group performance and between-group gaps;
- explicit evaluation scopes for `all_logged`, `random_display`,
  `position_corrected`, and `context_conditioned`;
- step-level U/U_TILDE parity, including candidate reconstruction, ascending-rank
  semantics, tie handling, product aggregation, and population correction.

Legacy `NLLH`, `AUC`, `DP`, `U`, and `U_TILDE` outputs must remain available.

### Deliverables

```text
configs/fairjob/task_protocols.yaml
configs/fairjob/fairness_protocols.yaml
fuxictr_ext/fairjob/metrics_extended.py
fuxictr_ext/fairjob/calibration.py
fuxictr_ext/fairjob/selection_protocols.py
tests/fairjob/test_metrics_extended.py
tests/fairjob/test_task_protocols.py
tests/fairjob/test_utility_step_parity.py
```

### Exit Gate

- Existing anchor metrics reproduce within numerical tolerance.
- New signed/absolute DP and calibration fixtures pass.
- Every protocol declares model features separately from evaluator metadata.
- Reports use scientific regime names without breaking old experiment IDs.

## 7. M2 - Deep Backbones and Representation Export

### Tasks

1. Audit the existing FuxiCTR DeepFM and DCNv2 implementations and reuse their
   established model/config style.
2. Add FairJob experiment configurations for both protocols and both proxy
   regimes.
3. Add model-agnostic representation hooks with named extraction points:
   input/embeddings, FM interaction, DCNv2 cross layers, DNN hidden layers,
   logit, and final probability.
4. Store representation shards under the Stage1.1 workdir, keyed by `row_id`.
5. Add schema, shape, ordering, and finite-value validation before a shard is
   accepted by probe code.

### Run Order

1. Single-seed smoke: DeepFM then DCNv2.
2. Verify prediction and representation alignment.
3. Single-seed full diagnostic run.
4. Three-seed screening only after the single-seed full run is valid.
5. Expand to five seeds only for configurations entering formal conclusions.

### Deliverables

```text
configs/fairjob/deepfm_model_config.yaml
configs/fairjob/dcnv2_model_config.yaml
fuxictr_ext/fairjob/representation_io.py
tests/fairjob/test_representation_io.py
```

### Exit Gate

- DeepFM and DCNv2 both produce aligned full predictions under `pre_ranking`.
- Named representations are reproducible for the same commit and seed.
- Representation export does not change predictions beyond tolerance.

## 8. M3 - Leakage, Identity, and Path Diagnosis

### Probe Protocol

- Keep CTR training, representation extraction, and probe training separate.
- Fit probe hyperparameters only on a probe validation split.
- Report linear and bounded-capacity nonlinear probes on the same frozen probe
  test rows.
- Report input leakage, layer leakage, balanced accuracy, AUC, sample count,
  confidence interval, amplification, and input-normalized amplification.
- Separate ID features from non-ID features and add conditional leakage by
  senior/product/category/position/random-display context.

### Required Ablations

- no-user-id;
- no-product-id;
- no-user/product-ID;
- only category/numeric features;
- seen-user versus unseen/low-frequency user;
- field leave-one-out;
- pairwise/path masking;
- matched-sparsity random masking;
- layer masking.

### Deliverables

```text
fuxictr_ext/fairjob/proxy_probe.py
fuxictr_ext/fairjob/interaction_ablation.py
fuxictr_ext/fairjob/run_protocol_matrix.py
tests/fairjob/test_proxy_probe_splits.py
tests/fairjob/test_interaction_ablation.py
```

### Exit Gate

Interaction amplification is considered supported only when it:

1. appears in at least DeepFM and DCNv2;
2. has consistent direction across at least three screening seeds;
3. exceeds the input leakage baseline;
4. remains after no-user-id controls or is explicitly reclassified as identity
   memorization;
5. responds more strongly to targeted path masking than matched random masking.

A failed gate is a valid result and triggers a different Stage2 direction.

## 9. M4 - Outcome, Calibration, and Selection Diagnosis

### Tasks

1. Relate layer/path leakage to `DP_signed`, `DP_abs`, group calibration, Brier,
   NLLH, AUC, U, and U_TILDE without using causal language prematurely.
2. Implement DP within-context/composition decomposition using documented
   reference distributions.
3. Compare `all_logged`, `random_display`, `position_corrected`, and
   `context_conditioned` protocols.
4. Add symmetric/asymmetric proxy-label noise and missing-proxy sensitivity.
5. Use user or impression cluster bootstrap for primary confidence intervals.

### Deliverables

```text
fuxictr_ext/fairjob/dp_decomposition.py
fuxictr_ext/fairjob/bootstrap.py
tests/fairjob/test_dp_decomposition.py
tests/fairjob/test_cluster_bootstrap.py
```

### Exit Gate

- Aggregate disparity is separated into within-context and composition terms.
- Direction reversals across logging protocols are explicitly reported.
- Leakage is called harmful only when intervention and robustness evidence link
  it to outcome changes.

## 10. M5 - Minimal Fairness Baselines

Implement on DeepFM first, then validate the main conclusion on DCNv2:

1. group/sample reweighting;
2. direct DP regularization;
3. MMD or HSIC representation alignment;
4. gradient-reversal adversarial removal;
5. projection/decorrelation baseline;
6. group-blind calibration, with group-aware calibration reported separately if
   used;
7. fixed-weight versus adaptive dual/Pareto optimization;
8. random interaction suppression;
9. selective interaction-path suppression only if the M3 gate is satisfied.

Use matched tuning budgets, early stopping, seeds, and validation-only model
selection. Report Pareto curves rather than a single favorable fairness weight.

### Exit Gate

- All methods use the same backbone, data protocol, seed set, and tuning budget.
- Pareto reports include AUC-DP, NLLH-DP, U_TILDE-DP, leakage-AUC,
  leakage-DP, and calibration-gap-utility views.
- Selective suppression is compared with matched random suppression and global
  removal.

## 11. M6 - Formal Statistics and Stage2 Decision

Formal result tables require at least five independent seeds and report mean,
standard deviation, and 95% cluster-bootstrap intervals. Each main claim must
list alternative explanations and the evidence used to reject or retain them.

The final decision report must select exactly one primary direction:

1. stable interaction leakage amplification -> selective path suppression;
2. leakage without outcome association -> privacy/process-fairness audit;
3. stable group calibration gap -> proxy-uncertain fair calibration;
4. selection mechanism dominates -> selection-aware fairness;
5. no stable phenomenon -> systematic audit, proxy measurement error, or data
   expansion rather than forced method development.

Deliverable:

```text
docs/fairjob/stage1_1/stage1_1_decision_report.md
```

## 12. Local-to-Server Workflow

For every implementation batch:

1. Inspect the current local and server commits before editing.
2. Implement locally without modifying FuxiCTR core unless an extension hook is
   demonstrably necessary.
3. Run local compile, unit, synthetic parity, and smoke checks.
4. Review the staged diff for data, checkpoints, credentials, absolute local
   paths, and unrelated changes.
5. Commit and push the exact tested code.
6. Update the server to that commit. If GitHub TLS remains unstable, transfer a
   verified incremental Git bundle and fast-forward the server branch.
7. Assert `local HEAD == GitHub branch == server HEAD` before training.
8. Generate runtime configs under the Stage1.1 workdir, never in the server
   source checkout.
9. Run one smoke job, validate row/representation alignment, then submit full
   jobs sequentially or with an explicit resource allocation plan.
10. Save logs, hparams, commit, environment versions, hashes, and reports under
    the Stage1.1 workdir.
11. Keep both source worktrees clean after each training batch.

## 13. First Implementation Batch

The first code batch is deliberately limited to M0 and M1:

1. create the Stage1.1 branch and server workdir;
2. add task/fairness protocol configs;
3. add scientific regime-name mapping at report time;
4. implement signed/absolute DP, group details, Brier, and ECE;
5. add U/U_TILDE step-level parity fixtures;
6. re-evaluate existing LR/XGB predictions without retraining;
7. produce a protocol-validation report;
8. stop for gate review before modifying DeepFM or DCNv2.

This order resolves semantic and evaluator ambiguity before expensive training
or representation-diagnostic work begins.
