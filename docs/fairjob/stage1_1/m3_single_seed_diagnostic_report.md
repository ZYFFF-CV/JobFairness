# Stage1.1 M3 Single-Seed Proxy Diagnostic Report

## Status and scope

This report records the bounded M3 screening run on the full FairJob split. It
is an integration-mode, single-seed diagnostic (`seed=2019`), not formal
multi-seed evidence. It does not support a paper-level mechanism claim.

- CTR models: the completed DeepFM and DCNv2 pre-ranking runs.
- Proxy target: raw `protected_attribute` from evaluator metadata.
- Probe rows: 50,000 train, 50,000 validation, and 50,000 test rows.
- Export source: 200,000 sampled train, 50,000 full validation, and 200,000
  sampled test representations.
- Alignment: all four model exports passed shard hash, schema, finiteness, row
  order, and cross-model `row_id` equality checks.
- Probe split use: train fits the probe, validation selects capacity or L2 C,
  and test is used once for reporting.
- Code revisions: `81e65fd`/`0b7b5f4` for initial probes, `528950d` for
  identity-intervention training, `88f262d` for no-user layer probes, and
  `58e75b9` for matched-capacity nonlinear input/layer probes.

The raw-input probe target-encodes categorical fields using only training data.
Training rows use five-fold cross-fitting; validation and test use transform
from the fitted training encoder. This avoids protected-target leakage across
splits.

## Raw-input identity controls

| Input feature set | Test AUC | Balanced accuracy |
|---|---:|---:|
| All pre-ranking proxy-excluded inputs | 0.992032 | 0.942711 |
| `user_id` and `product_id` only | 0.991283 | 0.940155 |
| Non-ID fields only | 0.618277 | 0.582703 |
| No `user_id` | 0.626609 | 0.587000 |
| No `product_id` | 0.992011 | 0.942550 |
| All proxy-included inputs (positive control) | 1.000000 | 1.000000 |

The result is unambiguous at screening level: raw-input proxy predictability is
almost entirely associated with `user_id`. Removing `product_id` has negligible
effect, while removing `user_id` reduces test AUC by 0.365423. This is evidence
of an identity-linked input proxy, not evidence that the CTR network amplifies
that proxy.

## Linear representation probes

`Amp = Leak(H, A) - Leak(X, A)` and
`NAmp = (Leak(H, A) - 0.5) / (Leak(X, A) - 0.5)`, where the all-input
proxy-excluded test AUC (0.992032) is `Leak(X, A)`.

| Model | Representation | Test AUC | Balanced accuracy | Amp | NAmp |
|---|---|---:|---:|---:|---:|
| DeepFM | embedding_flat | 0.609948 | 0.574596 | -0.382084 | 0.223456 |
| DeepFM | fm_logit | 0.521972 | 0.519668 | -0.470059 | 0.044656 |
| DeepFM | dnn_linear_0 | 0.608866 | 0.573741 | -0.383166 | 0.221258 |
| DeepFM | dnn_linear_1 | 0.590195 | 0.561012 | -0.401836 | 0.183312 |
| DeepFM | dnn_linear_2 | 0.573869 | 0.548333 | -0.418162 | 0.150131 |
| DeepFM | dnn_linear_3 | 0.520719 | 0.510834 | -0.471313 | 0.042109 |
| DeepFM | logit | 0.496532 | 0.500169 | -0.495500 | -0.007049 |
| DCNv2 | embedding_flat | 0.628043 | 0.589609 | -0.363988 | 0.260234 |
| DCNv2 | cross_layer_0 | 0.660329 | 0.610052 | -0.331703 | 0.325850 |
| DCNv2 | cross_layer_1 | 0.660159 | 0.608637 | -0.331872 | 0.325506 |
| DCNv2 | cross_layer_2 | 0.652284 | 0.604873 | -0.339748 | 0.309500 |
| DCNv2 | parallel_dnn_linear_0 | 0.623109 | 0.586205 | -0.368922 | 0.250206 |
| DCNv2 | parallel_dnn_linear_1 | 0.606110 | 0.576349 | -0.385922 | 0.215656 |
| DCNv2 | final | 0.655575 | 0.605441 | -0.336457 | 0.316188 |
| DCNv2 | logit | 0.495748 | 0.497580 | -0.496283 | -0.008641 |

All representations have negative amplification relative to the raw-input
baseline. DCNv2 cross layers are more linearly decodable than its input
embedding, but never approach the identity-dominated raw-input leakage.

Five sklearn linear fits emitted an iteration-limit warning at 1,000 steps.
Those screening files predate explicit convergence metadata. Their AUC values
must not be treated as formal estimates until a targeted convergence stability
check is completed.

## Bounded nonlinear probes

| Model | Representation | Test AUC | Balanced accuracy | Amp | Iterations | Converged |
|---|---|---:|---:|---:|---:|---|
| DeepFM | embedding_flat | 0.719068 | 0.653002 | -0.272963 | 100 | No |
| DeepFM | dnn_linear_0 | 0.722969 | 0.649434 | -0.269063 | 100 | No |
| DeepFM | dnn_linear_3 | 0.523366 | 0.516977 | -0.468666 | 14 | Yes |
| DCNv2 | embedding_flat | 0.725981 | 0.659740 | -0.266051 | 100 | No |
| DCNv2 | cross_layer_0 | 0.736374 | 0.664528 | -0.255658 | 100 | No |
| DCNv2 | final | 0.774933 | 0.696648 | -0.217099 | 100 | No |

The DCNv2 nonlinear probe rises by 0.048951 from embedding to final, so a
bounded nonlinear decoder finds additional proxy information along that path.
This is a within-model diagnostic, not input-normalized amplification: the final
representation remains 0.217099 AUC below the raw-input baseline. Five selected
MLPs reached the fixed 100-iteration cap; their values are capacity-bounded
lower estimates and are reported with that limitation.

## Positive controls

Both proxy-included `embedding_flat` representations achieve AUC 1.0 and
balanced accuracy 1.0 with a linear probe. DeepFM and DCNv2 logits remain near
chance (0.494465 and 0.490571). The positive controls verify that row alignment
and probe direction can recover an explicitly embedded protected proxy.

## Identity-intervention full runs

All six identity-control models completed at commit `528950d`. Predictions
cover all 214,446 test rows, metrics use the unchanged raw evaluator metadata,
and train/valid/test representations were exported for each run.

| Model | Feature protocol | NLLH | AUC | DP | U | U_TILDE |
|---|---|---:|---:|---:|---:|---:|
| DeepFM | original proxy-excluded | 0.066489 | 0.715530 | 0.000610 | 0.010253 | 0.011359 |
| DeepFM | no user ID | 0.079691 | 0.653801 | 0.001149 | 0.010029 | 0.010704 |
| DeepFM | no product ID | 0.060602 | 0.719300 | 0.000791 | 0.009964 | 0.012130 |
| DeepFM | no user/product IDs | 0.052612 | 0.699319 | 0.000887 | 0.010248 | 0.012570 |
| DCNv2 | original proxy-excluded | 0.039060 | 0.781174 | 0.000735 | 0.010274 | 0.011962 |
| DCNv2 | no user ID | 0.042120 | 0.767565 | 0.000043 | 0.009993 | 0.012309 |
| DCNv2 | no product ID | 0.039425 | 0.783307 | 0.000213 | 0.010222 | 0.011745 |
| DCNv2 | no user/product IDs | 0.039864 | 0.761959 | 0.000656 | 0.010520 | 0.012675 |

Removing user ID has backbone-dependent outcome effects. DCNv2 loses 0.013609
AUC while DP decreases by 0.000692; DeepFM loses 0.061729 AUC while DP
increases by 0.000539. Consequently, user ID is a major raw proxy source, but
feature removal is not a model-independent fairness intervention.

## No-user-ID representation probes

All probe samples were regenerated after adding the no-user-ID exports. Six
representation roots passed full hash, schema, finiteness, ordering, and exact
cross-model `row_id` checks. The same 50,000 rows per split are used below.

The linear raw-input baseline is AUC 0.626609. DeepFM has no positive linear
amplification. DCNv2 has small positive values at `cross_layer_0` (0.636998,
Amp +0.010389), `cross_layer_1` (0.639681, +0.013072), `cross_layer_2`
(0.633870, +0.007261), and `final` (0.635215, +0.008606). Several selected
LogisticRegression fits reached the fixed 1,000-iteration limit, so these are
screening estimates.

The first nonlinear comparison was invalid for amplification because it used
an MLP on hidden representations but the linear input baseline. It is retained
only as a diagnostic artifact and is not used for the gate. The corrected
comparison uses the same two-candidate MLP grid on both target-encoded raw input
and hidden representations:

| Model/input | Representation | Test AUC | Amp | NAmp | Iterations | Converged |
|---|---|---:|---:|---:|---:|---|
| raw no-user input | matched MLP | 0.687182 | 0.000000 | 1.000000 | 100 | No |
| DeepFM no-user | embedding_flat | 0.669760 | -0.017422 | 0.906927 | 100 | No |
| DeepFM no-user | dnn_linear_0 | 0.704068 | +0.016886 | 1.090213 | 100 | No |
| DeepFM no-user | dnn_linear_3 | 0.529599 | -0.157582 | 0.158131 | 18 | Yes |
| DCNv2 no-user | embedding_flat | 0.661478 | -0.025703 | 0.862682 | 100 | No |
| DCNv2 no-user | cross_layer_0 | 0.701990 | +0.014808 | 1.079110 | 100 | No |
| DCNv2 no-user | final | 0.722255 | +0.035074 | 1.187377 | 100 | No |

This matched-capacity result preserves a small positive intermediate-layer
signal in both backbones after user ID removal. It also shows compression at
the embeddings and at DeepFM's final DNN/logit path. The selected positive MLPs
reached the intentional 100-iteration cap; the result is bounded-capacity
screening evidence, not a converged optimum estimate.

## Updated M3 gate decision

The candidate now satisfies three prerequisites:

1. positive matched-capacity amplification appears in both DeepFM and DCNv2;
2. the selected layer AUC exceeds the matched raw-input baseline;
3. the increase remains after full no-user-ID CTR retraining.

The preregistered M3 gate still does **not** pass because:

1. only seed 2019 has been trained and probed, rather than at least three
   independent screening seeds;
2. targeted path/layer masking has not been compared with matched-sparsity
   random masking;
3. probe-capacity stability and confidence intervals remain outstanding.

The defensible current conclusion is **identity-linked raw proxy plus a small,
cross-backbone candidate interaction signal that survives user-ID removal**.
It must not yet be described as a stable amplification mechanism or used to
justify a selective-suppression method.

## Stop boundary

The next valid step requires long training: at least two additional no-user-ID
CTR seeds for both backbones, followed by targeted and matched-random path
interventions if the layer signal is directionally stable. This report stops
before those runs. Stage2 method selection remains unfrozen until that evidence
is available.

The four seed-2020/2021 jobs have isolated run directories and are prepared in
the `identity_multiseed` matrix group. After replacing `COMMIT_SHA` with the
final code revision, run them in the foreground with:

```text
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python fuxictr_ext/fairjob/run_protocol_matrix.py --group identity_multiseed --mode foreground --resume --expected_commit COMMIT_SHA
```
