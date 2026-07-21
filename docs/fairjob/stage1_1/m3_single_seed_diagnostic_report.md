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
- Code revisions: `81e65fd` for input/linear probes and `0b7b5f4` for nonlinear
  probes with explicit convergence metadata.

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

## M3 gate decision

The preregistered interaction-amplification gate does not pass:

1. A positive input-normalized amplification does not appear in either model.
2. Only one CTR seed has been screened.
3. Every measured `Amp` relative to raw input is negative.
4. The raw control identifies `user_id` as the dominant leakage source, but no
   no-user-ID CTR model has yet been trained.
5. Targeted path masking and matched random masking have not been run and are
   not justified before the identity intervention.

The current result should therefore be classified as **identity-linked input
proxy with model-dependent compression**, plus a provisional DCNv2 nonlinear
within-network increase. It must not be described as a general FairJob
interaction-amplification mechanism.

## Required next experiment and stop boundary

The next causal diagnostic is full retraining of DeepFM and DCNv2 under three
feature controls: no `user_id`, no `product_id`, and no user/product IDs. Smoke
runs should verify the generated configs first. Full runs are long training and
are intentionally not started as part of this report.

Only if a repeatable layer increase remains in the no-user-ID models should M3
continue to targeted path/layer masking, matched-sparsity random masking, at
least three screening seeds, and confidence intervals. If it disappears, M3 is
a valid failed gate and Stage2 should pivot toward identity-aware protocol or
generalization controls rather than selective interaction suppression.
