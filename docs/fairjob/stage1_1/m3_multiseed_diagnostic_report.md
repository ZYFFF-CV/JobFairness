# Stage1.1 M3 Multiseed Diagnostic Report

## Scope and claim boundary

This report extends the single-seed M3 screen to CTR seeds 2019, 2020, and
2021. It evaluates proxy-excluded, no-user-id DeepFM and DCNv2 models on the
same frozen 50,000 train, validation, and test row IDs. It is a bounded
screening experiment, not a formal paper result and not evidence that a
fairness intervention improves CTR outcomes.

The code revisions used here are:

- `6247314`: fixed-row representation re-export and multiseed probes;
- `35e9d6f`: targeted versus matched-random representation masking.

All ten representation roots passed full shard hash, schema, finiteness, order,
and cross-model row-ID validation. Source exports contain 200,000 train,
50,000 validation, and 200,000 test rows. The frozen probe sample contains
50,000 rows from each split. CTR training seeds vary, while representation
sampling and probe initialization remain fixed at 2019.

## No-user-id CTR results

| Backbone | CTR seed | NLLH | AUC | DP abs | U | U_TILDE |
|---|---:|---:|---:|---:|---:|---:|
| DeepFM | 2019 | 0.079691 | 0.653801 | 0.001149 | 0.010029 | 0.010704 |
| DeepFM | 2020 | 0.070760 | 0.663830 | 0.000569 | 0.010124 | 0.011175 |
| DeepFM | 2021 | 0.067422 | 0.645181 | 0.000914 | 0.010208 | 0.011233 |
| DCNv2 | 2019 | 0.042120 | 0.767565 | 0.000043 | 0.009993 | 0.012309 |
| DCNv2 | 2020 | 0.040982 | 0.760368 | 0.000358 | 0.010274 | 0.012335 |
| DCNv2 | 2021 | 0.043177 | 0.766847 | 0.001130 | 0.010365 | 0.012272 |

Identity removal is not itself a stable fairness intervention. DeepFM and
DCNv2 differ in both utility response and DP response, so no-user-id is retained
as a mechanism control rather than proposed as a method.

## Matched-capacity nonlinear probes

The raw no-user-id input baseline is a train-only target-encoded MLP with test
AUC `0.687182`. Hidden representations use the same two-candidate MLP grid.
Validation selects capacity and test is read once.

| Backbone | CTR seed | Representation | Test probe AUC | Amp | NAmp | Converged |
|---|---:|---|---:|---:|---:|---|
| DeepFM | 2019 | dnn_linear_0 | 0.704068 | +0.016886 | 1.090213 | No |
| DeepFM | 2020 | dnn_linear_0 | 0.705452 | +0.018270 | 1.097607 | No |
| DeepFM | 2021 | dnn_linear_0 | 0.703265 | +0.016084 | 1.085926 | No |
| DCNv2 | 2019 | cross_layer_0 | 0.701990 | +0.014808 | 1.079110 | No |
| DCNv2 | 2020 | cross_layer_0 | 0.700798 | +0.013617 | 1.072747 | No |
| DCNv2 | 2021 | cross_layer_0 | 0.702186 | +0.015005 | 1.080162 | No |
| DCNv2 | 2019 | dcnv2_final | 0.722255 | +0.035074 | 1.187377 | No |
| DCNv2 | 2020 | dcnv2_final | 0.736255 | +0.049073 | 1.262169 | No |
| DCNv2 | 2021 | dcnv2_final | 0.730577 | +0.043396 | 1.231837 | No |

Positive Amp survives user-ID removal and has the same direction in both
backbones and all three CTR seeds. The effect is small at the first amplified
layer and larger at the DCNv2 final representation. Every positive MLP reached
the intentional 100-iteration cap, so convergence stability remains open.

## Targeted versus matched-random masking

For each representation, units are ranked by absolute protected-group
standardized mean difference using probe-train rows only. The top 10% are set
to their train means at test time. Ten random masks remove the same number of
units. Probe architecture and fitted weights are frozen; all nine baseline AUCs
reproduce their source JSON exactly.

| Backbone/layer | CTR seed | Targeted AUC drop | Random mean drop | Random max drop | Targeted extra drop | Empirical p |
|---|---:|---:|---:|---:|---:|---:|
| DeepFM dnn0 | 2019 | 0.073694 | 0.038342 | 0.046929 | +0.035353 | 0.091 |
| DeepFM dnn0 | 2020 | 0.094105 | 0.041024 | 0.057477 | +0.053081 | 0.091 |
| DeepFM dnn0 | 2021 | 0.044022 | 0.040215 | 0.064812 | +0.003807 | 0.273 |
| DCNv2 cross0 | 2019 | 0.044156 | 0.026953 | 0.029098 | +0.017203 | 0.091 |
| DCNv2 cross0 | 2020 | 0.059914 | 0.026490 | 0.032483 | +0.033424 | 0.091 |
| DCNv2 cross0 | 2021 | 0.063397 | 0.029112 | 0.043484 | +0.034286 | 0.091 |
| DCNv2 final | 2019 | 0.119918 | 0.043007 | 0.055903 | +0.076911 | 0.091 |
| DCNv2 final | 2020 | 0.146176 | 0.051981 | 0.072760 | +0.094195 | 0.091 |
| DCNv2 final | 2021 | 0.132704 | 0.049189 | 0.070137 | +0.083515 | 0.091 |

Targeted masking has a larger drop than the random-mask mean in all nine
comparisons. DCNv2 is strong and consistent. DeepFM seed 2021 is weak and does
not exceed the largest random draw. Ten random repeats give a coarse empirical
test whose smallest possible corrected p-value is `1/11`; these values must not
be presented as formal significance tests.

## M3 gate decision

The preregistered **screening gate passes directionally**:

1. positive matched-capacity Amp appears in DeepFM and DCNv2;
2. its direction is consistent across three CTR seeds;
3. selected layer AUC exceeds the matched raw-input baseline;
4. the effect remains after the no-user-id control;
5. targeted masking reduces leakage more than matched-random masking on average.

The stronger formal claim is **not established**. Outstanding limitations are:

- nonlinear probes do not converge within 100 iterations;
- masking is a representation-level leakage diagnostic, not a trained CTR
  intervention;
- DeepFM targeted sensitivity is weak in one seed;
- only three CTR seeds and ten random masks are available;
- no cluster-bootstrap confidence interval or model-level AUC/DP/U_TILDE
  intervention result is available.

The defensible conclusion is: **a small cross-backbone interaction-leakage
signal survives user-ID removal, is directionally stable across three screening
seeds, and is concentrated enough to respond to train-selected unit masking,
especially in DCNv2.** This supports developing selective path suppression as
a candidate, but does not yet justify claiming a fairness improvement or a
validated journal-level method.

## Next boundary

Proceed with outcome and selection diagnostics using existing prediction files.
Before any long fairness training, freeze a model-level intervention matrix that
compares selective suppression with matched-random and global suppression under
identical backbones, seeds, early stopping, and tuning budgets. Formal results
require at least five independent seeds and cluster-bootstrap intervals.
