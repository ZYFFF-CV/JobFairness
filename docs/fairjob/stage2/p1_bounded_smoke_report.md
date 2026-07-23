# Stage2 P1 Bounded Smoke Report

Status: engineering pass; not scientific evidence.

## Runtime

- Commit: `4c80ec6629b09ffecb2f3a1dfb65bdc78705204c`
- GPU: NVIDIA RTX 4090 D
- Data: 36,000 smoke-train, 4,000 smoke-validation, 10,000 smoke-test rows
- Seed: 2019
- Epochs: 1
- Protocol: pre-ranking, raw user ID excluded, protected proxy excluded from
  model input and retained as train-only adversary metadata

Artifacts are under:

```text
/root/autodl-tmp/workdirs/JobFairness/stage2/smoke/seed2019/
```

## Engineering Checks

- Real forward graph audit passed on GPU.
- Every run exported a checkpoint, prediction CSV, metrics, hparams, and run
  manifest.
- Forward and representation-export probabilities matched exactly.
- Frozen baseline checkpoint initialization succeeded.
- Evaluation and prediction did not require the protected proxy.
- Multi-node and joint adversaries produced finite balanced BCE values.
- Learned and random gates spent the same total suppression budget.
- Matched-capacity control kept proxy-head capacity but used zero encoder
  gradient reversal and identity gates.
- Loss and graph-risk telemetry appeared every 10 steps in the foreground.

## Stability Correction

The first primary-method smoke used raw centered-logit MSE. DCNv2 CrossNet
logits reached a dynamic range above 4,000, causing the preservation loss to
grow to 2,444 and total loss to 122.6. This run is excluded and retained only
as a debugging artifact:

```text
multi_layer_path_gate_full/
```

The correction normalizes student and frozen-teacher centered logits by the
teacher batch standard deviation and uses SmoothL1. The rerun kept centered
loss between 0.017 and 0.037 at logged steps and total loss between 0.123 and
0.163:

```text
multi_layer_path_gate_full_v2/
```

## Smoke Metrics

| Method | Test AUC | Test NLLH | Brier | ECE | DP abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.615559 | 0.056893 | 0.005888 | 0.002897 | 0.000848 |
| primary full | 0.644513 | 0.039290 | 0.005261 | 0.002928 | 0.001737 |
| structured random gate | 0.644956 | 0.039384 | 0.005243 | 0.002913 | 0.001666 |
| matched capacity | 0.669685 | 0.037529 | 0.005214 | 0.003048 | 0.001050 |

`U` and `U_TILDE` were identical across these short runs
(`0.005128` and `0.004447`). This smoke subset and one training epoch are not
adequate for a utility conclusion.

## Interpretation

P1 passes as an implementation and stability check. It does not pass the
Stage2 scientific pilot gate because no full-data, three-seed, independent
probe, calibration, migration, or paired non-inferiority evidence has been
collected.

The primary and structured-random methods are effectively tied in this smoke.
This is expected because the learned gate allocation moved only slightly from
its uniform initialization. It must not be presented as evidence that the
learned gate succeeds or fails.

## Stop Point

The next step is the preregistered P2 full-data matrix for seeds 2019, 2020,
and 2021. It is a long GPU training phase and requires explicit approval before
execution. Weight or threshold changes based on these smoke test metrics are
not allowed.
