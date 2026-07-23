# Stage2 P2 Three-Seed Training Audit

## Status

P2 completed all 27 preregistered full-data jobs:

- seeds: `2019`, `2020`, `2021`
- methods: one baseline and eight frozen Stage2 variants per seed
- training commit: `7574ed45fcacc41dba046e10f962de8472ca500f`
- dataset: `fairjob_m5_pre_ranking_no_user_id_proxy_excluded_full`
- test rows per job: `214446`

The artifact audit found 27 complete run manifests, checkpoints, prediction
files, metric files, hparameter files, and logs. Every nonbaseline job names the
exact same-seed baseline checkpoint in its hparameters. Logs contain no NaN,
OOM, traceback, or runtime error markers.

## Three-Seed Means

These are raw P2 evaluation values. They are not probe or containment results.
Delta columns are paired against the baseline from the same seed before taking
the mean.

| Method | AUC | dAUC | NLLH | dNLLH | Brier | dBrier | DP | U | U_TILDE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.764927 | +0.000000 | 0.042093 | +0.000000 | 0.006948 | +0.000000 | 0.000510 | 0.010210 | 0.012306 |
| multi_layer_path_gate_full | 0.768455 | +0.003528 | 0.042001 | -0.000091 | 0.006925 | -0.000023 | 0.001239 | 0.010227 | 0.012355 |
| multi_layer_no_gate | 0.766078 | +0.001151 | 0.041610 | -0.000482 | 0.006948 | +0.000001 | 0.001000 | 0.010198 | 0.012034 |
| full_without_joint_adversary | 0.766639 | +0.001713 | 0.040792 | -0.001300 | 0.006917 | -0.000031 | 0.001157 | 0.010182 | 0.011994 |
| structured_random_gate | 0.764712 | -0.000214 | 0.043172 | +0.001079 | 0.007068 | +0.000120 | 0.001206 | 0.010223 | 0.012202 |
| matched_capacity_regularization | 0.772117 | +0.007190 | 0.040662 | -0.001431 | 0.006932 | -0.000016 | 0.001162 | 0.010276 | 0.012128 |
| layer_risk_sum | 0.600764 | -0.164163 | 0.414900 | +0.372807 | 0.017633 | +0.010685 | 0.001086 | 0.010154 | 0.012132 |
| final_only_path_gate | 0.559141 | -0.205785 | 0.410927 | +0.368834 | 0.018109 | +0.011162 | 0.002129 | 0.010192 | 0.012403 |
| final_only_no_gate | 0.549212 | -0.215714 | 0.753097 | +0.711004 | 0.026395 | +0.019448 | 0.002446 | 0.010305 | 0.012593 |

The primary `multi_layer_path_gate_full` method improves AUC in every seed; its
worst paired seed delta is `+0.001707`. Its mean NLLH and Brier do not show
prediction collapse. `layer_risk_sum`, `final_only_path_gate`, and
`final_only_no_gate` do collapse on ranking and calibration metrics and remain
negative controls rather than viable methods. `structured_random_gate` fails
the preregistered per-seed AUC tolerance in seed 2021 with a paired delta of
`-0.013194`.

## Interpretation Boundary

P2 establishes that the primary model trains, preserves click prediction
quality, and is distinct from the collapsed controls. It does not establish
that protected-proxy information is contained or removed. That decision
requires P3 row-disjoint and raw-user-disjoint node/joint probes, calibrated
prediction checks, migration counts, and the frozen machine gate.

Persistent export of every checkpoint representation was rejected because the
server had approximately 22 GB free after P2, while a complete export was
estimated to require substantially more space. P3 therefore reconstructs one
checkpoint at a time, keeps frozen-sample representations in memory, and saves
only compact JSON probe results.
