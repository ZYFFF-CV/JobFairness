# S12-M1B post-intervention export report

- Status: `pass`.
- Export commit: `49fa33953a30cce1e19e10a7f12460ba17a15ef7`.
- Jobs: `18`; split manifests: `54`.
- Rows per train/valid/test split and job: `50000/50000/50000`.
- Maximum prediction difference: `1.1102230246251565e-16` (tolerance `1e-06`).
- Maximum component reconstruction error: `0.0`.
- Sampling: exact frozen raw row IDs from Stage1.1 `probe_rows.npz`.
- Training: none; all exports loaded existing M5A checkpoints.

| Job | Seed | Suppressed dims | Prediction diff | Component error |
|---|---:|---:|---:|---:|
| baseline_seed2019 | 2019 | 0 | 9.999e-17 | 0.000e+00 |
| baseline_seed2020 | 2020 | 0 | 9.996e-17 | 0.000e+00 |
| baseline_seed2021 | 2021 | 0 | 9.996e-17 | 0.000e+00 |
| global_suppression_seed2019 | 2019 | 800 | 9.996e-17 | 0.000e+00 |
| global_suppression_seed2020 | 2020 | 800 | 9.999e-17 | 0.000e+00 |
| global_suppression_seed2021 | 2021 | 800 | 9.996e-17 | 0.000e+00 |
| matched_random_suppression_seed2019 | 2019 | 93 | 1.110e-16 | 0.000e+00 |
| matched_random_suppression_seed2020 | 2020 | 93 | 9.996e-17 | 0.000e+00 |
| matched_random_suppression_seed2021 | 2021 | 93 | 9.998e-17 | 0.000e+00 |
| selective_suppression_seed2019 | 2019 | 93 | 9.999e-17 | 0.000e+00 |
| selective_suppression_seed2020 | 2020 | 93 | 9.996e-17 | 0.000e+00 |
| selective_suppression_seed2021 | 2021 | 93 | 9.996e-17 | 0.000e+00 |
| dp_regularization_seed2019 | 2019 | 0 | 9.999e-17 | 0.000e+00 |
| dp_regularization_seed2020 | 2020 | 0 | 9.996e-17 | 0.000e+00 |
| dp_regularization_seed2021 | 2021 | 0 | 9.999e-17 | 0.000e+00 |
| adversarial_seed2019 | 2019 | 0 | 1.110e-16 | 0.000e+00 |
| adversarial_seed2020 | 2020 | 0 | 1.110e-16 | 0.000e+00 |
| adversarial_seed2021 | 2021 | 0 | 1.110e-16 | 0.000e+00 |

The matrix is accepted for S12-M2 leakage and migration probes. This receipt does not claim that any intervention reduced leakage or improved outcome fairness.
