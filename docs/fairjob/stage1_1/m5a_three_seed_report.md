# M5A DCNv2 three-seed result

- Status: `complete_m5b_gate_failed`.
- Position-corrected: `unavailable_missing_external_propensity`.
- Selection protocols: `all_logged`, `random_display`, and `context_conditioned`.
- Values are mean +/- sample standard deviation over seeds 2019, 2020, and 2021.

| Method | All AUC | All NLLH | All ECE | All DP | Random DP | Context DP | U | U_TILDE | All ECE gap | Random ECE gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.764927 +/- 0.003964 | 0.042093 +/- 0.001098 | 0.000839 +/- 0.000895 | 0.000510 +/- 0.000560 | 0.000602 +/- 0.000792 | 0.000492 +/- 0.000558 | 0.010210 +/- 0.000194 | 0.012306 +/- 0.000032 | 0.000347 +/- 0.000186 | 0.001757 +/- 0.000868 |
| global_suppression | 0.762338 +/- 0.003014 | 0.038876 +/- 0.000053 | 0.000596 +/- 0.000283 | 0.000605 +/- 0.000141 | 0.000591 +/- 0.000097 | 0.000585 +/- 0.000141 | 0.010460 +/- 0.000087 | 0.012357 +/- 0.000451 | 0.000225 +/- 0.000145 | 0.001424 +/- 0.000417 |
| matched_random_suppression | 0.762699 +/- 0.005113 | 0.040553 +/- 0.000091 | 0.001005 +/- 0.000878 | 0.000627 +/- 0.000477 | 0.000512 +/- 0.000363 | 0.000608 +/- 0.000475 | 0.010360 +/- 0.000074 | 0.012508 +/- 0.000077 | 0.000338 +/- 0.000140 | 0.001826 +/- 0.000954 |
| selective_suppression | 0.761707 +/- 0.006766 | 0.040982 +/- 0.000146 | 0.000687 +/- 0.000506 | 0.000599 +/- 0.000145 | 0.000539 +/- 0.000148 | 0.000578 +/- 0.000140 | 0.010253 +/- 0.000083 | 0.012292 +/- 0.000286 | 0.000211 +/- 0.000160 | 0.001462 +/- 0.000677 |
| dp_regularization | 0.764372 +/- 0.002943 | 0.041439 +/- 0.002041 | 0.000792 +/- 0.000866 | 0.000597 +/- 0.000496 | 0.000690 +/- 0.000829 | 0.000579 +/- 0.000496 | 0.010227 +/- 0.000160 | 0.012379 +/- 0.000144 | 0.000279 +/- 0.000236 | 0.001620 +/- 0.001006 |
| adversarial | 0.536575 +/- 0.001734 | 0.154763 +/- 0.004686 | 0.008709 +/- 0.000716 | 0.000142 +/- 0.000090 | 0.000653 +/- 0.000503 | 0.000146 +/- 0.000115 | 0.010376 +/- 0.000155 | 0.013116 +/- 0.000329 | 0.000332 +/- 0.000194 | 0.002573 +/- 0.000345 |

## M5B gate

The M5B gate failed. No intervention improved mean DP in all three available-data protocols while also matching or improving baseline AUC, NLLH, U, and U_TILDE.

The adversarial baseline reduced all-logged and context DP but collapsed ranking and calibration performance (mean all-logged AUC 0.536575 and ECE 0.008709, versus baseline 0.764927 and 0.000839). The remaining methods did not improve both all-logged and context-conditioned DP means over baseline.

Five-seed DCNv2 expansion and DeepFM generalization are therefore not approved by the frozen gate. Further mitigation design or weight sweeps require a new decision; they are not part of this completed M5A screen.

This is an available-data association study. Position-corrected results and causal claims remain unavailable.
