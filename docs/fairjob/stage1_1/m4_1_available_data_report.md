# M4.1 Available-data calibration and utility audit

- M4 status: `conditional_pass` / `available_data_pass`.
- Position-corrected status: `unavailable_missing_external_propensity`.
- Position-corrected results are excluded from current method selection.
- Current selection protocols: `all_logged`, `random_display`, and `context_conditioned`.
- Calibration gap is `ECE(group 1) - ECE(group 0)`; both signed and absolute values are reported.

| Model | Scope | G0 NLLH | G1 NLLH | G0 Brier | G1 Brier | G0 ECE | G1 ECE | ECE gap signed | ECE gap abs | U | U_TILDE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deepfm_no_user_id_seed2019 | all_logged | 0.081203 | 0.078175 | 0.008290 | 0.007357 | 0.005567 | 0.005022 | -0.000545 | 0.000545 | 0.010029 | 0.010704 |
| deepfm_no_user_id_seed2019 | random_display | 0.058633 | 0.117245 | 0.006619 | 0.008519 | 0.003660 | 0.006262 | 0.002602 | 0.002602 | 0.010029 | 0.010704 |
| deepfm_no_user_id_seed2020 | all_logged | 0.068045 | 0.073481 | 0.007785 | 0.007315 | 0.004522 | 0.004727 | 0.000205 | 0.000205 | 0.010124 | 0.011175 |
| deepfm_no_user_id_seed2020 | random_display | 0.052688 | 0.104321 | 0.006565 | 0.008465 | 0.004068 | 0.005897 | 0.001829 | 0.001829 | 0.010124 | 0.011175 |
| deepfm_no_user_id_seed2021 | all_logged | 0.065733 | 0.069115 | 0.008063 | 0.007297 | 0.005476 | 0.005066 | -0.000410 | 0.000410 | 0.010208 | 0.011233 |
| deepfm_no_user_id_seed2021 | random_display | 0.044063 | 0.098556 | 0.006398 | 0.008324 | 0.004393 | 0.005814 | 0.001422 | 0.001422 | 0.010208 | 0.011233 |
| dcnv2_no_user_id_seed2019 | all_logged | 0.044586 | 0.039647 | 0.007357 | 0.006789 | 0.001559 | 0.002119 | 0.000560 | 0.000560 | 0.009993 | 0.012309 |
| dcnv2_no_user_id_seed2019 | random_display | 0.042463 | 0.043266 | 0.005928 | 0.007816 | 0.003484 | 0.001133 | -0.002352 | 0.002352 | 0.009993 | 0.012309 |
| dcnv2_no_user_id_seed2020 | all_logged | 0.041267 | 0.040696 | 0.007145 | 0.006517 | 0.000061 | 0.000282 | 0.000221 | 0.000221 | 0.010274 | 0.012335 |
| dcnv2_no_user_id_seed2020 | random_display | 0.031811 | 0.046044 | 0.005511 | 0.007582 | 0.001746 | 0.000985 | -0.000761 | 0.000761 | 0.010274 | 0.012335 |
| dcnv2_no_user_id_seed2021 | all_logged | 0.047052 | 0.039293 | 0.007332 | 0.006544 | 0.000707 | 0.000447 | -0.000260 | 0.000260 | 0.010365 | 0.012272 |
| dcnv2_no_user_id_seed2021 | random_display | 0.051903 | 0.044189 | 0.006111 | 0.007542 | 0.002914 | 0.000756 | -0.002158 | 0.002158 | 0.010365 | 0.012272 |

`U` and `U_TILDE` internally evaluate only `displayrandom=1` rows. They are nevertheless listed under both declared outer scopes so the evaluation path remains explicit and auditable.

This is an available-data diagnostic pass, not evidence that the position-corrected protocol has passed. No logging propensity was estimated from outcomes.
