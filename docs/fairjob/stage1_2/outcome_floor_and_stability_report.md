# S12-M4 outcome floor and stability report

- Status: `complete`.
- DP role: `DP_near_floor_for_current_setting`.
- Position-corrected: `unavailable_missing_external_propensity`.
- Scope: 6 methods x 3 matched seeds; no CTR model was retrained.
- Intervals are the existing 20-repeat impression-cluster screening intervals, not publication-grade confidence intervals.

## Protocol DP

| Method | All signed | All abs | Random signed | Random abs | Context signed | Context abs |
|---|---:|---:|---:|---:|---:|---:|
| baseline | -0.000510 +/- 0.000560 | 0.000510 +/- 0.000560 | -0.000602 +/- 0.000792 | 0.000602 +/- 0.000792 | -0.000492 +/- 0.000558 | 0.000492 +/- 0.000558 |
| global_suppression | -0.000605 +/- 0.000141 | 0.000605 +/- 0.000141 | -0.000591 +/- 0.000097 | 0.000591 +/- 0.000097 | -0.000585 +/- 0.000141 | 0.000585 +/- 0.000141 |
| matched_random_suppression | -0.000627 +/- 0.000477 | 0.000627 +/- 0.000477 | -0.000512 +/- 0.000363 | 0.000512 +/- 0.000363 | -0.000608 +/- 0.000475 | 0.000608 +/- 0.000475 |
| selective_suppression | -0.000599 +/- 0.000145 | 0.000599 +/- 0.000145 | -0.000539 +/- 0.000148 | 0.000539 +/- 0.000148 | -0.000578 +/- 0.000140 | 0.000578 +/- 0.000140 |
| dp_regularization | -0.000597 +/- 0.000496 | 0.000597 +/- 0.000496 | -0.000648 +/- 0.000878 | 0.000690 +/- 0.000829 | -0.000579 +/- 0.000496 | 0.000579 +/- 0.000496 |
| adversarial | 0.000111 +/- 0.000140 | 0.000142 +/- 0.000090 | 0.000653 +/- 0.000503 | 0.000653 +/- 0.000503 | 0.000134 +/- 0.000134 | 0.000146 +/- 0.000115 |

Context-conditioned group means are direct-standardized over `displayrandom x rank` common-support contexts. The machine JSON retains both standardized means and group counts.

## Matched-seed DP effects

Negative deltas favor the intervention. The final column compares the all-logged mean effect with baseline between-seed variation.

| Method | All abs delta | Random abs delta | Context abs delta | |All effect| / baseline seed std |
|---|---:|---:|---:|---:|
| global_suppression | 0.000095 +/- 0.000421 | -0.000011 +/- 0.000705 | 0.000094 +/- 0.000420 | 0.170 |
| matched_random_suppression | 0.000117 +/- 0.000096 | -0.000090 +/- 0.000455 | 0.000116 +/- 0.000097 | 0.209 |
| selective_suppression | 0.000089 +/- 0.000444 | -0.000063 +/- 0.000805 | 0.000086 +/- 0.000447 | 0.159 |
| dp_regularization | 0.000087 +/- 0.000065 | 0.000088 +/- 0.000111 | 0.000087 +/- 0.000064 | 0.155 |
| adversarial | -0.000368 +/- 0.000581 | 0.000051 +/- 0.000484 | -0.000346 +/- 0.000570 | 0.658 |

## Group quality and calibration

Worst-group values and absolute group gaps are averaged over the three seeds. For AUC, lower is worse; for NLLH/Brier/ECE, higher is worse.

| Method | Worst AUC | Worst NLLH | Worst Brier | Worst ECE | AUC gap | NLLH gap | Brier gap | ECE gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.753722 | 0.044302 | 0.007278 | 0.001036 | 0.021057 | 0.004423 | 0.000661 | 0.000347 |
| global_suppression | 0.758139 | 0.040368 | 0.007180 | 0.000710 | 0.007582 | 0.002988 | 0.000646 | 0.000225 |
| matched_random_suppression | 0.751558 | 0.040936 | 0.007215 | 0.001174 | 0.020812 | 0.000767 | 0.000555 | 0.000338 |
| selective_suppression | 0.750958 | 0.041905 | 0.007209 | 0.000794 | 0.020067 | 0.001848 | 0.000602 | 0.000211 |
| dp_regularization | 0.753250 | 0.042752 | 0.007227 | 0.000942 | 0.020835 | 0.002629 | 0.000618 | 0.000279 |
| adversarial | 0.529949 | 0.157388 | 0.009096 | 0.008875 | 0.012112 | 0.005252 | 0.000540 | 0.000332 |

The same group-wise table for `random_display`, together with fixed-bin group calibration curves for both scopes, is retained in the machine-readable audit.

## Distribution and ranking

| Method | All Wasserstein / std | All KS | Random Wasserstein / std | Random KS | U | U_TILDE | Within-impression rank gap | Group-stratified U gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.056613 | 0.040196 | 0.060692 | 0.053403 | 0.010210 | 0.012306 | 0.065858 | 0.000473 |
| global_suppression | 0.051277 | 0.039924 | 0.055650 | 0.053794 | 0.010460 | 0.012357 | 0.057109 | 0.000707 |
| matched_random_suppression | 0.059258 | 0.041994 | 0.057866 | 0.055994 | 0.010360 | 0.012508 | 0.059877 | 0.000699 |
| selective_suppression | 0.052628 | 0.040517 | 0.057474 | 0.051806 | 0.010253 | 0.012292 | 0.050303 | 0.000567 |
| dp_regularization | 0.059845 | 0.042057 | 0.065803 | 0.055400 | 0.010227 | 0.012379 | 0.064240 | 0.000560 |
| adversarial | 0.007275 | 0.046884 | 0.019960 | 0.054879 | 0.010376 | 0.013116 | 0.074004 | 0.000216 |

Mixed-proxy random-display impressions observed: `[78]`. The within-impression value is the mean group-1 minus group-0 normalized predicted-rank gap on these candidate sets. With only 78 mixed impressions, it is exploratory; group-stratified U is reported as a broader ranking-quality diagnostic.

## Bootstrap and proxy sensitivity

Baseline all-logged DP abs is `0.000510 +/- 0.000560`. `1/3` baseline screening intervals cross zero. `16/18` runs keep the same signed direction across all/random/context protocols.

| Method | 20% symmetric-flip / base | 20% random-missing / base |
|---|---:|---:|
| baseline | 0.788039 +/- 0.369736 | 0.996909 +/- 0.014625 |
| global_suppression | 0.585461 +/- 0.005777 | 0.996328 +/- 0.009163 |
| matched_random_suppression | 0.576755 +/- 0.021173 | 0.954863 +/- 0.082200 |
| selective_suppression | 0.571159 +/- 0.016609 | 0.986071 +/- 0.019438 |
| dp_regularization | 0.554989 +/- 0.046732 | 0.967195 +/- 0.072792 |
| adversarial | 1.856678 +/- 1.413262 | 1.697072 +/- 1.113374 |

## Decision

`DP_near_floor_for_current_setting` is selected because baseline mean absolute DP does not exceed its between-seed standard deviation and at least one baseline cluster interval crosses zero. DP remains a required reported metric, but it is not sufficiently resolved to serve as the sole or primary Stage2 optimization target.

All three available-data protocols are retained. Quantitative magnitudes vary by logging scope and seed; context conditioning does not create a stable intervention winner. Proxy flips attenuate the measured signal, while random missingness is comparatively stable. These are sensitivity diagnostics for a behavioral proxy, not claims about verified demographic attributes.

Position correction remains unavailable because no external logging propensity is present. No propensity is inferred from clicks or model predictions.
