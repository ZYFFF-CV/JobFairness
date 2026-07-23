# S12-M3 Tier 1 regularization, calibration, and collapse report

- No CTR model was retrained.
- Calibrators were fit and selected only within frozen validation rows.
- Test predictions were used only for final diagnostics.
- Position-corrected remains `unavailable_missing_external_propensity`.

| Method | Collapse seeds | AUC | Raw NLLH | Calibrated NLLH | Raw ECE | Calibrated ECE | Score std | Class gap | Selected calibrators |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0/3 | 0.764927 | 0.042093 | 0.038126 | 0.000839 | 0.000751 | 0.015611 | 0.015840 | isotonic, isotonic, isotonic |
| global_suppression | 0/3 | 0.762338 | 0.038876 | 0.038330 | 0.000596 | 0.000805 | 0.012287 | 0.014287 | platt, isotonic, isotonic |
| matched_random_suppression | 0/3 | 0.762699 | 0.040553 | 0.038104 | 0.001005 | 0.000767 | 0.015594 | 0.016462 | isotonic, isotonic, isotonic |
| selective_suppression | 0/3 | 0.761707 | 0.040982 | 0.038166 | 0.000687 | 0.000776 | 0.014525 | 0.015104 | isotonic, isotonic, isotonic |
| dp_regularization | 0/3 | 0.764372 | 0.041439 | 0.038432 | 0.000792 | 0.000563 | 0.015049 | 0.015810 | isotonic, identity, isotonic |
| adversarial | 3/3 | 0.536575 | 0.154763 | 0.041346 | 0.008709 | 0.000780 | 0.044335 | 0.004418 | isotonic, isotonic, platt |

Adversarial results are classified as fairness-by-collapse when all three seeds satisfy the frozen collapse rule. Calibration changes probability quality but cannot restore lost ranking AUC because every candidate is group-blind and fitted after the frozen CTR model.

Suppression NLLH/ECE gains that are matched by validation-only calibration are attributed to generic probability calibration rather than an independent fairness mechanism. Tier 2 regularization training remains conditional on this report and the M2 leakage classifications.
