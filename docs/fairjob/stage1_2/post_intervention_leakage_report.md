# S12-M2 post-intervention leakage and migration report

## Protocol receipt

- Completed checkpoint-layer combinations: `180/180`.
- Final family-level results: `360`; all converged.
- Linear retries: `68`; only initial nonconverged results were replaced.
- Frozen rows per train/valid/test split: `50000/50000/50000`.
- Test AUC direction is oriented only by validation AUC direction.
- Exploratory material-effect threshold: absolute AUC delta `0.005`.
- Deltas compare each method with the same-seed baseline at the same layer and probe family.

## Method classification

| Method | Classification | Linear final delta | Nonlinear final delta | Migration evidence |
|---|---|---:|---:|---|
| global_suppression | `redistributed` | -0.046926 | -0.045794 | cross_layer_2:linear (+0.005396) |
| matched_random_suppression | `inconclusive_probe_family_disagreement` | -0.004808 | +0.003704 | none above threshold |
| selective_suppression | `redistributed` | -0.005310 | -0.007289 | cross_layer_2:nonlinear (+0.006436) |
| dp_regularization | `not_reduced` | -0.000868 | -0.002563 | none above threshold |
| adversarial | `redistributed` | -0.009370 | -0.053380 | embedding_flat:linear (+0.012176), embedding_flat:nonlinear (+0.024281) |

## Selective versus matched-random

Negative values favor lower decodability for selective suppression.

| Representation | Linear delta | Nonlinear delta |
|---|---:|---:|
| pre-mitigation final | -0.001685 | +0.008739 |
| post-mitigation final | -0.000501 | -0.010993 |

Selective suppression lowers post-mitigation final leakage in both probe families and all three seeds. However, its nonlinear pre-mitigation final and cross-layer-2 leakage increase in all three seeds. The evidence therefore supports local gate effectiveness plus leakage redistribution, not complete removal.

Global suppression also lowers final leakage in both families but has an upstream linear increase at cross-layer 2. Matched-random suppression has probe-family disagreement. DP regularization does not materially reduce final leakage. Adversarial training lowers final leakage while increasing embedding leakage; prediction collapse is evaluated in S12-M3.

## M2 gate

S12-M2 passes as a diagnostic audit. Current local interventions can change final-layer decodability, but no method demonstrates clean, path-wide leakage removal. This result permits S12-M3 Tier 1 collapse and calibration attribution. It does not approve new long training, five seeds, DeepFM expansion, or Stage2 direction E.
