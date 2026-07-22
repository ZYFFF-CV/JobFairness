# S12-M2 probe convergence audit

## Initial full matrix

The first S12-M2 pass completed all 180 checkpoint-layer combinations on the frozen 50,000/50,000/50,000 train/validation/test row IDs. Each combination selected linear and matched-capacity nonlinear candidates using validation AUC; test labels were not used for candidate selection.

| Probe family | Results | Converged | Maximum observed iterations |
|---|---:|---:|---:|
| Linear | 180 | 112 | 1000 |
| Nonlinear with early stopping | 180 | 180 | 178 |

The nonlinear convergence gate passes. The linear gate does not yet pass: 68 selected linear probes reached the initial 1,000-iteration limit.

## Linear retry scope

Only nonconverged linear results are eligible for retry. The frozen retry matrix contains:

| Representation | Retry count |
|---|---:|
| `cross_layer_0` | 17 |
| `cross_layer_1` | 9 |
| `cross_layer_2` | 3 |
| `dcnv2_final_pre_mitigation` | 13 |
| `dcnv2_final` | 13 |
| `dcnv2_residual_component` | 13 |
| **Total** | **68** |

The retry changes only `linear_max_iter` from 1,000 to 3,000. It does not change samples, scaling, C candidates, model representations, nonlinear probes, validation selection, or test evaluation. Existing converged results are retained and skipped.

## Current interpretation boundary

Initial nonlinear results already indicate that global suppression and adversarial training substantially alter final-layer decodability, while selective suppression reduces final leakage relative to its own pre-mitigation state and matched-random control but shows possible redistribution at later cross layers. These are provisional diagnostics until the 68 linear retries and the full cross-seed aggregation are complete.

No S12-M2 method classification or Stage2 direction is frozen at this point.
