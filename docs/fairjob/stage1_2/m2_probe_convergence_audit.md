# S12-M2 probe convergence audit

## Initial full matrix

The first S12-M2 pass completed all 180 checkpoint-layer combinations on the frozen 50,000/50,000/50,000 train/validation/test row IDs. Each combination selected linear and matched-capacity nonlinear candidates using validation AUC; test labels were not used for candidate selection.

| Probe family | Results | Converged | Maximum observed iterations |
|---|---:|---:|---:|
| Linear | 180 | 112 | 1000 |
| Nonlinear with early stopping | 180 | 180 | 178 |

The nonlinear convergence gate passes. In the initial run, 68 selected linear probes reached the 1,000-iteration limit and entered the frozen retry scope below.

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

The retry changed only `linear_max_iter` from 1,000 to 3,000. It did not change samples, scaling, C candidates, model representations, nonlinear probes, validation selection, or test evaluation. Existing converged results were retained and skipped. All 68 retries converged; the slowest selected probe required 2,799 iterations.

## Current interpretation boundary

The merged audit confirms that global suppression and adversarial training substantially alter final-layer decodability, while selective suppression reduces final leakage relative to its own post-mitigation baseline and nonlinear matched-random control but redistributes leakage to its pre-mitigation final and later cross layer.

The formal method classifications are recorded in `post_intervention_leakage_report.md`. No Stage2 direction is frozen at this point.
