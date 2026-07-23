# Stage1.2 decision report

- Status: `stage1_2_complete_direction_frozen`.
- Selected Stage2 direction: `A - multi-layer/multi-path leakage intervention`.
- New long training: `not approved before the Stage2 pilot gate`.
- Position-corrected: `unavailable_missing_external_propensity`.

## Evidence chain

- M2: all probes converged; classifications are `{"adversarial": "redistributed", "dp_regularization": "not_reduced", "global_suppression": "redistributed", "matched_random_suppression": "inconclusive_probe_family_disagreement", "selective_suppression": "redistributed"}`.
- M2: clean leakage reduction methods: `[]`; redistribution methods: `['global_suppression', 'selective_suppression', 'adversarial']`.
- M3: adversarial collapse seeds: `3/3`.
- M3: validation-only group-blind calibration brings all suppression methods within 0.001 NLLH of calibrated baseline, so their raw NLLH advantage is not an independent fairness mechanism.
- M4: `DP_near_floor_for_current_setting`; baseline all-logged DP abs is `0.000510 +/- 0.000560`, with `1/3` baseline screening intervals crossing zero.
- M4: `16/18` runs preserve signed direction across all/random/context protocols, but no stable Pareto winner emerges.

## Direction decision

| Direction | Decision | Reason |
|---|---|---|
| A: multi-layer/multi-path containment | **Selected** | Existing local interventions do not cleanly reduce leakage and repeatedly move it to other layers or paths. |
| B: leakage-outcome decoupling | Not selected | No method first achieved stable clean leakage reduction without migration or collapse. |
| C: proxy-uncertain group calibration | Not selected | DP is near floor, but group-blind calibration already explains the suppression NLLH pattern; an unresolved calibration mechanism is absent. |
| D: measurement/reproducibility audit | Not selected | All frozen probes converged and 16/18 protocol signs agree; instability is not broad enough to be the sole Stage2 direction. |
| E: limited method recovery | Not selected | Selective suppression is not superior to matched-random/global controls and has no clean leakage-plus-outcome improvement. |

This selection does not claim that representation leakage causes outcome disparity. It identifies the earlier point of failure: current interventions do not contain proxy information across the representation graph.

## Frozen Stage2-A design

Research question: How can protected-proxy leakage be contained across DCNv2 layers and paths without migration or prediction collapse?

The proposed method attaches train-only proxy adversaries to embedding, cross0/1/2, and final representations; introduces separate cross-path and residual-path gates; penalizes the maximum layer-wise excess leakage; and adds prediction-preservation plus score-variance constraints to prevent fairness-by-collapse. The protected attribute is never required at inference.

| Method | Role |
|---|---|
| baseline | CTR_reference |
| current_selective_suppression | failed_local_gate_control_no_weight_sweep |
| multi_layer_path_containment | proposed_method |
| final_layer_only_containment | layer_distribution_ablation |
| multi_layer_without_path_gates | path_control_ablation |
| matched_capacity_regularization | generic_regularization_control |

Development uses DCNv2 no-user-id and seeds 2019/2020/2021. Primary endpoints are max-layer excess probe AUC, migration count, AUC, NLLH, worst-group NLLH/Brier, and score variance. DP remains secondary because M4 places it near the current measurement floor.

The pilot is selected on validation only. It must reduce leakage at all targeted layers without increasing an untargeted layer, keep matched-seed AUC loss within 0.01 and NLLH increase within 0.002, and retain at least half of baseline prediction-score standard deviation. Test metrics cannot select hyperparameters.

Five-seed DCNv2 expansion is conditional on the three-seed pilot gate. DeepFM cross-backbone validation follows only after that expansion. Current selective-suppression weight sweeps, a logging policy model, position-corrected claims, and new-backbone search remain unapproved.

## Claim boundary

- `protected_attribute` is a behavioral proxy, not a verified demographic attribute.
- Available results are descriptive associations, not causal effects.
- Missing external propensities keep position correction unavailable.
- Stage1.2 does not establish a successful fairness mitigation method; it freezes a falsifiable Stage2 mechanism hypothesis.
