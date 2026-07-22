# M5 DCNv2 mitigation protocol

## Scope and status

M5A uses the proxy-excluded, pre-ranking-no-user-id DCNv2 backbone because its
M3 final-representation amplification and targeted sensitivity were the most
stable across seeds 2019, 2020, and 2021. DeepFM is not an M5A gatekeeper.

The current method-selection protocols are:

- `all_logged`: overall click performance, group calibration, and senior-scope DP;
- `random_display`: randomized-display diagnostics and FairJob U/U_TILDE;
- `context_conditioned`: senior-scope DP decomposed by `displayrandom` and `rank`.

`position_corrected = unavailable_missing_external_propensity`. It is excluded
from method selection. M5 does not estimate logging propensity and does not add
a logging-policy model.

## M5A matrix

Each method uses the same sequential FairJob split, model-facing features,
optimizer settings, and seeds:

| Method | Frozen intervention |
|---|---|
| baseline | Native DCNv2 representation and click loss |
| global suppression | Zero the complete 800-dimensional cross-interaction path |
| matched-random suppression | Zero 93 of 928 final coordinates using an independent deterministic seed |
| selective suppression | Zero the 93 train-only M3 proxy-ranked coordinates for the corresponding model seed |
| DP regularization | Add squared senior-scope batch DP with weight 0.1 |
| adversarial baseline | Gradient-reversal proxy classifier on senior rows with weight 0.1 |

The raw `protected_attribute` and `senior` values are copied into FuxiCTR
`type: meta` aliases. FuxiCTR excludes these aliases from model inputs; they are
available only to the two mitigation losses. Selective coordinates were frozen
from M3 train representations before M5 outcome evaluation.

## M5B gate

M5B starts only if at least one non-baseline method is non-dominated across the
three available-data protocols and shows a consistent DP improvement without a
material click-performance or random-display utility collapse. The selected
DCNv2 comparison is then extended to five seeds. DeepFM is used afterward only
to test cross-backbone generalization of the selected intervention.

No position-corrected result is required or used for this gate.

## Server commands

After deployment, first regenerate the runtime config so the M5 meta aliases
and model entries are visible, then run the complete smoke matrix in the
foreground:

```text
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python configs/fairjob/make_runtime_config.py --processed_dir /root/autodl-tmp/datasets/FairJob/processed --data_root /root/autodl-tmp/datasets/FairJob/fuxictr --out_dir /root/autodl-tmp/workdirs/JobFairness/stage1_1/runtime_config
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python fuxictr_ext/fairjob/run_protocol_matrix.py --matrix configs/fairjob/stage1_1_m5_matrix.yaml --group m5a_dcnv2_three_seed --mode foreground --dry_run --resume --expected_commit COMMIT_SHA
```

The formal foreground command is intentionally deferred until the smoke matrix
passes. It will stream FuxiCTR loss and validation metrics to the terminal and
also retain the same output in each isolated M5 work directory.

The smoke matrix constructs each model and checks one forward batch. It does
not take optimizer steps and must not be reported as an M5 experimental result.
The 18 formal M5A training runs remain pending until the explicit foreground
training command is started.
