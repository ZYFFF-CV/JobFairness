# Stage2 Foreground Training Runbook

## Runtime Contract

- Repository: `/root/autodl-tmp/code/JobFairness`
- Dataset: `/root/autodl-tmp/datasets/FairJob`
- Artifacts: `/root/autodl-tmp/workdirs/JobFairness/stage2`
- Python: `/root/autodl-tmp/workdirs/JobFairness/venv/bin/python`
- GPU: one RTX 4090, selected with `--gpu 0`
- Training is foreground only. Loss and Stage2 telemetry remain visible in the
  active SSH terminal; closing that terminal stops the run.

Stage2 does not modify FuxiCTR or model-zoo source. All checkpoints,
predictions, representations, probes, and reports remain outside the Git
repository.

## Refresh Generated Config

```bash
cd /root/autodl-tmp/code/JobFairness
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python \
  configs/fairjob/make_model_config.py
```

## Bounded Seed-2019 Smoke

Train the paired baseline first:

```bash
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python \
  fuxictr_ext/fairjob/run_fuxictr_model.py \
  --config_dir configs/fairjob \
  --expid Stage2DCNv2_baseline_smoke \
  --dataset_id fairjob_m5_pre_ranking_no_user_id_proxy_excluded_smoke \
  --gpu 0 --seed 2019 --epochs 1 \
  --run_dir /root/autodl-tmp/workdirs/JobFairness/stage2/smoke/seed2019/baseline
```

The paired baseline checkpoint is:

```text
/root/autodl-tmp/workdirs/JobFairness/stage2/smoke/seed2019/baseline/checkpoints/fairjob_m5_pre_ranking_no_user_id_proxy_excluded_smoke/Stage2DCNv2_baseline_smoke.model
```

Run one containment method by replacing `METHOD` below. The bounded P1 methods
are `multi_layer_path_gate_full`, `structured_random_gate`, and
`matched_capacity_regularization`.

```bash
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python \
  fuxictr_ext/fairjob/run_fuxictr_model.py \
  --config_dir configs/fairjob \
  --expid Stage2DCNv2_METHOD_smoke \
  --dataset_id fairjob_m5_pre_ranking_no_user_id_proxy_excluded_smoke \
  --gpu 0 --seed 2019 --epochs 1 \
  --backbone_checkpoint /root/autodl-tmp/workdirs/JobFairness/stage2/smoke/seed2019/baseline/checkpoints/fairjob_m5_pre_ranking_no_user_id_proxy_excluded_smoke/Stage2DCNv2_baseline_smoke.model \
  --run_dir /root/autodl-tmp/workdirs/JobFairness/stage2/smoke/seed2019/METHOD
```

Smoke results validate execution, gradients, telemetry, artifact alignment,
and memory use. They are not pilot evidence.

## Formal Pilot Boundary

P1 smoke passed and the three-seed P2 matrix was approved on 2026-07-23. Start
the foreground, resumable scheduler from the cloud provider's server console:

```bash
cd /root/autodl-tmp/code/JobFairness
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python -u \
  fuxictr_ext/fairjob/run_stage2_pilot.py
```

The default invocation runs all 27 jobs in frozen order. For each seed it
trains `Stage2DCNv2_baseline_full` first, then passes that exact checkpoint to
the eight dependent methods. A rerun skips only jobs whose manifest is
complete and whose Git commit matches the current checkout.

To inspect the exact command sequence without training:

```bash
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python -u \
  fuxictr_ext/fairjob/run_stage2_pilot.py --dry_run
```

To run one seed at a time:

```bash
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python -u \
  fuxictr_ext/fairjob/run_stage2_pilot.py --seeds 2019
```

Repeat with `2020` and `2021`. This remains foreground execution: logs are
visible in the server console and no daemon or background process is created.

The formal pilot is evaluated only by
`configs/fairjob/stage2_pilot_gate.yaml`. Test metrics must not be used to
change objective weights, thresholds, gate budgets, or probe capacities.
