#!/usr/bin/env bash
set -euo pipefail

# Run XGBoost smoke baselines on the prepared FairJob smoke splits. XGB_TUNE_TRIALS
# defaults to 0; only explicit 5 or 10 is allowed by the Python entrypoint.
python fuxictr_ext/fairjob/baselines/xgb_baseline.py \
  --data_dir data/FairJob/processed \
  --regime unaware \
  --mode smoke \
  --hparams configs/fairjob/xgb_reference_hparams.yaml \
  --tune_trials "${XGB_TUNE_TRIALS:-0}" \
  --out_pred results/fairjob/pred/XGB_fairjob_unaware_smoke.csv

python fuxictr_ext/fairjob/baselines/xgb_baseline.py \
  --data_dir data/FairJob/processed \
  --regime unfair \
  --mode smoke \
  --hparams configs/fairjob/xgb_reference_hparams.yaml \
  --tune_trials "${XGB_TUNE_TRIALS:-0}" \
  --out_pred results/fairjob/pred/XGB_fairjob_unfair_smoke.csv
