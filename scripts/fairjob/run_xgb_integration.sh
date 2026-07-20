#!/usr/bin/env bash
set -euo pipefail

# Run XGBoost full-data integration baselines. XGB_TUNE_TRIALS defaults to 0 so
# this script never starts paper-level tuning by accident.
python fuxictr_ext/fairjob/baselines/xgb_baseline.py \
  --data_dir data/FairJob/processed \
  --regime unaware \
  --mode integration \
  --hparams configs/fairjob/xgb_reference_hparams.yaml \
  --tune_trials "${XGB_TUNE_TRIALS:-0}" \
  --out_pred results/fairjob/pred/XGB_fairjob_unaware_full.csv

python fuxictr_ext/fairjob/baselines/xgb_baseline.py \
  --data_dir data/FairJob/processed \
  --regime unfair \
  --mode integration \
  --hparams configs/fairjob/xgb_reference_hparams.yaml \
  --tune_trials "${XGB_TUNE_TRIALS:-0}" \
  --out_pred results/fairjob/pred/XGB_fairjob_unfair_full.csv
