#!/usr/bin/env bash
set -euo pipefail

# Run the two FuxiCTR native LR full-data integration jobs. Set FAIRJOB_GPU to a
# CUDA device id, or leave it unset for CPU execution.
python configs/fairjob/make_dataset_config.py

python fuxictr_ext/fairjob/run_fuxictr_model.py \
  --config_dir configs/fairjob \
  --expid LR_fairjob_unaware_full \
  --dataset_id fairjob_unaware_full \
  --prediction_out results/fairjob/pred/LR_fairjob_unaware_full.csv \
  --gpu "${FAIRJOB_GPU:--1}"

python fuxictr_ext/fairjob/run_fuxictr_model.py \
  --config_dir configs/fairjob \
  --expid LR_fairjob_unfair_full \
  --dataset_id fairjob_unfair_full \
  --prediction_out results/fairjob/pred/LR_fairjob_unfair_full.csv \
  --gpu "${FAIRJOB_GPU:--1}"
