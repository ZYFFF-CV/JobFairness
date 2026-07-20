#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke validation. FAIRJOB_RAW_PATH may override the default raw
# data location; model scripts use their own documented environment variables.
raw_path="${FAIRJOB_RAW_PATH:-data/FairJob/raw/fairjob.csv.gz}"

python fuxictr_ext/fairjob/prepare_fairjob.py \
  --raw_path "$raw_path" \
  --out_dir data/FairJob/processed \
  --test_ratio 0.2 \
  --valid_rows 50000 \
  --smoke_rows 50000 \
  --format csv

python configs/fairjob/make_dataset_config.py

bash scripts/fairjob/run_lr_smoke.sh
bash scripts/fairjob/run_xgb_smoke.sh
bash scripts/fairjob/eval_all.sh
python fuxictr_ext/fairjob/build_report.py \
  --results_dir results/fairjob \
  --processed_dir data/FairJob/processed \
  --mode smoke \
  --out docs/fairjob/integration_validation_report.md
