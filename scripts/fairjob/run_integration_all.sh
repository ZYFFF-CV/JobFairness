#!/usr/bin/env bash
set -euo pipefail

# End-to-end full-data validation. Optional --xgb_tune_trials exports
# XGB_TUNE_TRIALS for the XGBoost scripts; default behavior is no tuning.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --xgb_tune_trials)
      export XGB_TUNE_TRIALS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

python configs/fairjob/make_dataset_config.py

bash scripts/fairjob/run_lr_integration.sh
bash scripts/fairjob/run_xgb_integration.sh
bash scripts/fairjob/eval_all.sh
python fuxictr_ext/fairjob/build_report.py \
  --results_dir results/fairjob \
  --processed_dir data/FairJob/processed \
  --mode integration \
  --out docs/fairjob/integration_validation_report.md
