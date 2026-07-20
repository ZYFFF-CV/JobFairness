#!/usr/bin/env bash
set -euo pipefail

# Evaluate every saved FairJob prediction CSV. File suffix selects the smoke or
# full meta file so prediction rows stay aligned with the correct split.
shopt -s nullglob
for pred in results/fairjob/pred/*.csv; do
  base="$(basename "${pred%.csv}")"
  if [[ "$base" == *"_smoke" ]]; then
    meta="data/FairJob/processed/smoke_test_meta.csv"
  else
    meta="data/FairJob/processed/test_meta.csv"
  fi
  python fuxictr_ext/fairjob/evaluator.py \
    --pred "$pred" \
    --meta "$meta" \
    --out "results/fairjob/metrics/${base}.json"
done
