#!/usr/bin/env bash
set -euo pipefail

# Evaluate one prediction CSV against its matching meta CSV. The evaluator will
# fail if row_id, y_true, or probability constraints are violated.
if [ "$#" -lt 2 ]; then
  echo "Usage: scripts/fairjob/eval_one.sh PRED_CSV META_CSV [OUT_JSON]" >&2
  exit 2
fi

pred="$1"
meta="$2"
out="${3:-results/fairjob/metrics/$(basename "${pred%.csv}").json}"

python fuxictr_ext/fairjob/evaluator.py \
  --pred "$pred" \
  --meta "$meta" \
  --out "$out"
