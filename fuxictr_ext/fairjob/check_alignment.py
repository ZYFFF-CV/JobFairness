"""CLI for strict FairJob prediction/meta row alignment checks.

Use this before metrics/reporting when debugging a single prediction file. The
same validation is also called by the evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fuxictr_ext.fairjob.prediction_io import check_prediction_alignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--meta", required=True)
    args = parser.parse_args()

    result = check_prediction_alignment(args.pred, args.meta)
    print(json.dumps({"ok": True, **result}, indent=2))


if __name__ == "__main__":
    main()
