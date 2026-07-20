# FairJob Integration Validation Report

This report is generated from single-run smoke/integration artifacts.
It is not a strict reproduction of the NeurIPS FairJob paper results.

## Data

Processed data directory: `data\FairJob\processed`

## Split Manifest

```json
{
  "format": "csv",
  "full": {
    "actual_valid_rows": 50000,
    "input_rows": 1072226,
    "prefix": "full",
    "requested_valid_rows": 50000,
    "test_rows": 214446,
    "test_start_original_offset": 857780,
    "train_candidate_rows": 857780,
    "train_rows": 807780,
    "valid_policy": "tail_slice_from_train_candidate",
    "valid_rows": 50000
  },
  "out_dir": "data\\FairJob\\processed",
  "raw_path": "D:\\code\\datasets\\FairJob\\fairjob.csv.gz",
  "smoke": {
    "actual_valid_rows": 4000,
    "input_rows": 50000,
    "prefix": "smoke",
    "requested_valid_rows": 5000,
    "test_rows": 10000,
    "test_start_original_offset": 40000,
    "train_candidate_rows": 40000,
    "train_rows": 36000,
    "valid_policy": "tail_slice_from_train_candidate",
    "valid_rows": 4000
  },
  "test_ratio": 0.2
}
```

## Feature Manifest

```json
{
  "binary_context_features": [
    "senior",
    "displayrandom"
  ],
  "categorical_features": [
    "cat0",
    "cat1",
    "cat2",
    "cat3",
    "cat4",
    "cat5",
    "cat6",
    "cat7",
    "cat8",
    "cat9",
    "cat10",
    "cat11",
    "cat12"
  ],
  "categorical_id_features": [
    "user_id",
    "impression_id",
    "product_id"
  ],
  "label": "click",
  "numeric_features": [
    "rank",
    "num16",
    "num17",
    "num18",
    "num19",
    "num20",
    "num21",
    "num22",
    "num23",
    "num24",
    "num25",
    "num26",
    "num27",
    "num28",
    "num29",
    "num30",
    "num31",
    "num32",
    "num33",
    "num34",
    "num35",
    "num36",
    "num37",
    "num38",
    "num39",
    "num40",
    "num41",
    "num42",
    "num43",
    "num44",
    "num45",
    "num46",
    "num47",
    "num48",
    "num49",
    "num50"
  ],
  "protected_feature": "protected_attribute",
  "protected_feature_for_model": "protected_attribute_feat",
  "unaware_features": [
    "user_id",
    "impression_id",
    "product_id",
    "cat0",
    "cat1",
    "cat2",
    "cat3",
    "cat4",
    "cat5",
    "cat6",
    "cat7",
    "cat8",
    "cat9",
    "cat10",
    "cat11",
    "cat12",
    "senior",
    "displayrandom",
    "rank",
    "num16",
    "num17",
    "num18",
    "num19",
    "num20",
    "num21",
    "num22",
    "num23",
    "num24",
    "num25",
    "num26",
    "num27",
    "num28",
    "num29",
    "num30",
    "num31",
    "num32",
    "num33",
    "num34",
    "num35",
    "num36",
    "num37",
    "num38",
    "num39",
    "num40",
    "num41",
    "num42",
    "num43",
    "num44",
    "num45",
    "num46",
    "num47",
    "num48",
    "num49",
    "num50"
  ],
  "unfair_features": [
    "user_id",
    "impression_id",
    "product_id",
    "cat0",
    "cat1",
    "cat2",
    "cat3",
    "cat4",
    "cat5",
    "cat6",
    "cat7",
    "cat8",
    "cat9",
    "cat10",
    "cat11",
    "cat12",
    "senior",
    "displayrandom",
    "rank",
    "num16",
    "num17",
    "num18",
    "num19",
    "num20",
    "num21",
    "num22",
    "num23",
    "num24",
    "num25",
    "num26",
    "num27",
    "num28",
    "num29",
    "num30",
    "num31",
    "num32",
    "num33",
    "num34",
    "num35",
    "num36",
    "num37",
    "num38",
    "num39",
    "num40",
    "num41",
    "num42",
    "num43",
    "num44",
    "num45",
    "num46",
    "num47",
    "num48",
    "num49",
    "num50",
    "protected_attribute_feat"
  ]
}
```

## Integration Results

| model | regime | NLLH | AUC | DP | U | U_TILDE | n_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LR | unaware | 0.0719612 | 0.469955 | 0.00128846 | 0.00963965 | 0.0111363 | 214446 |
| LR | unfair | 0.0537267 | 0.581341 | 0.000107413 | 0.0106362 | 0.0127049 | 214446 |
| XGB | unaware | 0.0822224 | 0.746586 | 0.0120756 | 0.0101259 | 0.0113337 | 214446 |
| XGB | unfair | 0.0821649 | 0.764238 | 0.0116612 | 0.0100444 | 0.0108663 | 214446 |

## Quality Gates

- QG1 row alignment is enforced by `check_alignment.py` and `evaluator.py`.
- QG2 unaware configs exclude protected features.
- QG3 unfair configs include `protected_attribute_feat`.
- QG4 evaluator outputs NLLH/AUC/DP/U/U_TILDE.
- QG5 smoke completion requires four metric rows.
- QG6 integration completion requires four metric rows or recorded failure.
- QG7 scripts default to zero XGBoost trials and single-seed runs.

This validation confirms that FairJob has been integrated into the FuxiCTR engineering workflow and that LR/XGBoost baselines can be evaluated with FairJob-specific metrics. The reported results are single-run integration checks using available or reference hyperparameters. They are not claimed as strict reproduction of the NeurIPS FairJob paper results.

Current results are not paper-level strict reproduction and do not include 100 trials, multi-seed runs, or a full fairness multiplier grid.
