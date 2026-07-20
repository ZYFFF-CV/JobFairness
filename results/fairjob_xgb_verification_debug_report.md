# FairJob XGB Verification Debug Report

## Scope

This report checks the current XGB integration chain for FairJob inside the FuxiCTR workspace. It verifies engineering integration and evaluator behavior only. The current run is integration mode with fixed/manual reference parameters, not strict paper mode and not the FairJob paper's 100-trial Optuna reproduction.

## Data Split and Feature Construction

- Raw data path recorded in manifest: `D:\code\datasets\FairJob\fairjob.csv.gz`
- Full input rows: `1072226`
- Train candidate rows, first 80%: `857780`
- Test rows, final 20%: `214446`
- Test start offset: `857780`
- Validation policy: `tail_slice_from_train_candidate`, validation rows `50000`
- Train positives/negatives: `5999` / `851781`
- Test positives/negatives: `1490` / `212956`

The split matches the FairJob reference helper's sequential 80/20 split. There is no shuffle for the outer test split.

Required feature/metadata fields:

```json
{
  "rank": true,
  "displayrandom": true,
  "senior": true,
  "user_id": true,
  "impression_id": true,
  "product_id": true,
  "protected_attribute": true
}
```

Feature protocol:

- `unaware` feature count: `54`; protected feature included: `False`
- `unfair` feature count: `55`; `protected_attribute_feat` included: `True`
- categorical id features: `user_id, impression_id, product_id`
- binary context features included as categorical inputs: `senior, displayrandom`
- numeric features include `rank` and `35` num* columns, ending at `num50`

Evaluation uses `test_meta.csv` raw columns (`protected_attribute`, `senior`, `displayrandom`, `impression_id`, `product_id`) rather than FuxiCTR-tokenized IDs.

## XGBoost Training Settings

| regime | source | encoder | n_features | scale_pos_weight | max_depth | min_child_weight | subsample | learning_rate | colsample_bytree | reg_lambda | gamma | n_estimators | tree_method |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| unaware | manual_reference_yaml | sklearn.preprocessing.TargetEncoder | 54 | 141.9871645 | 6 | 1 | 0.8 | 0.05 | 0.8 | 1 | 0.1 | 100 | hist |
| unfair | manual_reference_yaml | sklearn.preprocessing.TargetEncoder | 55 | 141.9871645 | 6 | 1 | 0.8 | 0.05 | 0.8 | 1 | 0.1 | 100 | hist |

The implementation sets `scale_pos_weight = negative_count / positive_count`, yielding `141.9871645`. `predict_proba(X)[:, 1]` is exported as `y_pred`, so all metrics use the click=1 probability.

Target encoding path:

- Current code uses `sklearn.preprocessing.TargetEncoder`.
- Encoder is fitted on train categorical columns only, then used to transform test columns.
- No validation/test labels are used to fit the encoder.

Reference-code difference:

- The FairJob reference `example_fit.py` creates TargetEncoder-transformed `X_xgb_train`/`X_xgb_test`, and uses encoded matrices in the tuning objective.
- In the final XGB fit/predict block, the local reference file calls `model_xgb.fit(X_extended_train, ...)` and `predict_proba(X_extended_test)` on raw extended tensors despite having built encoded arrays.
- Current integration consistently uses target-encoded categorical features plus numeric features. This is deliberate and should be documented as an integration-mode implementation choice, not as byte-for-byte paper reproduction.

## Metric Implementation Path

- NLLH: local binary log loss on test probabilities, clipping only for numerical stability; no training weights or `scale_pos_weight` enter test NLLH.
- AUC: rank-based ROC-AUC over all test rows using `y_pred` as click probability.
- DP: senior-only scope, absolute difference between A=1 and A=0 predicted click means.
- U: displayrandom-only rows, impression-level ascending probability rank, averaged over impressions.
- U_TILDE: displayrandom-only rows, product-level aggregation with FairJob population/campaign correction.
- Ratios: `FEMALE_RATIO = 1.36519119`, `MALE_RATIO = 0.7639356516`.

## DP Debug

| regime | senior rows | A=0 rows | A=1 rows | mean p(A=0,S=1) | mean p(A=1,S=1) | signed A1-A0 | DP abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unaware | 142751 | 69172 | 73579 | 0.03060657929 | 0.01853102099 | -0.0120755583 | 0.0120755583 |
| unfair | 142751 | 69172 | 73579 | 0.02991358444 | 0.01825239062 | -0.01166119382 | 0.01166119382 |

DP is not mixing `senior=0` rows. The signed value is A=1 mean minus A=0 mean; the reported DP is its absolute value.

## U and U_TILDE Debug

| regime | displayrandom rows | impressions | products | U local | U official-argsort | delta | U_TILDE local | U_TILDE official-argsort | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unaware | 21145 | 11757 | 9505 | 0.01012588245 | 0.01024779564 | -0.0001219131865 | 0.01133368431 | 0.01207916479 | -0.0007454804717 |
| unfair | 21145 | 11757 | 9505 | 0.01004437073 | 0.01001389243 | 3.047829662e-05 | 0.01086625965 | 0.01096250901 | -9.624936631e-05 |

The current evaluator uses true ascending ranks within each impression (`pandas rank(method="first", ascending=True)`). The FairJob reference code's literal tensor operation is `torch.argsort(prob_click) + 1`; reproduced exactly, that assigns the sorted original indices plus one back to the original row order. The difference is small here but nonzero. If strict parity with the reference implementation is required, the evaluator should be changed to the literal argsort-index behavior and the synthetic tests updated accordingly. If the intended metric is semantic rank where higher probability gets larger rank, the current evaluator is the coherent implementation.

## Current Results Versus Table 1

| regime | metric | current | Table 1 | current - Table 1 |
| --- | --- | ---: | ---: | ---: |
| unaware | NLLH | 0.0822224181 | 0.05491 | 0.0273124181 |
| unaware | AUC | 0.7465860862 | 0.75787 | -0.01128391378 |
| unaware | DP | 0.0120755583 | 0.00278 | 0.009295558295 |
| unaware | U | 0.01012588245 | 0.01017 | -4.411754699e-05 |
| unaware | U_TILDE | 0.01133368431 | 0.01276 | -0.001426315685 |
| unfair | NLLH | 0.0821648785 | 0.05736 | 0.0248048785 |
| unfair | AUC | 0.7642380422 | 0.76201 | 0.002228042178 |
| unfair | DP | 0.01166119382 | 0.00323 | 0.00843119382 |
| unfair | U | 0.01004437073 | 0.01037 | -0.0003256292705 |
| unfair | U_TILDE | 0.01086625965 | 0.01236 | -0.001493740353 |

## Interpretation

- AUC is close to the paper values: unaware is lower by about `-0.01128391378`, unfair is higher by about `0.002228042178`. This supports that labels, main features, class direction, and row alignment are broadly correct.
- NLLH is higher than Table 1 by about `0.0273124181` for unaware and `0.0248048785` for unfair. The main likely causes are fixed manual parameters, no 100-trial Optuna search, and different probability calibration.
- DP is materially higher than Table 1. The DP scope and group means are implemented correctly, so the remaining likely causes are model parameter/calibration differences and the integration-mode target-encoding choice versus the paper/reference path.
- U is close to Table 1. U_TILDE is lower, especially for unfair, and should be interpreted together with the rank implementation note above.

## Conclusion

The XGB integration chain is engineering-correct for integration mode: sequential split, feature protocols, raw evaluator metadata, positive-class probabilities, class imbalance weighting, hparams logging, and row alignment are all verified. Current numbers should not be described as strict Table 1 reproduction. The main non-blocking discrepancy to document is that the integration XGB uses a consistent TargetEncoder matrix for final fit/predict, while the inspected FairJob reference file's final fit/predict uses raw extended features. A second evaluator detail is the current true-rank U/U_TILDE implementation versus the literal reference `argsort + 1` behavior.
