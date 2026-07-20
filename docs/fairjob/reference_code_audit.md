# FairJob Reference Code Audit

Reference code path used for inspection only:

```text
codex-related/fairjob-dataset-main
```

This directory is ignored by git and must not be imported as a runtime dependency
by the FairJob integration code.

## Observations

- `paper_results.sh` calls `example_fit.py --lr=1`, while the local
  `example_fit.py` registers `--dummy`, `--lr_fair`, `--xgb`, and `--unfair`,
  but not `--lr`. Regular LR logic is implemented more completely in
  `example_simulations_LR.py`.
- The official split helper uses a deterministic first-80-percent train,
  last-20-percent test split despite the docstring saying "random split".
- The XGBoost block builds TargetEncoder-transformed arrays, but the final
  `model_xgb.fit()` and `predict_proba()` calls use the unencoded extended
  tensors. The new implementation will use a stable train-only TargetEncoder
  path and record that choice in hparams/report outputs.
- FairJob official LR uses categorical embeddings plus numeric features and a
  two-class softmax. FuxiCTR native LR is a feature-weight LR with sigmoid
  output, so LR results are integration checks rather than exact paper-model
  reproductions.
