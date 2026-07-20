# FairJob x FuxiCTR Handoff Notes

This document is for a new GPT/agent that has no memory of the previous work.
It summarizes the current FairJob integration state in `D:\code\JobFairness`.

## 1. Project Goal

The repository is a FuxiCTR project. The task was to integrate the FairJob
dataset into the FuxiCTR engineering workflow and validate LR/XGBoost baselines.

Important boundary:

- This is **integration mode**, not strict paper reproduction.
- Do **not** claim current numbers reproduce FairJob paper Table 1 exactly.
- Do **not** run paper-level defaults such as XGBoost 100 Optuna trials, LR 10
  seeds, or fairness multiplier grids unless explicitly requested.
- `codex-related/` contains task cards and FairJob reference code. It is local
  reference material only and must not become runtime project code.

## 2. Local Data

Raw FairJob data is outside the repo:

```text
D:\code\datasets\FairJob\fairjob.csv.gz
```

It is a valid gzip CSV. It does not need manual decompression. Pandas can read it
directly.

Raw dataset facts already checked:

```text
rows excluding header: 1,072,226
expected sequential 20% test rows: 214,446
columns: click, protected_attribute, senior, displayrandom, rank,
         user_id, impression_id, product_id, cat0-cat12, num16-num50
```

Generated processed data lives under:

```text
data/FairJob/processed/
```

That directory is ignored by git. Do not commit raw or processed data.

## 2.1 Remote Server Layout

The current single-GPU server mounts its persistent data disk at
`/root/autodl-tmp` (not `/autodl-tmp`). Keep this project isolated from other
repositories with the following project-specific paths:

```text
/root/autodl-tmp/code/JobFairness
/root/autodl-tmp/datasets/FairJob
/root/autodl-tmp/workdirs/JobFairness/checkpoints
/root/autodl-tmp/workdirs/JobFairness/logs
/root/autodl-tmp/workdirs/JobFairness/predictions
/root/autodl-tmp/workdirs/JobFairness/reports
```

Generate a runtime `dataset_config.yaml` with absolute dataset paths on the
server. Pass `--model_root /root/autodl-tmp/workdirs/JobFairness/checkpoints`
to `run_fuxictr_model.py` so FuxiCTR checkpoints and its file logs do not land
inside the source checkout. Do not modify sibling repositories under
`/root/autodl-tmp/code`.

The server uses Python 3.12 from the shared Conda base through a project venv.
Install `configs/fairjob/requirements-server.txt` in that venv. It pins
scikit-learn 1.4.2 because FuxiCTR 2.3.9 still passes the deprecated `eps`
argument to `sklearn.metrics.log_loss`; scikit-learn 1.5 and newer reject it.
XGBoost is pinned to 2.1.4 to match the locally validated baseline version.

## 3. Implemented Files

New FairJob code is under:

```text
fuxictr_ext/fairjob/
```

Important modules:

```text
prepare_fairjob.py                  Prepare raw FairJob CSV into full/smoke splits.
run_fuxictr_model.py                FuxiCTR LR wrapper that exports aligned predictions.
baselines/xgb_baseline.py           XGBoost baseline using same splits and evaluator format.
prediction_io.py                    Shared prediction CSV schema and row alignment checks.
check_alignment.py                  CLI for prediction/meta alignment checks.
hparams.py                          Hyperparameter source resolution and recording.
metrics.py                          FairJob NLLH/AUC/DP/U/U_TILDE implementation.
evaluator.py                        Model-agnostic evaluator.
check_metric_parity.py              Synthetic metric parity smoke check.
build_report.py                     Builds smoke/integration tables and markdown report.
```

New configs are under:

```text
configs/fairjob/
```

Important config files:

```text
dataset_config.yaml                 Generated from processed feature_manifest.json.
model_config.yaml                   LR experiment definitions.
feature_schema.yaml                 Feature schema generated/updated by prepare script.
reference_hparams.yaml              Manual/fallback parameter policy.
xgb_reference_hparams.yaml          Fixed XGB integration-mode parameters.
make_dataset_config.py              Generates dataset_config.yaml.
```

Scripts exist under `scripts/fairjob/`, but this machine is Windows. Prefer
direct `python ...` commands instead of bash scripts.

## 4. Git Ignore State

`.gitignore` was updated to avoid committing local-only or heavy files:

```text
codex-related/
data/FairJob/raw/
data/FairJob/processed/
data/FairJob/fuxictr/
results/fairjob/
```

`results/fairjob/` contains local metrics/predictions/hparams and is ignored.
There are also two debug files currently under `results/`:

```text
results/fairjob_xgb_verification_debug_report.md
results/fairjob_xgb_verification_debug.json
```

Those are not inside `results/fairjob/`. Decide explicitly whether to keep,
move, ignore, or commit them.

Previous upload/commit was paused. A `git add` attempt failed because creating
`.git/index.lock` was denied by Windows ACL permissions. No commit was created.

## 5. Data Split and Feature Protocol

The FairJob split is sequential, not random:

```text
full input rows: 1,072,226
train candidate rows, first 80%: 857,780
test rows, final 20%: 214,446
test_start_original_offset: 857,780
```

For FuxiCTR training:

```text
train_full.csv: 857,780 rows
train.csv:      807,780 rows
valid.csv:       50,000 rows
test.csv:       214,446 rows
test_meta.csv:  214,446 rows
```

Smoke split:

```text
smoke_train.csv:      36,000 rows
smoke_valid.csv:       4,000 rows
smoke_test.csv:       10,000 rows
smoke_test_meta.csv:  10,000 rows
```

Feature protocol:

- `unaware` input features: `user_id`, `impression_id`, `product_id`, `cat*`,
  `senior`, `displayrandom`, `rank`, `num*`
- `unfair` input features: same as unaware plus `protected_attribute_feat`
- `protected_attribute` is **not** a model input for unaware.
- Evaluation always uses raw metadata from `test_meta.csv`, not FuxiCTR-tokenized
  ids.

Required raw evaluator fields:

```text
row_id
click
protected_attribute
senior
displayrandom
impression_id
product_id
```

All full models were checked against the same `test_meta.csv`, with matching
`row_id` and `y_true`.

## 6. Windows Commands

Use PowerShell and the `openmmlab` conda environment.

Prepare data:

```powershell
conda activate openmmlab
cd D:\code\JobFairness
python fuxictr_ext/fairjob/prepare_fairjob.py --raw_path D:/code/datasets/FairJob/fairjob.csv.gz --out_dir data/FairJob/processed --test_ratio 0.2 --valid_rows 50000 --smoke_rows 50000 --format csv
python configs/fairjob/make_dataset_config.py
```

LR full:

```powershell
python fuxictr_ext/fairjob/run_fuxictr_model.py --config_dir configs/fairjob --expid LR_fairjob_unaware_full --dataset_id fairjob_unaware_full --prediction_out results/fairjob/pred/LR_fairjob_unaware_full.csv --gpu -1
python fuxictr_ext/fairjob/run_fuxictr_model.py --config_dir configs/fairjob --expid LR_fairjob_unfair_full --dataset_id fairjob_unfair_full --prediction_out results/fairjob/pred/LR_fairjob_unfair_full.csv --gpu -1
```

XGB full:

```powershell
python fuxictr_ext/fairjob/baselines/xgb_baseline.py --data_dir data/FairJob/processed --regime unaware --mode integration --hparams configs/fairjob/xgb_reference_hparams.yaml --tune_trials 0 --out_pred results/fairjob/pred/XGB_fairjob_unaware_full.csv
python fuxictr_ext/fairjob/baselines/xgb_baseline.py --data_dir data/FairJob/processed --regime unfair --mode integration --hparams configs/fairjob/xgb_reference_hparams.yaml --tune_trials 0 --out_pred results/fairjob/pred/XGB_fairjob_unfair_full.csv
```

Evaluate full predictions:

```powershell
python fuxictr_ext/fairjob/evaluator.py --pred results/fairjob/pred/LR_fairjob_unaware_full.csv --meta data/FairJob/processed/test_meta.csv --out results/fairjob/metrics/LR_fairjob_unaware_full.json
python fuxictr_ext/fairjob/evaluator.py --pred results/fairjob/pred/LR_fairjob_unfair_full.csv --meta data/FairJob/processed/test_meta.csv --out results/fairjob/metrics/LR_fairjob_unfair_full.json
python fuxictr_ext/fairjob/evaluator.py --pred results/fairjob/pred/XGB_fairjob_unaware_full.csv --meta data/FairJob/processed/test_meta.csv --out results/fairjob/metrics/XGB_fairjob_unaware_full.json
python fuxictr_ext/fairjob/evaluator.py --pred results/fairjob/pred/XGB_fairjob_unfair_full.csv --meta data/FairJob/processed/test_meta.csv --out results/fairjob/metrics/XGB_fairjob_unfair_full.json
python fuxictr_ext/fairjob/build_report.py --results_dir results/fairjob --processed_dir data/FairJob/processed --mode integration --out docs/fairjob/integration_validation_report.md
```

Alignment checks:

```powershell
python fuxictr_ext/fairjob/check_alignment.py --pred results/fairjob/pred/XGB_fairjob_unaware_full.csv --meta data/FairJob/processed/test_meta.csv
python fuxictr_ext/fairjob/check_alignment.py --pred results/fairjob/pred/XGB_fairjob_unfair_full.csv --meta data/FairJob/processed/test_meta.csv
```

Lightweight checks:

```powershell
python -m compileall fuxictr_ext configs/fairjob tests/fairjob
python fuxictr_ext/fairjob/check_metric_parity.py
python -m pytest tests/fairjob/test_metrics_reference_parity.py -q
```

## 7. Current Full Integration Results

Current integration table:

```text
model  regime    NLLH       AUC       DP        U         U_TILDE    n_rows
LR     unaware   0.07196    0.46996   0.00129   0.00964   0.01114    214446
LR     unfair    0.05373    0.58134   0.00011   0.01064   0.01270    214446
XGB    unaware   0.08222    0.74659   0.01208   0.01013   0.01133    214446
XGB    unfair    0.08216    0.76424   0.01166   0.01004   0.01087    214446
```

`results/fairjob/integration_report.json` showed:

```json
{
  "complete_four_runs": true,
  "mode": "integration",
  "n_rows": 4,
  "table_path": "results\\fairjob\\integration_table.csv"
}
```

## 8. XGB Verification Summary

The XGB verification debug report is:

```text
results/fairjob_xgb_verification_debug_report.md
```

Important verified points:

- Split is sequential 80/20.
- All compared methods use the same `test_meta.csv`.
- `scale_pos_weight = negative_count / positive_count = 141.9871645`.
- XGB uses `sklearn.preprocessing.TargetEncoder`.
- TargetEncoder is fit on train only and transforms test; no test label leakage.
- XGB exports `predict_proba(X)[:, 1]`, the click=1 probability.
- Test NLLH is unweighted binary log loss with clipping only for numerical
  stability.
- DP uses only `senior=1` rows.
- U and U_TILDE use only `displayrandom=1` rows.
- Female/male correction ratios are:
  - `FEMALE_RATIO = 0.536 / 0.392619 = 1.36519119`
  - `MALE_RATIO = 0.464 / 0.607381 = 0.76393565`

XGB versus paper Table 1:

```text
XGB unaware:
  current: NLLH=0.08222, AUC=0.74659, DP=0.01208, U=0.01013, U_TILDE=0.01133
  paper:   NLLH=0.05491, AUC=0.75787, DP=0.00278, U=0.01017, U_TILDE=0.01276

XGB unfair:
  current: NLLH=0.08216, AUC=0.76424, DP=0.01166, U=0.01004, U_TILDE=0.01087
  paper:   NLLH=0.05736, AUC=0.76201, DP=0.00323, U=0.01037, U_TILDE=0.01236
```

Interpretation:

- AUC is close to paper values, so labels, major features, probability direction,
  and row alignment are broadly correct.
- NLLH is higher, probably due to fixed/manual params, no paper 100-trial Optuna
  search, and probability calibration differences.
- DP is higher than Table 1. DP scope/group means are implemented correctly, so
  remaining likely causes are model params/calibration and integration-mode
  feature encoding choices.
- U is close. U_TILDE is lower and should be interpreted with the evaluator rank
  detail below.

## 9. Known Implementation Differences and Risks

XGB reference-code difference:

- FairJob reference `example_fit.py` builds TargetEncoder-transformed
  `X_xgb_train` and `X_xgb_test`.
- Its tuning objective uses encoded matrices.
- The final fit/predict block in the inspected local reference file uses raw
  `X_extended_train` and `X_extended_test` despite having built encoded arrays.
- Current integration consistently uses target-encoded categorical features plus
  numeric features for final XGB fit/predict. This is coherent for integration
  mode but is not byte-for-byte paper reproduction.

U/U_TILDE rank detail:

- Current evaluator uses semantic ascending rank within each impression:
  `pandas rank(method="first", ascending=True)`.
- The FairJob reference code literally uses `torch.argsort(prob_click) + 1`.
- Reproducing that literal tensor behavior gives small but nonzero differences.
- If strict reference parity becomes required, decide whether to change evaluator
  to literal argsort-index behavior and update tests.

LR note:

- FuxiCTR native LR is not the same architecture as FairJob official PyTorch LR.
- LR full is useful for integration validation, not exact paper-model
  reproduction.
- LR unaware AUC is below 0.5 in the current single run. This is a result risk to
  document, not an engineering-link failure.

## 10. What To Do Next

For another GPT/agent:

1. Do not rerun expensive full training unless needed.
2. If checking correctness, start from row alignment and existing metrics.
3. If improving paper closeness, focus first on XGB parameter search/calibration
   and clarify the official final-fit target-encoding ambiguity.
4. If committing code, avoid committing raw data, processed data, prediction CSVs,
   or `codex-related/`.
5. If git staging fails, inspect Windows ACL on `.git`; a previous `git add`
   failed with permission denied while creating `.git/index.lock`.
