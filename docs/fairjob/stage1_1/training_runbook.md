# Stage1.1 服务器训练运行手册

更新日期：2026-07-21

## 当前边界

- 本阶段代码准备不启动正式训练。
- 正式训练只在服务器单卡 GPU 上运行。
- 本地、GitHub 与服务器必须使用同一 Git commit。
- 数据位于 `/root/autodl-tmp/datasets/FairJob`。
- checkpoint、预测、表示、日志和运行时配置统一位于
  `/root/autodl-tmp/workdirs/JobFairness/stage1_1`。
- FuxiCTR 原生核心代码不修改；DeepFM/DCNv2 诊断能力位于
  `fuxictr_ext/fairjob` 扩展层。

服务器 Python：

```text
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python
```

## 一次性数据与配置准备

Stage1.1 增加了 train/valid/test 原始 evaluator metadata，并修正了 FuxiCTR
full 配置中 train/valid 重叠的问题。因此更新代码后需重新生成 processed
文件，但不需要重新下载或解压数据。

```bash
cd /root/autodl-tmp/code/JobFairness
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python fuxictr_ext/fairjob/prepare_fairjob.py --raw_path /root/autodl-tmp/datasets/FairJob/raw/fairjob.csv.gz --out_dir /root/autodl-tmp/datasets/FairJob/processed --test_ratio 0.2 --valid_rows 50000 --smoke_rows 50000 --format csv --feature_schema /root/autodl-tmp/workdirs/JobFairness/stage1_1/runtime_config/feature_schema.yaml
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python configs/fairjob/make_runtime_config.py --processed_dir /root/autodl-tmp/datasets/FairJob/processed --data_root /root/autodl-tmp/datasets/FairJob/fuxictr --out_dir /root/autodl-tmp/workdirs/JobFairness/stage1_1/runtime_config
```

## 无训练调试

该命令会依次构建 4 个 smoke 数据集和模型，执行一个 batch 的前向传播，
并验证原生预测与表示导出路径数值一致。它不会调用 optimizer step。

```bash
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python fuxictr_ext/fairjob/run_protocol_matrix.py --group primary_screening --mode foreground --dry_run --resume --expected_commit COMMIT_SHA
```

查看调试状态：

```bash
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python fuxictr_ext/fairjob/run_protocol_matrix.py --group primary_screening --mode status --dry_run
```

## 正式后台训练

只有 dry-run 全部完成、代码 commit 冻结并确认服务器 GPU 空闲后，才运行：

```bash
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python fuxictr_ext/fairjob/run_protocol_matrix.py --group primary_screening --mode detach --expected_commit COMMIT_SHA
```

`detach` 使用独立服务器进程会话，标准输出写入 Stage1.1 workdir。本地关机或
SSH 断开不会终止训练。`--resume` 只跳过已经具有 success marker 的完整实验；
FuxiCTR 当前 checkpoint 不包含 optimizer state，因此失败的单个实验会从头重跑，
不能宣称为 epoch 内严格续训。

查看正式任务状态：

```bash
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python fuxictr_ext/fairjob/run_protocol_matrix.py --group primary_screening --mode status
```

## 表示诊断

主矩阵训练成功后，每个主实验会导出固定种子、最多 200,000 行/数据 split
的分片表示。完整 test 预测和 FairJob 指标不采样。示例 probe：

```bash
/root/autodl-tmp/workdirs/JobFairness/venv/bin/python fuxictr_ext/fairjob/proxy_probe.py --representation_root /root/autodl-tmp/workdirs/JobFairness/stage1_1/training/deepfm_pre_ranking_proxy_excluded/representations --representation embedding_flat --probe_type both --max_rows_per_split 200000 --seed 2019 --out /root/autodl-tmp/workdirs/JobFairness/stage1_1/probes/deepfm_pre_ranking_proxy_excluded_embedding.json
```

probe 仅在 train 拟合、valid 选择容量、test 报告最终 AUC 和 balanced
accuracy。逐层结果必须与 input/embedding baseline、去 ID 消融、随机 masking
对照共同解释，不能仅凭某层可预测保护代理就声称公平伤害。

## 科学门槛

DeepFM/DCNv2 是诊断 backbone，不是创新点本身。Stage2 方法方向只有在以下证据
成立后才冻结：跨 backbone 和随机种子稳定；泄漏超过输入基线；排除 ID 记忆；
目标路径 masking 强于等稀疏随机 masking；并且泄漏变化与校准、DP 或 utility
变化具有稳健关联。未通过门槛时应报告负结果或转向 calibration/selection-aware
方向，不能为追求期刊创新度强行增加新模型。
