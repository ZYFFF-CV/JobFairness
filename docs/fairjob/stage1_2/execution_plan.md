# Stage1.2 执行计划

## 目标与边界

Stage1.2 用于定位 Stage1.1 公平干预失败发生在证据链的哪个环节，并据此只选择一个 Stage2 主方向。当前阶段复用已有 DCNv2 六种方法、三个训练种子的 18 个 checkpoint，不新增 backbone、五种子训练、logging-policy 模型或大规模权重搜索。

固定边界如下：

- FairJob 保持顺序切分，测试集固定为 214,446 行；所有表征和预测按原始 `row_id` 对齐。
- `position_corrected = unavailable_missing_external_propensity`，不进入当前方法选择标准。
- 代码保存在 `/root/autodl-tmp/code/JobFairness`，数据保存在 `/root/autodl-tmp/datasets/FairJob`，Stage1.2 运行产物保存在 `/root/autodl-tmp/workdirs/JobFairness/stage1_2`。
- 本地和服务器使用同一 Git commit；训练或 GPU 导出只在服务器执行。
- 在 S12-M2 完成前不批准任何新的长训练。

## S12-M0：冻结 Stage1.1 证据

状态：已完成。正式清单位于服务器
`stage1_2/manifests/stage1_1_closure.json`，可读报告同步为
`docs/fairjob/stage1_2/stage1_1_closure.md`。

1. 校验六种方法乘三个种子的 checkpoint、prediction、hparams、metrics、run manifest 和 success marker 完整。
2. 从原始 run manifest 恢复 M5A 训练 commit、seed、config hash 和运行环境，不进行人工猜测。
3. 记录数据 manifest、`test_meta.csv`、冻结 probe sample、Stage1.1 报告及所有 M5A 产物的 SHA-256。
4. 固化 Stage1.1 结论：M3 directional pass、M4 conditional pass、M5B failed、selective suppression 未获 Stage2 批准。

退出门槛：18 组产物完整、测试预测均为 214,446 行、训练版本唯一、冻结清单可重复生成。

## S12-M1：干预后表征导出

### M1A 导出链路验证

状态：已完成。`baseline_seed2019` 的 2,048 行 test smoke 已通过，详见
`docs/fairjob/stage1_2/post_intervention_export_smoke.md`。

1. 直接加载 M5A checkpoint，不重新训练。
2. 导出 `embedding_flat`、`cross_layer_0/1/2`、`dcnv2_final_pre_mitigation`、`dcnv2_final`、`dcnv2_suppressed_component`、`dcnv2_residual_component`、`dcnv2_logit` 和 `dcnv2_probability`。
3. 在 manifest 中记录方法、训练 seed、mask 维度、被抑制维度、索引和 mask hash。
4. 用原始预测 CSV 按 `row_id` 对齐复核测试概率，要求最大绝对误差不超过 `1e-6`。

退出门槛：单 checkpoint 的表征有限值、行号递增、组件满足 `pre = suppressed + residual`、预测一致性通过。

### M1B 完整矩阵导出

状态：等待 M1A 通过后执行。

对 18 个 checkpoint 使用同一抽样协议导出 train/valid/test 表征。每个 split 最多 200,000 行，抽样基准 seed 为 2019，split 偏移规则沿用 Stage1.1；完整导出在服务器前台串行运行并支持 `--resume`。

退出门槛：18 组表征的层名、维度、抽样 `row_id` 和元信息一致，且每组测试概率复核通过。

## S12-M2：冻结 Probe 与泄漏迁移审计

状态：等待 M1B。

1. 复用冻结 probe 行集合，禁止按方法重新抽样。
2. 每个方法、seed、层分别运行线性 probe 和等容量非线性 probe。
3. 使用 train-only fit、valid-only 选择与早停，test 仅做最终评价。
4. 对非线性 probe 执行 100/300/500 预算或等价早停审计，报告收敛状态而非只报告单点 AUC。
5. 计算相对同 seed baseline 的 leakage 变化，并判定 `reduced`、`not_reduced`、`redistributed`、`collapsed` 或 `inconclusive`。
6. 直接比较 selective suppression 与 matched-random suppression，不能省略不利结果。

退出门槛：回答每种方法是否真正降低目标泄漏、是否迁移到相邻层或输出，以及 selective 是否优于随机对照。

## S12-M3：失败归因

### Tier 1：不训练诊断

状态：等待 M2。

使用现有预测和导出表征检查 prediction/logit 均值、标准差、分位数、熵、动态范围、正负样本分数间隔、AUC、NLLH、Brier、ECE 和 DP 的联合变化。优先判断 adversarial 的低 DP 是否来自预测坍缩，并用 validation-only temperature scaling、beta calibration 或 isotonic calibration 判断 NLLH/ECE 改善是否可由后处理复现。

### Tier 2：条件性轻量训练

默认不执行。只有 M2 证明 leakage 稳定下降且 Tier 1 无法解释结果时，才允许在预先固定的小矩阵中比较 L2、dropout 或容量匹配对照；不得使用 test 指标选参数。

退出门槛：对 global、random、selective 和 adversarial 分别给出泄漏、迁移、坍缩、通用正则化/校准解释及剩余机制效应。

## S12-M4：Outcome 地板与稳定性

状态：等待 M2/M3。

1. 对 `all_logged`、`random_display`、`context_conditioned` 报告 DP signed/abs、两组预测均值与组样本数。
2. 报告 group-wise AUC/NLLH/Brier/ECE、worst-group 指标、组间 gap、分数分布距离、U 和 U_TILDE。
3. 复用 impression-cluster bootstrap 和代理 flip/missingness sensitivity，量化方法效应相对 seed 波动与区间宽度。
4. 明确把 DP 判定为 `DP_informative_primary_outcome`、`DP_secondary_high_variance_metric` 或 `DP_near_floor_for_current_setting` 三者之一。

退出门槛：说明三种可用协议是否方向一致，以及 DP 是否仍足以承载主要方法结论。

## S12-M5：唯一 Stage2 方向

状态：等待 M2-M4。

严格按照 Stage1.2 文档中的触发条件，从以下方向中只选择一个：

- A：重新设计多层/多路径 leakage intervention；
- B：转向 leakage 与 outcome fairness 脱钩审计；
- C：研究代理不确定性下的组间校准；
- D：研究公平测量与可重复性；
- E：仅在严格恢复门槛全部满足时恢复方法路线。

最终报告必须列出被选择方向的证据、未选择方向缺失的证据、下一阶段冻结矩阵和不能作出的因果或人口属性声明。

## 当前执行顺序

```text
M0 正式冻结清单
-> M1A 单 checkpoint smoke
-> M1B 18 组表征导出
-> M2 冻结 probe 与迁移审计
-> M3 Tier 1 无训练归因
-> 条件判断是否批准 M3 Tier 2
-> M4 outcome 稳定性
-> M5 唯一方向冻结
```

只有 M1B 属于当前已知的较长 GPU/I/O 操作；它不更新模型权重。任何新的训练需求都必须在 M2 结果形成后重新检查门槛。
