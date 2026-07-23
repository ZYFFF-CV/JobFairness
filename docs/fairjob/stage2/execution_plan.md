# Stage2-A 执行计划：Representation-Graph Proxy Containment

## 1. 当前状态

- Pro 评审状态：`major_revision_required_before_pilot`。
- Stage2-A 状态：`conditionally_approved_as_one_falsifiable_pilot`。
- 当前允许事项：协议冻结、真实 forward graph 审计、代码开发、单种子机制
  smoke 和离线评价链路验证。
- 当前禁止事项：三种子长训练、五种子扩展、第二 backbone、logging-policy
  模型、position-corrected 声明和 test-driven sweep。
- Stage1.1/Stage1.2 的冻结证据与判定保持不变；本计划只替换
  `stage1_2_decision_report.md` 中尚未执行的 provisional Stage2 设计。

本阶段研究对象统一称为 `behavioral protected proxy`。研究目标不是证明
表示中不存在敏感信息，而是在预注册 probe family 范围内控制
representation-graph proxy decodability redistribution。

## 2. 代码审计得到的真实 DCNv2 图

当前 `FairJobDCNv2` 使用 FuxiCTR 原生 parallel DCNv2：

```text
embedding_flat
├── CrossNetV2
│   ├── cross_layer_0
│   ├── cross_layer_1
│   └── cross_layer_2 = cross_path_output
└── parallel DNN
    ├── parallel_dnn_linear_0
    └── parallel_dnn_linear_1 = parallel_dnn_path_output

[cross_path_output, parallel_dnn_path_output]
→ fusion_pre_logit
→ linear output
→ logit
→ probability
```

`dcnv2_suppressed_component` 和 `dcnv2_residual_component` 是 Stage1.1
mitigation 对 final vector 的代数分解，不是原生 DCNv2 branch。它们只用于
旧方法审计和对照，不进入 Stage2 原生表示图定义。

Stage2 的结构化 gate 优先作用于 cross block 增量、cross branch 输出、
parallel DNN block/branch 输出和 fusion 前分支。`embedding_flat` 首先作为
migration sentinel；只有其相对同 seed baseline 出现材料性上升时，训练约束
才允许增强。

## 3. 冻结的科学问题

Stage2-A 回答：

> 局部干预为何会使行为保护代理可解码性沿 CTR 表示图的层、分支和联合路径
> 重新分配，以及如何在保持 CTR utility 的前提下控制最危险节点和路径？

预期贡献按以下顺序建立：

1. 现象：干预诱发的 node/path/joint-path decodability redistribution。
2. 测量：probe-family、user-disjoint、joint-path 和 collapse-aware 审计。
3. 方法：migration-aware graph containment。

只有 pilot full pass 后，第三项才能作为成功的方法贡献。无论方法是否成功，
前两项审计贡献都保留。

## 4. S2-P0：协议与表示图冻结

状态：`in_progress`。本阶段不训练模型。

### P0.1 文档与证据边界

1. 归档 Pro 审核意见，不修改 Stage1.1/1.2 冻结结论。
2. 在 Stage2 配置中记录 source commit、Stage1.2 M2-M5 报告 hash。
3. 固定术语和 claim boundary：
   - 不使用 true gender、gender discrimination removal 或 causal fairness；
   - 不声称 complete information removal；
   - position correction 保持
     `unavailable_missing_external_propensity`。

交付物：

```text
docs/fairjob/stage2/stage2_review_opinion.md
docs/fairjob/stage2/execution_plan.md
docs/fairjob/stage2/representation_graph_audit.md
```

### P0.2 Forward graph 审计

1. 对原生 `FairJobDCNv2` 和旧 `FairJobMitigatedDCNv2` 分别导出节点、维度、
   父子依赖和输出别名。
2. 验证 cross、parallel DNN、fusion 和 output 路径均来自真实 forward，
   不把 mitigation 组件误记为原生 branch。
3. 冻结三类节点：
   - target：cross blocks、cross output、DNN blocks/output、fusion；
   - sentinel：embedding、logit、probability；
   - legacy control：suppressed/residual components。
4. 冻结 joint representations：
   - `cross_output + dnn_output`；
   - cross block increments；
   - branch outputs + fusion；
   - legacy suppression/residual 拼接，仅用于旧方法审计。
5. 验证 gate 构成输入到 logit 主要路径的结构化切集，无未声明 bypass。

交付物：

```text
configs/fairjob/stage2_representation_graph.yaml
fuxictr_ext/fairjob/representation_graph.py
tests/fairjob/test_representation_graph.py
```

退出门槛：所有节点可以在同一次 forward 中稳定导出；维度和依赖可由测试验证；
baseline prediction 与原生 DCNv2 在 `1e-6` 内一致。

### P0.3 Probe 与数据划分冻结

1. 保留 Stage1.2 row-disjoint probe，保证历史证据连续。
2. 增加基于原始 `user_id` 的 user-disjoint train/valid/test probe split。
3. 审计用户跨 row split 的重叠、低频/未见用户覆盖和 proxy 组分布。
4. 在元数据允许时设计 temporal/block probe；若缺少可靠时间字段，明确记为
   unavailable，不根据行顺序冒充 temporal split。
5. 训练 adversary 与评价 probe 解耦。评价固定包含：
   - linear；
   - Stage1.2 matched-capacity nonlinear；
   - 架构不同的 bounded evaluator；
   - joint-path probe；
   - row-disjoint 和 user-disjoint 两种协议。
6. AUC 方向、early stopping 和 probe capacity 只由 validation 决定。

交付物：

```text
configs/fairjob/stage2_probe_protocol.yaml
fuxictr_ext/fairjob/user_disjoint_probe.py
fuxictr_ext/fairjob/joint_path_probe.py
tests/fairjob/test_user_disjoint_probe.py
tests/fairjob/test_joint_path_probe.py
```

阻塞条件：无法从处理后数据或外部原始数据稳定恢复 `row_id -> user_id` 对齐时，
必须停止 user-disjoint 实现并提交数据审计结果，不允许用 tokenizer ID 代替。

### P0.4 训练目标冻结

首个 falsification pilot 使用平滑最坏节点风险作为主目标：

```text
CTR loss
+ structured gate regularization
+ weak ranking/logit preservation
+ smooth-max(node risks + joint-path risk)
```

普通 layer-risk sum 作为必要消融。Adaptive dual 与 CVaR 接口先实现为独立
risk aggregator，但不在首次 pilot 中同时开放参数搜索；只有主 pilot 通过后，
才进入正式 objective 消融。

约束原则：

- 各训练 adversary 使用 balanced proxy loss；
- embedding 默认只监控，材料性迁移时才提高其约束；
- 不逐样本强制复制 baseline probability；
- 不把 post-hoc probe AUC 直接当作可微训练损失；
- 推理阶段删除 adversary，且不输入 protected proxy。

交付物：

```text
fuxictr_ext/fairjob/multinode_adversary.py
fuxictr_ext/fairjob/adaptive_constraints.py
tests/fairjob/test_multinode_adversary.py
tests/fairjob/test_adaptive_constraints.py
```

### P0.5 Structured gate 冻结

1. Gate 单位限定为 branch/block/interaction group，不使用任意 hidden-unit
   index 作为主方法。
2. 首版采用固定预算的 scalar/group gate，保证跨 seed 语义一致。
3. 冻结 gate sparsity budget、初始化、温度和 inference 行为。
4. 实现相同预算的 structured-random gate control。
5. 旧 selective hidden-unit mask 只作为冻结失败参考，不继续 sweep。

交付物：

```text
configs/fairjob/stage2_path_gates.yaml
fuxictr_ext/fairjob/structured_path_gates.py
tests/fairjob/test_structured_path_gates.py
```

### P0.6 Pilot gate 冻结

Full pass 必须同时满足：

- 每个主要 target 的最大 held-out probe AUC 平均下降至少 `0.005`，且
  `3/3` seed 同方向；
- joint target 平均下降至少 `0.005`，其余 joint path 不上升；
- 任一非目标节点平均增量小于 `0.005`，且同一节点不能有 `2/3` seed
  增量达到 `0.005`；
- `material_migration_count = 0`；
- paired AUC 平均变化不低于 `-0.005`，任一 seed 不低于 `-0.010`；
- group-blind calibration 后 NLLH 平均增量不超过 `0.001`，任一 seed
  不超过 `0.002`；
- calibrated Brier 的平均/单 seed screening margin 初始上限分别为
  `1e-4`/`2e-4`，最终值须在训练前用 baseline validation 波动冻结；
- class gap 平均至少保留 baseline 的 `80%`；单 seed catastrophic floor
  须在训练前由 baseline validation 分布冻结；
- validation paired ranking 不材料性恶化；
- 任一 seed 触发 AUC/class-gap/calibrated-loss collapse 即 fail；
- 完整方法优于 final-only、multi-layer-no-gate、structured-random 和
  matched-capacity regularization。

Score variance 和 dynamic range 仅作双侧告警，不单独决定通过。
DP 保留为 secondary report，不参与 pilot pass 的必要充分判定。

Partial pass 只允许一次预先限定的机制修复，不自动进入五种子；Fail 不允许
通过放宽门槛或扩大搜索恢复。

交付物：

```text
configs/fairjob/stage2_pilot_gate.yaml
fuxictr_ext/fairjob/stage2_pilot_gate.py
tests/fairjob/test_stage2_pilot_gate.py
```

## 5. S2-P1：代码实现与单种子机制 Smoke

状态：等待 P0 完成。只允许短时服务器运行。

### P1.1 模型实现

新增独立的 `FairJobGraphContainmentDCNv2`，不修改 FuxiCTR 原始
`model_zoo/DCNv2`。模型复用原生 embedding、CrossNetV2、parallel DNN 和
output layer，只在 FairJob 扩展层中增加：

- named graph-node forward；
- structured branch/block gates；
- node adversaries；
- joint-path adversary；
- smooth-max risk aggregator；
- ranking/logit preservation；
- anti-collapse telemetry。

旧 `FairJobMitigatedDCNv2` 保持冻结，不重构其历史行为。

### P1.2 单元与合成测试

1. Baseline mode 与原生 DCNv2 prediction parity。
2. Gate 关闭/全开边界和梯度传播。
3. Joint representation 的维度、顺序和 hash 稳定。
4. Training adversary、encoder 和 gate 的梯度方向正确。
5. Smooth-max 在单节点、同风险和极端风险下数值稳定。
6. Proxy 不进入 CTR feature tensor，推理接口不要求 proxy。
7. Loss/telemetry 在单组缺失、极端类别不平衡和空 senior scope 下可控。
8. Checkpoint 保存/加载后 gate、adversary 和 dual/risk state 一致。

### P1.3 Seed-2019 机制 smoke

使用 smoke 数据和短 epoch，只检查：

- forward/反向传播无 NaN；
- 4090 显存可容纳 node + joint adversaries；
- 各节点 adversary 有有限 loss 且可以学习；
- gate 形成预期切集；
- smooth-max 不长期只由数值尺度最大的一个 loss 锁死；
- representation export 不改变 prediction；
- 无立即 AUC collapse、class-gap collapse 或 score explosion。

Smoke 不读取 test 指标做结构选择。任何结构修改都必须回到 P0 更新配置 hash。

退出门槛：代码测试通过、单种子短程数值稳定、manifest 完整。通过后才提交
三种子 pilot 训练批准。

## 6. S2-P2：冻结的三种子 Validation Pilot

状态：未批准，等待 P1。

初始矩阵使用 DCNv2 no-user-id、seeds `2019/2020/2021`：

| 条件 | 目的 |
|---|---|
| `baseline` | CTR reference |
| `current_selective_suppression` | 冻结失败局部方法 |
| `final_only_no_gate` | final adversarial baseline |
| `multi_layer_no_gate` | multi-layer 主效应 |
| `final_only_path_gate` | structured path gate 主效应 |
| `multi_layer_path_gate_full` | 完整方法 |
| `full_without_joint_adversary` | joint containment 消融 |
| `layer_risk_sum` | smooth-max objective 对照 |
| `structured_random_gate` | 结构稀疏/容量对照 |
| `matched_capacity_regularization` | 普通正则化对照 |
| `calibration_only` | 离线评价对照，不训练新 CTR |

这是 2×2 multi-layer × structured-gate 因子设计加必要控制。所有配置、预算、
early stopping、calibration candidate set 和 gate threshold 在运行前冻结。

训练在服务器前台运行，loss、node risks、joint risk、gate 值、AUC 和
anti-collapse telemetry 实时可见。Checkpoint 存放于：

```text
/root/autodl-tmp/workdirs/JobFairness/stage2/checkpoints/
```

代码、数据和其他产物继续遵循：

```text
/root/autodl-tmp/code/JobFairness
/root/autodl-tmp/datasets/FairJob
/root/autodl-tmp/workdirs/JobFairness/stage2
```

## 7. S2-P3：独立 Containment 审计

状态：等待 P2。

1. 对全部 pilot checkpoint 导出相同 row IDs 的 train/valid/test 表征。
2. 运行 linear、matched nonlinear、independent evaluator 和 joint probes。
3. 同时报告 row-disjoint 与 user-disjoint 结果。
4. 计算每个 node/joint path 相对同 seed baseline 的：
   - probe-family max AUC；
   - delta；
   - migration type；
   - migration count；
   - 最大迁移幅度。
5. 统一运行 validation-only group-blind calibration。
6. 联合审计 AUC、class gap、calibrated NLLH/Brier、ranking、score/logit
   range、worst-group quality、U/U_TILDE 和 secondary DP。
7. 由机器可读 gate 只输出 `full_pass`、`partial_pass` 或 `fail`。
8. 报告训练 adversary 与独立 post-hoc probe 的能力差，禁止用训练 adversary
   loss 代替 containment 证据。

交付物：

```text
fuxictr_ext/fairjob/stage2_containment_audit.py
fuxictr_ext/fairjob/stage2_collapse_audit.py
docs/fairjob/stage2/pilot_report.md
```

## 8. S2-P4：一次性决策

### A1：Full pass

批准五种子 DCNv2、paired intervals、anti-collapse/objective 正式消融，以及
第二 backbone 或受控半合成验证。

### A2：Clean containment，但 outcome 不改善

停止把 outcome 改善作为方法成功前提，正式转向
leakage-outcome decoupling/process-fairness measurement audit。

### A3：仍有迁移或只欺骗训练 adversary

停止继续增加 adversary/gate，不进行第二轮无边界方法搜索；转向局部去敏的
迁移失败模式、joint encoding、probe dependence 和不可识别性审计。

### A4：Utility collapse

方法路线 fail。不得放宽 AUC/NLLH/class-gap 门槛恢复。

## 9. 执行顺序

```text
P0.1 归档与声明边界
→ P0.2 真实 forward graph 审计
→ P0.3 row/user-disjoint probe 协议
→ P0.4 objective 与 adversary 协议
→ P0.5 structured gate 与 random control
→ P0.6 machine-readable pilot gate
→ P1 实现、测试、seed-2019 smoke
→ 请求一次三种子长训练批准
→ P2 冻结矩阵训练
→ P3 独立 containment 审计
→ P4 A1/A2/A3/A4 一次性决策
```

## 10. 当前停止点

当前只开始 P0，不启动训练。以下任一情况需要停下并提交决策：

- 实际 forward graph 存在未覆盖 bypass；
- 无法可靠构造 user-disjoint split；
- baseline validation 无法支持冻结 Brier、class-gap 或 ranking margin；
- joint adversary 导致 4090 显存不可接受且无法通过预注册的顺序更新解决；
- structured gate 无法在不同 seed 间保持结构语义；
- 需要新增数据、可靠时间戳或外部 propensity；
- P1 smoke 出现持续 NaN、梯度失控或立即 utility collapse；
- 开始 P2 三种子长训练前。
