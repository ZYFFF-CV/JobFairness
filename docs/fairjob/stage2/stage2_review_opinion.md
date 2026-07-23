# Stage2 方案评审意见：多层、多路径行为保护代理可解码性 containment

**评审日期：** 2026-07-23  
**项目：** FairJob × FuxiCTR CTR Fairness Research  
**评审对象：** Stage2-A — multi-layer/multi-path leakage intervention  
**评审性质：** 方案评审，不重新执行实验，不重新推翻 Stage1.1/Stage1.2 已冻结证据  
**评审状态：** `major_revision_required_before_pilot`  
**方向意见：** `A_continue_as_one_falsifiable_validation_only_pilot`  
**长训练状态：** `not_approved_before_revised_pilot_protocol_is_frozen`  
**位置校正状态：** `position_corrected = unavailable_missing_external_propensity`  

---

## 0. 执行摘要

Stage1.2 已完成失败归因，并按预先冻结的触发规则选择方向 A：重新设计多层、多路径行为保护代理泄漏干预。本评审不重新判断该选择是否符合 Stage1.2 规则，而只判断当前 Stage2-A 方案是否具有足够的科学合理性、可证伪性和期刊创新潜力。

评审结论如下：

1. **方向 A 可以继续，但只能批准一次严格受限、validation-only 的可证伪 pilot。** 当前证据尚不足以批准五种子、第二 backbone 或大规模权重搜索。
2. **“多层 adversary + 多个 gate”本身不足以形成 ACM TOIS 级 novelty。** 更有价值的创新对象是：局部公平干预导致行为保护代理可解码性在 CTR 表示图中发生层间、路径间和联合路径迁移，以及如何进行 graph-level containment。
3. **损失函数应从固定加权的局部 adversarial loss 求和，升级为表示图上的最坏节点/尾部风险约束与联合路径约束。** 推荐使用自适应对偶变量或平滑 max/CVaR，而非为每层人工搜索多个权重。
4. **gate 必须对应稳定、可解释的架构路径。** 若仍然作用于任意隐藏维度，则存在坐标旋转、跨种子不可比和“单元并非路径”的不可识别性问题。
5. **独立 layer-wise adversary 不足以排除联合编码。** 必须增加 cross path、parallel DNN path、融合前表示和必要组件拼接后的 joint-path adversary，并使用独立 post-hoc probe 家族评价。
6. **当前 score variance 下限不能识别已观察到的 adversarial collapse。** 已冻结结果中，adversarial 的 score std 是 baseline 的约 2.84 倍，但 AUC 从约 0.765 降至约 0.537，class gap 只剩 baseline 的约 28%。因此 variance 只能作为辅助告警，不能作为核心 anti-collapse gate。
7. **pilot 门槛应收紧并改成配对非劣标准。** 建议三种子平均 AUC 损失不超过 0.005、任一种子不超过 0.010；统一 group-blind calibration 后，平均 NLLH 增量不超过 0.001、任一种子不超过 0.002；目标节点最大 held-out probe AUC 至少下降 0.005，并且任何非目标节点或联合路径不得出现材料性迁移。
8. **Stage2 论文战略应同时保留“测量审计”贡献。** 当前已确认的 leakage redistribution、probe-family disagreement 和 fairness-by-collapse 本身可以构成论文的第一项实证贡献；只有 graph-level containment pilot 成功后，方法贡献才成立。
9. **`protected_attribute` 只能被称为行为保护代理。** 不能写成真实性别、真实人口属性、性别歧视或反事实公平，也不能把代理可解码性下降直接解释为现实招聘公平改善。

最终评审意见：

> **Stage2-A 有条件继续，当前方案需 major revision。其科学定位应从“在多个层增加 adversary”重构为“针对干预诱发的表示图代理可解码性迁移，进行结构化、多路径、最坏节点约束的 containment”。修订协议冻结前，不批准新的长训练。**

---

## 1. 评审边界

### 1.1 本评审不做的事情

本评审严格限定为方案审查，不做以下工作：

- 不重新质疑 Stage1.1 的 M3 directional pass；
- 不重新质疑 Stage1.1 的 M5B failed；
- 不重新运行或重新挑选 Stage1.2 probe；
- 不重新解释已经冻结的 checkpoint、prediction、representation 和 hash；
- 不根据本评审建议事后修改 Stage1.1/1.2 的判定门槛；
- 不把尚未执行的 Stage2 设计描述成已经取得的实验结果；
- 不批准 logging-policy 模型或从点击结果估计 propensity；
- 不恢复 `position_corrected` 结论；
- 不以 test AUC、test NLLH、test DP 或 test leakage 选择 Stage2 超参数。

### 1.2 评审所接受的冻结事实

以下内容作为本评审的既定前提：

| 冻结事实 | 当前状态 |
|---|---|
| Stage1.1 M3 | 小幅、跨 backbone、跨三种子方向一致的行为保护代理增量可解码性候选信号；无 user ID 后仍存在。 |
| Stage1.1 M5B | 当前 global、random、selective、DP regularization 和 adversarial 干预均未形成稳定公平—效用 Pareto 改善。 |
| Stage1.2 表征导出 | 18 个 checkpoint × 3 split 完成统一导出；冻结 row IDs、预测重算和组件重构均通过。 |
| Stage1.2 probe | 180 个 checkpoint-layer 组合、360 个线性/非线性结果全部收敛。 |
| Stage1.2 leakage 结论 | global、selective、adversarial 被分类为 `redistributed`；DP regularization 为 `not_reduced`；matched-random 为 probe-family disagreement。 |
| Stage1.2 collapse 结论 | adversarial 在 3/3 种子上触发 fairness-by-collapse。 |
| Stage1.2 calibration 结论 | suppression 的 raw NLLH/ECE 表面收益可由 validation-only、group-blind calibration 复现。 |
| Stage1.2 outcome 结论 | 当前 DCNv2 no-user-id 设置中，DP 为 `DP_near_floor_for_current_setting`，保留报告但不作为唯一或主要优化目标。 |
| Stage1.2 方向 | 方向 A 被唯一选择；新长训练必须先通过 Stage2-A pilot gate。 |
| Position correction | 因缺少外部 logging propensity，继续标记为 unavailable。 |

### 1.3 关键冻结数值

以下数值直接影响 Stage2 pilot 设计：

- baseline AUC：约 `0.764927`；
- adversarial AUC：约 `0.536575`；
- baseline calibrated NLLH：约 `0.038126`；
- adversarial calibrated NLLH：约 `0.041346`；
- baseline score std：约 `0.015611`；
- adversarial score std：约 `0.044335`；
- baseline class gap：约 `0.015840`；
- adversarial class gap：约 `0.004418`；
- baseline all-logged DP abs：`0.000510 ± 0.000560`；
- selective final leakage delta：linear `-0.005310`，nonlinear `-0.007289`；
- selective migration：`cross_layer_2` nonlinear `+0.006436`；
- global migration：`cross_layer_2` linear `+0.005396`；
- adversarial migration：embedding linear `+0.012176`、nonlinear `+0.024281`。

这些结果说明 Stage2 的首要问题不是“如何再压低一点 final-layer probe AUC”，而是：

> **如何避免代理可解码性从被约束节点迁移到未约束层、分支或联合表示，同时避免通过预测坍缩获得表面改善。**

---

## 2. 总体评审表

| 评审维度 | 评审意见 | 当前风险 | 启动 pilot 前要求 |
|---|---|---|---|
| 科学问题 | 有价值 | 若只写成多层 adversary，则问题过于常规 | 改写为 representation-graph leakage redistribution and containment |
| 期刊创新度 | 当前不足，重构后有潜力 | 组件堆叠容易被认为是工程组合 | 建立迁移定义、图级目标、联合路径约束及因子消融 |
| 损失函数 | 基本方向合理，但形式不足 | 固定求和允许层间此消彼长 | 使用平滑 max/CVaR 或约束式 adaptive dual |
| 路径 gate | 必须重设计 | 任意 hidden-unit gate 不等于 path gate | 使用 branch/block/interaction-group gate，构成关键路径切集 |
| Layer-wise adversary | 必要但不充分 | 可欺骗训练 adversary；无法识别联合编码 | 加 joint-path adversary 与独立 post-hoc probe |
| Anti-collapse | 当前 score variance gate 不科学 | 高方差无信息预测仍能通过 | 用 AUC、class gap、calibrated NLLH/Brier 和动态范围联合判定 |
| Pilot AUC 门槛 | 当前过宽 | `-0.01` 允许约 2.5 个 baseline seed std 的损失 | 平均 `ΔAUC ≥ -0.005`；任一种子 `≥ -0.010` |
| Pilot NLLH 门槛 | raw NLLH 不合适 | 容易把概率尺度或校准变化当作方法收益 | 统一 group-blind calibration 后判定 |
| Score variance | 仅可辅助 | 当前 adversarial collapse 反而方差更大 | 设为双侧告警，不单独决定 pass/fail |
| 对照与消融 | 起点合理但不充分 | 无法分离多层、多路径及其交互贡献 | 2×2 因子设计 + joint adversary + structured random gate |
| 不可识别性 | 中高 | probe family、user overlap、hidden-coordinate gate | user-disjoint probe、独立强 probe、结构化 gate |
| 方向 A vs 审计 | A 有条件继续 | 纯方法承诺过早 | 一次 falsification pilot；失败后停止方法扩张 |
| 声明边界 | 必须严格 | 行为代理被误写成真实性别 | 全文使用 behavioral protected proxy 和 proxy-conditioned language |

---

## 3. 多层、多路径 containment 的创新度评审

### 3.1 当前组件组合不足以单独构成 TOIS 级创新

当前冻结方案包含：

- embedding、cross0/1/2、final 的 train-only proxy adversaries；
- cross path 和 residual path gates；
- maximum layer-wise excess leakage；
- prediction-preservation；
- score-variance constraint。

这些单独组件分别已有成熟先例：对抗公平表示、表示分布对齐、结构稀疏 gate、任务保持和防坍缩正则都不是新的研究对象。因此，不能把创新主张写成：

> “我们首次在 CTR 多个层增加 adversary，并为不同路径增加 gate。”

即使工程实现完整，这种表述仍容易被认为是既有技术的组合。

### 3.2 真正具有潜力的创新对象：干预诱发的 leakage redistribution

Stage1.2 已确认：

- final-layer decodability 降低时，上游 cross 层可能升高；
- selective local gate 有局部效果，但 pre-mitigation final 或后期 cross 层出现迁移；
- adversarial 降低 final leakage 的同时，embedding leakage 上升；
- 没有任何现有干预实现 clean path-wide reduction。

因此，Stage2 应把研究问题定义为：

> **局部公平干预为什么会使行为保护代理的可解码性沿 CTR 表示图重新分配，以及如何在保留 CTR utility 的前提下，对最危险节点、分支和联合路径实施图级 containment。**

这一定位包含三个比“多层 adversary”更强的贡献层次：

1. **现象贡献：** 系统描述局部干预诱发的层间与路径间代理可解码性迁移；
2. **测量贡献：** 建立 node、path、joint-path 与 collapse-aware 的迁移审计协议；
3. **方法贡献：** 提出 graph-aware、migration-aware、anti-collapse 的 containment 方法。

### 3.3 建议使用“可解码性/可提取性”，谨慎使用“信息放大”

在确定性网络中，表示是输入的函数。严格的信息论意义下，网络不能凭空产生超过输入所含的 Shannon 信息。当前 probe 结果支持的是：

> 某些表示使行为保护代理对指定 probe family 更容易解码或提取。

因此，论文建议使用：

```text
proxy decodability amplification
proxy extractability amplification
representation-graph leakage redistribution
probe-family guardedness
```

谨慎使用：

```text
sensitive information creation
true information amplification
complete information removal
```

### 3.4 达到期刊创新度的最低要求

要使该方向具备 TOIS 级潜力，至少需要满足：

- 给出表示图、节点、路径、迁移和 containment 的明确形式化；
- 方法结构与 Stage1.2 观察到的迁移机制逐项对应；
- 独立评价联合路径，而不是只看每个层的边际 probe；
- 使用结构化 gate，而非任意 hidden-unit mask；
- 证明 graph-aware objective 优于普通 layer-loss sum；
- 证明 multi-layer 与 multi-path 贡献可以分别识别；
- 证明收益不是普通正则化、校准或容量削减；
- 证明方法不是通过 AUC collapse 或 score degeneration 获得表面公平数字；
- 至少完成五种子 DCNv2 与一个第二 backbone 或受控半合成机制验证；
- 明确限制所有结论只针对行为保护代理和指定日志协议。

---

## 4. Stage2 科学问题与形式化建议

### 4.1 表示图

将 CTR backbone 表示为有向计算图：

\[
G=(V,E),
\]

其中节点集合至少覆盖当前实际可导出的关键表示：

```text
embedding_flat
cross_layer_0
cross_layer_1
cross_layer_2
parallel_dnn_linear_0
parallel_dnn_linear_1
cross_path_output / cross component
parallel_dnn_path_output
final_pre_mitigation
suppressed_component
residual_component
final
logit
probability
```

具体节点名称必须由代码审计确认，不得为了论文叙事虚构不存在的 branch。若某一 component 只是当前 mitigation 模块中的代数分解，应明确标注，而不能当作 DCNv2 原生路径。

### 4.2 节点可解码性

对节点 \(v\) 的表示 \(H_v\)，定义在冻结 probe 协议下的行为代理可解码性：

\[
D_v(Q)=\max_{q\in Q}\operatorname{AUC}(q(H_v),A),
\]

其中：

- \(A\) 是行为保护代理；
- \(Q\) 是预注册的 held-out probe family；
- 至少包含 linear probe、matched-capacity nonlinear probe 和一个与训练 adversary 不同的独立 evaluator；
- AUC 方向仅由 validation 决定；
- test 只用于最终读取。

### 4.3 联合路径可解码性

单层边际不可解码不代表多路径联合不可解码。定义：

\[
D_J(Q)=\max_{q\in Q}\operatorname{AUC}
\left(q([H_{p_1},H_{p_2},\ldots,H_{p_m}]),A\right),
\]

其中拼接对象至少包括：

- cross path output；
- parallel DNN path output；
- mitigation residual/suppressed components；
- 融合前表示；
- 必要时相邻层增量表示。

### 4.4 迁移定义

相对于同 seed baseline，定义节点变化：

\[
\Delta D_v=D_v^{method}-D_v^{baseline}.
\]

建议将迁移分类冻结为：

| 类型 | 定义 |
|---|---|
| Local reduction | 目标节点 \(\Delta D_v\leq-\delta\) |
| Upstream migration | 上游节点 \(\Delta D_u\geq+\delta\) |
| Downstream migration | 后续节点或 logit/probability \(\Delta D_w\geq+\delta\) |
| Cross-path migration | 未直接约束的并行路径 \(\Delta D_p\geq+\delta\) |
| Joint-only migration | 各边际节点变化不大，但 \(\Delta D_J\geq+\delta\) |
| Probe disagreement | 不同 probe family 对同一节点给出相反材料性方向 |
| Clean containment | 所有目标节点下降，所有非目标与 joint path 无材料性上升，且 utility 非劣 |

Stage1.2 使用的 exploratory material threshold \(\delta=0.005\) 可以继续作为 pilot 筛选阈值，但必须明确它是预注册的工程阈值，不是普遍统计显著性界限。

### 4.5 Stage2 目标

Stage2-A 不应优化“平均层 leakage”，而应控制：

1. 最大节点可解码性；
2. 联合路径可解码性；
3. 迁移计数和迁移幅度；
4. CTR ranking utility；
5. calibrated probability quality；
6. collapse safety。

DP 保持报告，但在当前设置中为 secondary outcome，不作为唯一 pilot 成功标准。

---

## 5. 损失函数评审

### 5.1 固定加权求和的问题

简单目标：

\[
L=L_{CTR}-\sum_v\lambda_v L_{adv,v}
+\lambda_g\Omega(g)
+\lambda_kL_{keep}
\]

存在以下缺陷：

- 不同层维度、类别难度和 adversary capacity 不同，loss scale 不可直接比较；
- 求和允许一个层显著下降、另一个层升高，恰好重现 Stage1.2 的 migration；
- 多个 \(\lambda_v\) 容易演化为大规模权重搜索；
- 独立层目标不能排除 joint-path 协同编码；
- 训练 adversary 的失败不能证明独立 post-hoc probe 失败。

因此，不建议将“若干 adversarial loss 相加”作为最终方法的核心形式。

### 5.2 推荐的约束式目标

设 CTR 参数为 \(\theta\)，结构化 gates 为 \(g\)，训练 adversaries 为 \(\phi_v\) 和 \(\phi_J\)。建议主问题写为：

\[
\min_{\theta,g}
L_{CTR}(\theta,g)
+\lambda_{gate}\Omega(g)
+\lambda_{rank}L_{rank\text{-}preserve}
\]

满足：

\[
R_v(\theta,g)\leq\epsilon_v,
\quad v\in V_c,
\]

\[
R_J(\theta,g)\leq\epsilon_J,
\]

以及 validation-level utility non-inferiority constraints。

这里 \(R_v\) 可用平衡 proxy BCE advantage、pairwise surrogate 或其他可微依赖指标近似。最终评价仍使用独立 probe AUC，不把 AUC 直接当作可微训练损失。

### 5.3 自适应对偶变量

建议使用：

\[
\mu_v\leftarrow
[\mu_v+\eta(R_v-\epsilon_v)]_+,
\]

\[
\mu_J\leftarrow
[\mu_J+\eta(R_J-\epsilon_J)]_+.
\]

其优点是：

- 当前最危险节点自动获得更大约束；
- 已达到阈值的节点不会持续被过度压制；
- 减少人工为每层搜索不同权重；
- 与“contain the worst migrating node”叙事一致。

### 5.4 平滑最大风险

若实现约束式训练成本过高，可以使用平滑最大风险：

\[
R_G=\tau\log\sum_{v\in V_c\cup\{J\}}
\exp(\widetilde R_v/\tau),
\]

或 top-k/CVaR 风险。与普通求和相比，它更强调最危险节点，但比硬 max 稳定。

需要加入以下消融：

```text
sum of layer risks
vs
log-sum-exp max risk
vs
CVaR/top-k risk
```

否则无法证明“migration-aware worst-node objective”本身有贡献。

### 5.5 Task preservation 的边界

不建议强制逐样本完全复制 baseline probability，因为 baseline 本身可能包含代理相关结构。建议优先使用：

- CTR label loss；
- pairwise ranking preservation；
- 弱权重、中心化 logit distillation；
- validation-calibrated probability non-inferiority；
- 必要时只保存局部排序而非绝对分数。

任务保持应防止无信息坍缩，而不是冻结 baseline 的所有行为。

---

## 6. 路径 gate 评审

### 6.1 任意隐藏维度 gate 的不可识别性

如果 gate 作用于 final representation 的任意坐标，则：

- 隐藏空间可以发生等价旋转；
- 同一功能可由不同坐标组合实现；
- 不同种子间 gate index 缺少可比语义；
- “高风险单元”不等于“高风险交互路径”；
- 模型可以把代理相关函数迁移到未被 gate 的坐标。

因此，逐神经元 gate 只能作为弱 baseline，不能支撑“path-level containment”主张。

### 6.2 推荐的结构化 gate 单元

优先 gate 以下结构：

- 每个 cross block 的增量输出；
- cross branch 总输出；
- parallel DNN block 或 DNN branch 输出；
- cross/DNN 融合前的分支；
- 当前 mitigation 的 suppressed/residual components；
- 有明确字段含义时的 field-group interaction blocks。

实现可选：

- scalar branch gate；
- group-wise hard-concrete gate；
- group lasso gate；
- monotonic keep/suppress gate；
- 具有明确稀疏预算的 structured mask。

### 6.3 gate 必须形成关键路径切集

需要通过实际 forward graph 审计，确认从输入到 logit 的每条主要路线至少经过一个受控点。例如：

```text
embedding → cross blocks → cross output → fusion → logit
embedding → parallel DNN → DNN output → fusion → logit
mitigation residual/suppressed components → final → logit
```

若存在未覆盖的 bypass，泄漏迁移不是意外，而是预期结果。

### 6.4 embedding 的处理

Stage1.1 no-user-id 结果显示，embedding 在匹配容量 probe 下低于原始输入，而 cross/final 才出现正向增量信号。因此建议：

- embedding 先作为 migration sentinel；
- cross、fusion、final 和 joint paths 作为主要优化节点；
- 仅当 embedding decodability 相对 baseline 材料性增加时，由 adaptive dual 激活更强约束。

这样可以避免把“阻止交互层提高代理可解码性”悄悄扩大为“删除输入中所有与行为代理有关的信息”。

---

## 7. Layer-wise adversary 与评价 probe 评审

### 7.1 训练 adversary 的合理角色

训练阶段可以使用行为保护代理监督 adversary；推理阶段不得要求该代理。训练 adversary 应：

- 使用 balanced loss 或 class-weighted loss；
- 只在 train/validation 上选择容量和权重；
- 明确每层输入的 stop-gradient/GRL 位置；
- 记录每个 adversary 的 capacity、优化步数和实际收敛状态；
- 避免某一层 adversary 过强而主导 CTR 更新。

### 7.2 独立 layer-wise adversary 不足以证明 containment

必须额外加入 joint adversary：

\[
q_J([H_{cross},H_{dnn},H_{residual},H_{fusion}]).
\]

理由是：

- 每个路径可能只编码代理的一部分；
- 独立 marginal probe 接近随机时，拼接后仍可能恢复代理；
- 多路径模型尤其容易形成 distributed code；
- 只约束各层边际无法排除协同编码。

### 7.3 训练 adversary 与评价 probe 必须解耦

评价至少包含：

1. linear probe；
2. 与 Stage1.2 对齐的 matched-capacity nonlinear probe；
3. 与训练 adversary 架构不同的 bounded evaluator；
4. joint-path probe；
5. user-disjoint probe；
6. 可行时 temporal/block probe。

正式结论只能写：

> 在预注册的 probe families、split 和容量范围内，行为代理不再容易解码。

不能写：

> 表示中不存在行为代理信息。

---

## 8. Anti-collapse 约束评审

### 8.1 当前 score variance 下限不成立

原 pilot 要求：

\[
\sigma_{score}^{method}
\geq0.5\sigma_{score}^{baseline}.
\]

但冻结结果表明：

- baseline score std：`0.015611`；
- adversarial score std：`0.044335`；
- adversarial 是 baseline 的约 `2.84×`；
- adversarial AUC：`0.536575`；
- adversarial class gap：`0.004418`，仅约为 baseline 的 `27.9%`。

所以，高 score variance 可能来自无信息噪声或错误拉伸，并不代表预测仍有判别力。当前 adversarial collapse 会轻松通过“至少保留 50% 方差”的门槛。

### 8.2 推荐的联合 anti-collapse gate

anti-collapse 必须至少同时检查：

1. **AUC 非劣；**
2. **正负样本 class gap；**
3. **统一 calibration 后的 NLLH；**
4. **统一 calibration 后的 Brier；**
5. **pairwise ranking loss 或排序一致性；**
6. **score/logit dynamic range；**
7. **score variance 双侧告警；**
8. **每种子 catastrophic failure。**

score variance 只作为诊断项：过低可能是收缩，过高可能是无信息膨胀。它不能独立决定通过。

### 8.3 建议的 class-gap pilot 阈值

建议 pilot 中要求：

\[
\frac{Gap_{method}}{Gap_{baseline}}\geq0.80
\]

并且三种子均不低于预注册的 catastrophic floor。

`0.80` 是 screening safety margin，不是理论常数。正式五种子阶段应使用 paired interval 和敏感性分析，不把 80% 写成普遍标准。

### 8.4 Calibration 的统一处理

Stage1.2 已证明 suppression 的 raw NLLH/ECE 改善可由 group-blind calibration 复制。因此 pilot 比较必须：

- 对所有方法使用同一 calibration candidate set；
- 只在 validation 上选择 calibrator；
- 对 test 只读取最终 calibrated metrics；
- raw NLLH/ECE 保留报告，但不用于判定独立公平机制；
- group-aware calibrator 若使用，必须单列，不与 group-blind 主结果混合。

---

## 9. 修订后的 pilot 门槛

### 9.1 总体原则

三种子 pilot 的目的仅是回答：

> 新的 graph-level containment 是否值得进入正式五种子阶段？

它不是最终统计结论。所有模型选择、gate 稀疏度、dual 参数、adversary 容量和早停必须由 train/validation 决定。

### 9.2 门槛表

| 项目 | 原冻结门槛 | 修订建议 | 判定理由 |
|---|---:|---:|---|
| 目标节点 leakage | 所有目标层下降 | 每个主要目标节点的最大 held-out probe AUC 平均下降至少 `0.005`，且 `3/3` 种子方向一致 | 避免单一 probe 或单一种子驱动 |
| Joint-path leakage | 未明确 | joint-path 最大 probe AUC 平均不得上升；目标 joint path 若被约束，应下降至少 `0.005` | 排除协同编码 |
| 非目标节点迁移 | 不得增加 | 任何节点平均增加不得达到 `+0.005`；同一节点不得有 `2/3` 种子增加超过 `+0.005` | 直接对应 Stage1.2 migration |
| Migration count | 作为 endpoint | `material migration count = 0` 为完整 pass；若为 1，仅允许 exploratory partial pass，不进入五种子 | 防止“平均下降掩盖局部上升” |
| AUC | 损失不超过 `0.01` | 平均 paired `ΔAUC ≥ -0.005`；任一种子 `ΔAUC ≥ -0.010` | 原门槛约为 baseline seed std 的 2.5 倍，过宽 |
| Raw NLLH | 增加不超过 `0.002` | 不作为主 gate | 受 calibration 和概率尺度影响 |
| Calibrated NLLH | 未明确 | 平均 `ΔNLLH_cal ≤ +0.001`；任一种子 `≤ +0.002` | 排除普通校准差异 |
| Calibrated Brier | 未明确 | 预注册 validation-derived non-inferiority margin；pilot 建议平均增量不超过约 `1e-4`，任一种子不超过约 `2e-4` | 与概率质量相互验证；数值仅为 screening margin |
| Score variance | 至少保留 baseline 的 50% | 降级为双侧告警，不独立判定 | 已观察到 collapse 模型方差反而更高 |
| Class gap | 未明确 | 平均至少保留 baseline 的 `80%`；任一种子不得触发 catastrophic floor | 直接检查标签分离能力 |
| Pairwise/ranking | 未明确 | validation paired ranking loss 不得材料性恶化；test 只报告 | 防止仅靠概率尺度满足 NLLH |
| Collapse | 间接 | AUC、class gap、calibrated NLLH/Brier 和动态范围联合判定；任何种子 collapse 即 fail | 防止均值掩盖灾难性运行 |
| DP | 主要公平结果之一 | 保留 secondary report，不作为 pilot pass 的必要充分条件 | 当前为 near-floor |
| Worst-group quality | endpoint | 必须报告；不得出现稳定材料性恶化 | 防止总体 utility 掩盖组内损失 |

### 9.3 Pilot 判定等级

#### Full pass

同时满足：

- 目标节点与 joint paths 达到预注册下降；
- `material migration count = 0`；
- AUC、calibrated NLLH/Brier 和 class gap 全部非劣；
- 无种子 collapse；
- 完整方法优于 final-only、multi-layer-no-gate 和 structured-random controls。

#### Partial pass

只允许用于诊断，不自动进入五种子：

- 目标节点下降；
- 仅一个边缘节点出现接近阈值的轻微迁移；
- utility 全部非劣；
- joint-path 无上升。

Partial pass 需要一次预先限定的机制修复，不得开放大规模 sweep。

#### Fail

任一条件成立即 fail：

- 目标节点未稳定下降；
- 任何关键节点或 joint path 出现材料性迁移；
- 任一种子 AUC collapse；
- 平均 AUC 损失超过 0.005；
- calibrated NLLH/Brier 超过非劣边界；
- class gap 明显退化；
- 完整方法不优于 structured-random 或普通正则化对照。

---

## 10. 对照与消融评审

### 10.1 必须采用 2×2 因子设计

为识别 multi-layer 与 path gate 的独立贡献，至少包含：

| Layer constraint | Structured path gate | 条件名称 | 作用 |
|---|---|---|---|
| Final only | No | `final_only_no_gate` | 普通 final adversarial baseline |
| Multi-layer | No | `multi_layer_no_gate` | 识别多层约束贡献 |
| Final only | Yes | `final_only_path_gate` | 识别路径 gate 独立贡献 |
| Multi-layer | Yes | `multi_layer_path_gate_full` | 完整方法及交互贡献 |

当前 `current_selective_suppression` 只有在 gate 位置、容量、优化预算、稀疏度和其他损失与 `final_only_path_gate` 完全一致时，才能作为该格的严格对照。否则必须重新实现匹配版本。

### 10.2 必需对照矩阵

建议 pilot 至少包含：

| 方法 | 必要性 |
|---|---|
| baseline | CTR reference |
| current selective suppression | 冻结失败局部方法，不做权重 sweep |
| final-only adversary, no gate | 普通局部 adversarial baseline |
| multi-layer adversaries, no gate | 多层独立贡献 |
| final-only + structured path gates | path gate 独立贡献 |
| full multi-layer/multi-path containment | proposed method |
| full method without joint adversary | joint-path 约束消融 |
| layer-risk sum objective | 与 max/CVaR 比较 |
| structured-random gate | 排除结构稀疏正则效应 |
| matched-capacity regularization | 排除容量/正则化效应 |
| calibration-only | 解释 NLLH/ECE |
| adversarial failure reference | 只保留冻结结果，除非代码需要，不再作为搜索对象 |

### 10.3 Anti-collapse 消融

至少比较：

```text
no anti-collapse
prediction/ranking preservation only
utility non-inferiority only
full anti-collapse package
```

用于回答：

- 预测保持是否必要；
- class-gap 和 calibrated probability 约束是否真正阻止 collapse；
- anti-collapse 是否反过来阻止 containment；
- 方法收益是否仅来自更强 CTR 正则化。

### 10.4 正式阶段的附加分析

pilot 通过后，正式阶段应报告：

- 每层 linear/nonlinear/independent/joint probe；
- migration count 和最大迁移幅度；
- gate sparsity 与计算开销；
- 不同种子的 gate 选择稳定性，如 branch-level Jaccard；
- 训练 adversary 与 post-hoc probe 的能力差；
- worst-group AUC/NLLH/Brier/ECE；
- U/U_TILDE 和 ranking diagnostics；
- all_logged、random_display、context_conditioned 三协议；
- 五种子 paired intervals；
- 第二 backbone 或受控半合成验证。

---

## 11. 泄漏迁移、标签泄漏与不可识别性

### 11.1 泄漏迁移已被确认

Stage1.2 的 360 个最终 probe 结果均已收敛，因此当前 migration 不能再用“probe 没收敛”解释。Stage2 必须主动覆盖：

- layer-to-layer migration；
- cross-to-DNN 或 DNN-to-cross migration；
- final-to-embedding/upstream migration；
- suppressed-to-residual component migration；
- marginally hidden but jointly decodable migration；
- probe-family-specific migration。

### 11.2 Probe-family dependence

有限 probe 接近随机只能说明对该 probe family 具有一定 guardedness。需要防止：

- 训练 adversary 被欺骗，但独立 MLP 仍可恢复；
- linear probe 下降，nonlinear probe 上升；
- 单层 probe 下降，joint probe 上升；
- row-level probe 下降，user-disjoint probe 不下降。

任何“完全去除”或“表示独立”声明都不成立，除非提供远强于当前方案的可证明证书；本项目当前不应作此承诺。

### 11.3 User overlap 与身份相关构念

即使输入删除 `user_id`，相同用户仍可能通过其他稳定行为特征出现在不同 row split。必须增加：

- 现有 row-disjoint probe，用于与冻结证据连续比较；
- user-disjoint probe，用于测试未见用户；
- 低频/未见用户分层；
- 可行时 temporal/block split。

user-disjoint 结果不会推翻 row-level 证据，但会决定论文可以把现象外推到什么范围。

### 11.4 `protected_attribute` 的构念重叠

行为保护代理由用户行为模式构造，而 CTR 输入也包含用户与职位行为特征。因此，probe 恢复代理可能部分是在恢复代理构造规则，而不是恢复一个外部真实人口属性。

论文必须明确：

- 研究对象是 behavioral protected proxy；
- proxy decodability 可能反映行为模式可恢复性；
- 不能把结果直接解释为真实性别可恢复；
- 不能把 proxy-conditioned disparity 等同现实招聘歧视。

### 11.5 代理构造的时间耦合风险

现有项目材料没有充分证明行为代理构造只使用了每个样本时点之前的互动，也没有证明使用了未来行为。正确表述是：

> `proxy-construction temporal alignment is undocumented and remains a measurement limitation.`

在未核验构造代码或时间戳前，不能直接写“存在未来标签泄漏”，也不能写“已排除未来信息使用”。

### 11.6 普通点击标签泄漏

Stage1.2 表征导出直接加载冻结 checkpoint，预测复核误差接近机器精度，组件重构误差为零，没有重新训练。现有材料不支持存在普通 test click-label leakage。

Stage2 需要继续确保：

- test proxy 不参与 gate 或超参数选择；
- test click 不参与 calibration 选择；
- test leakage 不参与方法选择；
- 所有阈值、结构和 early stopping 只由 train/validation 决定；
- 失败配置不能根据 test 结果回炉调参。

### 11.7 Gate 的函数不可识别性

任意 hidden-unit gate 缺少稳定语义。解决方式：

- branch/block-level gate；
- 同构 seed 间结构对齐；
- gate budget 固定；
- structured-random control；
- 报告结构级而非坐标级稳定性。

---

## 12. Stage2 应继续方向 A，还是转向测量审计

### 12.1 当前正式判断

**继续方向 A，但批准性质是 `falsification pilot approval`，不是完整方法路线承诺。**

原因：

- 现有方法没有先实现 clean containment；
- 因此尚不能严格检验“泄漏被干净降低后，outcome 是否仍不变”；
- 直接宣布 leakage–outcome decoupling 会保留“干预没有真正打到目标”的替代解释；
- Stage1.2 的方向 A 选择逻辑仍然成立。

### 12.2 测量审计已经是可写贡献，但尚不是唯一主方向

当前已具备的审计贡献包括：

- 局部 suppression 可能导致代理可解码性迁移；
- 不同 probe family 可能对同一干预给出不同判断；
- final-layer leakage 下降不代表 path-wide containment；
- adversarial fairness 可能通过预测坍缩获得表面 DP 改善；
- raw NLLH/ECE 改善可能只是普通 calibration；
- DP 在当前设置中接近测量地板。

这些内容可以作为论文的诊断部分，即使 Stage2-A 方法成功也应保留。

### 12.3 Pilot 后的唯一决策规则

#### A1：clean containment + utility non-inferiority

进入：

```text
5-seed DCNv2
→ formal paired intervals
→ second backbone or semi-synthetic mechanism validation
→ outcome association analysis
```

#### A2：clean containment 成功，但 outcome 不改善

正式转向：

> `leakage–outcome decoupling / process-fairness measurement audit`

此时可以合理声称：在实现指定 probe-family 下的 clean containment 后，proxy-conditioned outcome 仍未出现稳定改善。

#### A3：仍然迁移或只能欺骗训练 adversary

停止继续堆叠更多 adversary 和 gate，不批准第二轮无边界方法搜索。转向：

> 局部去敏的迁移失败模式、probe dependence、联合编码和不可识别性审计。

#### A4：utility collapse

方法路线直接 fail。不能通过放宽 AUC/NLLH 门槛继续恢复；只能修改基本优化设计，并且修改次数应预先限定。

---

## 13. 结论声明边界

### 13.1 当前已经可以写的结论

- FairJob 的 `protected_attribute` 是行为保护代理，而非已验证人口属性。
- 在 no-user-id 条件下，部分 CTR 交互表示对该行为代理表现出小幅增量可解码性。
- 该候选信号在 DeepFM/DCNv2 与三种子 screening 中方向一致。
- 当前局部 selective suppression 能降低 post-mitigation final 的代理可解码性，但泄漏会迁移到其他表示。
- global 和 adversarial 干预也出现表示图中的 leakage redistribution。
- 当前 DP regularization 未稳定降低表示 leakage。
- adversarial baseline 在三个种子上发生预测能力坍缩，其较低 all/context DP 不能解释为有效公平改善。
- suppression 的部分 raw NLLH/ECE 优势可以由 group-blind calibration 解释。
- 当前 DCNv2 no-user-id 设置中的 DP 接近测量地板，应作为 secondary metric。
- 现有结果是指定数据、协议和 probe family 下的描述性关联证据。

### 13.2 Stage2-A pilot 成功后可以写的结论

- 方法降低了预注册表示图节点和联合路径中行为保护代理的 post-hoc decodability；
- 方法减少了相对于局部去敏基线的 representation-graph leakage redistribution；
- 在指定 probe families、split 和 backbone 中，实现了更好的 containment–utility trade-off；
- 方法训练阶段使用行为代理监督，但推理阶段不需要该字段；
- 若 outcome 指标稳定改善，可以写 proxy-conditioned disparity、worst-group quality 或 group calibration 改善；
- 若第二 backbone/半合成验证通过，可以写方法不完全依赖单一 DCNv2 实现。

### 13.3 不能写的结论

- 模型放大了真实用户性别信息；
- 方法消除了性别信息；
- 方法改善了真实性别公平；
- 行为代理就是准确的人口属性；
- 代理可解码性下降证明现实歧视减少；
- representation leakage 是 DP 的因果来源；
- 方法实现了 counterfactual fairness；
- 表示与敏感属性统计独立；
- probe AUC 接近 0.5 证明不存在代理信息；
- 方法消除了 logging-policy bias 或 position bias；
- 方法提供无偏 off-policy evaluation；
- 方法防止了现实招聘歧视；
- all_logged/random_display 结果可外推为完全无选择偏差的现实效果。

### 13.4 推荐术语

全文推荐：

```text
behavioral protected proxy
proxy decodability
proxy extractability
proxy-conditioned disparity
representation-graph leakage redistribution
joint-path decodability
probe-family guardedness
available logged/random-display scopes
descriptive association
train-time proxy supervision, no proxy at inference
```

避免：

```text
true gender leakage
gender discrimination removal
causal fairness improvement
complete sensitive-information erasure
unbiased position-corrected fairness
```

---

## 14. 建议的 Stage2-A pilot 执行顺序

### P0：方案冻结与代码图审计

在任何长训练前完成：

1. 导出实际 DCNv2 forward graph；
2. 明确 cross、parallel DNN、fusion、suppressed/residual components 的真实依赖关系；
3. 冻结目标节点、sentinel 节点和 joint-path 组合；
4. 冻结结构化 gate 单元和稀疏预算；
5. 冻结训练 adversary 与评价 probe 的不同架构；
6. 冻结 calibration candidate set；
7. 冻结所有 validation-only 门槛；
8. 冻结一次 pilot 后的停止规则；
9. 记录 Git commit、config hash、seed、数据和 probe row hash。

### P1：单种子机制 smoke

只验证：

- forward 正确；
- gate 构成预期路径切集；
- joint adversary 可训练；
- dual/max objective 数值稳定；
- representation export 不改变原预测定义；
- 无明显 NaN、梯度爆炸或立即 collapse。

不得根据 test 指标选择结构。

### P2：三种子 validation pilot

运行冻结的最小矩阵：

```text
baseline
final_only_no_gate
multi_layer_no_gate
final_only_path_gate
multi_layer_path_gate_full
full_without_joint_adversary
layer_risk_sum
structured_random_gate
matched_capacity_regularization
```

其中 adversarial failure reference 使用 Stage1.2 冻结结果，不重复进行无意义搜索。

### P3：独立 containment 审计

对全部 pilot checkpoint：

- 导出统一 train/valid/test 表征；
- 运行 linear、nonlinear、independent 和 joint probes；
- 运行 row-disjoint 与 user-disjoint 评价；
- 计算 node deltas、joint deltas、migration count；
- 完成 calibration、AUC、class gap、Brier 和 collapse audit；
- 只按冻结 gate 判定 full/partial/fail。

### P4：条件扩展

只有 full pass 才允许：

1. 五种子 DCNv2；
2. publication-grade cluster bootstrap；
3. 第二 backbone；
4. 半合成已知迁移路径实验；
5. 正式 Pareto；
6. outcome 关系分析。

不允许：

- test-driven weight sweep；
- logging-policy 模型；
- position-corrected 主张；
- 在 pilot fail 后直接增加大量 backbone；
- 把 partial pass 包装成方法成功。

---

## 15. 建议交付物

### 15.1 方案文件

```text
docs/fairjob/stage2/stage2_review_opinion.md
configs/fairjob/stage2_representation_graph.yaml
configs/fairjob/stage2_pilot_gate.yaml
configs/fairjob/stage2_ablation_matrix.yaml
```

### 15.2 代码模块

```text
fuxictr_ext/fairjob/representation_graph.py
fuxictr_ext/fairjob/structured_path_gates.py
fuxictr_ext/fairjob/multinode_adversary.py
fuxictr_ext/fairjob/joint_path_probe.py
fuxictr_ext/fairjob/adaptive_constraints.py
fuxictr_ext/fairjob/collapse_audit.py
fuxictr_ext/fairjob/user_disjoint_probe.py
```

### 15.3 结果产物

```text
stage2/manifests/
stage2/checkpoints/
stage2/representations/
stage2/probes/
stage2/calibration/
stage2/reports/
stage2/bootstrap/
```

每个正式运行至少记录：

```text
git_commit
model/backbone
seed
feature_protocol
proxy_regime
representation_graph_hash
gate_definition_hash
adversary_definition_hash
joint_path_definition_hash
pilot_gate_hash
calibration_protocol
prediction_hash
representation_manifest_hash
```

---

## 16. 最终评审决定

### 16.1 决定

```text
Stage2 direction A: conditionally approved
Current design: major revision required
New long training: not approved yet
Approved next action: freeze revised graph-level pilot protocol
```

### 16.2 必须修改的十项内容

1. 将研究对象改写为表示图中的 proxy decodability redistribution；
2. 明确定义 representation graph、node、path、joint path 和 migration；
3. 使用架构级 branch/block gate，替代任意 hidden-unit gate；
4. 将 parallel DNN、fusion、pre-mitigation final 和 joint paths 纳入覆盖；
5. 增加 joint-path adversary 与独立 joint probe；
6. 使用 adaptive dual 或 smooth max/CVaR，而非只求和层级 loss；
7. 将 score variance 降级为辅助告警；
8. 使用 AUC、class gap、calibrated NLLH/Brier 联合 anti-collapse；
9. 完成 2×2 multi-layer × path-gate 因子消融及 structured-random control；
10. 增加 user-disjoint probe 和一次 pilot 后的严格停止规则。

### 16.3 论文层面的最终定位

当前最有潜力的论文结构不是：

> 我们提出了一个多层 adversarial fairness 模块。

而是：

> 我们发现局部公平干预会使行为保护代理可解码性在 CTR 特征交互表示图中重新分配；建立层级、路径级、联合路径和 collapse-aware 的审计框架；并提出一种 migration-aware graph containment 方法，在不依赖推理时代理的条件下控制最危险节点和路径，同时保持 CTR utility。

该表述中的“首次”必须在投稿前完成系统文献检索后再决定，现阶段不得直接写入摘要或贡献列表。

---

## 附录 A：建议的训练与评价目标摘要

### A.1 训练阶段

\[
\min_{\theta,g}
L_{CTR}
+\lambda_g\Omega(g)
+\lambda_rL_{rank\text{-}preserve}
+\sum_v\mu_v(R_v-\epsilon_v)_+
+\mu_J(R_J-\epsilon_J)_+.
\]

其中：

- \(R_v\)：节点级可微 proxy advantage；
- \(R_J\)：联合路径 proxy advantage；
- \(\mu_v,\mu_J\)：自适应对偶变量；
- \(g\)：结构化 branch/block gates；
- 训练阶段使用行为代理，推理阶段移除 adversaries，且不输入代理。

### A.2 评价阶段

主 containment 指标：

\[
D_v(Q)=\max_{q\in Q}AUC(q(H_v),A),
\]

\[
D_J(Q)=\max_{q\in Q}AUC(q(H_J),A).
\]

主迁移指标：

\[
M_{count}=\sum_{v\in V}
\mathbf{1}[\Delta D_v\geq0.005]
+\mathbf{1}[\Delta D_J\geq0.005].
\]

utility 与安全指标：

```text
AUC
calibrated NLLH
calibrated Brier
class gap
pairwise ranking preservation
score/logit dynamic range
worst-group AUC/NLLH/Brier/ECE
U / U_TILDE
DP as secondary outcome
```

---

## 附录 B：证据来源映射

| 本文使用的冻结结论 | 对应项目文档 |
|---|---|
| Stage1.2 范围、里程碑和唯一方向选择规则 | `execution_plan(1).md` |
| Probe 收敛状态 | `m2_probe_convergence_audit.md` |
| 统一 checkpoint 表征导出与对齐 | `post_intervention_export_report.md`、`post_intervention_export_smoke.md` |
| Leakage reduction、redistribution 和 probe-family disagreement | `post_intervention_leakage_report.md` |
| Adversarial collapse 与 calibration 归因 | `regularization_and_collapse_report.md` |
| DP 地板、组内质量和三协议稳定性 | `outcome_floor_and_stability_report.md` |
| Stage1.2 最终方向 A 与原 pilot 门槛 | `stage1_2_decision_report.md` |
| Stage1.1 M5B 失败与三种子基线 | `m5a_three_seed_report.md` |
| Stage1.1 冻结产物、commit 与 hash | `stage1_1_closure.md` |

---

## 附录 C：方法学参考定位

以下工作仅用于判断方案新颖性和方法风险，不替代 FairJob 项目内部证据：

1. Madras et al. **Learning Adversarially Fair and Transferable Representations.** ICML 2018。
2. Elazar and Goldberg. **Adversarial Removal of Demographic Attributes from Text Data.** EMNLP 2018。
3. Gitiaux and Rangwala. **Learning Smooth and Fair Representations.** AISTATS 2021。
4. Jang et al. **Achieving Fairness through Separability: A Unified Framework for Fair Representation Learning.** AISTATS 2024。
5. Chen et al. **Causality-Inspired Fair Representation Learning for Multimodal Recommendation.** ACM Transactions on Information Systems, 2025。
6. Wang et al. **DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems.** 2020。

这些文献表明，adversarial fair representation、单层属性去除和公平表示本身并不新；本项目需要依靠“CTR 表示图中的干预诱发迁移、联合路径审计和 graph-level containment”形成差异化贡献。
