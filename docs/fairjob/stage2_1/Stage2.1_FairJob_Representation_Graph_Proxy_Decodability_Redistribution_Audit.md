# Stage2.1 — FairJob 表示图行为保护代理可解码性重分布与机制审计

**最后更新：** 2026-07-23  
**项目：** FairJob × FuxiCTR CTR Fairness Research  
**阶段名称：** Stage2.1  
**英文名称：** Representation-Graph Behavioral-Proxy Decodability Redistribution and Mechanism Audit  
**文档定位：** 独立、完整的阶段说明、执行规范与决策门槛；阅读本文不需要预先了解 Stage1.1、Stage1.2 或 Stage2-A。  
**阶段状态：** `protocol_proposed_pending_freeze`  
**Stage2-A 科学结论：** `containment_hypothesis_rejected_by_preregistered_gate`  
**Stage2-A 工程结论：** `engineering_and_training_chain_passed`  
**Stage2.1 默认运行政策：** `checkpoint_only_audit_first`  
**新增 CTR 长训练：** `not_approved`  
**位置校正状态：** `position_corrected = unavailable_missing_external_propensity`  

---

## 0. 一句话记忆锚点

Stage2.1 不再尝试修补已经失败的 graph-wide containment 方法，而是检验并解释一个新的、独立预注册的科学问题：

> 当多层对抗约束与结构化路径 gate 在保持 CTR 预测质量的同时只降低某一局部路径的行为保护代理可解码性时，该代理为何会在 DCNv2 的 cross、fusion、joint 和 embedding 表示中重新变得更容易解码；这种重分布是否跨种子、跨用户划分稳定，分别由多层压力、路径 gate、joint adversary 还是普通正则化驱动。

阶段流程：

```text
正式关闭 Stage2-A containment 假设
→ 在未用于原主结论的种子上复现重分布
→ 用有限关键 control 做组件归因
→ 冻结 conditional/path-specific usable-information 指标
→ 区分局部下降、跨层迁移、联合恢复和 probe-family 依赖
→ 判断 measurement/mechanism 论文是否成立
→ 只有得到明确机制后，才允许提出全新的后续干预假设
```

---

## 1. 执行摘要

FairJob × FuxiCTR 工程链路、Stage2 训练矩阵和独立表示审计均已成功执行：

- 预注册的 `3 seeds × 9 methods = 27` 个全量训练任务全部完成；
- 每个预测文件覆盖同一顺序测试集的 214,446 行；
- checkpoint、prediction、manifest、hparams 和 metrics 完整且按 same-seed baseline 对齐；
- seed-2019 baseline 与 primary method 的表示由 checkpoint 逐个重建；
- checkpoint 重建预测与原 P2 预测的最大绝对误差约为 `1e-16`；
- row-disjoint 和 raw-user-disjoint probe 均使用 train/validation/test 分离及 validation-only 模型选择；
- primary method 没有发生 prediction collapse，三种子平均 AUC、NLLH 和 Brier 均保持健康。

但 Stage2-A 的主要科学假设按预注册门槛明确失败：

> 多层 adversarial pressure、joint adversary 与结构化 path gates 没有实现 DCNv2 表示图中的 graph-wide behavioral-proxy containment。

seed-2019 的独立审计显示：

- `parallel_dnn_path_output` 的代理可解码性下降；
- `cross_layer_0/1/2`、多个 joint paths、`fusion_pre_logit` 和 `embedding_flat` 的代理可解码性上升；
- raw-user-disjoint 协议下的 cross、fusion 和 joint path 增幅更明显；
- 预测 AUC、NLLH 和 Brier 没有坍缩，因此该现象不能用 constant-score 或 destroyed-ranking 解释；
- 预注册规则要求每个目标节点在所有种子上均为负向变化，seed-2019 已出现多个正向目标节点，因此剩余种子不可能挽救原 gate。

Stage2.1 因此作出以下边界调整：

1. **Stage2-A 永久保持 failed；不得通过放宽阈值、追加权重搜索或选择有利 probe family 改写该结论。**
2. **Stage2.1 是新的 measurement/mechanism 阶段，不是 Stage2-A 的补充实验。**
3. **默认只审计既有 checkpoint；不批准新的公平干预训练。**
4. **剩余 probe 只用于复现与机制归因，不能改变原 containment gate。**
5. **Stage2.1 只有在重分布跨种子、跨划分稳定并可由组件对照解释时，才具备形成期刊级测量/机制论文的基础。**
6. **若重分布不能复现，则 Stage2 结果保留为严谨的内部负结果，不再围绕该现象继续扩张研究范围。**

最准确的阶段定位是：

> Stage2.1 研究的不是“如何让已失败的方法成功”，而是“公平干预如何重写显式特征交互网络中行为代理的可解码性几何结构，以及现有 probe 和局部去敏指标为什么可能误判 graph-wide containment”。

---

## 2. 项目背景、变量与声明边界

### 2.1 数据和任务

主数据为 FairJob，主任务为展示前候选点击打分：

```text
protocol: pre_ranking
backbone: DCNv2 parallel cross/DNN structure
proxy regime: proxy-excluded
raw user_id model input: excluded
protected_attribute model input: excluded
protected_attribute training use: permitted only for train-time adversaries
protected_attribute inference use: not required
```

FairJob 使用官方顺序切分：前 80% 为训练候选池，后 20% 为冻结测试集。所有预测与元数据必须按原始 `row_id` 对齐。

### 2.2 关键变量

- 商业点击标签：\(Y\)；
- CTR 模型输入：\(X\)；
- 数据中观测到的 `protected_attribute`：\(A\)；
- 不可直接观测的潜在人口属性：\(S^*\)；
- 表示图节点：\(H_v\)；
- CTR 预测：\(\hat Y\)。

本文中的 \(A\) 是由行为构造的**行为保护代理**，不是已验证的真实性别或其他人口统计 ground truth。Stage2.1 研究的是：

```text
behavioral-proxy decodability
probe-family-specific usable information
representation redistribution
proxy-conditioned outcome diagnostics
```

不得将其改写为：

```text
true gender information
gender discrimination
demographic causal harm
counterfactual fairness
complete sensitive-information erasure
```

### 2.3 Position correction

当前数据不包含真实 logging propensity，且项目不从点击结果或 CTR 预测反推 propensity。因此：

```text
position_corrected = unavailable_missing_external_propensity
```

Stage2.1 不是 selection-aware 或 off-policy evaluation 研究，不作无偏反事实展示效果声明。

---

## 3. Stage2-A 冻结结论

### 3.1 冻结方法

Stage2-A 的 primary method 为：

```text
multi_layer_path_gate_full
```

包含：

- `cross_layer_0`、`cross_layer_1`、`cross_layer_2` 的 train-only adversaries；
- `parallel_dnn_path_output` adversary；
- `fusion_pre_logit` adversary；
- `cross_dnn_outputs` joint adversary；
- 结构化 path gates 与冻结 suppression budget；
- smooth-max graph risk；
- same-seed baseline teacher；
- teacher-standardized centered-logit preservation；
- pairwise ranking preservation；
- anti-collapse constraints。

代理只在训练中用于 adversary，不在推理时使用。

### 3.2 冻结训练矩阵

```text
training commit: 7574ed45fcacc41dba046e10f962de8472ca500f
seeds: 2019, 2020, 2021
methods: 9
full runs: 27
rows per test prediction: 214446
```

九种方法：

1. `baseline`；
2. `multi_layer_path_gate_full`；
3. `multi_layer_no_gate`；
4. `full_without_joint_adversary`；
5. `structured_random_gate`；
6. `matched_capacity_regularization`；
7. `layer_risk_sum`；
8. `final_only_path_gate`；
9. `final_only_no_gate`。

### 3.3 三种子 outcome 结果

| Method | AUC | ΔAUC | NLLH | ΔNLLH | Brier | ΔBrier |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.764927 | +0.000000 | 0.042093 | +0.000000 | 0.006948 | +0.000000 |
| multi_layer_path_gate_full | 0.768455 | +0.003528 | 0.042001 | -0.000091 | 0.006925 | -0.000023 |
| multi_layer_no_gate | 0.766078 | +0.001151 | 0.041610 | -0.000482 | 0.006948 | +0.000001 |
| full_without_joint_adversary | 0.766639 | +0.001713 | 0.040792 | -0.001300 | 0.006917 | -0.000031 |
| structured_random_gate | 0.764712 | -0.000214 | 0.043172 | +0.001079 | 0.007068 | +0.000120 |
| matched_capacity_regularization | 0.772117 | +0.007190 | 0.040662 | -0.001431 | 0.006932 | -0.000016 |
| layer_risk_sum | 0.600764 | -0.164163 | 0.414900 | +0.372807 | 0.017633 | +0.010685 |
| final_only_path_gate | 0.559141 | -0.205785 | 0.410927 | +0.368834 | 0.018109 | +0.011162 |
| final_only_no_gate | 0.549212 | -0.215714 | 0.753097 | +0.711004 | 0.026395 | +0.019448 |

解释边界：

- primary method 没有 collapse；
- primary method 的三种子平均 AUC、NLLH 和 Brier 优于 baseline；
- matched-capacity regularization 的 outcome 结果更强；
- 因此 primary 的 CTR 改善不能归因于 containment；
- outcome 健康不等于表示 containment 成功。

### 3.4 冻结 containment gate

对每个节点、joint path 和划分协议：

1. 每个 probe family 只使用 validation 选择容量与方向；
2. 冻结后在 test 读取一次；
3. 取预注册 probe families 的最大 held-out AUC；
4. 与 same-seed baseline 的对应最大 AUC 做差。

primary gate 要求：

- 每个目标节点在三个种子上均为负向 delta；
- 目标节点平均 delta 至少达到 `-0.005`；
- joint paths 不得材料性上升；
- 非目标或 sentinel 节点不得出现材料性迁移；
- CTR 质量和校准保持在冻结容忍范围内；
- controls 不能解释结果。

其中“所有种子负向”是不可逆规则：任一目标节点在任一种子上出现非负 delta，原方法即不能 full pass 或 partial pass。

### 3.5 seed-2019 冻结审计

#### Row-disjoint 目标节点

| Target | Baseline max AUC | Primary max AUC | Delta |
|---|---:|---:|---:|
| cross_layer_0 | 0.672472 | 0.701167 | +0.028695 |
| cross_layer_1 | 0.685083 | 0.694357 | +0.009274 |
| cross_layer_2 | 0.685876 | 0.697374 | +0.011499 |
| parallel_dnn_path_output | 0.658539 | 0.632413 | -0.026126 |
| fusion_pre_logit | 0.698080 | 0.702644 | +0.004563 |

#### Raw-user-disjoint 目标节点

| Target | Baseline max AUC | Primary max AUC | Delta |
|---|---:|---:|---:|
| cross_layer_0 | 0.563942 | 0.591582 | +0.027640 |
| cross_layer_1 | 0.569084 | 0.598257 | +0.029173 |
| cross_layer_2 | 0.573928 | 0.597033 | +0.023105 |
| parallel_dnn_path_output | 0.556614 | 0.557916 | +0.001303 |
| fusion_pre_logit | 0.563015 | 0.592621 | +0.029607 |

#### Joint paths 与 sentinel

| Joint path | Row-disjoint delta | Raw-user-disjoint delta |
|---|---:|---:|
| cross_dnn_outputs | +0.004563 | +0.029607 |
| cross_block_increments | +0.004272 | +0.024630 |
| branch_outputs_with_fusion | -0.001546 | +0.027707 |

`embedding_flat`：

```text
row-disjoint:      +0.018416
raw-user-disjoint: +0.025836
```

### 3.6 Stage2-A 正式决定

```yaml
stage2_a:
  engineering_status: pass
  training_matrix_status: complete
  scientific_hypothesis: rejected
  containment_gate: early_fail
  remaining_confirmation_probes: not_required_for_original_gate
  additional_weight_tuning: prohibited
  five_seed_expansion: not_approved
  cross_backbone_method_expansion: not_approved
  final_decision_commit: 0cc94639cdfba1546f257af3e0f14abcf181cfcf
```

Stage2.1 不得修改上述任何状态。

---

## 4. 为什么需要 Stage2.1

Stage2-A 已经回答“该方法是否实现 graph-wide containment”，答案是否定的。但 seed-2019 暴露了一个尚未完成验证的机制候选：

```text
局部 DNN 路径 decodability 下降
同时
cross / fusion / joint / embedding decodability 上升
且
CTR prediction quality 未坍缩
```

这产生四个新的科学问题：

1. 该现象是否跨训练种子稳定，而不是 seed-2019 的表示几何偶然性？
2. 该现象是否在 raw-user-disjoint 样本中稳定，而不是同一用户的行级重复造成？
3. 迁移主要由 multi-layer adversarial pressure、learned gates、joint adversary 还是普通容量/正则化改变驱动？
4. 绝对节点 AUC 上升究竟代表子节点新增了可用代理信息，还是原有信息被重参数化后对当前 probe family 更容易提取？

Stage2.1 只在这些问题上开展工作。它不再以降低 DP 或提出新公平模块为阶段目标。

---

## 5. Stage2.1 核心研究问题

### RQ1：重分布是否跨种子稳定

比较 primary 与 same-seed baseline 在 seeds 2019、2020、2021 下的：

- `embedding_flat`；
- `cross_layer_0/1/2`；
- `parallel_dnn_path_output`；
- `fusion_pre_logit`；
- `cross_dnn_outputs`；
- `cross_block_increments`；
- `branch_outputs_with_fusion`。

同时运行：

- row-disjoint；
- raw-user-disjoint。

### RQ2：哪个组件导致迁移或放大

重点对照：

| Control | 主要归因问题 |
|---|---|
| `multi_layer_no_gate` | 没有 gate 时，多层 adversarial pressure 是否已经引起 cross/joint 重编码？ |
| `full_without_joint_adversary` | joint adversary 是抑制联合恢复，还是诱导其他路径重新编码？ |
| `structured_random_gate` | learned gate 的效应是否超出相同预算的结构扰动？ |
| `matched_capacity_regularization` | CTR 改善和表示变化是否可由普通容量/正则化解释？ |

严重 collapse 的 `layer_risk_sum`、`final_only_path_gate`、`final_only_no_gate` 不进入第一轮机制审计；除非后续研究问题明确转向 fairness-by-collapse，才单独批准。

### RQ3：节点增幅是新增可用信息还是可提取性重参数化

使用条件预测增益回答：

> 在已知父节点表示后，子节点是否仍增加了对行为代理的 held-out 预测能力？

Stage2.1 不直接估计完整 Shannon mutual information，也不声称识别真实信息流。使用固定 probe family 下的 held-out cross-entropy 构造操作性、可复现的 usable-information 指标。

### RQ4：联合路径是否恢复单路径中被削弱的代理

检查：

- cross path 单独；
- DNN path 单独；
- cross + DNN；
- branch outputs + fusion；
- cross increments 拼接。

回答是否存在：

> 单路径 decodability 下降，但联合表示重新恢复代理的现象。

不得在没有正式 partial information decomposition estimator 的情况下把这一现象直接称为 PID synergy。

### RQ5：probe-family disagreement 是否改变科学结论

保留两类终点：

1. **保守安全终点：** 预注册 family 的最大 held-out AUC；
2. **机制归因终点：** linear-to-linear、MLP-to-MLP、ExtraTrees-to-ExtraTrees 的配对变化。

Stage2.1 不允许根据 test 结果挑选最有利 family，也不允许以单个下降的 tree probe 覆盖上升的 linear/MLP probe。

### RQ6：CTR outcome 改善是否只是 generic regularization

重点比较：

```text
baseline
primary
matched_capacity_regularization
full_without_joint_adversary
structured_random_gate
```

分析：

- AUC、NLLH、Brier；
- 训练 epoch、早停、参数量和有效容量；
- score/logit distribution；
- 表示图 decodability；
- 是否出现“更高 CTR 质量同时更高 cross/joint decodability”。

若 matched-capacity regularization 同样引起 decodability 重整却取得更强 outcome，primary 的 CTR 提升应归因于训练/容量效应，而非 containment。

---

## 6. 新预注册假设

以下假设完全独立于已失败的 Stage2-A gate。seed-2019 是 discovery evidence；seeds 2020/2021 是主要确认范围。

### H2.1-A：跨路径重分布可复现

primary 相对 same-seed baseline 在 confirmatory seeds 2020/2021 中满足：

1. 至少一个预注册局部路径在两个种子上均呈下降；
2. 至少两个预注册 cross/fusion/joint targets 在两个种子上均呈正向变化；
3. 至少一个 joint target 在 raw-user-disjoint 协议中达到材料性正向变化；
4. CTR AUC 未触发 collapse。

材料性阈值暂定为 held-out AUC 绝对 delta `0.005`。该阈值表示工程/科学上的最小关注幅度，不等同统计显著性。

### H2.1-B：迁移不是 learned gate 的单一副作用

若 `multi_layer_no_gate` 也在 cross/joint targets 出现相似正向变化，则迁移主要与 multi-layer adversarial pressure 或共享表示重编码相关，而不是 learned gate 独有。

若仅 primary 显著迁移，而 `multi_layer_no_gate` 不迁移，则 learned gate 或其与 adversary 的交互成为主要候选机制。

### H2.1-C：joint adversary 不保证 joint-path containment

若 `full_without_joint_adversary` 与 primary 在 joint targets 上差异很小，或 primary 反而更高，则当前 joint adversary 没有形成可验证的联合路径 guardedness。

若 primary 稳定低于 without-joint control，则 joint adversary 有局部贡献，但仍需检查是否把 decodability 推向其他节点。

### H2.1-D：CTR 改善主要由通用正则化解释

若 `matched_capacity_regularization`：

- outcome 优于 primary；
- 并产生相近的表示重整或 decodability 上升；

则 Stage2 outcome 改善不得归因于 fairness containment，最合理解释是 generic regularization、容量或优化路径变化。

### H2.1-E：条件预测增益能够区分简单重参数化与路径新增可用性

若子节点绝对 AUC 上升，但：

```text
CE(A | parent) - CE(A | [parent, child]) ≈ 0
```

则更可能是同一代理信号被重新编码得更易由边际 probe 读取，而非子节点在父节点之外增加可用预测信息。

若该条件增益稳定为正，则子节点或联合路径提供了超出父节点表示的额外 probe-family-specific predictive utility。

---

## 7. 表示图与审计对象

### 7.1 节点集合

Stage2.1 的预注册节点至少包括：

```text
embedding_flat
cross_layer_0
cross_layer_1
cross_layer_2
parallel_dnn_path_output
fusion_pre_logit
```

如 checkpoint forward 能稳定重建并与 P2 manifest 对齐，可附加：

```text
parallel_dnn_linear_0
parallel_dnn_linear_1
logit
probability
```

### 7.2 Joint targets

```text
cross_dnn_outputs
cross_block_increments
branch_outputs_with_fusion
```

所有 joint feature 拼接顺序、维度和 hash 必须记录在 manifest 中，不得按 test 结果更改组合。

### 7.3 图边与条件关系

推荐的父子关系：

```text
embedding_flat → cross_layer_0
cross_layer_0 → cross_layer_1
cross_layer_1 → cross_layer_2
embedding_flat → parallel_dnn_path_output
[cross_layer_2, parallel_dnn_path_output] → fusion_pre_logit
```

条件 usable-information 分析必须预先声明父节点和子节点，不能在查看 test 结果后选择最有利配对。

---

## 8. Probe 协议

### 8.1 Probe families

冻结三类独立评价器：

1. linear logistic probe；
2. matched-capacity nonlinear MLP probe；
3. bounded ExtraTrees probe。

训练 adversary 不得被直接当作独立评价器。

### 8.2 Split 规则

每个 protocol 均使用：

```text
probe train → 拟合
probe validation → 选择容量、早停、方向
probe test → 冻结后读取一次
```

禁止：

- 使用 test 选择 family；
- 使用 test 选择 capacity；
- 根据 test 结果改变 orientation；
- 按方法重新抽取更有利的 row IDs；
- 在 primary 和 baseline 间使用不同用户集合。

### 8.3 Row-disjoint

延续 seed-2019 的固定规模：

```text
train: 50,000 rows
valid: 50,000 rows
test:  50,000 rows
```

若 seeds 2020/2021 使用相同冻结行集合，必须记录 row-ID hash。

### 8.4 Raw-user-disjoint

使用原始 evaluator metadata 中的 `user_id` 构造互斥用户集合。模型本身不输入 raw `user_id`，但该字段可以用于离线 split 审计。

seed-2019 当前规模：

```text
train: 29,939 rows
valid: 9,881 rows
test:  10,217 rows
```

新种子必须复用完全相同的 raw-user-disjoint row IDs，不得重新挑选用户。

### 8.5 Primary endpoint

对表示 \(H_v\) 与预注册 family 集合 \(\mathcal Q\)：

\[
D(v)=\max_{q\in\mathcal Q}\operatorname{AUC}_{test}(q(H_v),A).
\]

方法相对 same-seed baseline：

\[
\Delta D_m(v)=D_m(v)-D_{base}(v).
\]

该 endpoint 用于保守判断：只要任一注册 family 能更容易恢复代理，就不能声称该节点已经被 containment。

### 8.6 Secondary family-specific endpoints

同时报告：

\[
\Delta D_{m,q}(v)
=
AUC_{m,q}(v)-AUC_{base,q}(v),
\]

其中 \(q\) 分别为 linear、MLP 和 ExtraTrees。

这组结果只用于机制归因和 family disagreement，不用于覆盖 primary max-family 结论。

### 8.7 固定容量敏感性

除 validation-selected capacity 外，至少增加一个预注册固定容量的 paired sensitivity：

- baseline 与 method 使用同一 family；
- 使用同一固定 capacity；
- 使用同一输入标准化；
- 报告配对差异。

它用于判断 difference-of-maxima 是否主要由两边选择了不同容量造成，但不替代 primary endpoint。

---

## 9. 重分布与迁移指标

### 9.1 节点状态

对阈值 \(\delta=0.005\)：

```text
reduced:      ΔD(v) ≤ -δ
stable:       |ΔD(v)| < δ
amplified:    ΔD(v) ≥ +δ
```

“amplified”只表示对冻结 probe family 更容易解码，不表示 Shannon information 增加。

### 9.2 Reduction count 与 migration count

\[
N_{reduce}=
\sum_{v\in V}\mathbf 1[\Delta D(v)\le -\delta],
\]

\[
N_{migrate}=
\sum_{v\in V}\mathbf 1[\Delta D(v)\ge +\delta].
\]

### 9.3 Reduction mass 与 migration mass

\[
M_{reduce}=
\sum_{v\in V}\max(-\Delta D(v),0),
\]

\[
M_{migrate}=
\sum_{v\in V}\max(\Delta D(v),0).
\]

这些量用于描述变化在表示图中的总体分布，不被解释为信息守恒或信息流。

### 9.4 Redistribution index

探索性定义：

\[
RI=
\frac{M_{migrate}}
{M_{migrate}+M_{reduce}+\epsilon}.
\]

解释：

- 接近 0：变化主要表现为下降；
- 接近 1：变化主要表现为上升；
- 约 0.5：下降和上升的质量相近。

该指标是 descriptive audit summary，不作为因果量或理论信息度量。

### 9.5 Protocol consistency

分别计算：

```text
row-disjoint RI
raw-user-disjoint RI
```

并报告方向一致节点数。不能把两个 protocol 任意平均成一个数字后隐藏不一致。

---

## 10. Conditional/path-specific usable-information 审计

### 10.1 使用 held-out cross-entropy

对父节点 \(H_p\) 和子节点 \(H_c\)，定义：

\[
G_{\mathcal V}(c\mid p)
=
CE_{\mathcal V}^{*}(A\mid H_p)
-
CE_{\mathcal V}^{*}(A\mid[H_p,H_c]),
\]

其中：

- \(\mathcal V\) 是冻结 probe family；
- \(CE_{\mathcal V}^{*}\) 只在 validation 上选择模型；
- test 只读取选定模型的 held-out cross-entropy；
- 正值表示加入子节点后，行为代理在该 probe family 下更容易预测。

这应被称为：

```text
conditional predictive gain
probe-family-specific usable-information gain
```

不得写成完整 mutual information 或可识别因果信息流。

### 10.2 预注册父子比较

```text
cross_layer_0 | embedding_flat
cross_layer_1 | cross_layer_0
cross_layer_2 | cross_layer_1
parallel_dnn_path_output | embedding_flat
fusion_pre_logit | [cross_layer_2, parallel_dnn_path_output]
```

### 10.3 Joint-path predictive gain

对 cross 和 DNN 两条分支：

\[
J_{\mathcal V}
=
\min\{
CE_{\mathcal V}^{*}(A\mid H_{cross}),
CE_{\mathcal V}^{*}(A\mid H_{dnn})
\}
-
CE_{\mathcal V}^{*}(A\mid[H_{cross},H_{dnn}]).
\]

若 \(J_{\mathcal V}>0\)，说明联合表示比任一单分支更有利于代理预测。它只能被称为 joint predictive gain，不能未经额外理论直接称为 synergy information。

### 10.4 新鲜证据要求

seed-2019 已用于形成这一指标假设，因此：

- 指标和节点配对必须在分析 seeds 2020/2021 前冻结；
- seeds 2020/2021 是主要确认范围；
- 若仍使用原 test 用户，结论必须标注为 internal confirmatory across training seeds，而非完全新数据确认；
- 若能保留尚未分析的 user holdout，则将其设为最终确认集；
- 无新鲜 holdout 时，不得把 exploratory decomposition 包装成严格 confirmatory result。

---

## 11. 最小 checkpoint 审计矩阵

Stage2.1 不运行全部剩余 25 个 P3 checkpoint。采用分层审批。

### Tier 1：主现象跨种子确认

必须运行：

```text
baseline_seed2020
multi_layer_path_gate_full_seed2020
baseline_seed2021
multi_layer_path_gate_full_seed2021
```

对每个 checkpoint 执行：

- 18 个预注册 node/joint targets；
- row-disjoint；
- raw-user-disjoint；
- linear、MLP、ExtraTrees；
- max-family endpoint；
- family-specific endpoint；
- conditional predictive gain。

Tier 1 的唯一问题是：

> seed-2019 的局部下降与 cross/joint 重分布是否跨训练种子复现？

### Tier 2：seed-2019 组件归因

仅在 Tier 1 发现至少方向性复现后运行：

```text
multi_layer_no_gate_seed2019
full_without_joint_adversary_seed2019
structured_random_gate_seed2019
matched_capacity_regularization_seed2019
```

它们回答 gate、joint adversary、多层压力和 generic regularization 的归因问题。

### Tier 3：条件性多种子扩展

只将 Tier 2 中能够明显解释迁移的 control 扩展到 seeds 2020/2021。审批条件：

- control 与 primary 在至少两个预注册节点上存在材料性差异；
- 差异对应一个预先定义的机制问题；
- 扩展结果有能力推翻或支持具体归因；
- 不是为了增加表格数量或寻找有利结果。

### 默认不批准

```text
layer_risk_sum
final_only_path_gate
final_only_no_gate
```

这些方法已经发生严重 outcome collapse，不适合用于“预测健康条件下的重分布”主分析。

---

## 12. 组件归因方法

### 12.1 Same-seed paired delta

所有 control 与 same-seed baseline 比较：

\[
\Delta D_c(v)=D_c(v)-D_{base}(v).
\]

### 12.2 Primary-control contrast

\[
\Gamma_c(v)=
\Delta D_{primary}(v)-
\Delta D_c(v).
\]

解释：

- \(\Gamma_c(v)>0\)：primary 在该节点比 control 更容易解码代理；
- \(\Gamma_c(v)<0\)：primary 在该节点比 control 更难解码代理。

该 contrast 是组件归因线索，不自动构成因果效应，因为方法间可能同时存在多项结构差异。

### 12.3 归因表

| 比较 | 可支持的结论 | 不能直接支持的结论 |
|---|---|---|
| primary vs multi-layer no-gate | gate 加入后表示图如何变化 | gate 是唯一因果来源 |
| primary vs without-joint | joint adversary 的增量影响 | joint adversary 已消除联合信息 |
| primary vs structured-random | learned gate 是否优于同预算随机结构扰动 | learned gate 识别了真实有害路径 |
| primary vs matched-capacity | outcome/表示变化是否可由普通正则化解释 | regularization 是唯一机制 |

---

## 13. Outcome 指标在 Stage2.1 中的角色

Stage2.1 不以 DP 优化或公平 mitigation 为主要目标。Outcome 指标用于排除替代解释：

- AUC：确认排序能力未坍缩；
- NLLH/Brier：确认概率质量；
- score/logit 分布：检查异常扩张或收缩；
- DP：继续报告，但当前设置接近 measurement floor；
- U/U_TILDE：检查排序效用；
- group-wise AUC/NLLH/Brier/ECE：描述性补充。

### 13.1 Prediction-collapse 判定

任何 checkpoint 若出现以下组合，应从健康重分布分析中单列：

- AUC 大幅下降；
- class separation 严重减弱；
- NLLH/Brier 显著恶化；
- score variance 变化无法由有效排序解释。

score variance 不能单独定义 collapse，因为无信息噪声也可能产生高方差。

### 13.2 Generic regularization 分析

`matched_capacity_regularization` 当前具有最强三种子 outcome：

```text
AUC:   0.772117
NLLH:  0.040662
Brier: 0.006932
```

因此必须检查：

- 它是否也提高 cross/joint decodability；
- 它与 primary 是否具有相似训练 dynamics；
- primary 的 outcome 改善是否只是蒸馏、容量或优化正则化效应。

Stage2.1 不将 matched-capacity 方法重新包装为 fairness method。

---

## 14. 确认门槛与停止规则

### 14.1 迁移机制确认

只有同时满足以下条件，才能把现象从“seed-2019 失败分析”升级为“稳定重分布机制证据”：

1. seeds 2020 和 2021 均完成 same-seed baseline/primary audit；
2. 预注册局部下降路径在至少两个训练种子上方向一致；
3. 至少两个预注册 cross/fusion/joint targets 在至少两个训练种子上材料性上升；
4. 至少一个 joint target 在 raw-user-disjoint 下跨种子方向一致；
5. family-specific 结果透明报告，不能只给 max-family；
6. primary 的 CTR AUC/NLLH/Brier 未发生 collapse；
7. 结果不依赖临时更换 row IDs、用户集合、probe family 或阈值。

### 14.2 机制归因确认

至少一个组件 control 必须满足：

- 在预注册节点上系统性改变 migration pattern；
- 差异跨 family 或跨 seed 具有可解释一致性；
- generic regularization control 不能完全复制该模式；
- 归因不依赖 collapsed methods。

否则只能写“重分布被观察到，但触发组件尚未识别”。

### 14.3 Stage2.1 失败条件

若发生以下任一情形，停止将该现象发展为主论文：

- seeds 2020/2021 未复现 cross/joint 正向变化；
- raw-user-disjoint 结果方向不稳定；
- 结果完全由单一 probe family 驱动且其他 family 反向；
- fixed-capacity sensitivity 不支持主要方向；
- primary 与 generic regularization 的表示变化无法区分；
- conditional predictive gain 不稳定且没有其他机制证据；
- 需要修改 Stage2-A 失败阈值才能形成叙事。

此时最准确状态为：

```text
rigorous_internal_negative_result
not_approved_for_method_or_measurement_paper_expansion
```

### 14.4 新干预的批准条件

Stage2.1 默认不提出新干预。只有出现明确、跨种子机制后才可单独立项。例如：

> 单路径 decodability 下降后，cross+DNN joint predictive gain 稳定上升，并且该模式由 joint objective 或 gate 交互触发。

此时可以提出针对 joint-path predictive gain 的新假设，但必须：

- 新建下一阶段；
- 新预注册；
- 不复用 Stage2-A 的成功声明；
- 不在已观察的 test 结果上继续调参；
- 使用新鲜确认样本或第二设置。

---

## 15. Stage2.1 里程碑

### S21-M0：正式关闭 Stage2-A

**任务：**

1. 保存 Stage2-A machine decision、commit、run manifests 和 probe protocol hash；
2. 写入 `scientific_hypothesis = rejected`；
3. 禁止原方法的权重 retuning；
4. 声明剩余 probes 只能用于新阶段机制审计；
5. 为 Stage2.1 创建独立 workdir 和 manifest。

**退出门槛：**

- Stage2-A 状态不可被 Stage2.1 覆盖；
- 训练和决策 commit 唯一；
- 27 个 P2 run manifest 完整；
- Stage2.1 audit plan hash 已冻结。

### S21-M1：主现象跨种子复现

**任务：**

- 运行 seeds 2020/2021 baseline 与 primary 的完整 row/user-disjoint audit；
- 复用 seed-2019 的节点、joint targets、families 和 split hashes；
- 输出 max-family 和 family-specific deltas；
- 计算 migration/reduction count、mass 和 RI；
- 执行 conditional predictive gain。

**退出门槛：**

- 所有 probe 收敛或按预注册 retry 规则处理；
- seed-2019 discovery 与 2020/2021 confirmatory 分开报告；
- 给出 `replicated`、`partially_replicated` 或 `not_replicated` 判定。

### S21-M2：关键组件归因

**前置条件：** M1 至少 `partially_replicated`。

**任务：**

- seed-2019 四个关键 controls；
- primary-control contrasts；
- gate、joint adversary、multi-layer pressure 和 generic regularization 归因；
- 只按冻结规则选择最多一个 control 扩展到其他种子。

**退出门槛：**

- 至少一个组件获得可解释支持，或明确记录 `trigger_component_unidentified`；
- 不运行与机制问题无关的完整 control 矩阵。

### S21-M3：Probe 与 conditional measurement 审计

**任务：**

- max-family、family-specific 与 fixed-capacity paired results；
- held-out cross-entropy；
- conditional predictive gain；
- joint predictive gain；
- family disagreement map；
- user-disjoint 稳健性。

**退出门槛：**

- 明确哪些结论依赖 probe family；
- 不把 near-chance probe 解释为信息不存在；
- 不把 predictive gain 解释为完整 mutual information。

### S21-M4：Outcome 与 regularization 归因

**任务：**

- 对 primary、matched-capacity、without-joint、random-gate 和 baseline 汇总 outcome；
- 检查训练 dynamics、参数量、早停 epoch；
- 对照表示重分布与 CTR 改善；
- 明确 outcome 改善是否可由 generic regularization 解释。

**退出门槛：**

- 不把 CTR 改善归因于已经失败的 containment；
- 对 primary outcome 提供至少一个可信替代解释的检验。

### S21-M5：唯一后续方向冻结

只能选择一个：

| 方向 | 触发条件 |
|---|---|
| Measurement/mechanism paper | 重分布跨种子、跨划分复现，并至少部分完成组件归因 |
| New intervention hypothesis | measurement 发现明确、可操作的新机制，并有新鲜确认设计 |
| Internal negative result closure | 迁移不复现、无法归因或完全由 probe/regularization 偶然性解释 |
| Broader measurement expansion | family disagreement 本身稳定，且可在第二 setting 复现 |

不得同时宣称“机制论文已经成立”和“立即开始新方法训练”。

---

## 16. Artifact 与运行布局

### 16.1 仓库与数据

```text
local repository:  D:\code\JobFairness
server repository: /root/autodl-tmp/code/JobFairness
branch:             codex/stage1-2
FairJob data:       /root/autodl-tmp/datasets/FairJob
Stage2 artifacts:   /root/autodl-tmp/workdirs/JobFairness/stage2
```

### 16.2 Stage2.1 新目录

```text
/root/autodl-tmp/workdirs/JobFairness/stage2_1/
  manifests/
  reconstructed_representations/
  probes/
  conditional_probes/
  attribution/
  bootstrap/
  reports/
  logs/
  runtime_config/
```

checkpoint 表示可采用逐个重建、内存审计、只持久化 compact JSON 的模式，以避免重复占用大量磁盘。

### 16.3 每个 audit 必须记录

```text
stage2_training_commit
stage2_decision_commit
stage2_1_code_commit
checkpoint_sha256
prediction_sha256
row_disjoint_hash
user_disjoint_hash
representation_target
joint_target_definition
probe_family
probe_candidate_grid
validation_selection
orientation_rule
seed
protocol
result_hash
```

---

## 17. 建议新增文件

```text
configs/fairjob/stage2_1_audit_protocol.yaml
configs/fairjob/stage2_1_representation_graph.yaml
configs/fairjob/stage2_1_probe_families.yaml

fuxictr_ext/fairjob/stage2_1/reconstruct_checkpoint.py
fuxictr_ext/fairjob/stage2_1/run_representation_graph_audit.py
fuxictr_ext/fairjob/stage2_1/conditional_probe.py
fuxictr_ext/fairjob/stage2_1/redistribution_metrics.py
fuxictr_ext/fairjob/stage2_1/component_attribution.py
fuxictr_ext/fairjob/stage2_1/build_stage2_1_report.py

tests/fairjob/test_stage2_1_graph_targets.py
tests/fairjob/test_stage2_1_split_freeze.py
tests/fairjob/test_stage2_1_conditional_probe.py
tests/fairjob/test_stage2_1_redistribution_metrics.py

docs/fairjob/stage2_1/stage2_a_closure.md
docs/fairjob/stage2_1/stage2_1_protocol.md
docs/fairjob/stage2_1/stage2_1_multiseed_replication.md
docs/fairjob/stage2_1/stage2_1_component_attribution.md
docs/fairjob/stage2_1/stage2_1_decision_report.md
```

上述文件是建议职责划分，不表示当前已经存在。

---

## 18. 统计与报告规范

### 18.1 种子角色

- seed-2019：discovery；
- seeds 2020/2021：internal confirmatory across training initialization；
- 若未来新增五种子：必须由 Stage2.1 M5 重新批准，不能自动延续；
- 不以三个训练种子替代独立数据集泛化。

### 18.2 Bootstrap

对 probe test 可使用 user-cluster bootstrap；row-disjoint 结果可同时提供 row bootstrap 作为补充。至少报告：

- paired delta；
- 95% interval；
- cluster unit；
- repeats；
- 是否跨零。

若计算预算不足，screening interval 必须明确标注，不能称 publication-grade CI。

### 18.3 多重比较

Stage2.1 节点较多。主节点与 joint targets 必须预注册；探索性附加层单独列出。正式机制判断优先基于：

- 预注册节点集合；
- 方向一致性；
- effect-size threshold；
- paired interval；
- 跨 protocol 复现。

不能仅在几十个节点中挑选显著结果。

### 18.4 结果展示

至少提供：

- representation-graph heatmap；
- 节点 family-specific delta；
- row/user-disjoint 对照；
- migration/reduction mass；
- conditional predictive gain；
- component contrast；
- outcome–decodability scatter；
- probe-family disagreement matrix。

所有图应同时显示不利结果。

---

## 19. 期刊创新定位

### 19.1 当前不能依赖的 novelty

下列广义结论不足以构成主要创新：

- 去偏可能失败；
- adversarial representation learning 可能泄漏；
- 一个 probe 下降、另一个 probe 上升；
- 多层网络中信息可以重新编码；
- generic regularization 可以提高 CTR。

### 19.2 可能形成的具体创新组合

若 Stage2.1 通过，可以形成以下组合贡献：

1. **CTR 表示图中的局部去敏—跨路径重分布现象。** 证明在显式 cross/DNN 交互结构中，局部 DNN decodability 下降可与 cross、fusion 和 joint decodability 上升同时发生，并且不依赖 raw user-ID 输入或 prediction collapse。
2. **保守、独立的表示图审计协议。** 同时使用 row-disjoint、raw-user-disjoint、max-family endpoint、family-specific attribution 和 joint targets。
3. **conditional/path-specific usable-information 测量。** 区分边际 probe AUC 上升与父节点之外的额外预测增益。
4. **组件归因。** 区分 multi-layer pressure、learned gate、joint adversary 和 generic regularization 对重分布的作用。
5. **负结果的预注册可信度。** containment 假设失败后不放宽门槛，而将其作为 discovery 证据建立新的确认阶段。

### 19.3 达到 TOIS 级别仍需的最低外部证据

即使 Stage2.1 在 FairJob/DCNv2 内通过，正式期刊论文还应至少补充以下之一：

- 第二个显式特征交互 backbone；
- 第二个 CTR/推荐数据集；
- 一个具有已知行为代理生成、已知有害路径和已知干预机制的半合成环境。

半合成验证尤其适合回答：

- 哪条路径真正携带新增代理信号；
- probe 是否正确定位迁移；
- joint predictive gain 是否对应已知生成机制；
- 普通重参数化与真实路径新增如何区分。

在完成第二 setting 前，不宜声称这是全部 CTR 网络的普遍规律。

---

## 20. 可以写、条件成立后可以写、不能写

### 20.1 当前已经可以写

- FairJob 中的 `protected_attribute` 是行为保护代理，不是已验证人口属性。
- Stage2-A 的 27 个训练任务和工程产物链完整通过。
- multi-layer/path-gate primary method 没有发生 prediction collapse。
- primary method 没有通过预注册 graph-wide containment gate。
- seed-2019 中，`parallel_dnn_path_output` 的代理可解码性下降，而多个 cross、joint、fusion 和 embedding 表示的可解码性上升。
- raw-user-disjoint 协议下也观察到 cross/joint 增幅，因此该结果不能仅由模型显式输入 raw `user_id` 解释。
- 当前证据只支持 seed-2019 的机制候选，不支持跨种子普遍结论。
- matched-capacity regularization 的 CTR outcome 优于 primary，因此 primary 的 outcome 改善不能归因于 containment。
- `position_corrected` 因缺少真实 logging propensity 而不可用。

### 20.2 Stage2.1 通过后可以写

- 在指定 FairJob、DCNv2、probe families 和 split 协议下，局部行为代理去可解码化伴随跨层、跨路径或联合表示中的可解码性重分布。
- 该模式跨多个训练种子并在 raw-user-disjoint 样本中复现。
- 某个方法组件与重分布模式存在稳定的配对关联。
- 条件 predictive-gain 结果表明，某些子节点或 joint representation 在父节点之外增加了 probe-family-specific 代理预测效用。
- CTR outcome 改善主要与 generic regularization 或容量变化一致，而不是成功 containment 的证据。
- 现有 final-layer 或单路径 probe 不足以支持 graph-wide guardedness 声明。

### 20.3 不能写

- 模型放大了真实用户性别；
- 方法增加或减少了真实 demographic information；
- cross layer 导致现实招聘歧视；
- proxy decodability 上升证明 outcome fairness 恶化；
- proxy decodability 下降证明歧视减少；
- 方法实现或没有实现 counterfactual fairness；
- user-disjoint 结果排除了全部身份、行为或上下文代理；
- max-family AUC 是完整 Shannon mutual information；
- conditional predictive gain 是可识别的因果信息流；
- 一个有利 probe family 可以推翻预注册最大 family 结果；
- Stage2.1 结果能够挽救 Stage2-A 的 containment gate；
- 当前结果已经控制 logging-policy 或 position bias；
- 行为保护代理是准确、稳定、无测量误差的敏感属性。

全文建议统一使用：

```text
behavioral protected proxy
proxy decodability
proxy extractability
probe-family-specific usable information
representation redistribution
joint predictive gain
proxy-conditioned outcome disparity
descriptive association
```

---

## 21. 风险登记

| 风险 | 影响 | Stage2.1 处理 |
|---|---|---|
| seed-2019 偶然性 | 负结果无法形成机制论文 | 先运行 2020/2021 primary-baseline 确认 |
| test 已参与新假设形成 | confirmatory 强度受限 | 冻结新指标；优先保留新 user holdout 或第二 setting |
| probe-family dependence | decodability 结论不稳定 | max-family + family-specific + fixed-capacity 三重报告 |
| raw-user overlap | 行级 probe 高估泛化 | raw-user-disjoint 作为必要协议 |
| 行为代理构念重叠 | 可能只恢复代理构造规则 | 限制术语；增加条件与半合成分析 |
| hidden geometry 重参数化 | AUC 上升不等于新增信息 | conditional predictive gain；不作 Shannon information 声明 |
| 多节点多重比较 | 容易挑选有利结果 | 预注册主节点、effect threshold 和 paired interval |
| generic regularization | outcome 提升被误归因 | matched-capacity、random-gate 和 without-joint controls |
| 存储压力 | 完整表征难以持久化 | checkpoint-by-checkpoint、内存审计、compact JSON |
| 继续方法修补诱惑 | 破坏预注册可信度 | 禁止 Stage2-A retuning；新方法必须另立阶段 |
| 缺少 propensity | 无法做 position-corrected | 保持 unavailable；不作 causal/off-policy 声明 |

---

## 22. Stage2.1 完成标准

Stage2.1 完成必须同时满足：

### 工程与冻结

- [ ] Stage2-A closure manifest 已生成并不可被覆盖；
- [ ] Stage2.1 protocol、节点、joint targets、families、阈值和 split hashes 冻结；
- [ ] 所有 checkpoint 重建预测通过一致性检查；
- [ ] Stage2.1 不修改任何 Stage2-A checkpoint。

### 主现象复现

- [ ] seeds 2020/2021 baseline-primary row-disjoint audit 完成；
- [ ] seeds 2020/2021 baseline-primary raw-user-disjoint audit 完成；
- [ ] max-family、family-specific 和 fixed-capacity 结果均报告；
- [ ] 迁移判定明确为 replicated、partial 或 not replicated。

### 机制归因

- [ ] 至少完成四个关键 controls 的 seed-2019 审计，或根据 M1 结果记录停止原因；
- [ ] gate、joint adversary、multi-layer pressure 与 generic regularization 分别获得判定；
- [ ] conditional/path-specific predictive gain 完成；
- [ ] family disagreement 透明报告。

### 决策输出

- [ ] 完成 Stage2.1 decision report；
- [ ] 唯一选择 measurement paper、新干预假设、broader audit 或 internal closure；
- [ ] 未通过时不继续增加控制、权重或 backbone；
- [ ] 通过时明确第二 setting 与新鲜确认要求；
- [ ] 所有结论遵守行为保护代理与描述性关联边界。

---

## 23. 建议论文结构：仅在 Stage2.1 通过后启用

```text
1. Introduction
   - local debiasing cannot be equated with graph-wide containment

2. Problem Setting
   - behavioral proxy, CTR representation graph, claim boundary

3. A Preregistered Failed Containment Attempt
   - engineering success, scientific early fail, no threshold relaxation

4. Representation-Graph Redistribution Audit
   - nodes, joint paths, row/user-disjoint protocols, probe families

5. Conditional and Joint Predictive Gain
   - parent-conditioned and cross-path usable information

6. Mechanism Attribution
   - multi-layer pressure, gates, joint adversary, generic regularization

7. Outcome and Measurement Separation
   - healthy CTR, near-floor DP, regularization explanation

8. Cross-Seed and Second-Setting Validation

9. Limitations
   - behavioral proxy, probe-family guardedness, no propensity, no causal claim
```

论文主贡献不应表述为“提出了成功的公平方法”，而应表述为：

> 对显式特征交互 CTR 模型中局部去代理化、跨路径重分布、联合恢复和普通正则化进行可复现、保守且不依赖 prediction collapse 的表示图测量与机制归因。

---

## 24. 最终阶段定位

Stage2.1 的最准确标签是：

> **Stage2-A 的 graph-wide containment 假设已经按预注册门槛失败；Stage2.1 在不修改该结论、不继续调参和默认不新增训练的前提下，检验局部路径去可解码化是否稳定伴随 cross、fusion、joint 与 embedding 表示中的行为代理可解码性重分布，并通过跨种子、raw-user-disjoint、conditional probing 和关键组件对照判断其是否足以形成 measurement/mechanism 研究。**

Stage2.1 的成功标准不是找到一个新的、表面更低的 proxy AUC，也不是把失败方法改写成 partial success，而是回答：

1. 重分布是否真实、稳定、可复现；
2. 哪个组件触发或放大它；
3. 边际 decodability 与条件可用信息如何区分；
4. CTR outcome 改善是否只是 generic regularization；
5. 该证据是否足够支持期刊级测量/机制论文，或者应当严谨关闭项目主线。

---

## 25. 依据文件与结果来源

本阶段建立在以下已冻结材料之上：

```text
Stage1.2_FairJob_Intervention_Failure_Attribution_and_Direction_Refreezing.md
Stage2_FairJob_MultiLayer_MultiPath_Containment_Review.md
docs/fairjob/stage1_2/stage1_2_decision_report.md
docs/fairjob/stage2/representation_graph_audit.md
docs/fairjob/stage2/probe_protocol_audit.md
docs/fairjob/stage2/p1_bounded_smoke_report.md
docs/fairjob/stage2/p2_training_audit.md
docs/fairjob/stage2/p3_seed2019_early_stop_audit.md
docs/fairjob/stage2/training_runbook.md
/root/autodl-tmp/workdirs/JobFairness/stage2/reports/p3_seed2019_early_stop.json
```

其中 Stage2-A 当前结论为：工程通过、预测未坍缩、graph-wide containment 科学假设失败。Stage2.1 的任何结果均不得覆盖这一历史结论。
