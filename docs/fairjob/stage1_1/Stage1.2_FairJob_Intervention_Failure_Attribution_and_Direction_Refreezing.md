# Stage1.2 — FairJob 干预失败归因、泄漏—结果关系验证与研究方向重新冻结

**最后更新：** 2026-07-22  
**阶段名称：** Stage1.2  
**项目：** FairJob × FuxiCTR CTR Fairness Research  
**文档定位：** 独立、完整的阶段说明与执行规范；阅读本文不需要预先了解 Stage1.0 或 Stage1.1。  
**当前状态：** Stage1.1 已完成候选机制筛选和首轮模型级公平干预；M5B 冻结门槛未通过，因此原“选择性路径抑制改善结果公平”路线暂不批准进入 Stage2。  
**Stage1.2 目标：** 确定失败发生在“干预没有真正降低代理泄漏”，还是“代理泄漏虽被降低但未转化为结果公平改善”，并据此重新冻结唯一的 Stage2 主方向。  
**复现边界：** 当前属于 FairJob × FuxiCTR integration-mode research，不声称严格复现 FairJob 原论文，也不作无 logging propensity 支撑的无偏反事实或因果公平声明。

---

## 0. 一句话记忆锚点

Stage1.2 不再继续盲目调节公平损失，而是回答一个决定论文方向的问题：

> M5A 为什么没有形成稳定的公平—效用改进：是现有方法没有成功干预表示中的行为保护代理信息，还是表示泄漏本身并不是当前结果差异的可靠替代指标？

阶段逻辑：

```text
Stage1.1 候选现象：小幅、跨模型的代理信息放大
→ M5A 方法结果：冻结门槛失败
→ 干预后表示复测
→ 泄漏迁移、普通正则化和预测坍缩排查
→ DP 地板效应与结果指标可识别性评估
→ 重新冻结 Stage2 唯一主方向
```

---

## 1. 为什么必须进入 Stage1.2

### 1.1 Stage1.1 已完成的科学判断

Stage1.1 的原始目标是判断：现代 CTR 特征交互模型是否会在原始输入已有信息之外产生行为保护代理信息放大，以及这种放大是否可以通过选择性路径抑制转化为更好的公平—效用权衡。

现有证据形成了两个必须同时保留、不能互相替代的结论。

#### 结论 A：候选代理放大现象得到方向性支持

在移除 `user_id` 后，使用相同容量的非线性 probe 比较原始输入与隐藏表示：

| Backbone / 表示 | 三个 CTR 种子的 probe AUC | 相对 no-user 输入的 Amp |
|---|---:|---:|
| 原始 no-user 输入 | 0.687182 | 0 |
| DeepFM `dnn_linear_0` | 0.703265–0.705452 | +0.016084 至 +0.018270 |
| DCNv2 `cross_layer_0` | 0.700798–0.702186 | +0.013617 至 +0.015005 |
| DCNv2 `final` | 0.722255–0.736255 | +0.035074 至 +0.049073 |

三个种子方向一致；针对性屏蔽在九组比较中均比随机屏蔽的平均 leakage drop 更大，DCNv2 final 表示上的集中性最稳定。该结果支持“隐藏表示中存在小幅、可定位的代理信息增量”这一候选现象，但仍受 probe 未充分收敛、仅三种子和表示级干预等限制。

#### 结论 B：当前选择性抑制方法没有通过结果公平门槛

M5A 在 DCNv2、种子 2019/2020/2021 上比较了：

- baseline；
- global suppression；
- matched-random suppression；
- selective suppression；
- DP regularization；
- adversarial removal。

冻结门槛要求某个干预在 `all_logged`、`random_display` 和 `context_conditioned` 三个可用协议中改善平均 DP，同时至少匹配 baseline 的 AUC、NLLH、U 和 U_TILDE。没有任何方法满足该条件。

因此，Stage1.1 的正确结项不是“候选机制被否定”，而是：

```text
候选表示现象：方向性支持
现有训练级干预：未形成稳定结果公平收益
选择性抑制作为 Stage2 主方法：暂不批准
失败原因：尚未识别
```

### 1.2 为什么不能直接继续调权重

继续进行大规模 suppression 权重搜索会产生三个问题：

1. **破坏冻结门槛。** M5B 已预先规定方法扩展条件，门槛失败后不能通过更大规模 test-driven 搜索反向制造成功结果。
2. **无法区分失败位置。** 未测量干预后表示泄漏，就无法判断方法是否真正作用于目标机制。
3. **可能把普通正则化或输出压缩误认为公平改进。** 部分方法改善 NLLH/ECE，但并未改善 DP；adversarial 则以严重 AUC 崩塌换取较小 DP。

因此，需要一个独立阶段专门进行失败归因，而不是把失败实验继续塞入原方法开发阶段。

---

## 2. 当前证据快照

### 2.1 M3：代理泄漏与路径集中性

当前可防守结论：

- 原始全特征中的保护代理可预测性几乎由 `user_id` 主导；
- 移除 `user_id` 后，DeepFM 与 DCNv2 的部分中间表示仍出现小幅正 amplification；
- DCNv2 final 表示的 amplification 大于首个 cross layer；
- train-selected targeted masking 通常比 matched-random masking 更能降低 probe AUC；
- 该结果是表示级诊断，不是结果公平干预成功的证据。

当前不能声称：

- 代理放大必然造成 DP；
- targeted masking 已经是有效公平方法；
- 表示可预测性等价于实际歧视或因果伤害；
- 三种子筛选结果已经达到正式期刊统计标准。

### 2.2 M4：结果、日志范围与代理测量敏感性

六个 no-user-id DeepFM/DCNv2 运行显示：

- `all_logged` 与 `random_display` 下 DP 符号没有反转；
- random-display 会改变 DP 大小和预测性能，但影响方向不统一；
- 在 `displayrandom × rank` 上下文分解中，composition residual 约为 \(1.6\times10^{-5}\) 至 \(2.5\times10^{-5}\)，多数聚合 DP 来自 within-context 项；
- 20% 对称代理翻转会使非忽略 DP 大致衰减 39%–43%；随机缺失对 DP 大小影响较小；
- 当前没有证据表明日志选择是主要方向反转来源。

### 2.3 `position_corrected` 状态

当前数据不包含真实 `logging_propensity`，且无法从数据提供方获取额外信息。因此正式冻结：

```yaml
selection_protocols:
  all_logged:
    status: available
  random_display:
    status: available
  context_conditioned:
    status: available
  position_corrected:
    status: unavailable_missing_external_propensity
```

Stage1.2 不批准单独开发 logging-policy 模型，原因是当前缺少对真实候选集合、历史策略可见特征、动作空间和真实展示概率的充分观测，无法验证估计 propensity 是否对应实际 logging policy。

这一缺失：

- **不阻塞**表示泄漏、模型干预和观测日志范围内的公平研究；
- **限制**无偏 off-policy evaluation、完整位置校正和因果公平声明；
- **不能解释**当前 M5A 失败，因为三个已有评价协议均未形成稳定 Pareto 改善，且 M4 未观察到方向反转。

### 2.4 M5A 三种子结果

数值为种子 2019、2020、2021 的均值 ± 样本标准差。

| Method | All AUC | All NLLH | All ECE | All DP | Random DP | Context DP | U | U_TILDE | All ECE gap | Random ECE gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.764927 ± 0.003964 | 0.042093 ± 0.001098 | 0.000839 ± 0.000895 | 0.000510 ± 0.000560 | 0.000602 ± 0.000792 | 0.000492 ± 0.000558 | 0.010210 ± 0.000194 | 0.012306 ± 0.000032 | 0.000347 ± 0.000186 | 0.001757 ± 0.000868 |
| global_suppression | 0.762338 ± 0.003014 | 0.038876 ± 0.000053 | 0.000596 ± 0.000283 | 0.000605 ± 0.000141 | 0.000591 ± 0.000097 | 0.000585 ± 0.000141 | 0.010460 ± 0.000087 | 0.012357 ± 0.000451 | 0.000225 ± 0.000145 | 0.001424 ± 0.000417 |
| matched_random_suppression | 0.762699 ± 0.005113 | 0.040553 ± 0.000091 | 0.001005 ± 0.000878 | 0.000627 ± 0.000477 | 0.000512 ± 0.000363 | 0.000608 ± 0.000475 | 0.010360 ± 0.000074 | 0.012508 ± 0.000077 | 0.000338 ± 0.000140 | 0.001826 ± 0.000954 |
| selective_suppression | 0.761707 ± 0.006766 | 0.040982 ± 0.000146 | 0.000687 ± 0.000506 | 0.000599 ± 0.000145 | 0.000539 ± 0.000148 | 0.000578 ± 0.000140 | 0.010253 ± 0.000083 | 0.012292 ± 0.000286 | 0.000211 ± 0.000160 | 0.001462 ± 0.000677 |
| dp_regularization | 0.764372 ± 0.002943 | 0.041439 ± 0.002041 | 0.000792 ± 0.000866 | 0.000597 ± 0.000496 | 0.000690 ± 0.000829 | 0.000579 ± 0.000496 | 0.010227 ± 0.000160 | 0.012379 ± 0.000144 | 0.000279 ± 0.000236 | 0.001620 ± 0.001006 |
| adversarial | 0.536575 ± 0.001734 | 0.154763 ± 0.004686 | 0.008709 ± 0.000716 | 0.000142 ± 0.000090 | 0.000653 ± 0.000503 | 0.000146 ± 0.000115 | 0.010376 ± 0.000155 | 0.013116 ± 0.000329 | 0.000332 ± 0.000194 | 0.002573 ± 0.000345 |

关键解释：

- selective suppression 的 AUC、all/context DP 和 U_TILDE 均没有优于 baseline；
- global suppression 改善 NLLH、ECE 和部分效用，但 DP 反而升高，可能主要体现正则化或校准效应；
- matched-random suppression 在部分指标上不差于 selective suppression，削弱了“目标路径选择具有独立价值”的证据；
- adversarial 的 all/context DP 较低，但 AUC 从 0.764927 降至 0.536575，NLLH 和 ECE 大幅恶化，不能被解释为有效公平改进；
- baseline 的 All DP 为 0.000510 ± 0.000560，跨种子标准差与均值同量级，提示可能存在指标地板或高方差问题。

---

## 3. Stage1.2 修订后的科学问题

### 3.1 核心问题

> 对同一组 CTR 模型和公平干预，表示级代理泄漏是否真的被改变；若被改变，这种变化是否稳定关联于 DP、组间校准、最差组预测质量和排序效用；若不关联，失配来自结果指标地板、泄漏迁移、普通正则化，还是过程泄漏本身与结果公平的结构性脱钩？

### 3.2 研究问题

#### RQ1：M5A 干预是否真正降低了目标表示中的代理泄漏？

对 baseline 与五种干预重新导出同一组表示，用冻结 probe 协议比较：

\[
\Delta \mathrm{Leak}_{m,l}
=
\mathrm{Leak}(H^{m}_{l},A)
-
\mathrm{Leak}(H^{baseline}_{l},A)
\]

其中 \(m\) 表示干预方法，\(l\) 表示层或路径。

#### RQ2：泄漏是否从被抑制层迁移到了其他层？

同时检查 embedding、cross layers、final representation、logit 和 probability。若目标层泄漏下降而相邻层或最终表示上升，则应判定为 leakage redistribution，而非成功去敏。

#### RQ3：结果变化是否可以由普通正则化或后处理校准解释？

比较 suppression 与：

- 更强 L2；
- dropout；
- 等容量随机剪枝或 width reduction；
- temperature scaling；
- beta calibration 或 isotonic calibration。

若简单对照能够复制 NLLH/ECE/效用改善，则 suppression 不能被解释为独立公平机制。

#### RQ4：adversarial 的低 DP 是否来自预测坍缩？

分析：

- prediction mean/std；
- logit mean/std；
- 正负样本分数间距；
- 分数熵和有效动态范围；
- 两组分数是否共同向常数收缩；
- AUC、NLLH、Brier、ECE 与 DP 的联合变化。

#### RQ5：当前 DP 是否有足够信噪比承载主要结论？

检查：

- `DP_signed`、`DP_abs` 和两组预测均值；
- impression-cluster bootstrap；
- group-wise AUC/NLLH/Brier/ECE；
- worst-group metric；
- calibration curve；
- score-distribution distance；
- impression 内排序或曝光差异；
- 三种可用 selection protocol 的一致性。

#### RQ6：哪一个 Stage2 方向得到证据支持？

Stage1.2 结束时必须从预先定义的方向中选择一个，不允许同时保留多个模糊主线。

---

## 4. Stage1.2 非协商约束

1. 保持 FairJob 顺序切分不变：前 80% 训练候选、后 20% 测试，冻结测试集仍为 214,446 行。
2. 所有预测和表示按同一 `row_id` 对齐，并使用同一原始 `test_meta.csv`。
3. `protected_attribute` 仍被定义为行为保护代理，不等同于已验证人口属性。
4. 报告使用 `proxy-excluded` / `proxy-included` 等科学命名；旧实验 ID 仅为兼容代码保留。
5. 不使用 test AUC、test DP 或 test utility 选择权重、模型或停止点。
6. 不因为 M5B 失败进行无边界大规模 suppression 权重搜索。
7. 在完成干预后 leakage 复测前，不宣称“泄漏与结果公平无关”。
8. 在证明模型确实降低 leakage 前，不宣称“当前方法成功去敏”。
9. `position_corrected` 保持 `unavailable_missing_external_propensity`；Stage1.2 不开发 logging-policy 模型。
10. 在 Stage1.2 决策门槛通过前，不扩展到五种子 DCNv2、DeepFM 泛化或更多近期 backbone。
11. 正式训练继续在服务器进行；源代码、数据、checkpoint、表示和预测保持目录隔离，并记录 git commit、配置和文件 hash。
12. 负结果必须保留，不得通过更换指标或选择性汇报隐藏。

---

## 5. Stage1.2 执行里程碑

| 里程碑 | 主要工作 | 是否需要新增长训练 | 退出门槛 |
|---|---|---:|---|
| S12-M0 | 冻结 Stage1.1 证据、代码和运行清单 | 否 | M3/M4/M5A 文件、commit、配置与 hash 可追溯 |
| S12-M1 | 干预后表示重新导出与一致性检查 | 通常否，优先复用 checkpoint | 六种方法 × 三种子表示完整、对齐且不改变预测 |
| S12-M2 | 冻结 probe 的 post-intervention leakage 审计 | 否或轻量 | 明确每种方法是否降低 leakage、是否发生迁移 |
| S12-M3 | 正则化、校准和预测坍缩归因 | 少量、分层批准 | suppression 效果与普通正则化/坍缩可区分 |
| S12-M4 | Outcome 信噪比、地板效应和协议稳健性 | 否 | 判断 DP 是否适合继续作为主 outcome |
| S12-M5 | Stage2 方向重新冻结 | 否 | 选择且只选择一个主方向，给出证据与边界 |

---

## 6. S12-M0 — 证据和运行基础冻结

### 6.1 必须冻结的输入

```text
execution_plan.md
m3_single_seed_diagnostic_report.md
m3_multiseed_diagnostic_report.md
m4_outcome_screening_report.md
m5a_three_seed_report.md
```

### 6.2 必须记录

```text
stage1_2_baseline_commit
m5a_training_commit
representation_export_commit
probe_commit
environment_versions
data_manifest_hash
prediction_hashes
checkpoint_hashes
probe_sample_row_hash
```

如当前报告未包含 M5A 训练 commit，不得猜测；应在 Stage1.2 第一个操作中从运行目录、日志或 checkpoint manifest 中恢复并记录。

### 6.3 Stage1.1 正式结项状态

```yaml
stage1_1:
  m3_interaction_leakage_screen:
    status: directional_pass
  m4_available_data_selection_screen:
    status: conditional_pass
  position_corrected:
    status: unavailable_missing_external_propensity
  m5b_fairness_utility_gate:
    status: failed
  selective_suppression_stage2_approval:
    status: not_approved
```

### 6.4 退出门槛

- 所有运行均可追溯到明确代码版本；
- 三个种子和六种 M5A 方法的预测、checkpoint、配置完整；
- 当前结果表能够由原始预测重新生成；
- 未把任何运行产物写入源代码 checkout。

---

## 7. S12-M1 — 干预后表示导出

### 7.1 必须覆盖的方法

| 方法 | 作用 |
|---|---|
| baseline | 参考模型 |
| global_suppression | 判断全局去敏是否改变 leakage |
| matched_random_suppression | 排除容量和随机抑制效应 |
| selective_suppression | 检验目标方法是否作用于其声称的高风险表示 |
| dp_regularization | 判断输出 DP 约束是否间接改变表示泄漏 |
| adversarial | 判断显著性能崩塌是否伴随真实 leakage removal |

### 7.2 必须覆盖的种子

```text
2019
2020
2021
```

Stage1.2 首轮只使用已有三种子。没有证据通过 S12-M2/M3 前，不新增五种子。

### 7.3 必须导出的表示

DCNv2 至少导出：

```text
embedding_flat
cross_layer_0
cross_layer_1
cross_layer_2
dcnv2_final
logit
probability
```

如现有 suppression 直接作用于某个更细粒度 path/gate，还应导出：

```text
pre_gate_representation
post_gate_representation
gate_values_or_mask
suppressed_component
residual_component
```

### 7.4 表示验收

每个表示 root 必须通过：

- schema；
- shape；
- finite value；
- shard hash；
- row ordering；
- train/valid/test split；
- 跨方法完全一致的 probe sample row IDs；
- representation export 前后 prediction tolerance。

### 7.5 退出门槛

六种方法 × 三种子均能在同一冻结行集合上提取可比较表示；任何缺失或错位运行必须在 probe 前修复。

---

## 8. S12-M2 — 干预后泄漏审计

### 8.1 Probe 协议

保持 CTR 训练、表示导出和 probe 训练分离：

- probe train 只用于拟合；
- probe validation 只用于选择容量和正则；
- probe test 只读一次；
- 所有方法、种子和层使用同一候选容量、相同样本和相同评估代码；
- 同时报告线性 probe 与有限容量非线性 probe；
- 显式记录 `converged`、迭代数和停止原因。

M3 中多数非线性 probe 达到 100 迭代上限。Stage1.2 必须进行 convergence stability 检查，例如比较 100、300、500 次迭代或等价的验证早停设置；若 AUC 仍显著漂移，不得把单一数值当作稳定估计。

### 8.2 主要指标

对每种方法 \(m\)、种子 \(s\)、层 \(l\) 报告：

```text
probe_auc
probe_balanced_accuracy
probe_brier_or_logloss
converged
n_iterations
input_probe_auc
amplification
normalized_amplification
change_vs_baseline
change_vs_matched_random
```

定义：

\[
\Delta Leak_{m,s,l}
=Leak_{m,s,l}-Leak_{baseline,s,l}
\]

\[
TargetedAdvantage_{s,l}
=\Delta Leak_{random,s,l}-\Delta Leak_{selective,s,l}
\]

当数值越低表示 leakage 越少时，`TargetedAdvantage > 0` 表示 selective 比 matched-random 更有效。

### 8.3 泄漏控制成功的筛选条件

一个方法在 Stage1.2 中被判定为“成功作用于目标 leakage”，至少需要：

1. 目标层 leakage 相对 baseline 在三个种子上方向一致下降；
2. selective 的平均下降大于 matched-random；
3. 结果对合理 probe 容量和收敛预算稳定；
4. 下降不是由表示方差接近零或模型整体坍缩产生；
5. 最终表示、logit 或相邻层未出现同量级的 compensating leakage increase；
6. 同一结论能由至少一种线性和一种非线性 probe 支持，或对二者不一致给出明确解释。

该条件是阶段筛选门槛，不等同于正式期刊显著性标准。正式主张仍需五种子和适当 bootstrap/统计检验。

### 8.4 关键输出分类

#### A. 未成功降低 leakage

```text
目标层 leakage 不降
或 selective 不优于 random
或 leakage 转移到其他层
```

结论只能是“当前训练机制未成功干预目标表示”，不能推断 leakage 与 outcome 无关。

#### B. 成功降低 leakage，但 outcome 未改善

```text
leakage 稳定下降
AUC/DP/校准/效用不随之改善
且不存在明显坍缩或迁移
```

该结果支持“representation leakage 与 outcome fairness 脱钩”的候选主线。

#### C. leakage 与 outcome 同时改善

若出现此前报告未覆盖的可靠结果，必须确认其来源不是 test-driven 重新选择。只有在预先冻结的现有运行或严格 validation-only 新运行中出现，才可考虑恢复方法路线。

### 8.5 退出门槛

必须给出每种 M5A 方法的明确标签：

```text
leakage_reduced
leakage_not_reduced
leakage_redistributed
inconclusive_probe_instability
model_collapsed
```

不允许只给 probe 表而不作机制分类。

---

## 9. S12-M3 — 普通正则化、校准和预测坍缩归因

### 9.1 分层执行原则

优先使用低成本、无需重训的分析；只有其不能回答问题时，才批准少量新训练。

### Tier 1：无需或低成本重训

- temperature scaling；
- beta calibration；
- isotonic calibration；
- score mean/variance/entropy；
- logit distribution；
- 正负样本分离度；
- group-wise score contraction；
- 与 baseline 相同预测上的后处理对照。

所有校准器只在 validation 上拟合，test 只用于最终读取。

### Tier 2：有限训练对照

仅在 Tier 1 无法解释 global/selective suppression 的改善时运行：

- stronger L2；
- dropout；
- width reduction；
- matched-capacity random pruning。

每类最多使用与现有 suppression 相匹配的有限候选数，并保持相同种子、早停和 tuning budget。不得开启大规模超参数搜索。

### 9.2 Adversarial 坍缩判定

adversarial 不能因为 DP 较低就被认为成功。至少检查：

- AUC 是否接近随机；
- prediction std 是否显著低于 baseline；
- logit 动态范围是否显著收缩；
- 正负样本分数间距是否消失；
- 两组预测均值是否同时靠近全局均值；
- ECE/NLLH/Brier 是否恶化；
- leakage 是否下降，以及下降是否仅来自表示坍缩。

如果低 DP 主要来自输出接近常数，则标记为：

```text
fairness_by_prediction_collapse
```

不得作为有效 Pareto 点。

### 9.3 普通正则化归因

如果 L2/dropout/width reduction 或 calibration 可以在不使用保护代理监督的情况下复制 global/selective suppression 的 NLLH、ECE 或效用变化，则报告：

```text
observed_gain_explained_by_generic_regularization_or_calibration
```

这不否定工程收益，但否定其作为公平机制 novelty 的独立性。

### 9.4 退出门槛

对 global、random、selective、adversarial 分别回答：

1. 是否降低 leakage；
2. 是否发生 leakage migration；
3. 是否发生 prediction collapse；
4. 指标改善是否可由普通正则化或校准复制；
5. 是否仍存在不可由简单对照解释的公平机制效应。

---

## 10. S12-M4 — Outcome 指标信噪比与地板效应

### 10.1 为什么必须检查

DCNv2 baseline 的三种子 All DP 为：

\[
0.000510\pm0.000560
\]

标准差与均值同量级，且 M4 中 DCNv2 seed 2019 的 bootstrap 区间跨零。若主 outcome 接近地板或对随机初始化高度敏感，任何小幅干预效应都难以稳定识别。

### 10.2 必须报告的结果

#### 预测差异

```text
DP_signed
DP_abs
group_0_mean_prediction
group_1_mean_prediction
prediction_ratio
group_counts
```

#### 组内预测质量

```text
group-wise AUC
group-wise NLLH
group-wise Brier
group-wise ECE
worst-group metric
between-group gap
```

#### 分布与排序

```text
score distribution distance
calibration curves
within-impression ranking gap
U
U_TILDE
```

#### 稳健性

```text
all_logged
random_display
context_conditioned
impression-cluster bootstrap
proxy flip and missingness sensitivity
```

### 10.3 DP 可继续作为主 outcome 的条件

DP 只有在以下条件下才继续承担 Stage2 主结果：

- 多数 baseline 种子不处于近零不可分辨区；
- 干预效应相对种子波动和 cluster-bootstrap 区间足够大；
- all/random/context 三种协议方向不矛盾；
- 改善不是由统一分数压缩造成；
- 至少有一个组内质量或排序效用指标与其形成可解释的联合变化。

若上述条件不满足，DP 仍保留为报告指标，但不能继续作为唯一或主要 optimization target。

### 10.4 退出门槛

必须给出以下三选一判断：

```text
DP_informative_primary_outcome
DP_secondary_high_variance_metric
DP_near_floor_for_current_setting
```

并说明证据。

---

## 11. Stage2 方向重新冻结规则

Stage1.2 结束时必须选择以下一个方向。选择依据是 S12-M2 至 S12-M4 的结果，而不是研究者偏好。

### 方向 A：重新设计 leakage intervention

**触发条件：**

- M3 候选现象仍稳定；
- 当前 selective/global 方法没有真正降低 leakage，或出现明显 leakage migration；
- 现有结果不能用于判断 leakage—outcome 关系。

**Stage2 主问题：**

> 如何防止行为保护代理信息绕过局部 gate，并在多层、多路径之间迁移？

**允许的方法方向：**

- layer-distributed leakage constraint；
- path-level adversary；
- leakage redistribution penalty；
- adaptive sparse gate；
- target-layer + final-layer joint control。

**限制：** 不得沿用当前 selective suppression 并仅扩大权重搜索。

### 方向 B：泄漏—结果公平脱钩审计

**触发条件：**

- 一个或多个方法稳定降低隐藏表示 leakage；
- 无明显 leakage migration 或 prediction collapse；
- DP、校准、组内质量或效用未稳定改善；
- matched-random/global/DP/adversarial 对照共同表明 leakage reduction 不是可靠 outcome proxy。

**Stage2 主问题：**

> 表示中的行为保护代理可预测性在什么条件下才构成“有害泄漏”，为什么降低一般 representation leakage 不能保证结果公平？

**潜在贡献：**

- 区分 input proxy、representation amplification 与 outcome disparity；
- 提出 harmful-leakage 判定协议；
- 揭示 adversarial fairness-by-collapse；
- 系统比较泄漏、DP、校准和效用的失配。

### 方向 C：代理不确定性下的组间校准

**触发条件：**

- DP 接近地板或高方差；
- group-wise ECE/Brier/NLLH gap 比 DP 更稳定；
- suppression 的主要可重复效果集中在校准，而不是 DP；
- 简单 group-blind calibration 不能完全解释或解决该差异。

**Stage2 主问题：**

> 在训练时可访问行为保护代理、推理时不依赖该代理的条件下，如何改善不确定代理分组下的最差组概率质量？

### 方向 D：公平测量与可重复性审计

**触发条件：**

- leakage、DP、校准和效用均对种子、probe、协议或代理扰动高度不稳定；
- 当前数据无法支持稳定 mitigation target；
- 多数方法差异落在估计噪声范围内。

**Stage2 主问题：**

> CTR 公平结论对行为保护代理、随机种子、日志范围和指标选择有多敏感，现有评估协议何时产生不可重复或相互矛盾的结论？

### 方向 E：有限恢复方法路线

**仅在严格条件下允许：**

- 预先冻结、非 test-driven 的现有或 validation-only 新运行显示 leakage 与至少一个结果公平指标共同稳定改善；
- selective 明确优于 matched-random 和 global；
- 无 prediction collapse；
- 三种 selection protocol 方向一致；
- M5B 失败原因可由已修复的实现或目标失配明确解释，而不是单纯更换成功标准。

满足后才批准：

```text
DCNv2 扩展至五种子
→ DeepFM 跨 backbone 验证
→ 正式 Pareto 与 bootstrap
```

该方向默认不被批准，除非 Stage1.2 产生新的、可审计证据。

---

## 12. 推荐的最小实验矩阵

### 12.1 必须完成

| 类别 | 配置 |
|---|---|
| Backbone | DCNv2 no-user-id |
| Seeds | 2019、2020、2021 |
| Methods | baseline、global、matched-random、selective、DP regularization、adversarial |
| Representations | embedding、cross0/1/2、final、logit、probability |
| Probe | linear + matched-capacity nonlinear + convergence audit |
| Outcomes | AUC、NLLH、Brier、ECE、DP signed/abs、group metrics、U/U_TILDE |
| Protocols | all_logged、random_display、context_conditioned |
| Diagnostics | score/logit distributions、collapse、leakage migration |

### 12.2 条件执行

只有 S12-M2 发现 leakage 确实下降，才运行有限正则化/校准对照；只有 Stage1.2 方向 E 被触发，才扩展五种子或 DeepFM。

### 12.3 暂不执行

```text
logging-policy model
position-corrected evaluation
更多近期 CTR backbone
大规模 suppression weight sweep
新的复杂多模块组合
test-guided Pareto selection
```

---

## 13. 统计与报告规范

### 13.1 当前三种子的用途

三种子用于 Stage1.2 机制判定和方向筛选，不直接作为最终期刊主表。

### 13.2 正式主张要求

进入 Stage2 后的正式结论至少需要：

- 五个独立 CTR seeds；
- mean、sample standard deviation；
- impression 或 user cluster-bootstrap interval；
- validation-only hyperparameter selection；
- matched tuning budget；
- 预先定义的 Pareto 选择规则；
- 对每个主张列出替代解释和排除证据。

### 13.3 多重比较

Stage1.2 会跨方法、层和指标进行多项比较。报告必须区分：

- exploratory diagnostic；
- preregistered gate；
- formal statistical claim。

M3 中十个 random masks 的最小经验 p 值为 1/11，不能作为正式显著性证据。Stage1.2 如继续使用随机干预，应增加重复次数或使用适合的配对 bootstrap/置换检验。

### 13.4 不允许的结果选择

- 只汇报有利种子；
- 只选择最有利层；
- 根据 test DP 选择 suppression strength；
- adversarial AUC 坍缩后只强调 DP；
- selective 不优于 random 时省略 random；
- DP 不显著后临时把 ECE 改为唯一主指标而不说明方向调整。

---

## 14. 工程与产物规划

### 14.1 建议目录

```text
/root/autodl-tmp/workdirs/JobFairness/stage1_2/
  manifests/
  representations/
  probes/
  leakage_audit/
  collapse_diagnostics/
  regularization_controls/
  calibration_controls/
  bootstrap/
  reports/
  runtime_config/
```

### 14.2 建议新增代码或脚本

```text
fuxictr_ext/fairjob/post_intervention_representation_export.py
fuxictr_ext/fairjob/leakage_migration_audit.py
fuxictr_ext/fairjob/probe_convergence_audit.py
fuxictr_ext/fairjob/prediction_collapse_diagnostics.py
fuxictr_ext/fairjob/regularization_controls.py
fuxictr_ext/fairjob/calibration_controls.py
fuxictr_ext/fairjob/outcome_floor_audit.py
fuxictr_ext/fairjob/build_stage1_2_report.py
```

文件名是职责建议，可根据现有项目结构合并；不应为满足文档而无必要拆分代码。

### 14.3 建议报告文件

```text
docs/fairjob/stage1_2/
  stage1_1_closure.md
  post_intervention_leakage_report.md
  leakage_migration_report.md
  regularization_and_collapse_report.md
  outcome_floor_and_stability_report.md
  stage1_2_decision_report.md
```

### 14.4 统一实验主键

```text
backbone
protocol
proxy_regime
fairness_method
seed
feature_ablation
selection_protocol
representation_name
probe_type
config_hash
git_commit
```

---

## 15. Stage1.2 完成标准

Stage1.2 只有在以下条件全部满足后才能结项。

### 15.1 证据冻结

- [ ] M3、M4、M5A 的代码、配置、checkpoint 和结果 hash 可追溯；
- [ ] Stage1.1 正式记录为 M3 directional pass、M4 conditional pass、M5B failed；
- [ ] `position_corrected` 明确标记 unavailable，不再阻塞当前研究。

### 15.2 干预有效性

- [ ] 六种 M5A 方法均完成相同层级的 post-intervention representation export；
- [ ] probe convergence stability 已检查；
- [ ] 每种方法均被分类为 leakage reduced / not reduced / redistributed / collapsed / inconclusive；
- [ ] selective 与 matched-random 的 leakage 差异已直接比较。

### 15.3 失败归因

- [ ] 已判断 suppression 的 NLLH/ECE 改善是否可由普通正则化或校准复制；
- [ ] 已判断 adversarial 的低 DP 是否来自预测坍缩；
- [ ] 已检查 leakage 是否迁移到其他层；
- [ ] 已区分“方法未干预机制”与“机制—结果脱钩”。

### 15.4 Outcome 有效性

- [ ] 报告 DP signed/abs、两组均值和 cluster bootstrap；
- [ ] 报告 group-wise AUC/NLLH/Brier/ECE 和 worst-group 指标；
- [ ] 明确 DP 是主 outcome、次要高方差指标，还是当前设置下接近地板；
- [ ] all/random/context 三种协议的差异已解释。

### 15.5 方向决策

- [ ] 从 A–E 中选择且只选择一个 Stage2 主方向；
- [ ] 记录未选择方向及其缺失证据；
- [ ] 冻结下一阶段 backbone、seed、方法和评价矩阵；
- [ ] 不通过扩大搜索或更换成功标准掩盖 M5B 负结果。

---

## 16. 论文创新性的当前判断

### 16.1 当前不能作为核心 novelty 的内容

- “选择性路径抑制稳定改善 FairJob 结果公平”；
- “降低 representation leakage 必然降低 DP”；
- “adversarial 去敏形成良好公平—效用权衡”；
- “缺少 position correction 是 M5A 失败主因”；
- “M3 的 probe AUC 增长已经证明实际歧视机制”。

### 16.2 Stage1.2 可能形成的 TOIS 级问题

若 S12-M2 证明 leakage 稳定下降而结果公平不改善，潜在创新主线可重构为：

> 代理信息的输入可预测性、表示增量放大和结果群体差异是三个不能互换的对象；现有公平表示方法可能降低可解码性，却无法保证 outcome fairness，甚至可能通过预测坍缩制造表面公平。

这一方向要达到 TOIS 水平，仍需：

- 跨 backbone 或跨数据验证；
- 更充分的 probe 和干预统计；
- 有害泄漏的形式化判定；
- 对 leakage、DP、calibration、utility 失配的机制解释；
- 半合成或辅助数据中的受控验证。

若当前方法根本没有降低 leakage，则 Stage1.2 的价值是防止得出错误结论，并为新的多层泄漏控制方法提供明确问题定义；此时论文 novelty 仍需在 Stage2 重新建立。

---

## 17. 风险与应对

| 风险 | 含义 | 应对 |
|---|---|---|
| Probe 不收敛 | leakage 数值可能是容量下界或训练不稳定 | 增加受控预算、早停和容量稳定性分析 |
| Leakage migration | 目标层下降但其他层上升 | 全层审计，联合约束而非局部 gate |
| DP 地板效应 | 干预空间小、种子噪声大 | 保留 DP，增加组内质量与排序结果，重新判断主 outcome |
| Prediction collapse | 低 DP 来自接近常数预测 | 分数分布、AUC/NLLH/Brier/ECE 联合判定 |
| Generic regularization | suppression 改善不是公平机制独有 | L2/dropout/width/calibration 对照 |
| Proxy measurement uncertainty | 行为代理误差改变结果差异 | 保留 flip/missingness 敏感性，不等同真实性别 |
| Missing propensity | 无法做完整 position correction | 标记 unavailable，限制因果声明，不开发不可验证策略模型 |
| Scope expansion | 失败后不断增加模型和搜索 | 采用分层 gate，未通过不得扩展 |

---

## 18. 建议的立即执行顺序

```text
1. 冻结 Stage1.1 closure 与 M5A 运行 manifest
2. 复用六种方法 × 三种子的 checkpoint 导出统一表示
3. 使用冻结 probe 做 convergence audit 和 post-intervention leakage 测量
4. 判断 selective 是否真的降低 leakage、是否优于 random、是否发生迁移
5. 对 adversarial 做 prediction-collapse 诊断
6. 对 global/selective 的 NLLH/ECE 改善做 calibration/regularization 对照
7. 完成 DP 地板、组内指标和三协议稳定性报告
8. 依据预定规则选择唯一 Stage2 方向
```

任何新的长训练都应在第 4 步之后、且有明确诊断理由时才批准。

---

## 19. 最终阶段定位

Stage1.2 的成功标准不是把 M5A 调成“成功”，而是把失败定位到证据链中的准确位置：

```text
原始输入代理
→ 表示增量放大
→ 训练级泄漏干预
→ 输出分布和校准
→ DP / 组内质量 / 排序效用
```

只有明确哪一个箭头没有成立，Stage2 才有可靠的研究问题。

最准确的阶段标签是：

> FairJob × FuxiCTR 已观察到小幅、跨模型的表示级代理放大候选现象，但首轮选择性抑制未形成稳定结果公平收益。Stage1.2 通过干预后泄漏复测、泄漏迁移审计、普通正则化与预测坍缩对照、结果指标地板分析，决定后续应重新设计干预、转向泄漏—结果脱钩审计、研究代理不确定性校准，还是开展公平测量可重复性研究。

---

## 20. 本文依据的阶段材料

1. `execution_plan.md` — Stage1.1 的里程碑、冻结协议和 Stage2 五方向决策框架。
2. `m3_single_seed_diagnostic_report.md` — 单种子身份代理、层级 probe 和 no-user-id 候选信号。
3. `m3_multiseed_diagnostic_report.md` — 三种子 amplification 与 targeted/matched-random masking 结果。
4. `m4_outcome_screening_report.md` — all/random selection scope、上下文分解、代理噪声和 propensity blocker。
5. `m5a_three_seed_report.md` — DCNv2 三种子公平干预结果及 M5B gate failed 决策。

