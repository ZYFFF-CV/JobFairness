# Stage1.1 — FairJob × FuxiCTR 科学问题校准、诊断验证与研究路线冻结

**最后更新：** 2026-07-20  
**阶段名称：** Stage1.1  
**项目：** FairJob × FuxiCTR CTR Fairness Research  
**文档定位：** 独立、完整的阶段说明与执行规范；阅读本文不需要预先了解 Stage1.0。  
**当前研究定位：** 以可靠工程基础为起点，完成任务语义、保护代理、指标与实验协议校准，验证可发表的公平失效机制，再决定 Stage2 方法方向。  
**复现边界：** 当前属于 FairJob × FuxiCTR integration-mode research，不声称逐项严格复现 FairJob 原论文。

---

## 0. 一句话记忆锚点

Stage1.1 的任务不是继续堆叠 CTR 模型，而是回答一个决定后续论文成败的问题：

> 现代 CTR 特征交互模型是否会在原始输入已有信息之外，进一步放大行为保护代理的信息；这种放大在何种任务、日志选择和位置条件下转化为结果差异；能否只抑制有害交互路径，而不是全局删除所有与保护代理相关的预测信息。

阶段流程：

```text
FairJob 数据与统一 evaluator
→ 任务语义和保护代理边界冻结
→ DeepFM/DCNv2 层级与路径级诊断
→ 泄漏、校准、DP、排序效用和选择偏差分解
→ 公平方法基线与反事实对照
→ Stage2 研究方向决策门槛
```

---

## 1. 阶段结论与调整原因

### 1.1 当前方向是否有达到 ACM TOIS 水平的潜力

有潜力，但必须调整研究组织方式。

FairJob 提供了一个较少见的真实研究组合：点击日志、行为型保护代理、位置、随机展示、用户与职位标识、展示上下文以及公平效用指标。这些信息能够支持对以下关系的系统研究：

```text
原始代理信息
→ CTR 表示与高阶交互
→ 预测分数和校准
→ 群体差异
→ 排名/展示效用
→ 日志选择与位置偏差
```

但是，现有工程结果尚不能支持“模型利用敏感信息提高准确率并必然恶化公平性”的简单叙事。当前 LR/XGB 结果表现为模型、指标和任务条件相关的不一致关系。因此，Stage1.1 必须从“预设公平退化并设计模块”调整为“先验证现象、排除替代解释、再选择方法”。

### 1.2 核心调整

Stage1.1 对原研究路线做以下实质性修改：

| 原计划倾向 | Stage1.1 调整 |
|---|---|
| 将 `unaware` 直接解释为模型不知道敏感信息 | 改为 `proxy-excluded`；不输入保护代理不代表模型无法从其他特征恢复代理信息。 |
| 将 `unfair` 直接解释为不公平模型 | 改为 `proxy-included`；输入保护代理不必然使所有公平指标恶化。 |
| 默认把 `protected_attribute` 当作真实性别 | 明确其为行为保护代理，不直接等同潜在真实人口属性。 |
| 默认研究一个统一 CTR 任务 | 分离展示前排序与展示后点击估计，分别规定 `rank`、`displayrandom` 和 `impression_id` 的角色。 |
| 先实现大量近期 CTR 模型 | 先完成代表性 backbone、机制诊断和公平基线；近期模型只按研究需要选取。 |
| 只使用 AUC/NLLH/DP/U/U_TILDE | 增加组内性能、校准、层级泄漏、增量泄漏、上下文分解和选择偏差诊断。 |
| 用固定加权公平损失直接进入 Stage2 | 优先考虑约束优化、自适应对偶或 Pareto 分析；方法必须由诊断结果触发。 |
| 只在 FairJob 上验证最终主张 | 主结果使用 FairJob，同时规划半合成受控验证和一个真实敏感属性辅助数据集。 |

---

## 2. 项目科学问题与论文目标

### 2.1 修订后的核心研究问题

本项目研究：

> 在具有行为保护代理、位置偏差和展示选择机制的 job-ad CTR 场景中，现代特征交互模型是否会产生超出原始输入可预测性的代理信息放大；哪些字段、交互路径和模型层负责这种放大；代理信息放大是否与人口统计差异、组间校准差和排序效用变化稳定相关；选择性路径抑制是否能够比全局去敏保留更多 CTR utility。

### 2.2 预期论文贡献结构

目标论文应形成以下证据链，而不是只提交一个新的正则项：

1. **新现象：** 识别 CTR 表示或高阶特征交互中的增量代理泄漏，并证明其不是原始 ID 记忆或输入代理信息的平凡结果。
2. **机制定位：** 识别泄漏集中在哪些字段、交互路径、层级和业务上下文。
3. **结果关联：** 区分过程泄漏、DP、校准差、排序效用和日志选择偏差之间的关系。
4. **针对性方法：** 只抑制被诊断为高风险的路径，避免全局删除任务有效信息。
5. **优化依据：** 使用明确的公平约束、Pareto 前沿或自适应对偶优化，而不是任意固定权重。
6. **稳健验证：** 多种子、置信区间、随机路径对照、ID 消融、代理噪声、随机展示与选择校正。
7. **边界透明：** 不把行为代理直接等同真实性别，不把相关性结果包装为因果结论。

### 2.3 当前不得使用的结论

在 Stage1.1 完成前，不应声称：

- 深度 CTR 模型必然比线性或树模型更不公平；
- 模型不输入保护代理就实现了 fairness through unawareness；
- `protected_attribute` 是准确的真实性别标签；
- 表示可预测保护代理就必然造成结果伤害；
- DP 的单个数值变小就证明公平性全面改善；
- 当前实验严格复现 FairJob 原论文；
- 观察到的相关性证明了因果作用。

---

## 3. 关键术语、变量与声明边界

### 3.1 保护代理与潜在真实属性

本文使用以下符号：

- \(A\)：数据中观测到的 `protected_attribute`，即行为保护代理；
- \(S^*\)：不可直接观测的潜在真实人口属性；
- \(X\)：模型可用输入特征；
- \(Y\)：点击标签；
- \(H_l\)：模型第 \(l\) 层或某一交互模块的表示；
- \(\hat{Y}\)：CTR 预测概率。

FairJob 原始数据没有直接提供真实性别，而是提供由非保护行为信息构造的代理。因此，Stage1.1 默认研究：

- behavioral protected proxy；
- proxy leakage；
- proxy-conditioned disparity；
- robustness under uncertain protected labels。

除非后续明确建立 \(S^*\)、\(A\) 与测量误差机制，并给出可辨识假设，否则不使用“真实性别的反事实公平”或“消除真实性别因果效应”等表述。

### 3.2 训练阶段与推理阶段的代理使用

项目默认部署目标是：

> 允许训练阶段使用行为保护代理进行审计、约束、对抗训练或校准监督，但主方法在推理阶段不依赖该代理。

任何需要推理时代理输入的方法必须单独标识，不得与 no-proxy-inference 方法混合比较。

### 3.3 实验组命名

代码内部短期可以保留旧实验 ID 以避免破坏流水线，但报告、表格和论文统一使用以下名称：

| 旧名称 | Stage1.1 报告名称 | 定义 |
|---|---|---|
| `unaware` | `proxy-excluded` | 模型输入中不包含 `protected_attribute`；不代表模型无法利用其他代理特征。 |
| `unfair` | `proxy-included` | 模型显式输入行为保护代理；名称不预设公平结果。 |
| `ours` | `mitigated-train-proxy` | 训练时可使用代理进行公平监督，推理时默认不输入代理。 |
| 新增 | `mitigated-no-proxy` | 训练与推理均不直接访问代理，仅在离线审计中使用代理。 |

---

## 4. 当前工程基础与数据状态

### 4.1 仓库与数据位置

主仓库：

```text
D:\code\JobFairness
```

原始 FairJob 数据：

```text
D:\code\datasets\FairJob\fairjob.csv.gz
```

处理后数据：

```text
data/FairJob/processed/
```

处理目录应保持在 git ignore 中，不提交原始数据、处理数据、预测文件或本地大规模结果。

### 4.2 数据规模与字段

| 项目 | 数值 |
|---|---:|
| 总行数，不含表头 | 1,072,226 |
| 顺序测试集 20% | 214,446 |
| 前 80% 训练候选 | 857,780 |
| 测试集原始起始 offset | 857,780 |

主要字段：

```text
click, protected_attribute, senior, displayrandom, rank,
user_id, impression_id, product_id, cat0-cat12, num16-num50
```

### 4.3 已实现模块

代码目录：

```text
fuxictr_ext/fairjob/
```

| 文件 | 当前职责 |
|---|---|
| `prepare_fairjob.py` | 生成 full 与 smoke 顺序切分。 |
| `run_fuxictr_model.py` | FuxiCTR 模型运行包装与对齐预测导出。 |
| `baselines/xgb_baseline.py` | XGBoost 锚点基线。 |
| `prediction_io.py` | 统一预测 CSV schema 与对齐检查。 |
| `check_alignment.py` | 预测与元数据 CLI 对齐检查。 |
| `hparams.py` | 超参数来源解析和记录。 |
| `metrics.py` | NLLH、AUC、DP、U、U_TILDE。 |
| `evaluator.py` | 模型无关 evaluator。 |
| `check_metric_parity.py` | 合成数据指标 parity 检查。 |
| `build_report.py` | 构建集成结果表和 Markdown 报告。 |

配置目录：

```text
configs/fairjob/
```

| 文件 | 当前职责 |
|---|---|
| `dataset_config.yaml` | 由特征 manifest 生成的数据配置。 |
| `model_config.yaml` | 已有模型实验定义。 |
| `feature_schema.yaml` | 特征 schema。 |
| `reference_hparams.yaml` | 手工或 fallback 参数策略。 |
| `xgb_reference_hparams.yaml` | XGB integration-mode 固定参数。 |
| `make_dataset_config.py` | 生成数据配置。 |

### 4.4 已冻结的数据切分

全量顺序切分：

| 文件 | 行数 |
|---|---:|
| `train_full.csv` | 857,780 |
| `train.csv` | 807,780 |
| `valid.csv` | 50,000 |
| `test.csv` | 214,446 |
| `test_meta.csv` | 214,446 |

Smoke 切分：

| 文件 | 行数 |
|---|---:|
| `smoke_train.csv` | 36,000 |
| `smoke_valid.csv` | 4,000 |
| `smoke_test.csv` | 10,000 |
| `smoke_test_meta.csv` | 10,000 |

Stage1.1 不随意修改主顺序切分。任何新切分仅用于诊断，并须保留原始顺序测试集作为主结果测试集。

### 4.5 统一评价元数据规则

所有模型必须使用同一 `test_meta.csv`，且以 `row_id` 对齐。公平指标必须读取原始元数据，不使用 FuxiCTR 编码后的离散 ID 替代原始保护代理或分组字段。

必需字段：

```text
row_id
click
protected_attribute
senior
displayrandom
impression_id
product_id
```

---

## 5. 当前锚点结果及其正确解释

### 5.1 Full integration 结果

| Model | 报告组名 | NLLH ↓ | AUC ↑ | 当前 DP | U | U_TILDE | n_rows |
|---|---|---:|---:|---:|---:|---:|---:|
| LR | proxy-excluded | 0.07196 | 0.46996 | 0.00129 | 0.00964 | 0.01114 | 214,446 |
| LR | proxy-included | 0.05373 | 0.58134 | 0.00011 | 0.01064 | 0.01270 | 214,446 |
| XGB | proxy-excluded | 0.08222 | 0.74659 | 0.01208 | 0.01013 | 0.01133 | 214,446 |
| XGB | proxy-included | 0.08216 | 0.76424 | 0.01166 | 0.01004 | 0.01087 | 214,446 |

### 5.2 可以得出的结论

- LR/XGB 流水线、标签方向、预测导出和行对齐已基本可用。
- 显式加入行为保护代理对 LR 和 XGB 的 AUC 均有提升。
- 不同公平与效用指标没有呈现统一方向。
- XGB 的 U 与 U_TILDE 在加入代理后略有下降，但变化较小。
- LR 在当前实现的五项指标上均向表面较优方向变化。
- 现有结果支持“公平影响依赖模型、指标和协议”，不支持“代理信息必然导致统一的公平退化”。

### 5.3 暂时不能得出的结论

- 当前 `DP` 尚未完成 signed/absolute 语义冻结，不能仅按“越小越好”解释。
- LR 的 AUC < 0.5 需要记录并诊断，但不能据此认定整体工程链路错误。
- XGB 与原论文结果的差距可能来自超参数、特征处理、校准和实现差异；当前不属于严格复现。
- 锚点结果不能替代 DeepFM/DCNv2 的层级与交互诊断。

---

## 6. 对标 ACM TOIS 相似工作的创新设计规律

Stage1.1 不要求复制任何一篇论文的方法，而是采用这些工作共同体现的创新组织方式。

### 6.1 Multi-FR：把固定加权重构为多目标优化

**论文：** *A Multi-Objective Optimization Framework for Multi-Stakeholder Fairness-Aware Recommendation*。

其核心不是再增加一个公平损失，而是：

```text
多利益相关方目标冲突
→ 多目标形式化
→ 可微公平目标近似
→ 自适应多梯度优化
→ Pareto 解与部署选择规则
```

对本项目的启示：DP、泄漏、校准和排序效用并不天然同向。Stage2 不应只使用未经解释的固定 \(\lambda\) 加权，应至少比较固定权重与自适应约束/Pareto 优化。

### 6.2 FAiR：从全局统一偏差升级为局部异质性偏差

**论文：** *Mitigating Popularity Bias for Users and Items with Fairness-centric Adaptive Recommendation*。

其创新是先证明不同用户和物品受到偏差影响的程度不同，再设计局部与全局两级、自适应的公平机制。

对本项目的启示：保护代理泄漏可能集中在少数用户、职位、类别、campaign、位置或交互路径。统一 adversary 可能过度删除任务信息；路径或上下文级选择性处理更有研究价值。

### 6.3 CFFair：先定义公平 estimand，再处理敏感标签缺失

**论文：** *Average User-Side Counterfactual Fairness for Collaborative Filtering*。

其结构是：

```text
统计公平语义不足
→ 反事实公平定义
→ 可辨识条件
→ propensity/标签缺失处理
→ 可训练目标
```

对本项目的启示：FairJob 的保护信息是代理，不是已验证的真实人口属性。除非建立代理测量误差模型和可辨识条件，否则应限制在 proxy-conditioned fairness 与稳健性分析，不使用过强因果语言。

### 6.4 FMMRec：先发现模型结构带来的新泄漏现象，再设计机制

**论文：** *Causality-Inspired Fair Representation Learning for Multimodal Recommendation*。

其高价值结构是：

```text
发现模态复杂度与敏感信息泄漏现象
→ 对泄漏来源进行表示级诊断
→ 设计与机制对应的表示分解和过滤
→ 同时评价推荐效果、结果公平和表示泄漏
```

对本项目的启示：最有希望的主线不是“实现通用公平正则项”，而是验证 CTR 特征交互是否产生增量代理泄漏，并把方法直接作用于被定位的高风险交互路径。

### 6.5 User Feature Balancing：把选择偏差、理想目标和可计算近似连成证据链

**论文：** *Debiased Recommendation with User Feature Balancing*。

其结构包括理想无偏风险、分布平衡、目标上界、可扩展近似以及合成/半合成/真实数据验证。

对本项目的启示：FairJob 中的 campaign selection、position 和 randomized display 不能只作为评价细节。若日志选择主导结果差异，表示去敏并未处理真实问题，应转向 selection-aware fairness。

### 6.6 TOIS 代表作的共同要求

| 层次 | 期刊级要求 | Stage1.1 对应动作 |
|---|---|---|
| 新问题或新现象 | 证明现有方法忽略了稳定、非平凡的失效机制 | 验证“特征交互产生增量代理泄漏”或发现替代主机制。 |
| 严格形式化 | 明确公平目标、变量与假设 | 区分代理与真实属性，冻结任务协议和公平指标。 |
| 方法与机制同构 | 每个模块对应一个被诊断的问题 | 选择性路径抑制、校准或选择校正由诊断结果触发。 |
| 优化或理论依据 | 约束、Pareto、上界或可辨识分析 | 自适应约束优化；可选表示差异上界或代理噪声敏感性界。 |
| 完整证据链 | 多数据条件、消融、敏感性、复杂度、置信区间 | ID 消融、随机路径对照、代理噪声、选择协议、多种子。 |

---

## 7. 当前项目与 TOIS 水平之间的差距

| 维度 | 当前基础 | 缺口 | Stage1.1 优先级 |
|---|---|---|---|
| 工程可复现性 | 顺序切分、统一预测、对齐 evaluator、LR/XGB 锚点已建立 | 需要扩展到深模型表示导出和统一协议矩阵 | P0 |
| 核心科学命题 | 推测现代 CTR 学习代理信息 | 尚未证明增量泄漏、路径集中性和结果关联 | P0 |
| 任务语义 | CTR、排序和展示后预测混合 | `rank`、`displayrandom` 的输入合法性未冻结 | P0 |
| 保护属性语义 | 当前称 protected attribute | 行为代理不等同真实人口属性 | P0 |
| 公平指标 | DP、U、U_TILDE 已实现 | DP signed/absolute、组均值和 parity 未完整冻结 | P0 |
| 机制诊断 | 仅在计划中 | 缺少输入基线、层级 probe、路径归因和随机对照 | P0 |
| 公平方法基线 | 尚不完整 | 需要 DP、MMD/HSIC、adversarial、calibration、selection correction | P0/P1 |
| 优化方式 | 尚未确定 | 固定权重不足以处理冲突目标 | P1 |
| 泛化证据 | 单一 FairJob | 需要半合成机制验证和辅助数据 | P1 |
| 统计证据 | 尚未冻结 | 需要多种子、cluster bootstrap、置信区间和 Pareto 曲线 | P0 |

---

## 8. Stage1.1 阶段边界

### 8.1 Stage1.1 包含

- 保留并复核现有 FairJob × FuxiCTR 工程基础；
- 冻结展示前与展示后两个任务协议；
- 冻结行为保护代理的术语和使用政策；
- 修订实验组名称；
- 完成 DP、U/U_TILDE 和组内指标协议；
- 实现 DeepFM、DCNv2 的表示导出与层级泄漏诊断；
- 完成 ID、字段、交互和日志选择消融；
- 加入最小但充分的公平方法基线；
- 形成 Stage2 方向决策报告；
- 给出半合成和辅助数据验证方案。

### 8.2 Stage1.1 不包含

- 在未验证现象前冻结最终方法；
- 为主表实现所有近期 CTR 模型；
- 在 test set 上选择公平强度；
- 将攻击器 AUC 降至 0.5 直接当作公平成功；
- 使用推理时不可获得的字段构造主部署结论；
- 把单次运行或单数据集现象包装为普遍规律；
- 声称严格复现 FairJob 原论文。

---

## 9. 任务协议冻结

### 9.1 主任务：展示前候选打分与排序

定义：预测候选职位在尚未由当前策略确定展示位置前的点击倾向，用于候选打分或排序。

默认输入：

```text
user features
product/job features
context features available before ranking
```

默认不作为模型输入：

```text
rank
displayrandom
impression_id as a one-off identifier
```

这些字段的角色：

- `rank`：位置偏差分析、propensity 估计和结果分层；
- `displayrandom`：随机展示子样本评价或无偏效用估计；
- `impression_id`：候选集合重建、组内排序、cluster bootstrap 和 evaluator 聚合。

主论文若讨论“公平 CTR ranking”或“候选排序”，应以此协议为主。

### 9.2 辅助任务：展示后点击概率估计

定义：预测已在给定位置展示的广告是否被点击：

\[
P(Y=1\mid X, \text{rank}, \text{display policy})
\]

在该协议下，`rank` 和 `displayrandom` 可以作为模型输入，但结论必须明确限定为：

- post-display response modeling；
- 展示后点击概率与校准；
- 不直接等同展示前排序公平。

### 9.3 协议命名

| 协议 | 建议 ID | 主要用途 |
|---|---|---|
| 展示前 | `pre_ranking` | 主论文任务与可部署 CTR 排序。 |
| 展示后 | `post_display` | 位置、校准和日志机制补充分析。 |

两种协议不得在同一主表中直接比较后得出统一模型结论。

---

## 10. 特征与身份泄漏审计

### 10.1 `impression_id`

需检查：

- cardinality；
- train/valid/test 重叠；
- 每个 impression 的候选数；
- 是否在推理时具有稳定、可泛化语义；
- 是否接近一次性标识。

默认将其用作分组字段而不是 embedding 特征，除非审计证明其在部署前可获得且具有可复用语义。

### 10.2 `user_id`

保护代理由用户行为构造，因此 `user_id` 可能承担强代理记忆。必须运行：

- full-feature；
- no-user-id；
- no-product-id；
- no-user/product-ID；
- seen-user；
- unseen-user 或低频用户；
- 仅 category/numeric 特征。

如果深模型泄漏优势在 no-user-id 后消失，则主要现象是身份记忆，不应包装为高阶交互放大。

### 10.3 `rank` 与 `displayrandom`

必须分别审计：

- 是否显著提升 AUC/NLLH；
- 是否改变保护代理可预测性；
- 是否造成不同群体的校准差；
- 是否只是编码现有展示策略；
- 移除后公平结论是否反转。

---

## 11. Stage1.1 研究问题

### RQ1：模型复杂度是否产生增量代理泄漏

比较：

- LR；
- XGB；
- DeepFM；
- DCNv2；
- 一个近期高阶/MLP 模型，优先 DS-MLP 或 FCN。

表示提取位置：

- 原始可用输入；
- field embeddings；
- FM 二阶交互；
- DCNv2 各 cross layer；
- DNN hidden layers；
- logit；
- 最终 probability。

定义基本增量：

\[
\mathrm{Amp}_l = \mathrm{Leak}(H_l, A)-\mathrm{Leak}(X, A)
\]

同时报告归一化版本：

\[
\mathrm{NAmp}_l =
\frac{\mathrm{Leak}(H_l,A)-0.5}
{\max(\mathrm{Leak}(X,A)-0.5,\epsilon)}
\]

其中 leakage 主要使用严格独立 probe 测试集上的 AUC，并辅以 balanced accuracy。

### RQ2：哪些字段与交互路径负责代理放大

重点分析：

- user_id × product_id；
- user_id × category；
- product_id × category；
- user features × senior；
- context × user/job；
- cross layer 中的高阶路径。

干预方法：

- field leave-one-out；
- pairwise interaction masking；
- layer masking；
- matched-sparsity random masking；
- 稀疏可学习 gate；
- 高泄漏路径与随机路径对照。

### RQ3：过程泄漏是否转化为结果差异

过程指标：

- layer-wise attack AUC；
- conditional leakage；
- input-normalized amplification；
- 高泄漏路径集中度。

结果指标：

- \(DP_{signed}\)；
- \(|DP|\)；
- 两组平均预测和比值；
- group-wise NLLH、AUC、Brier、ECE；
- U、U_TILDE；
- impression 内排序和曝光差异。

仅观察相关性时使用“associated with”。只有通过路径干预、选择机制控制和稳健性实验排除替代解释后，才能讨论更强机制证据。

### RQ4：选择性路径抑制是否优于全局去敏

比较：

- 全局 adversarial removal；
- MMD/HSIC 表示对齐；
- 直接 DP regularization；
- 全局 projection/decorrelation；
- 选择性高风险路径抑制；
- 随机等量路径抑制。

核心检验：在相近公平水平下，选择性方法是否保留更高 AUC、更低 NLLH 和更好的 U/U_TILDE。

### RQ5：结论是否对代理噪声和日志选择稳健

至少分析：

- 对称随机翻转保护代理；
- 非对称代理错误；
- 敏感标签缺失；
- 全部展示日志；
- 随机展示子样本；
- position/propensity correction；
- campaign-conditioned 结果；
- seen-user 与 unseen-user；
- pre-ranking 与 post-display。

---

## 12. 诊断方法与实现规范

### 12.1 保护代理 probe

Probe 必须满足：

1. CTR 模型训练与 probe 训练严格分离；
2. probe 超参数仅在 probe validation 上选择；
3. 最终 leakage 只在冻结 probe test 上报告；
4. 所有层使用同等 probe 容量与相同样本；
5. 同时提供线性 probe 和有限容量非线性 probe；
6. 报告类别基线、置信区间和样本量；
7. 不用 CTR test 标签训练 probe；
8. 对 ID 和非 ID 特征分别报告。

建议输出：

```text
model
protocol
regime
seed
layer_name
probe_type
probe_auc
probe_balanced_accuracy
input_probe_auc
leakage_amplification
n_probe_train
n_probe_test
```

### 12.2 条件泄漏

由于代理可预测性可能来自职位或 campaign 组成差异，应补充：

\[
\mathrm{Leak}(H_l,A\mid C)
\]

其中 \(C\) 可包括：

- senior；
- product/category；
- campaign 或可用上下文；
- position；
- random display 状态。

实现可采用分层 probe、残差化或条件互信息近似，但必须说明方法限制。

### 12.3 路径归因

路径重要性不能只依赖梯度可视化。至少结合：

- 交互屏蔽后的 leakage 变化；
- CTR 指标变化；
- 公平指标变化；
- 随机同规模屏蔽对照；
- 多种子稳定性。

路径被判定为高风险需同时满足：

1. 对 leakage 有稳定正贡献；
2. 干预后结果差异按预期变化；
3. 不可完全由单个 ID 字段解释；
4. 相比随机路径干预具有显著差异。

---

## 13. 指标协议调整

### 13.1 CTR 与组内性能

主指标：

- overall NLLH；
- overall AUC；
- Brier score；
- ECE；
- group-wise NLLH/AUC/Brier/ECE；
- 最差组性能；
- 组间性能差。

### 13.2 Demographic Parity

正式报告同时输出：

\[
DP_{signed}=E[\hat Y\mid A=1,senior=1]-E[\hat Y\mid A=0,senior=1]
\]

\[
DP_{abs}=|DP_{signed}|
\]

以及：

- 两组平均预测；
- 两组样本量；
- prediction ratio；
- 95% cluster-bootstrap 区间。

任何“公平改善”结论至少基于 \(|DP|\)、组均值与其他结果指标的共同变化，而不是单独比较 signed DP。

### 13.3 DP 上下文分解

聚合差异应分解为：

- 同一上下文内模型差异；
- 不同群体进入不同职位、campaign、位置或展示策略的 composition difference。

可使用共同参考分布 \(q(c)\) 表达 within-context 部分：

\[
\Delta_{within}=\sum_c q(c)
\left(E[\hat Y\mid A=1,c]-E[\hat Y\mid A=0,c]\right)
\]

其余差异归入 composition component，并对参考分布选择做敏感性分析。

### 13.4 U 与 U_TILDE

进入主结果前必须冻结：

- 每个 impression 的候选集合；
- 排名方向；
- tie 处理；
- 缺失或无效候选；
- `displayrandom=1` 过滤；
- position correction；
- 与 FairJob 参考实现的逐样本 parity。

Stage1.1 的工程 parity 检查不能只比较最终均值，还需比较中间排序和候选聚合结果。

### 13.5 Pareto 报告

公平方法不得只报告单个最优权重。至少展示：

- AUC–\(|DP|\)；
- NLLH–\(|DP|\)；
- U_TILDE–\(|DP|\)；
- leakage–AUC；
- leakage–\(|DP|\)；
- calibration gap–utility。

---

## 14. Baseline 体系重构

### 14.1 CTR backbone

Stage1.1 主表使用代表性而非数量优先的模型：

| 类别 | 模型 | 作用 |
|---|---|---|
| 线性/树锚点 | LR、XGB | 工程与非神经比较。 |
| 二阶交互 | DeepFM | 主诊断与方法开发 backbone。 |
| 显式高阶交互 | DCNv2 | cross path 机制验证。 |
| 近期高阶/MLP | DS-MLP 或 FCN 二选一 | 验证结论是否迁移到较新架构。 |
| 可选近期模型 | RFM、ME、GE4Rec、DLF 中按机制选一个 | 仅在与核心现象相关时加入。 |

不再把“完成六个近期 CTR 模型”作为 Stage1.1 的必要完成条件。

### 14.2 公平方法基线

至少实现：

| 类别 | 基线 |
|---|---|
| 输入协议 | proxy-excluded、proxy-included |
| 预处理 | group/sample reweighting |
| 直接结果约束 | DP regularization |
| 表示分布约束 | MMD 或 HSIC |
| 对抗去敏 | gradient-reversal adversarial removal |
| 全局线性处理 | projection/decorrelation |
| 多目标优化 | 固定权重与 adaptive dual/Pareto 版本 |
| 后处理 | group-blind calibration；必要时单列 group-aware calibration |
| 日志偏差 | position/selection-corrected training 或 evaluation |
| 关键反事实对照 | random interaction suppression |
| 候选方法 | selective interaction-path suppression |

### 14.3 Backbone 与方法承载关系

```text
DeepFM：主实现、机制消融
DCNv2：cross-layer 验证
DS-MLP/FCN：现代架构泛化
LR/XGB：只做锚点和部分后处理，不强行插入交互模块
```

---

## 15. 选择偏差与日志协议

FairJob 的研究价值之一是能区分部分日志机制。Stage1.1 至少形成四种评价协议：

| 协议 | 数据/校正 | 回答的问题 |
|---|---|---|
| `all_logged` | 全部日志 | 模型在观测分布上的总体表现。 |
| `random_display` | `displayrandom=1` 子样本 | 降低既有展示策略选择影响后的表现。 |
| `position_corrected` | 按位置 propensity 或校正权重 | 位置偏差对公平和效用的贡献。 |
| `context_conditioned` | campaign/product/category/senior 分层 | 组间组成差异与上下文内差异。 |

如果公平结论在四种协议间发生方向反转，论文主问题应优先转向 selection-aware fairness，而不是继续做纯表示去敏。

---

## 16. 统计与实验可信度规范

### 16.1 随机种子

- smoke 与代码调试：允许单种子；
- 现象筛选：至少 3 个种子；
- Stage1.1 决策与正式主表：至少 5 个独立种子；
- 报告均值、标准差和 95% 区间。

### 16.2 超参数选择

- 所有超参数仅使用 train/validation；
- 不根据 test DP、test AUC 或 test U_TILDE 选模型；
- 公平强度按验证集 Pareto 规则选取；
- 所有 baseline 使用可比的调参预算、早停和计算资源；
- 保存每次实验的配置、git commit、seed 和预测文件 hash。

### 16.3 Bootstrap 单位

- 行级指标可附 row bootstrap；
- 主置信区间使用 user 或 impression cluster bootstrap；
- campaign/context 结果使用对应 cluster；
- 时间分析使用 block bootstrap 或多个时间切片。

### 16.4 必须消融

| 消融 | 目的 |
|---|---|
| no gate | 验证 gate 的必要性。 |
| random gate | 排除普通稀疏正则效应。 |
| global adversary | 比较选择性与全局去敏。 |
| leakage-only | 检验结果约束是否必要。 |
| DP-only | 检验表示诊断是否有贡献。 |
| fixed weight | 检验自适应优化是否必要。 |
| no task preservation | 检验效用保持模块。 |
| no selection correction | 检验日志偏差贡献。 |
| no-user-id | 排除身份记忆。 |
| no-rank/displayrandom | 排除不可部署变量和策略编码。 |
| proxy noise | 检验代理误差稳健性。 |
| no-proxy inference | 验证部署阶段不依赖代理。 |

---

## 17. 条件候选方法：选择性代理路径抑制

该方向是 Stage1.1 的首选候选，但只有在决策门槛满足后才能冻结为 Stage2 主方法。

### 17.1 表示结构

设模型包含交互路径 \(\phi_k(X)\)，学习保留 gate \(g_k\in[0,1]\)：

\[
H_g = H_{base}+\sum_k g_k\phi_k(X)
\]

为了只修改少数高风险路径，可对抑制量使用稀疏约束：

\[
\lVert 1-g\rVert_1
\]

### 17.2 约束目标

\[
\min_{\theta,g} L_{CTR}(\theta,g)
+\lambda_{sp}\lVert 1-g\rVert_1
+\lambda_{keep}L_{task-preserve}
\]

满足：

\[
\mathrm{Leak}(H_g,A\mid C)\leq\epsilon
\]

\[
|DP|\leq\delta
\]

\[
\Delta ECE\leq\gamma
\]

建议使用自适应对偶变量或 Pareto 多目标方法，不把 \(\lambda\) 视为任意常数。

### 17.3 任务信息保留

候选方式：

- residual task path；
- teacher–student logit preservation；
- 高风险与低风险路径分路；
- task-consistent projection；
- 点击预测一致性约束。

必须证明公平变化不是由表示坍缩、模型容量下降或统一缩小预测分数造成。

### 17.4 可选理论方向

优先选择一个，不同时堆叠两套薄弱理论。

**方向 A：表示差异到结果 DP 的上界。**  
在 scorer 为 Lipschitz 函数的条件下，研究组间表示分布距离对输出均值差的控制，并进一步分解到 gated paths。

**方向 B：保护代理噪声敏感性界。**  
在给定对称或非对称代理误差范围下，给出观测 proxy-conditioned disparity 与潜在真实组差异之间的可行区间或敏感性分析。

---

## 18. 数据与泛化证据设计

### 18.1 FairJob：主现实数据

用于验证：

- 真实点击；
- 行为保护代理；
- position；
- randomized display；
- campaign/product context；
- U/U_TILDE；
- 真实日志中的选择机制。

所有主张必须限定在 job-ad CTR 场景，不声称代表全部 CTR 或推荐系统。

### 18.2 半合成数据：机制验证

基于 FairJob 统计结构构造：

- 潜在真实属性 \(S^*\)；
- 带噪代理 \(A\)；
- 可控代理错误率；
- 已知有害交互路径；
- 可控 position bias；
- 可控 campaign selection；
- 可控点击生成机制。

用于回答：

- 路径归因能否找回真实有害路径；
- 代理噪声到何种程度会导致结论失效；
- 模型偏差与选择偏差能否被正确分离；
- 理论上界是否与实验一致。

Stage1.1 至少冻结生成机制、参数范围和评价协议；完整结果可延伸到 Stage2。

### 18.3 辅助真实数据

选用一个包含真实人口属性的推荐数据集作为机制迁移验证，例如 MovieLens、LastFM 或相似公开数据。

该数据只用于验证方法是否依赖 FairJob 的特定代理构造，不替代 FairJob 主任务。

---

## 19. Stage1.1 工作包

### Work Package A — 任务、术语与指标有效性

- [ ] 新增 `pre_ranking` 与 `post_display` 配置；
- [ ] 冻结 `rank`、`displayrandom`、`impression_id` 使用政策；
- [ ] 报告名称改为 `proxy-excluded` / `proxy-included`；
- [ ] 实现 `DP_signed`、`DP_abs`、group means、ratio；
- [ ] 完成 U/U_TILDE 中间步骤 parity；
- [ ] 记录代理标签语义和声明边界。

### Work Package B — 深模型现象验证

- [ ] DeepFM smoke/full；
- [ ] DCNv2 smoke/full；
- [ ] 至少一个近期 backbone；
- [ ] 支持逐层表示导出；
- [ ] 线性与非线性 proxy probes；
- [ ] input leakage 与 layer amplification；
- [ ] 多种子与 cluster confidence intervals。

### Work Package C — 路径与身份诊断

- [ ] no-user-id / no-product-id；
- [ ] seen/unseen-user；
- [ ] field leave-one-out；
- [ ] pairwise/path masking；
- [ ] random matched masking；
- [ ] 高风险路径集中度和稳定性。

### Work Package D — 公平与日志基线

- [ ] DP regularization；
- [ ] adversarial removal；
- [ ] MMD 或 HSIC；
- [ ] calibration baseline；
- [ ] fixed-weight 与 adaptive constraint；
- [ ] all/random-display/position-corrected/context-conditioned 比较。

### Work Package E — Stage2 决策

- [ ] 汇总 RQ1–RQ5；
- [ ] 判定主失效机制；
- [ ] 选择 Stage2 方法方向；
- [ ] 形成不支持原假设时的转向方案；
- [ ] 冻结正式实验矩阵和论文主张边界。

---

## 20. Stage2 决策门槛

### 情形 1：确认稳定的交互泄漏放大

进入“选择性交互路径抑制”，前提是：

- 至少两个深 CTR backbone 出现稳定增量泄漏；
- 多种子方向一致；
- 泄漏集中在有限字段或路径；
- 高泄漏路径干预对结果公平的影响显著大于随机路径；
- 现象不完全由 user_id 记忆解释。

### 情形 2：泄漏存在，但与结果差异关系弱

不把表示泄漏直接称为公平伤害。可转向：

- 敏感代理隐私；
- 过程公平审计；
- 公平指标有效性研究。

### 情形 3：组间校准差最稳定

转向：

> 行为代理不确定性下、无需推理时保护代理的公平校准。

### 情形 4：选择偏差主导

若随机展示、全部日志和位置校正结果明显不同，且 composition component 解释大部分聚合 DP，则转向：

> selection-aware fairness diagnosis and correction for job-ad CTR。

### 情形 5：没有稳定现象

不强行开发公平模块。可选择：

- 扩展或更换数据；
- 构建系统性审计论文；
- 研究 CTR 公平指标不一致性；
- 研究行为保护代理的测量误差与可识别范围。

---

## 21. Stage1.1 完成标准

Stage1.1 完成需同时满足：

### 工程与协议

- [ ] 数据顺序切分、预测 schema 和 row alignment 保持稳定；
- [ ] pre-ranking/post-display 协议已实现且可复现；
- [ ] 报告组名完成修订；
- [ ] `DP_signed`、`DP_abs`、group means、ECE、Brier 可统一输出；
- [ ] U/U_TILDE 参考 parity 有逐步验证结果。

### 模型与诊断

- [ ] DeepFM 和 DCNv2 至少完成主协议下 full 结果；
- [ ] 正式结论至少基于 5 个种子；
- [ ] 表示导出和 proxy probe 可复现；
- [ ] input leakage、layer leakage 和 amplification 已报告；
- [ ] no-user-id、随机路径和日志协议对照已完成。

### 基线与证据

- [ ] 最小公平基线集已完成；
- [ ] 固定加权与 adaptive constraint 有比较；
- [ ] all_logged 与 random_display 至少完成对照；
- [ ] 形成包含置信区间和 Pareto 曲线的结果；
- [ ] 对每个主张列出替代解释及排除证据。

### 决策输出

- [ ] 完成 Stage1.1 decision report；
- [ ] 明确进入五种 Stage2 情形中的哪一种；
- [ ] 冻结 Stage2 主问题、方法边界、backbone 和数据协议；
- [ ] 若证据不支持原假设，明确记录转向而不是隐藏负结果。

---

## 22. 建议新增的项目文件

以下为计划文件，尚未存在时应按职责创建：

```text
configs/fairjob/task_protocols.yaml
configs/fairjob/fairness_protocols.yaml

fuxictr_ext/fairjob/metrics_extended.py
fuxictr_ext/fairjob/calibration.py
fuxictr_ext/fairjob/representation_io.py
fuxictr_ext/fairjob/proxy_probe.py
fuxictr_ext/fairjob/interaction_ablation.py
fuxictr_ext/fairjob/dp_decomposition.py
fuxictr_ext/fairjob/selection_protocols.py
fuxictr_ext/fairjob/run_protocol_matrix.py

results/fairjob/stage1_1/
docs/fairjob/stage1_1_decision_report.md
```

建议统一实验主键：

```text
model
backbone
protocol
proxy_regime
fairness_method
seed
feature_ablation
selection_protocol
```

---

## 23. 保留的基础验证命令

使用 PowerShell 和 `openmmlab` 环境。

### 数据准备

```powershell
conda activate openmmlab
cd D:\code\JobFairness
python fuxictr_ext/fairjob/prepare_fairjob.py --raw_path D:/code/datasets/FairJob/fairjob.csv.gz --out_dir data/FairJob/processed --test_ratio 0.2 --valid_rows 50000 --smoke_rows 50000 --format csv
python configs/fairjob/make_dataset_config.py
```

### 轻量检查

```powershell
python -m compileall fuxictr_ext configs/fairjob tests/fairjob
python fuxictr_ext/fairjob/check_metric_parity.py
python -m pytest tests/fairjob/test_metrics_reference_parity.py -q
```

### 对齐检查

```powershell
python fuxictr_ext/fairjob/check_alignment.py --pred results/fairjob/pred/XGB_fairjob_unaware_full.csv --meta data/FairJob/processed/test_meta.csv
python fuxictr_ext/fairjob/check_alignment.py --pred results/fairjob/pred/XGB_fairjob_unfair_full.csv --meta data/FairJob/processed/test_meta.csv
```

旧文件名可以保留，但生成报告时必须映射为新的科学命名。

---

## 24. 执行优先级

### P0：必须先完成

1. 任务协议、代理语义和组名冻结；
2. DP 与 U/U_TILDE 指标协议；
3. DeepFM/DCNv2 主协议运行；
4. 层级 proxy probe 和 input-normalized amplification；
5. no-user-id、random masking 和日志选择对照；
6. 最小公平基线；
7. Stage2 决策报告。

### P1：提高论文可信度

1. adaptive dual/Pareto 优化；
2. 半合成生成器；
3. 一个辅助真实数据集；
4. 理论上界或代理噪声敏感性；
5. 近期 backbone 泛化。

### P2：仅在主机制稳定后扩展

1. 更多近期 CTR 模型；
2. 大规模超参数搜索；
3. 复杂多模块组合；
4. 跨多个公平定义的完整扩展。

---

## 25. 最终阶段定位

Stage1.1 的最准确标签是：

> FairJob × FuxiCTR 工程基础已经建立；本阶段负责把一个宽泛的“CTR 模型可能学习敏感代理”想法，转化为可检验的科学命题，并通过任务语义冻结、层级与路径诊断、日志选择控制、公平基线和统计验证，决定 Stage2 应进入选择性交互抑制、校准、选择偏差校正、过程公平审计或其他方向。

Stage1.1 的成功标准不是得到一个看似更公平的数字，而是得到一个经得住替代解释检验的研究方向。

---

## 参考文献

1. Mariia Vladimirova, Federico Pavone, and Eustache Diemert. **FairJob: A Real-World Dataset for Fairness in Online Systems.** NeurIPS 2024 Datasets and Benchmarks Track. [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/142bff4f4c01dd55c4309860ff3a59f1-Paper-Datasets_and_Benchmarks_Track.pdf) · [Official repository](https://github.com/criteo-research/fairjob-dataset)
2. Haolun Wu et al. **A Multi-Objective Optimization Framework for Multi-Stakeholder Fairness-Aware Recommendation.** ACM Transactions on Information Systems, 41(2), 2022. [DOI](https://doi.org/10.1145/3564285)
3. Zihan Liu et al. **Mitigating Popularity Bias for Users and Items with Fairness-centric Adaptive Recommendation.** ACM Transactions on Information Systems, 41(3), 2023. [DOI](https://doi.org/10.1145/3564286)
4. Pengyang Shao et al. **Average User-Side Counterfactual Fairness for Collaborative Filtering.** ACM Transactions on Information Systems, 42(5), 2024. [DOI](https://doi.org/10.1145/3656639)
5. Weixin Chen et al. **Causality-Inspired Fair Representation Learning for Multimodal Recommendation.** ACM Transactions on Information Systems, 43(6), 2025. [DOI](https://doi.org/10.1145/3744240)
6. Mengyue Yang et al. **Debiased Recommendation with User Feature Balancing.** ACM Transactions on Information Systems, 41(4), 2023. [DOI](https://doi.org/10.1145/3580594)
