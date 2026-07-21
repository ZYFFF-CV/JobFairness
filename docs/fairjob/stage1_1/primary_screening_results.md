# Stage1.1 Primary Screening Results

更新日期：2026-07-21

## 运行口径

- Git commit：`39214fddfc571cd067a44bc58b1dfab4d11a836a`
- 任务协议：`pre-ranking`
- 数据切分：顺序外层 80/20；内部 `807,780 train + 50,000 valid`
- Test：`214,446` 行，四个实验完全一致
- 参数来源：`stage1_1_fixed_screening`
- Seed：`2019`
- 保护属性术语：behavioral protected proxy，不代表经验证的人口统计性别

## 完整 Test 结果

| Model | Regime | NLLH | AUC | Brier | ECE | DP abs | U | U_TILDE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DeepFM | proxy-excluded | 0.066489 | 0.715530 | 0.007976 | 0.007024 | 0.000610 | 0.010253 | 0.011359 |
| DeepFM | proxy-included | 0.069880 | 0.678310 | 0.008236 | 0.006868 | 0.000136 | 0.010257 | 0.011355 |
| DCNv2 | proxy-excluded | 0.039060 | 0.781174 | 0.006874 | 0.000727 | 0.000735 | 0.010274 | 0.011962 |
| DCNv2 | proxy-included | 0.039498 | 0.779070 | 0.006799 | 0.000896 | 0.000356 | 0.010140 | 0.011839 |

四个实验的 `DP_signed` 均为负，即 senior scope 中 proxy group 1 的平均预测
概率低于 group 0。`DP abs` 是该 signed difference 的绝对值。

## Regime 差值

以下差值均为 `proxy-included - proxy-excluded`：

| Model | Delta NLLH | Delta AUC | Delta DP abs | Delta U | Delta U_TILDE |
|---|---:|---:|---:|---:|---:|
| DeepFM | +0.003391 | -0.037220 | -0.000473 | +0.000004 | -0.000004 |
| DCNv2 | +0.000438 | -0.002104 | -0.000379 | -0.000134 | -0.000123 |

保护代理加入后两种模型的 DP abs 都下降，但预测性能没有改善。该现象可能来自
分数收缩、校准变化或模型选择，不能直接解释为公平改善，也不能据此声称代理输入
具有公平作用。

## 训练选择

| Model | Regime | Best validation epoch | Best validation AUC | Stop epoch |
|---|---|---:|---:|---:|
| DeepFM | proxy-excluded | 6 | 0.694670 | 8 |
| DeepFM | proxy-included | 4 | 0.691964 | 6 |
| DCNv2 | proxy-excluded | 1 | 0.779613 | 3 |
| DCNv2 | proxy-included | 1 | 0.776175 | 3 |

所有实验都回载 validation AUC 最优 checkpoint。日志中无 traceback、NaN、OOM
或训练失败；仅出现 FuxiCTR/Polars schema 性能警告。

## 概率诊断

Test 点击率为 `0.006948`，相同点击率常数预测的 NLLH 为 `0.041451`。

| Model | Regime | Median p | p99 | p99.9 | Max p | p > 0.9 | p < 1e-6 |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepFM | proxy-excluded | 0.000014 | 0.142960 | 0.672173 | 0.999993 | 67 | 63,115 |
| DeepFM | proxy-included | 0.000053 | 0.153721 | 0.702236 | 0.999996 | 94 | 48,436 |
| DCNv2 | proxy-excluded | 0.002785 | 0.058358 | 0.120949 | 0.935488 | 2 | 6,013 |
| DCNv2 | proxy-included | 0.002453 | 0.054995 | 0.113743 | 0.237542 | 0 | 5,088 |

DeepFM 的 NLLH 明显差于常数预测，主要风险信号是大量接近 0 的概率和少量接近 1
的极端概率。其 AUC 仍高于随机，但当前 checkpoint 不适合作为已校准概率模型。
DCNv2 的 NLLH 优于常数预测，且校准指标明显更稳定。

## 表示导出

四个实验均生成 train/valid/test representation manifest：

- train：从 `807,780` 行固定种子均匀采样 `200,000` 行；
- valid：完整 `50,000` 行；
- test：从 `214,446` 行固定种子均匀采样 `200,000` 行；
- sampling seeds：train `2019`、valid `2020`、test `2021`；
- 四个模型使用相同采样规则，完整指标仍使用全部 test 预测。

## 当前结论

1. 工程链路通过：四个模型配置、预测、评价、checkpoint 和表示导出均完成。
2. 在当前单 seed 固定参数下，DCNv2 是更可靠的主诊断 backbone。
3. DeepFM 需要先处理概率极化与 model-selection/calibration 问题，不能用当前
   NLLH 支持正式结论。
4. proxy-included 的 DP abs 下降伴随 AUC/NLLH 退化，尚无证据说明其改善了公平性。
5. 当前结果不能回答“高阶交互是否放大代理泄漏”；下一步必须运行严格分离的
   layer-wise proxy probes，并与 embedding/input baseline 比较。
6. Stage2 方法方向仍未冻结，不应提前实现选择性路径抑制。
