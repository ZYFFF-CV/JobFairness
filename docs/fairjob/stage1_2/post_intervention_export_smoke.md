# S12-M1A 干预后表征导出 Smoke

## 运行结论

S12-M1A 已在服务器单卡 RTX 4090 D 上完成。运行仅加载 Stage1.1 M5A 的 `baseline_seed2019` checkpoint，不重新训练、不更新权重。初始 2,048 行 smoke 通过后，进一步直接读取 Stage1.1 冻结的 `probe_rows.npz`，导出了 test split 中全部 50,000 个指定原始 `row_id`，没有重新随机抽样。

| 项目 | 结果 |
|---|---|
| Stage1.2 代码 commit | `f44a9f6d5eaa091ca0477b7fe0ab7d1f8c55f615` |
| M5A checkpoint | `baseline_seed2019` / `M5DCNv2_baseline_full` |
| Dataset | `fairjob_m5_pre_ranking_no_user_id_proxy_excluded_full` |
| Representation sampling seed | `2019` |
| 导出 split / 行数 | test / 50,000 frozen row IDs |
| 原预测一致性阈值 | `1e-6` |
| 最大预测绝对误差 | `9.999054535747565e-17` |
| 50,000 行单 split 实测占用 | `1.4 GB` |
| 状态 | `representation_export_complete` |

## 表征层

实际 forward 与导出清单覆盖：

| 表征 | 维度 |
|---|---:|
| `embedding_flat` | 800 |
| `cross_layer_0` | 800 |
| `cross_layer_1` | 800 |
| `cross_layer_2` | 800 |
| `parallel_dnn_linear_0` | 256 |
| `parallel_dnn_linear_1` | 128 |
| `dcnv2_final_pre_mitigation` | 928 |
| `dcnv2_final` | 928 |
| `dcnv2_suppressed_component` | 928 |
| `dcnv2_residual_component` | 928 |
| `dcnv2_logit` | 1 |
| `dcnv2_probability` | 1 |

其中 mask 不按行重复保存，而是在每个 split 的 manifest 中记录方法、训练 seed、final 维度、抑制维度、抑制索引和 mask SHA-256。代码测试同时验证
`final_pre_mitigation = suppressed_component + residual_component`。

## 运行环境

- Python 3.12.3
- FuxiCTR 2.3.9
- PyTorch 2.3.0+cu121
- CUDA 12.1
- scikit-learn 1.4.2
- NumPy 1.26.4
- pandas 2.3.3
- XGBoost 2.1.4

## 门槛判断

M1A 通过。checkpoint-only 导出保持原预测，要求的表征层与干预组件完整，允许进入 M1B 的 18 组统一导出。M1B 仍是只读 GPU/I/O 操作，不属于新训练。

容量审计显示，18 组乘 3 个 split 按实测约需 75.6 GB，而服务器数据盘在本次 smoke 后仅剩约 28 GB。M1B 因存储容量暂停，未进行部分矩阵导出，也未删除任何 Stage1.1 产物。
