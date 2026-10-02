# FlowPilot 五算法实验结果与清理报告（2026-09-27）

## 1. 完成状态与结论

正式训练、tune 评价、五算法 calibration 与两套冻结 test 均已成功完成，训练和最终阶段退出码均为 0。自动接续状态为 `complete`。本报告仅重算既有预测的统计量与任务组 bootstrap；未重新训练、推理、校准或更改模型。

- 当前固定配置中，LightGBM 的工具宏平均综合 pinball 最低，且进程内预测较快，是值得继续验证的主要候选。
- QRF 在 tune 的 Q50 MAE 和部分尾部指标较好，但当前实现预测 P95 约 43 ms；不能满足交接文档提出的 1 ms 初始热路径预算。
- Q99 尚未稳定达到逐工具 99% 覆盖；LightGBM 的全局校准改善了整体尾部，但明显增加短工具的 Q99 预测开销估计。
- EWMA 在线适应只在 tune 评估，改善了中位数误差，却恶化了宏平均 Q99 pinball；不能外推为在线 test 或调度收益。
- 本轮只有一个固定配置和随机种子，未做自动调参、特征消融、神经模型对照、真实 KV 调度或 SLO/goodput 实验。

| 阶段 | 开始 UTC | 完成 UTC | 耗时 |
| --- | --- | --- | --- |
| fit＋tune＋EWMA online | 17:00:43 | 17:01:19.724 | 约 36.7 s（开始时间按运行 ID / tmux 记录） |
| 自动接续校准＋test | 2026-09-27T17:15:30.130800+00:00 | 2026-09-27T17:17:25.021029+00:00 | 114.890 s |

运行目录：`runs/predictor_experiments/20260927T170043Z/`；最终目录：`runs/predictor_experiments/20260927T170043Z_final/`；接续日志与冻结快照：`runs/predictor_experiments/20260927T170043Z_final.autofinalize/`。

## 2. 数据与实验协议

目标为单工具客户端 RTT，单位 ms；输出 Q10/Q50/Q90/Q99。下表的组数只统计有有效 RTT 的任务组。原始 fit 清单有 253 个任务组，其中 252 个提供有效 RTT；此前交接中的 253 与这里 252 口径不同。

| 分区 | 有效 RTT | 有效 RTT 任务组 | 用途 |
| --- | --- | --- | --- |
| fit | 4056 | 252 | 训练基础模型 |
| tune | 694 | 52 | 固定配置评价；EWMA 在线对照 |
| calibration | 424 | 28 | 拟合全局分位数残差偏移 |
| test | 674 | 52 | 冻结模型最终评价 |

四分区 task_group 无交集。各评价流全部支持：tune 每流 694/694，test 每流 674/674，共六种工具；没有因 unsupported 排除工具。冻结模式允许使用 T1 前已观测的历史特征，但不在评价标签上更新模型参数。校准是全局 ms 残差偏移，不是逐工具校准，也不具有本任务下严格的覆盖保证。

当前设置：seed=20260927；每路线程配置 8；QRF 200 棵树；LightGBM 四个分位数头各 200 轮、learning_rate=0.05、max_depth=6、min_samples_leaf=20，标签 log1p；KMeans 最多 8 簇、10 次初始化；EWMA alpha=0.2、残差缓冲 512。没有合并 fit+tune 重训。

## 3. 指标解释

- 工具宏平均四分位 pinball：先在每种工具内对调用求均值，再对六种工具等权平均，最后对 Q10/Q50/Q90/Q99 等权平均；越低越好。
- Q50 MAE 是逐调用微平均，不能与工具宏平均混称。
- 覆盖率按 `y ≤ qτ` 统计；80% 区间为 `[Q10,Q90]`，同时报告宽度。覆盖高于名义值不一定更好，可能来自过宽预测。
- Q99 超出次数为 `y > Q99`；674 条 test 在名义 99% 下对应约 6.74 次超出，但调用相关，不能当独立二项试验。
- 预测 P95 包含特征抽取与模型预测，校准结果还包含偏移/重排；不含输入 JSON 解析、状态采集、RPC 和生产调度的全部开销。

## 4. Tune 结果

| 方法 | 工具宏平均四分位 pinball↓ / ms | 逐调用 Q50 MAE↓ / ms | 80% 区间覆盖 | 区间平均宽度 / ms | Q99 覆盖 | Q99 超出数 | 预测 P95 / ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| empirical | 90.146 | 522.395 | 75.94% | 1748.105 | 97.12% | 20 | 0.519 |
| ewma | 120.078 | 539.534 | 64.84% | 1226.568 | 97.55% | 17 | 0.321 |
| cluster | 95.960 | 516.472 | 76.22% | 1664.081 | 95.97% | 28 | 0.932 |
| qrf | 56.017 | 278.487 | 83.86% | 1269.408 | 99.42% | 4 | 43.926 |
| lightgbm | 53.317 | 330.224 | 72.62% | 977.197 | 98.85% | 8 | 0.795 |
| ewma_online | 104.401 | 398.234 | 79.68% | 1869.281 | 98.56% | 10 | 0.342 |

LightGBM 相比静态经验分布，宏平均 pinball 下降 **40.85%**，微平均 Q50 MAE 下降 **36.79%**。QRF 的 Q50 MAE 更低，但综合 pinball 和预测开销不占优。该结论只描述当前配置，并不证明已完成超参搜索。

EWMA 在线相对冻结：Q50 MAE 下降 26.19%，宏平均 pinball 下降 13.06%；但 Q99 宏平均 pinball 从 42.356 升到 88.050 ms。在线更新有收益与尾部代价，不能只汇报 MAE 改善。

## 5. Test 结果：原始预测

| 方法 | 工具宏平均四分位 pinball↓ / ms | 逐调用 Q50 MAE↓ / ms | 80% 区间覆盖 | 区间平均宽度 / ms | Q99 覆盖 | Q99 超出数 | 预测 P95 / ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| empirical_test_raw | 81.350 | 420.531 | 76.26% | 1467.007 | 97.33% | 18 | 0.502 |
| ewma_test_raw | 89.979 | 436.171 | 58.90% | 1160.817 | 96.29% | 25 | 0.316 |
| cluster_test_raw | 89.617 | 415.108 | 75.22% | 1421.758 | 95.85% | 28 | 0.931 |
| qrf_test_raw | 68.712 | 345.403 | 84.87% | 1134.670 | 98.81% | 8 | 43.130 |
| lightgbm_test_raw | 66.395 | 327.973 | 74.93% | 890.446 | 96.74% | 22 | 0.772 |

LightGBM 原始预测相对经验分布：宏平均 pinball 下降 **18.38%**，Q50 MAE 下降 **22.01%**。QRF 在 test 的 Q50 MAE 已不再优于 LightGBM，说明 tune 上局部排序不能直接外推。

## 6. Test 结果：校准后

| 方法 | 工具宏平均四分位 pinball↓ / ms | 逐调用 Q50 MAE↓ / ms | 80% 区间覆盖 | 区间平均宽度 / ms | Q99 覆盖 | Q99 超出数 | 预测 P95 / ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| empirical_test_calibrated | 81.365 | 420.590 | 72.11% | 1465.067 | 97.63% | 16 | 0.521 |
| ewma_test_calibrated | 89.914 | 438.377 | 75.22% | 1217.387 | 98.07% | 13 | 0.328 |
| cluster_test_calibrated | 89.591 | 414.945 | 71.36% | 1420.368 | 97.18% | 19 | 0.961 |
| qrf_test_calibrated | 68.738 | 345.509 | 77.30% | 1125.225 | 98.22% | 12 | 43.285 |
| lightgbm_test_calibrated | 65.087 | 328.131 | 78.64% | 896.635 | 98.52% | 10 | 0.774 |

| 算法 | 校准前宏 pinball | 校准后宏 pinball | 相对变化（负为改善） | Q99 超出数 前→后 |
| --- | --- | --- | --- | --- |
| empirical | 81.350 | 81.365 | +0.019% | 18 → 16 |
| ewma | 89.979 | 89.914 | -0.072% | 25 → 13 |
| cluster | 89.617 | 89.591 | -0.028% | 28 → 19 |
| qrf | 68.712 | 68.738 | +0.037% | 8 → 12 |
| lightgbm | 66.395 | 65.087 | -1.970% | 22 → 10 |

LightGBM 的 80% 区间覆盖由 74.93% 提升到 78.64%，Q99 覆盖由 96.74% 提升到 98.52%，但仍不是稳定的 99% 保证。经验分布的 80% 覆盖由 76.26% 降到 72.11%；QRF 的 80% 覆盖从 84.87% 变为 77.30%，Q99 超出从 8 增到 12。极小的综合指标变化不能夸大成确定收益。

### 6.1 校准偏移及短工具代价

| 算法 | Q10 偏移 / ms | Q50 偏移 / ms | Q90 偏移 / ms | Q99 偏移 / ms |
| --- | --- | --- | --- | --- |
| empirical | 0.820 | -0.366 | -1.120 | 1.073 |
| ewma | -61.986 | -27.689 | -1.348 | 5.333 |
| cluster | 1.076 | 1.040 | -0.315 | 2.145 |
| qrf | 3.854 | -0.661 | -5.758 | -5.083 |
| lightgbm | -4.610 | -1.034 | 1.579 | 804.681 |

LightGBM 的全局 Q99 偏移是 **+804.681 ms**，同样加到短文档读取和文件编辑工具上。该事实解释了整体搜索尾部改善与短工具过度保守同时发生；不能把“总体 Q99 覆盖提高”当作所有工具都变好。以下为 test 中每种工具预测 Q99 的中位数：

| 工具 | 原始 Q99 中位数 / ms | 校准后 Q99 中位数 / ms |
| --- | --- | --- |
| browsecomp/get_document | 313.363 | 1118.044 |
| browsecomp/search | 2826.890 | 3631.571 |
| hotpot/read_document | 277.206 | 1081.887 |
| hotpot/search | 4026.050 | 4830.731 |
| livecodebench/code_file_editor | 234.643 | 1039.323 |
| livecodebench/code_terminal | 499.894 | 1304.575 |

## 7. 逐工具 Test 分析

下表为各工具四分位平均 pinball（ms），越低越好。任务组在同一 benchmark 的工具之间可能重叠，不能把行内组数相加当独立任务数。

| 工具 | 行数 / 任务组 | 经验 raw | QRF raw | LightGBM raw | LightGBM 校准 |
| --- | --- | --- | --- | --- | --- |
| browsecomp/get_document | 15 / 13 | 1.943 | 2.258 | 2.533 | 4.645 |
| browsecomp/search | 336 / 20 | 166.692 | 131.265 | 143.526 | 139.937 |
| hotpot/read_document | 39 / 20 | 3.720 | 2.929 | 3.869 | 5.861 |
| hotpot/search | 89 / 20 | 261.633 | 207.708 | 208.750 | 199.742 |
| livecodebench/code_file_editor | 90 / 12 | 1.135 | 3.702 | 1.479 | 3.569 |
| livecodebench/code_terminal | 105 / 12 | 52.974 | 64.412 | 38.211 | 36.766 |

收益主要来自 search 和 code_terminal。对文件编辑与少样本文档读取，经验分布仍是有竞争力的廉价基线；LightGBM 的全局尾部校准使部分短工具损失增加。不能据 test 的逐工具排名直接拼装一个“每类选最优”的新模型并把本表当其无偏成绩；这种组合需在 tune 制定并用新的未使用留出集验证。

### 7.1 Q99 逐工具覆盖

| 工具 | 经验 raw | QRF raw | LightGBM raw | LightGBM 校准 |
| --- | --- | --- | --- | --- |
| browsecomp/get_document | 100.00% | 100.00% | 100.00% | 100.00% |
| browsecomp/search | 97.62% | 99.40% | 96.13% | 98.21% |
| hotpot/read_document | 100.00% | 100.00% | 100.00% | 100.00% |
| hotpot/search | 98.88% | 94.38% | 93.26% | 96.63% |
| livecodebench/code_file_editor | 90.00% | 100.00% | 100.00% | 100.00% |
| livecodebench/code_terminal | 100.00% | 99.05% | 97.14% | 99.05% |

LightGBM 校准后 Hotpot/search 的 Q99 覆盖仍只有 96.63%（3/89 次超出）；BrowseComp/search 为 98.21%（6/336 次超出）。短工具中的 100% 也不是精确保障：例如 get_document 只有 15 条 test，calibration 更只有 4 条。

## 8. 不确定性与尾部风险

原指标文件包含 500 次 task_group bootstrap。本报告另从既有预测进行 **2000 次配对任务组 bootstrap**，共同重采样 52 个 test 任务组并保持六种工具的等权口径；没有训练或推理。差值为 A−B，负值表示 A 更低。以下区间是本次单划分、单种子上的经验不确定性，不是多重比较校正后的显著性检验。

| A | B | 综合宏 pinball 差 / ms | 配对 95% 区间 / ms | Q99 宏 pinball 差 / ms | Q99 配对 95% 区间 / ms |
| --- | --- | --- | --- | --- | --- |
| lightgbm_test_raw | empirical_test_raw | -14.955 | [-27.782, -3.905] | 9.309 | [-11.938, 27.667] |
| lightgbm_test_calibrated | empirical_test_raw | -16.263 | [-27.581, -6.449] | 3.907 | [-12.863, 18.946] |
| lightgbm_test_raw | qrf_test_raw | -2.318 | [-9.005, 3.286] | -1.915 | [-17.927, 10.534] |
| lightgbm_test_calibrated | lightgbm_test_raw | -1.308 | [-3.262, 0.714] | -5.402 | [-13.150, 2.619] |
| qrf_test_calibrated | qrf_test_raw | 0.026 | [-0.034, 0.086] | 0.025 | [-0.022, 0.092] |

| 方法 | 整体 Q99 覆盖 | 原 500 次 bootstrap 95% 区间 |
| --- | --- | --- |
| empirical_test_raw | 97.33% | [95.62%, 98.91%] |
| qrf_test_raw | 98.81% | [97.61%, 99.56%] |
| lightgbm_test_raw | 96.74% | [94.60%, 98.47%] |
| lightgbm_test_calibrated | 98.52% | [97.25%, 99.52%] |

配对区间的解读：LightGBM 与经验分布的综合误差差值区间完全低于 0，支持它在本次划分上的改善；LightGBM 与 QRF 的差值区间跨过 0，因此尚不能断言两者精度存在稳定差距。LightGBM 校准前后的差值区间也跨过 0，校准收益仍有不确定性。

Q99 需同时看超出次数、超出幅度与过度保守成本。LightGBM 校准后的宏平均 Q99 pinball 为 34.815 ms，仍高于经验分布原始值 30.909 ms；因此它不是全部指标的优胜者。覆盖区间跨过 99% 也不能证明实现了严格或逐工具 99% 覆盖。

## 9. 运行开销与适用范围

| 算法 | fit 耗时 / s | tune 预测 P95 / ms | 顺序 test raw 预测 P95 / ms |
| --- | --- | --- | --- |
| empirical | 0.011 | 0.519 | 0.502 |
| ewma | 0.042 | 0.321 | 0.316 |
| cluster | 0.448 | 0.932 | 0.931 |
| qrf | 0.898 | 43.926 | 43.130 |
| lightgbm | 1.251 | 0.795 | 0.772 |

QRF 在顺序最终测试时仍约 43 ms，因此不能只归因于五路训练争抢资源。这是当前 QRF 实现的测量，不代表所有 QRF 实现的固有下限；逐条森林调用、线程调度和叶节点分布聚合可能有优化空间，需 profiling 验证。其他方法的进程内 P95 本轮低于 1 ms，但还不足以声称完整线上接口达标。GPU 未用于本轮训练。

本轮未预测整批 next-request-ready，未训练内部 executor 时间独立目标，未验证 hit/follower 路径、KV 迁移能力、正确且按时的 goodput 或端到端加速。不能从 RTT 回归改善推导系统收益。

## 10. 后续工作建议

1. 保留当前固定配置结果为基线；优先在 fit/tune 做 LightGBM 特征消融与参数搜索，不能再根据本次 test 反复调参并宣称盲测。
2. 针对 pooled Q99 校准的跨工具偏移，比较分后端/工具的收缩校准或相对尺度残差；calibration 仅 28 组且部分类极少，避免强行细分。
3. 对 QRF 独立 profiling，比较更低推理线程数、批处理与分布聚合实现，验证预测等价及真正热路径延迟。
4. 对 EWMA 在线尾部退化检查缓冲、漂移与反馈延迟；在线 test 需另立冻结协议。本轮没有运行它。
5. 若增加 MLP、冻结文本编码器或 BERT 对照，先用 fit/tune 开发；本 test 已被分析，后续反复利用时需保留新的 untouched holdout。
6. 调度接入前单独测 KV 能力/成本及 oracle 空间，再评价 SLO 与真实系统收益。

## 11. 证据与复现

- 正式 tune 汇总：[summary.csv](runs/predictor_experiments/20260927T170043Z/summary.csv)
- 正式 test 汇总：[summary.csv](runs/predictor_experiments/20260927T170043Z_final/summary.csv)
- 分析数值、配对区间与输入哈希：[analysis.json](evidence/predictor_results_20260927/analysis.json)
- 分析脚本：[analyze_time_predictor_results.py](scripts/analyze_time_predictor_results.py)
- 自动接续状态：[state.json](runs/predictor_experiments/20260927T170043Z_final.autofinalize/state.json)
- 每个算法的 metrics.json / predictions.jsonl / manifest.json，以及正式模型和冻结快照均保留。

本次重新核对全部 **10904 条** tune/test 预测记录的有限性、非负性、分位数顺序、pinball、覆盖和宏平均；正式模型与冻结快照 SHA256 校验通过。

复现分析（只读预测明细，不运行模型；使用新的输出文件名）：

```bash
cd /root/flowpilot_predictor
.venv-predictor/bin/python scripts/analyze_time_predictor_results.py \
  --run-dir runs/predictor_experiments/20260927T170043Z \
  --output-dir evidence/predictor_results_recheck \
  --report PREDICTOR_EXPERIMENT_REPORT_RECHECK.md
```

## 12. 进程与缓存清理记录

清理完成时间：`2026-09-27T17:34:13.445723+00:00`。检查时训练和最终测试进程均已自行退出，没有存活的预测计算进程需要终止；移除了以下四个已结束的 tmux 会话，并保存终端输出：

- `predictor-five-20260927T170043Z`
- `predictor-auto-20260927T170043Z`
- `predictor-review-check`
- `predictor-smoke-verified`

删除了以下冗余文件，共 **62.80 MiB**（文件逻辑大小 65,849,102 bytes；文件占用块约 62.99 MiB，未计目录开销）：

| 类别 | 清理范围 | 保留内容 |
| --- | --- | --- |
| 重复验证快照 | `runs/predictor_checks/auto_finalize_20260927/selected_models/` | 删除前逐文件确认与正式训练模型及清单 SHA256 相同；正式 `.autofinalize/selected_models/` 完整保留 |
| 旧调试数据 | `runs/predictor_prepared/schema_debug/` | 原始数据和正式 `runs/predictor_prepared/v1/` 保留 |
| 冒烟模型 | 四个 smoke 检查目录下共 23 个 `model.joblib` | 全部核对 `smoke_only=true`；保留清单、预测、指标、日志、回归检查 XML |
| 可再生缓存 | `.pytest_cache/`、`.ruff_cache/`，及 predictor/scripts/tests 下的 `__pycache__/` | 源码和 Python 虚拟环境保留 |

对正式训练、最终校准测试、正式冻结快照及 prepared/v1 共 **113 个文件**执行清理前后 SHA256 校验，全部一致。正式模型、校准模型、预测明细、汇总、配置与日志均可继续使用；已删除的 smoke 模型若要再次使用，需重新生成。

证据：[清理明细与保留文件哈希](evidence/predictor_results_20260927/cleanup.json)、[tmux 会话清理与终端输出](evidence/predictor_results_20260927/tmux_cleanup.json)。本次未启动新的训练或 test，也未更改已冻结模型与 predictor 实现。

## 13. 当前算法选择与下一轮实验建议

以下是依据本轮结果提出的下一轮候选方案，尚未执行，也不代表已经选出具有独立盲测验证的最终调度策略。

### 13.1 选择 LightGBM，按决策目的使用分位数

主预测器选 LightGBM，继续输出 Q10/Q50/Q90/Q99。普通耗时点估计优先用未校准 Q50；尾部预算实验优先试校准后 Q90；Q99 保留作尾部压力对照。若目标是 KV 卸载或预恢复，需在 ready/剩余时间目标上考察较早分位数 Q10，不能把 Q90/Q99 当作通用的“安全恢复时间”。这些是独立候选用途，不是已经实测的混合输出策略。

| 用途 | 建议起点 | 当前证据与限制 |
| --- | --- | --- |
| 普通单工具 RTT 点预测 | LightGBM raw Q50 | test 微平均 MAE 327.973 ms；Q50 实际覆盖 44.36%，并非已精确校准的中位数 |
| 单工具较晚返回预算 | LightGBM calibrated Q90 | 实际覆盖 87.98%，raw 为 86.05%；仍未达到名义 90% |
| 极端尾部对照 | LightGBM calibrated Q99，并保留经验 Q99 对照 | 实际覆盖 98.52%；全局 +804.681 ms 偏移会放大短工具估计；经验分布 raw 的宏 Q99 pinball 反而更低 |
| KV 卸载、提前恢复 | ready/剩余时间的较低分位数，例如 Q10 | 当前只验证单工具 RTT；应先实现与验证 ready 目标、运行中条件化及实测 KV 操作成本 |

Q90 表示目标条件分布的第 90 百分位，不是模型准确率 90%。不同分位数的 pinball 权重不同，不能看到 Q99 的绝对损失比 Q50 小就认定 Q99 更适合所有用途。分位数模型与区间解释可参阅 [scikit-learn 官方示例](https://scikit-learn.org/1.5/auto_examples/ensemble/plot_gradient_boosting_quantile.html)。

QRF raw 的 Q90 覆盖为 91.99%，宏 Q90 pinball 为 95.463 ms，LightGBM raw 对应 96.692 ms；但 QRF P95 预测约 43 ms，当前不宜作为低开销热路径首选。两者综合精度差异的配对区间跨 0，选择 LightGBM 同时考虑了实测计算成本。

### 13.2 下一轮按顺序开展

1. **冻结本轮证据，先在 fit/tune 开发。** 本 test 已参与分析与候选选择，保留作历史基线；后续算法、校准和调度策略选定后，使用新的、任务组不重叠的 test_v2 进行最终评估。按 task_group 划分，在线历史按事件时间回放，禁止把同任务调用随机分到训练和验证两边。防止用 test 调超参的原则见 [scikit-learn 官方交叉验证说明](https://scikit-learn.org/1.1/modules/cross_validation.html)。
2. **小规模 LightGBM 搜索与特征消融。** 初始候选为树数 {100,200,400} × 深度 {4,6} × 叶节点最少样本 {20,40} × 标签尺度 {ms,log1p}，共 24 组，固定学习率 0.05；先核对配置与实现支持。只在 fit 训练、tune 排序，再对前三组做 3 个随机种子的稳定性复核。随机种子重复不能替代任务组/时间划分验证。另依次比较工具/后端特征、加入参数特征、加入可用历史特征，报告增量收益。
3. **把分位数和校准方法作为明确对照。** 所有现有候选保留 Q10/Q50/Q90/Q99；如需探索折中再新增 Q95 预测头。校准比较 raw、当前全局 ms 偏移、按工具/后端且向上级收缩的校准、相对或 log 尺度残差校准。在 fit 内按任务组划出校准开发子集，并用 tune 选方法；方法冻结后才用保留的最终 calibration 拟合偏移，随后只评估 test_v2 一次。需要更多尾部数据时扩充独立 calibration_v2。不能在 calibration 上反复选择最有利的方法。
4. **固定评价指标。** 同一分位数横向比较逐工具/宏平均 pinball；分别报告 Q10/Q50/Q90/Q95/Q99 的单侧覆盖、Q99 超出次数与幅度、区间宽度、预测 P95/P99 和按任务组 bootstrap。当前 calibration 只有 424 次调用、28 个任务组，99% 名义尾部仅约 4.24 次调用，且存在相关性；增加独立任务组和跨时段数据比重复训练更有助于评估尾部。
5. **验证调度目标和收益。** 先训练独立 next-request-ready 预测器，比较经验分布与 LightGBM；不能把单工具 Q90 求和充当串行批次 Q90。运行中需要条件于“尚未完成”和已过去时间的剩余时间分布，不能简单将 RTT 减 elapsed 后截零。先做无实际操作的 shadow/replay，再按真实 KV 能力开展 KEEP/固定策略、经验预测、LightGBM 预测、oracle 对照；固定 arrival、deadline、正确性口径、资源和工作负载，报告恢复迟到、显存 byte-seconds、迁移量、端到端 P95/P99、正确且按时的 goodput 及调度开销。

KV 卸载的候选判断是：较早就绪分位数仍长于实测卸载＋恢复成本及余量，并确有内存压力和其他可运行工作。预恢复应参考较早就绪时间减恢复提前量，不能等到 Q90/Q99 时才启动。Q10 也只是经验概率估计，不是确定安全界；oracle 对照是收益空间诊断，其信息假设与具体策略仍需说明。

这一轮优先回答“LightGBM 参数/特征是否有效、校准是否损害短工具、分位数是否改善实际调度决策”。复杂神经网络可后续增加为独立对照；本轮证据尚未证明需要它。未启动上述新实验。
