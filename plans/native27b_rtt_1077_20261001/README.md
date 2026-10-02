# 27B 新轨迹预测器实验：准备完成，尚未训练

本目录提供独立的五算法训练、校准、测试及报告入口。已执行的操作仅为读取轨迹、构造训练输入、完整性/防泄漏检查、合成契约测试和命令 dry-run；没有拟合任何估计器，没有执行真实模型预测、校准或效果测试，也没有启动训练 tmux。

## 数据与分区

输入为 `/data1/ql_flowpilot_predictor/runs/native27b_c4_v1/` 的 600 题，以及 `runs/native27b_c4_increment_v2/` 的 477 题，共 1077 个不同题目。均为 Qwen3.5-27B、原生 vLLM、四任务并发、thinking 关闭的已完成轨迹。不需要模型服务、语料服务器、SSH 隧道或联网。

按用户确认，QuixBugs 的 40 题加入辅助训练，保留 `source_research_split=historical_dev`；它们不进入 tune、calibration 或 test，不报告 QuixBugs 泛化准确率。

| 用途 | 题目数 | 有效工具 RTT 行数 | 提供有效 RTT 的 task_group 数 |
|---|---:|---:|---:|
| fit，含 QuixBugs 辅助训练 | 665 | 9335 | 519 |
| tune | 155 | 2147 | 117 |
| calibration | 101 | 1588 | 81 |
| test | 156 | 2169 | 120 |
| 合计 | 1077 | 15239 | 分区之间无任务组交集 |

主 fit 原有 625 题产生 8941 条 RTT，QuixBugs 产生 394 条。4 道任务没有可用工具 RTT，保留在 `task_audit.jsonl`，不填 0，也没有因为最终答错而删除任务。合格题号 CSV 不作为筛选条件或特征。

准备后的数据位于 `/data1/ql_flowpilot_predictor/predictor_prepared/native27b_1077_v1/`。保留 `fit_primary.jsonl`（8941 行）、`quixbugs_auxiliary.jsonl`（394 行）及 `tune_forward.jsonl`（939 行）作为可追溯子集；这些是主数据的视图，不能重复计入总样本量。

逐条核对 input/target 对齐、唯一调用 ID、同一时钟域、T1→工具开始→返回顺序、历史结束时间早于快照、负载快照早于预测、原始参数与 schema 默认值。实际执行错误但有完整返回时间的调用可保留；控制动作和未执行动作排除。RTT 是客户端定义的耗时，不能与 executor 时长相加，也不是单纯网络延迟。超时返回可用于预测返回时间，原始自然完成时间的右删失标记保留。

冻结信息在 `protocol.json`、准备数据的 `manifest.json` 及 4354 个来源文件哈希中。后端、工具 schema 和版本身份保持原规则，不为提高支持率强行混合不同后端。三个主要评估分区的身份均在 fit 出现；稀疏样本仍由原模型支持/回退逻辑处理。

## 五种算法与流程

主实验并行运行 empirical、EWMA、TF-IDF/KMeans cluster、QRF、LightGBM。沿用此前算法实现和固定配置：seed 20260927；每路 8 线程；QRF 200 棵树；LightGBM 每个 quantile 头 200 轮、learning_rate 0.05、max_depth 6、min_samples_leaf 20，log1p 标签；KMeans 最多 8 簇；EWMA alpha 0.2、残差缓冲 512。所有输出统一还原到 ms，保证 Q10 ≤ Q50 ≤ Q90 ≤ Q99 且非负。

工作流为：

1. 五算法并行在 fit 拟合、在 tune 做冻结评价。词表、聚类、树和初始化状态只由 fit 学得。此轮不执行额外超参搜索，也不合并 fit+tune 重训。
2. 保存 `frozen_before_test.json`，固定五个模型及配置。按 tune 的四分位工具宏平均 pinball 记录全支持候选中的最低者，作为候选说明，不自动部署。
3. 五算法并行在 calibration 拟合全局毫秒分位残差偏移；与旧实验相同，偏移后非负截断、单调重排。这是经验校准，不是严格条件覆盖保证。
4. 对固定的五个模型分别执行原始与校准后 test，最多五个进程并行。测试结果不用于重新选参数或重新拟合。
5. 执行独立 EWMA 在线对照，见下节。
6. 所有训练/评价子进程退出后，逐算法测量校准后预测器的进程内开销；使用分层抽取的 tune context，不用标签更新模型。此独立计时避免五算法之间的 CPU 竞争。其他机器负载仍可能影响时延。
7. 自动导出报告及 CSV，结束所有本工作流的计算子进程，保留模型、预测和日志。没有自动替换现有调度器使用的预测模型。

只使用 CPU，无需 GPU。本机可见 128 CPU，cgroup 无 CPU quota，使用 5 路 × 每路 8 线程；BLAS 等额外线程池在入口限流。训练进程在 tmux 中运行，关闭电脑不会中断。

## EWMA 在线对照的时间边界

两轮 fit 合并后，第二轮 fit 已晚于第一轮 tune；QuixBugs 更晚于第二轮 tune。因此不能直接把合并训练模型放到整份 tune 上，声称进行了真实时间顺序的在线预测。

额外建立一个 EWMA 对照，只在 `fit_primary`（不含晚到的 QuixBugs）拟合，然后在第二轮 `tune_forward` 939 行上分别做冻结/在线评价。已验证所有训练观测都早于该评估流，使用同一时钟域；在线更新只在真实返回事件后发生，先评分再更新。同一初始模型用于两种模式，不能将这个子集的结果与合并 tune 的总体结果直接比较。

该对照不会在 calibration/test 上在线更新，也不会把冻结校准偏移套到持续变化的在线状态上。

## 以后执行的命令

仅检查部署与所有来源文件，不训练：

```bash
PLAN=/root/flowpilot_predictor/plans/native27b_rtt_1077_20261001
bash "$PLAN/run.sh" check
```

仅显示将运行的完整命令，不创建模型或训练会话：

```bash
bash "$PLAN/run.sh" dry-run
```

**真正启动全部训练、校准、测试和报告**，以后准备好再执行：

```bash
PLAN=/root/flowpilot_predictor/plans/native27b_rtt_1077_20261001
bash "$PLAN/run.sh" start
```

无需激活 conda。入口固定使用 `/root/flowpilot_predictor/.venv-predictor/bin/python` 和本目录 `code/predictor` 快照，避免影响旧版模型的严格代码哈希校验。

```bash
# 查看阶段、已完成步骤、活跃子进程和最终退出码
bash "$PLAN/run.sh" status

# 查看主日志
 tail -f /data1/ql_flowpilot_predictor/logs/predictor_native27b_1077_v1/native27b_1077_v1.workflow.log

# 本实验使用独立 tmux socket，默认 tmux ls 不显示它
 tmux -L flowpilot-predictor-27b ls
```

默认运行 ID 为 `native27b_1077_v1`。如需明确启动另一份实验，可以给 `start/status/stop` 的第二个参数指定新 ID；重复运行会重新访问同一 test，不能再声称它是新 holdout。已有输出或同名 tmux 会话会被拒绝，不覆盖或静默续跑。运行过程中不要修改封存代码、配置或数据；检查会拒绝哈希不匹配。

需要中断时运行 `bash "$PLAN/run.sh" stop`。它给本工作流发中断，父进程只终止自己创建的子进程组，不清理其他任务或删除结果。该操作是终止，不是可恢复暂停。

## 输出与“准度”的含义

默认结果根目录：

`/data1/ql_flowpilot_predictor/predictor_experiments/native27b_1077_v1/`

- `train/<algorithm>/model.joblib`：未校准模型及 tune 预测/指标。
- `final/<algorithm>_calibrated/model.joblib`：校准模型及校准偏移、来源说明。
- `final/<algorithm>_test_raw/`、`final/<algorithm>_test_calibrated/`：测试预测与完整指标。
- `online/ewma_forward_fit/`、`online/ewma_forward_online/`：时间合法的冻结/在线对照。
- `reports/REPORT.md`：五算法校准前后总表。
- `reports/overall.csv`、`by_tool.csv`、`by_benchmark.csv`：总体、逐工具、逐 benchmark 指标。
- `reports/predictions.csv`：每次调用的真实 RTT、Q10/Q50/Q90/Q99、分区、身份、支持/回退状态。
- `reports/isolated_latency.csv`：独立计时的预测 P50/P95/P99。
- `status.json`、`exit_code`、`logs/`、`deployment_snapshot/`：状态、日志和复现快照。

预测质量报告 Q10/Q50/Q90/Q99 各自的 pinball loss 与经验覆盖率、覆盖偏差百分点、Q50 MAE、Q10–Q90 中央 80% 区间覆盖/宽度、Q99 超出次数及超出幅度。覆盖率应接近 10%/50%/90%/99%，不是越高越好，也不等于“模型有 99% 置信度”。Q99 结合每类工具的有效样本、任务组数量和 tail support 看待。

按 task_group 做 500 次 bootstrap，原实现提供各分位工具宏平均 pinball 的 95% 区间及微平均 Q99 覆盖区间。工具宏平均和逐调用微平均分开报告，不用某类调用数量占优掩盖其他工具表现。所有 unsupported 行保留，报告分母与回退比例。

流水线中的预测耗时包含并行竞争，单独的 isolated_latency 才用于主要进程内成本比较；两者均不包含 RPC/调度器的完整端到端开销。

本轮保持已有研究分组，不把已查看过任务正确率及旧实验结果的 test 宣称为全研究过程完全未见的盲测集。结果仅描述串行 execute 工具 RTT，不能证明 tool reuse 命中、在途复用或联合 KV 调度的收益。
