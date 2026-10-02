# 工具耗时预测实验准备（2026-09-27）

状态：实现五个候选、P0 数据构造、四分位数评估与后台入口。用户已启动并完成首轮完整 fit+tune（20260927T170043Z）；真实 calibration/test 尚未启动。

复审版本：`tool-rtt-v2`。25 项测试通过，覆盖并发延迟反馈和合成数据的完整 fit→tune→calibration→test 流程。真实 test 未做模型效果评测。复审后的真实数据小规模工件在 `runs/predictor_checks/review_smoke_20260927/`，tmux 窗格为 `predictor-review-check`。旧版工件不再与新版代码混用；数据准备目录 v1 无需重建。

复审修正：EWMA 保存预测时刻均值并在延迟反馈时配对残差；初始 fit 同样按预测/返回事件回放。后端支持也不足时返回 unsupported，在线 EWMA 后续可积累支持。小规模校准工件继承 smoke 标记；冻结校准不能套用到在线变化的 EWMA。汇总结果显式区分算法、目标、分区、模式和校准版本。

## 目标与实现边界

- 首版预测已知调用从自身 dispatch 到客户端返回的 RTT（`round_trip_ms`），统一输出 Q10/Q50/Q90/Q99，ms。
- `executor_duration_ms` 可在配置中选为独立目标；单独启动、独立保存结果，不能与 RTT 相加。
- T1 用 `llm_response` 事件近似，尚未精确到解析完成；负载和历史来自 T0，显式带 age。
- 返回结果包含 `duration_ms`、method、support、fallback、版本，以及 schema v3 的 per_call envelope。输出整体记录另有真实标签与评分，仅供离线评估。离线 envelope 的 context_epoch/resolution_epoch=0 是占位值，不可直接用于生产调度验收。
- 首版只支持当前已知版本的串行 execute 路径。未知工具/schema/backend、hit、follower、未知路径返回 unsupported，不生成 0 ms。
- 批次 ready 标签已整理，但未训练 ready 模型，也未接入线上调度/KV 服务。
- `remaining_from_dispatch(context, elapsed_ms)` 提供历史分布的条件剩余 RTT；方法显式标为 empirical_conditional，存活样本不足返回 unsupported。它不是批次就绪服务，亦不从四个分位数推导完整生存函数。
- 在线适应本次提供 EWMA 的事件回放；其他四种保持冻结。尚未实现生产级 pending/invalidate 服务、GBDT 在线残差校正或后台重训。

## 五个算法

| 标识 | 实现 | 要点 |
|---|---|---|
| empirical | 后端/版本/工具分组经验分位数 | 少样本退回同后端分布 |
| ewma | log1p 耗时 EWMA + 有界的历史一步预测残差分位数 | 残差对齐预测时刻保存的均值；首轮可对照冻结/在线 |
| cluster | 每类工具的 TF-IDF + KMeans + 簇内经验分位数 | 词表、聚类仅 fit 拟合，稀疏/OOV 回退 |
| qrf | 随机森林叶节点加权经验 CDF | 每棵树把全部 fit 行映射到叶节点；每叶等权，再跨树平均；不是树均值分位数 |
| lightgbm | 四个独立 quantile 头 | alpha=0.1/0.5/0.9/0.99，默认 log1p 标签，返回 ms 后非负截断和单调重排 |

QRF 方法参考 [Meinshausen 2006](https://jmlr.org/papers/v7/meinshausen06a.html)。本实现把全部 fit 响应映射至每棵树的叶节点估计条件分布，包含非该树 bootstrap 内的行。
LightGBM 使用 [官方 quantile/alpha 接口](https://lightgbm.readthedocs.io/en/stable/Parameters.html)。

QRF/LightGBM 的 `feature_level` 支持 identity、parameters、history_state；聚类只使用工具分组与参数文本，经验/EWMA 基线按其定义使用分组/历史。`target_transform` 只控制树模型（QRF 的分裂目标和 LightGBM 的训练目标）；EWMA 固定使用 log1p。当前没有自动超参搜索器；手动调整配置后重复 train，由 tune 指标选配置。本次没有运行消融或超参搜索。

## 已完成的准备

项目独立环境：`/root/flowpilot_predictor/.venv-predictor`，Python 3.12。没有修改现有 Qwen/检索环境，不需要激活 conda。

准备数据：`runs/predictor_prepared/v1/`。RTT 数量 fit/tune/calibration/test = 4056/694/424/674；ready = 3768/645/401/613。四个分区无任务组交集。

输入关联键为 source_attempt + request_id（工具再加 tool_call_id）。仅从原始 arguments、冻结 schema 默认值、T0 已知配置/历史/负载构造 context。effective_arguments、实际 timeout、当前返回值与任务最终结果不进入模型。

特征版本中的工具 schema 指纹仅规范化 code_terminal 描述里的随机 `/tmp/flowpilot-code-local-actor-*/repo`；原始快照、事件、输入文件的 SHA256 保留在 manifest。schema 取每个 attempt 首次请求的冻结定义，这是本数据的适用范围。dataset_revision + 环境身份暂作为语料版本代理，不声称具备独立索引版本遥测。

准备目录中的 JSONL 包含独立 `context` 与 `labels`。算法的 predict 仅接收 context。批次调用数来自完整响应的已知工具列表，绝不按实际成功次数计数。

生成过程只做文件处理；默认计数断言针对本次冻结的 mixed_c4_v1 数据。若换采集批次，需显式修订数据协议。

## 环境和检查命令

当前机器已安装好环境，可直接执行：

```bash
cd /root/flowpilot_predictor
source .venv-predictor/bin/activate
python -m pip check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_time_predictor.py
bash -n scripts/run_predictor_suite.sh scripts/start_predictor_tmux.sh
```

重建隔离环境时，选择新的目录；不要覆盖正在使用的环境：

```bash
cd /root/flowpilot_predictor
/root/miniconda3/bin/python -m venv .venv-predictor-rebuilt
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL='' \
  .venv-predictor-rebuilt/bin/python -m pip install \
  --index-url https://pypi.org/simple -r scripts/requirements-predictor-lock.txt
export PREDICTOR_PYTHON="$PWD/.venv-predictor-rebuilt/bin/python"
```

默认 Python 已来自本机 miniconda，无需另建 conda 环境。若使用自备环境，只需把 `PREDICTOR_PYTHON` 指向其解释器；tmux 启动器会显式传入这个变量。

数据已经准备好，无需重复执行。要重新审计/构造，使用新的目录：

```bash
python -m predictor.cli prepare --output runs/predictor_prepared/v2
```

所有准备、训练、校准、评估输出目录必须不存在，避免意外覆盖。

## 小规模验证：可以立即复跑

```bash
cd /root/flowpilot_predictor
CHECK_ID="$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/start_predictor_tmux.sh smoke "predictor-smoke-$CHECK_ID" \
  runs/predictor_prepared/v1 "runs/predictor_checks/$CHECK_ID"
tmux attach -t "predictor-smoke-$CHECK_ID"
```

按 `Ctrl-b`，再按 `d` 脱离 tmux。后台执行不受终端断开影响。完成后窗格保留，显示 `Pane is dead (status 0)` 是正常成功状态。

smoke 模式每路仅取按事件时间排序的前 120 条 fit、前 36 条 tune；森林最多 8 棵树，LightGBM 每头最多 8 棵，线程数 1，bootstrap 10 次。结果明确标为 smoke_only；不能据此比较优劣。smoke 模型禁止直接用于完整评估。

五路之后追加 EWMA online 对照，随后自动汇总 CSV。退出码保存在输出根目录的 `exit_code`；任何一路失败，整体返回非零。详细日志在 `logs/*.log`。

## 以后正式启动：本次未执行

先检查要运行的命令（不训练，也不创建 tmux）：

```bash
cd /root/flowpilot_predictor
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/start_predictor_tmux.sh dry-run "predictor-five-$RUN_ID" \
  runs/predictor_prepared/v1 "runs/predictor_experiments/$RUN_ID" \
  configs/predictor/default.json
```

确定需要进行首轮固定配置实验时，再执行：

```bash
bash scripts/start_predictor_tmux.sh full "predictor-five-$RUN_ID" \
  runs/predictor_prepared/v1 "runs/predictor_experiments/$RUN_ID" \
  configs/predictor/default.json

tmux attach -t "predictor-five-$RUN_ID"
```

五个独立 CPU 进程并行，各自只用 fit 训练、tune 评估；不自动使用 calibration 或 test。默认模型线程数 8，原生 BLAS 池受到限制；默认隐藏 CUDA 卡。当前样本规模无需 GPU。

查看进度：

```bash
cat "runs/predictor_experiments/$RUN_ID/exit_code"
tail -n 50 "runs/predictor_experiments/$RUN_ID/logs/lightgbm.log"
cat "runs/predictor_experiments/$RUN_ID/summary.csv"
tmux list-sessions
```

`exit_code` 和 `summary.csv` 在运行完成后出现。完成后可清理对应窗格：`tmux kill-session -t "predictor-five-$RUN_ID"`。运行中停止会取消该会话的实验；不要误用其他任务的 session 名称。

单独运行某一算法：

```bash
python -m predictor.cli train --algorithm lightgbm \
  --data runs/predictor_prepared/v1 --config configs/predictor/default.json \
  --output runs/predictor_experiments/single_lightgbm
```

算法可选 empirical、ewma、cluster、qrf、lightgbm。`scripts/train_time_predictor.py` 和 `scripts/replay_time_predictor.py` 分别提供 train/evaluate 的兼容入口。

## 后续校准及最终评估：需冻结方案后手动执行

以下假设 `RUN_ID` 指向实际完成的 full 输出，不是 smoke。首轮调参只看 tune；确定算法、配置和报告口径后再校准。

```bash
cd /root/flowpilot_predictor
source .venv-predictor/bin/activate
ALGO=lightgbm
MODEL="runs/predictor_experiments/$RUN_ID/$ALGO/model.joblib"

python -m predictor.cli calibrate \
  --data runs/predictor_prepared/v1 --model "$MODEL" \
  --output "runs/predictor_experiments/$RUN_ID/${ALGO}_calibrated"

python -m predictor.cli evaluate \
  --data runs/predictor_prepared/v1 --model "$MODEL" \
  --split test --allow-test --mode frozen \
  --output "runs/predictor_experiments/$RUN_ID/${ALGO}_test_raw"

python -m predictor.cli evaluate \
  --data runs/predictor_prepared/v1 \
  --model "runs/predictor_experiments/$RUN_ID/${ALGO}_calibrated/model.joblib" \
  --split test --allow-test --mode frozen \
  --output "runs/predictor_experiments/$RUN_ID/${ALGO}_test_calibrated"
```

校准采用 calibration 上 pooled 的逐分位数 ms 残差分位数偏移，再做单调重排；它是经验边际校准，不声称严格 conformal 保证。校准后的模型禁止返回 tune 评估，避免反向泄漏。默认 suite 不调用校准或 test。

单独验证 EWMA 延迟反馈（每次重置为相同 fit 状态）：

```bash
python -m predictor.cli evaluate \
  --data runs/predictor_prepared/v1 \
  --model "runs/predictor_experiments/$RUN_ID/ewma/model.joblib" \
  --split tune --mode online \
  --output "runs/predictor_experiments/$RUN_ID/ewma_online_rerun"
```

事件先按同 clock_domain 的 monotonic_ns 合并；同时间用 source_attempt、源 seq 等确定顺序。predict 只保存预测；END 到达后先计分，再更新。重复 sample/反馈、跨时钟域、逆序、训练/评估组重叠会报错。在线回放要求 fit 所有标签早于评估，不能把未来 fit 状态用于历史模拟。

## 结果文件与 Q99 解读

每个算法：

- `model.joblib`：仅 fit 得到的模型，限本地可信文件；tune 回放状态不会覆盖它。
- `manifest.json`：配置、依赖版本、输入与代码哈希、样本量、smoke 标记、模型哈希。
- `predictions.jsonl`：逐调用预测、真实标签、按名称保存的 `score.pinball_ms.q10/q50/q90/q99`、身份/时刻及热路径耗时。
- `metrics.json`：Q50 MAE、四分位数 pinball/覆盖、80% 区间覆盖与宽度、Q99 超出次数及平均正超额、逐工具/benchmark/执行状态、宏/微平均、按任务组 bootstrap 区间。
- suite 根目录 `summary.csv`：五种冻结算法与 EWMA 在线对照的汇总。

Q99 是探索性输出。`q99_low_support` 初始启发式阈值是参考行数少于 1000 或任务组少于 30；QRF 还看有效权重样本量，EWMA 还看残差缓冲样本量。这不是可靠性定理。默认 EWMA 缓冲为 512，Q99 会保守标记为低支持。尾部不足时不能把“零超出”解释为严格 99% 覆盖。

覆盖指标使用 y <= Qτ；bootstrap 以整个 task_group 为单位，提供工具宏平均 pinball 和微平均 Q99 覆盖的 95% 经验区间。按分位数分别报告，不只看四者平均值。默认 bootstrap 500 次可在配置调整。

预测耗时包含 context 到特征提取及模型预测，不含文件读取/JSON 解析/线上 RPC。并行训练期间的延迟仅作诊断；正式评估热路径 P95 应在独占资源下单独 evaluate。当前不声称满足交接文档建议的 1 ms 预算。

## 文件组织

- `data.py`：P0 关联、时间/分区校验、标签/上下文隔离、哈希。
- `features.py`：三个级别的明确特征白名单。
- `models.py`：五种算法、稀疏回退、条件剩余经验分布。
- `evaluation.py`：延迟反馈回放、计分、任务组 bootstrap。
- `cli.py`：prepare/train/evaluate/calibrate。
- `tests/test_time_predictor.py`：数据泄漏、未知路径、QRF 分布语义、状态回放、边界条件及小规模兼容性测试。


## 完整阶段的启动顺序

`full` 不是自动调参或全分区一键实验，它执行固定配置的 fit 训练 + tune 评估。后续流程如下：

| 阶段 | 使用分区 | 是否学习 | 是否由 start_predictor_tmux.sh full 自动执行 |
|---|---|---|---|
| 基础模型训练 | fit，4056 条 | 是 | 是 |
| 配置评价 | tune，694 条 | 冻结模式不更新模型；EWMA online 单独计分后更新临时状态 | 是 |
| 选择超参数 | 仅看 tune | 需手动比较配置；每个候选仍只在 fit 训练 | 否，无自动搜索器 |
| 经验分位数校准 | calibration，424 条 | 只拟合偏移，不重训树/词表/聚类 | 否，单独启动 |
| 最终报告 | test，674 条 | 默认冻结；不再选参 | 否，必须显式 --allow-test |

不把四个分区合并训练。首轮也不自动 fit+tune 重训。若需要合并重训，应另行冻结规则与重新校准。

第一阶段完整命令（本次未执行正式运行）：

```bash
cd /root/flowpilot_predictor
source .venv-predictor/bin/activate
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$PWD/runs/predictor_experiments/$RUN_ID"
bash scripts/start_predictor_tmux.sh full "predictor-five-$RUN_ID"   "$PWD/runs/predictor_prepared/v1" "$RUN_DIR" "$PWD/configs/predictor/default.json"
tmux attach -t "predictor-five-$RUN_ID"
```

按 Ctrl-b 再 d 脱离，不会停止训练。回到原终端后检查 `cat "$RUN_DIR/exit_code"` 和 `cat "$RUN_DIR/summary.csv"`。重复试配置时指定新的输出目录，只看 tune 的结果。所有默认配置仍是起点，不是已调优配置。

冻结方案后，批量校准并测试五个模型：

```bash
cd /root/flowpilot_predictor
# RUN_DIR 应保留为上面已完成、最终选定的正式模型根目录。
FINAL_DIR="${RUN_DIR}_final"
bash scripts/finalize_predictor_suite.sh dry-run "$RUN_DIR" "$FINAL_DIR"   "$PWD/runs/predictor_prepared/v1"

tmux new-session -d -s "predictor-final-$RUN_ID" -c "$PWD"   bash "$PWD/scripts/finalize_predictor_suite.sh" full "$RUN_DIR" "$FINAL_DIR"   "$PWD/runs/predictor_prepared/v1" --allow-test

# 运行期间可查看；命令完成后该会话可能自动退出，结果文件仍保留。
tmux attach -t "predictor-final-$RUN_ID"
cat "$FINAL_DIR/exit_code"
cat "$FINAL_DIR/summary.csv"
```

finalize 脚本只处理已选定模型，不训练和调参。它逐算法执行 calibration、test_raw、test_calibrated；全量脚本拒绝 smoke/已校准/旧版工件。默认 dry-run 只打印命令。最终报告保留五算法各两种冻结结果，共 10 行；EWMA online 的 test 若要研究，应提前固定协议并另行 evaluate，不使用冻结校准偏移。

如果五个算法分别来自不同调参目录，可将每个选定算法的完整工件目录（包含 manifest 与 model）链接到一个 `selected/empirical`、`selected/ewma`、`selected/cluster`、`selected/qrf`、`selected/lightgbm` 根目录，再以该根目录作为 RUN_DIR。不要在看到 test 后更换选定模型。

## 输出格式（复审版本）

train / evaluate 每个输出目录包含 `predictions.jsonl`、`metrics.json`、`manifest.json`；train 另有 `model.joblib`。calibrate 只生成校准后的 `model.joblib` 与 `manifest.json`，其中记录 `offsets_ms` 和 `calibration_version`，不把 calibration 拟合效果当成最终成绩。

`predictions.jsonl` 每行是一个调用在 T1 的预测及返回后的评分。下例为简化的格式示例，数值是说明值：

```json
{
  "output_schema_version": 2,
  "split": "tune",
  "target": "round_trip_ms",
  "quantiles": {"q10": 0.1, "q50": 0.5, "q90": 0.9, "q99": 0.99},
  "sample_id": "sample-001",
  "task_group_id": "group-001",
  "adapter": "hotpot",
  "tool": "search",
  "y_ms": 600.0,
  "prediction": {
    "duration_ms": {"q10": 200.0, "q50": 500.0, "q90": 1000.0, "q99": 3000.0},
    "method": "lightgbm",
    "online_state_version": 0,
    "support": {"status": "supported", "reference_rows": 820, "reference_task_groups": 80, "q99_low_support": true},
    "fallback": {"used": false, "reason": null}
  },
  "score": {
    "pinball_ms": {"q10": 40.0, "q50": 50.0, "q90": 40.0, "q99": 24.0},
    "absolute_q50_error": 100.0
  },
  "predict_ms": 0.2,
  "update_ms": 0.0
}
```

实际行还包含 execution_status、observed_ns、完整 schema v3 envelope（per_call、clock_domain、as_of、模型版本、校准版本等）。unsupported 的 duration_ms 和 score 为 null，并带回退原因，不能补 0。

`metrics.json` 顶层：evaluation（算法/目标/训练分区/评估分区/模式/版本/smoke）、micro、by_tool、by_adapter、by_execution_outcome、tool_macro_pinball_ms、tool_macro_mean_pinball_ms、tools_total、tools_scored、macro_excludes_unsupported_tools、bootstrap、口径说明。

`summary.csv` 一行一个模型/模式/分区。主要列：experiment、algorithm、target、training_split、evaluation_split、evaluation_mode、model_version、calibration_version、smoke_only、rows、supported_rows、unsupported_rows、tools_total、tools_scored、q50_mae_ms、q10/q50/q90/q99_macro_pinball_ms、q10/q50/q90/q99_micro_coverage、q99_exceedances、q99_mean_excess_ms、central_80_coverage、central_80_width_ms、fallback_fraction、predict_p95_ms。


## 无人值守自动接续（2026-09-27）

新增独立脚本 `scripts/auto_finalize_predictors.py` 与 tmux 启动器 `scripts/start_auto_finalize_tmux.sh`。未修改 predictor 模型代码，当前已完成的 v2 模型仍可使用。25 项算法测试 + 7 项接续测试，共 32 项通过。

行为：等待指定运行目录出现 `exit_code=0` → 校验五个完整模型 → 保存模型与 manifest 的独立快照和 SHA256 → 逐算法 calibration、原始冻结 test、校准冻结 test → 汇总。训练失败、校验失败或超时会停止，不进入 test；相同输出目录只能接续一次，不自动重试。默认每 10 秒检查，最多等待 24 小时。脚本不做参数搜索或自动选型。

`--parameters-frozen --allow-test` 表示沿用指定目录的固定配置做最终评价。如果仍要根据 tune 改参数，应先完成调参，再对选定目录启动此脚本。

本轮完整启动命令，不依赖之前终端的 RUN_ID/RUN_DIR 变量：

```bash
cd /root/flowpilot_predictor
bash scripts/start_auto_finalize_tmux.sh predictor-auto-20260927T170043Z   --run-dir "$PWD/runs/predictor_experiments/20260927T170043Z"   --parameters-frozen --allow-test
```

这轮 fit+tune 已完成，因此执行上面的命令会直接进入后续阶段；其他仍在训练的目录会自动等待。SSH 断开不影响运行，服务器重启不会自动恢复任务。

只检查计划而不启动任何实际计算：

```bash
.venv-predictor/bin/python scripts/auto_finalize_predictors.py   --run-dir runs/predictor_experiments/20260927T170043Z --dry-run
```

观察和查看完成结果：

```bash
tmux attach -t predictor-auto-20260927T170043Z
cat runs/predictor_experiments/20260927T170043Z_final.autofinalize/state.json
tail -n 30 runs/predictor_experiments/20260927T170043Z_final.autofinalize/finalize.log
cat runs/predictor_experiments/20260927T170043Z_final/exit_code
cat runs/predictor_experiments/20260927T170043Z_final/summary.csv
```

按 Ctrl-b 再 d 脱离 tmux。状态为 waiting_for_training、validating_models、finalizing、complete、failed 或 cancelled。控制目录 `.autofinalize/` 另有 watcher.log、selection.json、selected_models/ 和自身 exit_code。最终阶段日志保存在 `_final/logs/`。失败时查看错误，不会自动重复 test。重试需明确选择新的输出路径。

当前耗时实测：首轮 17:00:43 UTC 启动，17:01:19 UTC 完成，约 36.7 秒。五算法 fit 都在 1.3 秒以内；主要耗时为 QRF 对 694 条 tune 的逐条预测，合计约 29.5 秒。后续每算法需预测 424 + 674 + 674 = 1772 条，以当前吞吐外推纯预测合计约 79 秒，计入进程启动、序列化和 bootstrap，预计后续整体约 2–4 分钟。此为估计，不是已运行 test 的计时结果。
