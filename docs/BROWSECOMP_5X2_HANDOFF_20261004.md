# BrowseComp-Plus 5 × 2 框架实机测试交接

更新日期：2026-10-04，北京时间。

**前置准备已完成，10 次实机任务尚未开始。** 用户因 GPU 资源占用要求本次停在准备阶段。远端 BrowseComp MCP 已启动并通过真实工具调用；本机没有启动 vLLM 或常驻 FlowPilot 网关。当前汇总为 `not_run: 10`，不能据此前置检查宣称完整框架已经通过实机测试。

## 固定的实验与 SLO

历史来源是 `/data/ql_flowpilot_predictor/manifests/native27b_c4_700_20260929`；最初消息中末尾为 `2026092` 的路径按前文确认使用 `20260929`。历史脚本中的 `/data1` 是容器内映射路径，本次在宿主机使用 `/data`。原 `fixed_experiment.sh` 是关闭预测器/调度器的采集流程，不能直接用它验证本次完整链路。

新实验目录：

```text
/home/liyachen/workspace/experiments/flowpilot/browsecomp5x2-20261003T161126Z
```

目录名采用 UTC，因此日期为 10 月 3 日。历史数据只通过临时 Docker 容器的只读挂载读取，没有改动权限或历史结果。

清单共有 200 道 BrowseComp 题目，其中 197 道满足 `execution_status=completed` 且历史 `duration_s` 为正有限数。按数字题号排序，使用 `random.Random(20261004).sample(..., 5)` 抽样；没有按答案正确率或耗时筛选。抽到的五题都来自历史 fit 分区，本次属于历史题目重放，不是未见题评估。

| 题号 | 历史完成耗时（秒） | 每次运行 SLO（秒） |
| --- | ---: | ---: |
| 275 | 789.756 | 1184.634 |
| 82 | 401.292 | 601.938 |
| 1238 | 659.803 | 989.704 |
| 563 | 172.261 | 258.392 |
| 695 | 407.628 | 611.443 |

用户已确认 **每题 SLO = 历史完成耗时 × 1.5**。文件中保留完整精度。第一轮运行以上五题，全部返回后再运行第二轮；每轮任务并发为 4，第五题等待空闲 worker。两轮沿用同一个新建缓存，观察重复任务的复用行为。

SLO 从本次 worker 的 `task_start` 开始，覆盖环境准备、会话、LLM/Tool 等待及清理，与历史 `duration_s` 对齐；不包含领取 worker 前的 campaign 排队时间或服务启动时间。第二轮使用新的开始时刻和相同预算。SLO 是完成目标，超时仍按原有任务预算继续执行，不替代 `task_timeout=14400`、60 iterations、160 Tool calls、100 LLM requests 等原配置。

## 已落地的代码入口

OpenHands benchmark 新增 `benchmark_adapters/workflow_slo.py`，由 `runner.py` 在创建 `LocalConversation` 前登记 `workflow_started_at` 和 `deadline`。配置入口为实验 profile 中：

```json
{
  "workload": {
    "baseline_latency_path": "baselines.json",
    "baseline_latency_sha256": "文件中已写入实际 SHA256",
    "slo_multiplier": 1.5
  }
}
```

基线按 dataset ID、revision、task ID、题目文本哈希匹配。每次运行生成新的 conversation UUID，对应 `job-<UUID>` 和 `line-<UUID>`；`run_id` 只标识轮次，不作为 job ID。SDK 再次注册同一 job 时保留先登记的 deadline。注册失败会明确报错，不会悄悄以无 SLO 模式运行。每次任务输出 `slo.json`，结果新增 `slo_met` 和 `slo_lateness_s`。

当前 SDK 实际提交为 `7a976b07332aa1b5e42fd4bbddb1762ce0e24a37`。本机适配器的旧版本 pin 会拒绝该已集成 FlowPilot 的 SDK，已将 pin 更新为这个实际提交；核心源码和 editable 安装校验仍启用。预测器仓库 `adapters/openhands_current` 已同步 SLO 改动，并保留其历史 SDK 版本支持及 live_web 后端。

## 服务与配置

```text
OpenHands -- HTTP :18852 --> FlowPilot + Predictor + Scheduler
                                      |
                                      +-- HTTP :18851 --> vLLM (GPU 0,1,2,3)
OpenHands -- MCP :18123 -- SSH --> qiulin_docker:127.0.0.1:8123
```

- MCP 部署：`qiulin_docker:/home/qiulin/flowpilot_predictor/deploy/native27b_fixed_c4_20260929`。仅启动 BrowseComp，未启动 Hotpot。
- 远端 tmux socket：`flowpilot-corpus-native27b`；session：`browsecomp`。启动时 MCP 进程 PID 为 `2233075`，索引报告 100195 篇文档。PID 会在重启后变化。
- 远端日志：`/home/qiulin/flowpilot_predictor/logs/native27b_fixed_c4/browsecomp.log`。监听 `127.0.0.1:8123/mcp`，通过 SSH alias `qiulin_docker` 访问。
- vLLM：现有 Qwen3.5-27B、TP=4、BF16、并发 4、上下文 262144；使用 GPU 0–3，CPU KV offload 64 GiB。保持当前 KV 扩展和已标定 cost model。
- 网关：单 worker、`prefill_slack` admission、真实预测器、retention、exact reuse、DCS；semantic 为 shadow。没有开启合成 Tool duration。
- 预测器：共享 LightGBM 模型，4 个 CPU worker，q50，在线分位数校准；加载 SHA256 为 `0432b075b8db58068c70326cddd84ec8aa02084a5ad0f9db4d00fb9b4a75db13`。
- MCP source commit 为 `046949032b0328319cc9a02663a759ec601d9402`；部署文件和服务入口哈希保存在 `mcp-provenance.json`。新检索契约的 `server_policy_revision` 根据这份实际部署证据生成。

## 下次运行

在当前宿主机执行，无需进入旧采集 Docker。启动器会使用各部件自己的 Python 环境；凭据保存在新实验目录的私有文件中，不需要手动粘贴到命令行。

先做可重复的 CPU/MCP 检查：

```bash
cd /home/liyachen/workspace/flowpilot_predictor
python3 plans/browsecomp5x2_20261004/experiment.py check
```

确认 GPU 0–3 可以用于实验后，执行完整的两轮：

```bash
python3 plans/browsecomp5x2_20261004/experiment.py run
```

`run` 会重新做前置检查，检查 GPU 0–3 是否已有计算进程及是否满足配置的显存需求，检查 18851/18852 端口，然后启动 SSH 隧道、vLLM、预测器网关，依次执行两轮。运行结束或失败时汇总已有结果，并关闭本次启动的本机网关、vLLM 和隧道；**远端 MCP 保持运行**。启动器按 PID 与进程启动时间识别自己创建的服务。

运行状态、结果汇总与单独清理：

```bash
python3 plans/browsecomp5x2_20261004/experiment.py status
python3 plans/browsecomp5x2_20261004/experiment.py report
python3 plans/browsecomp5x2_20261004/experiment.py stop-local
```

调试时可用 `start` 只启动服务；正式 `run` 自带启动步骤，不应在 `start` 后直接重复启动。若尝试已经开始，启动器会保留该次证据并拒绝覆盖，需为再次尝试另建实验目录。冻结清单会检测适配器源码变化；下次如修改代码，应重新准备清单，不要手改哈希绕过校验。

远端状态可直接检查：

```bash
ssh qiulin_docker 'tmux -L flowpilot-corpus-native27b list-panes -t browsecomp -F "#{pane_pid} #{pane_dead} #{pane_dead_status}"'
```

`pane_dead=0` 仅表示进程仍在，实际可用性以 `check` 的 MCP 调用结果为准。

## 已完成的验证与实机待验项

CPU/MCP 前置检查已通过，证据为 `preflight.json`、`mcp-check.json`、`embedding-check.json` 和 `services/check-*.log`：

- 两轮各 5 题已经通过官方 collector 的 prepare/validate，SDK provenance 和冻结输入哈希检查通过；五份历史结果副本的 SHA256 与原记录一致。
- 实际调用 MCP `search` 返回 5 条结果，再对返回的 docid 调用 `get_document` 成功；两种工具都提供真实 handler timing。该检查只用了通用检索词，没有运行选中的 benchmark 题目。
- 真实 LightGBM 模型加载成功；真实 Qwen3-Embedding-0.6B 在 CPU 上输出 1024 维归一化向量。
- vLLM 的 `launch vllm --check` 参数解析通过，只读取设备信息，没有启动引擎或加载 27B 模型。
- 本机 SDK/SLO/runner/collector 回归 31 项通过，FlowPilot admission/frontier/protocol 回归 30 项通过，发布版适配器回归 8 项通过。修改过的 SDK 文件逐文件 pre-commit 通过；新增脚本通过 Ruff 和语法检查。CPU 集成测试中的推理响应是测试夹具，不计入实机 benchmark。

下次仍需从真实结果核对：10 次任务的完成状态与独立身份，SLO 达成数，MCP 本地执行与缓存命中的区分，预测器估计/真实 RTT 反馈，调度队列/credit 归还，KV query/retention 回执及实际引擎恢复或重算。KEEP/OFFLOAD/DROP 命令被接收本身不能证明 CPU KV 已被后继推理消费；未触发的路径应记为未覆盖。答案正确率需要另外运行正式评价器，本次运行入口只做框架执行与 SLO 记录。

## 证据位置

以下路径都相对新实验目录：

| 路径 | 用途 |
| --- | --- |
| `selection.json` | 随机种子、197 题候选总体、5 题来源及原结果哈希 |
| `historical/<题号>/` | 原结果、原 profile、原公开任务副本 |
| `questions.jsonl` | 5 道任务的查询输入，不包含标准答案 |
| `baselines.json`、`profile.json`、`browsecomp.toml` | SLO、框架与 benchmark 配置 |
| `registry.json`、`mcp-provenance.json` | 复用 schema/scope 与远端部署证据 |
| `campaign.round1.json`、`campaign.round2.json` | 两轮运行规格 |
| `rounds/<run_id>/campaign.json`、`validation.json` | 官方 collector 冻结清单及验证结果 |
| `rounds/<run_id>/tasks/browsecomp--<id>/attempt-001/` | 下次生成的结果、事件、SLO、SDK 状态 |
| `services/` | 启动日志、PID 记录、各轮前后 predictor/scheduling/reuse/DCS/metrics 快照 |
| `gateway/` | 下次生成的 gateway trace、全新 reuse/DCS 数据库 |
| `benchmark-summary.json` | 10 次状态、耗时、SLO、job/conversation 映射；当前均为未运行 |
| `source-state.json`、`verification.json` | 当前源码版本与验证记录 |

题目、原答案、日志及密钥只保存在私有实验目录，未放入仓库文档。
