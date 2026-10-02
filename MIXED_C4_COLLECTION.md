# 三类任务 C4 工具耗时采集：准备与启动

本文件中的服务和采集命令由使用者手动执行。本次准备没有启动 Qwen、远端语料服务或 benchmark 任务。

## 采集规模与口径

每类先收集 150 个不同任务，共 450 个：训练 fit 100、调参 tune 20、校准 calibration 10、最终测试 test 20。三类为 HotpotQA、BrowseComp-Plus、LiveCodeBench，不含 QuixBugs。

这是轻量工具耗时预测器的起步规模，不保证足以训练复杂的上下文预测模型。先在 fit 的前 20–30 个已完成任务上审计工具分布、计时缺失和超时，再决定是否扩大。不要把相邻调用随机拆到训练与测试集，也不要将失败任务补换成成功任务。

LiveCodeBench 沿用原有题组和 split，仅从每个 split 选取子集。当前可用未暴露 LCB 池主要是 AtCoder easy/medium，不宣称覆盖所有编程任务。检索题按完整题面哈希分组，排除本地可识别的历史开发题；这不是预训练去污染保证。

每个 split 内总任务数严格 1:1:1，队列顺序为 Hotpot→BrowseComp→LCB 循环，空槽立即补位。最多同时运行 4 个任务，所有 benchmark 都可以占用任一槽；每个任务内部工具保持原适配器的串行语义。不同任务长短不同，瞬时运行比例与成功完成比例不保证严格相等。

所有任务配置与名单已在 `configs/mixed_c4_v1/`。已有的 `configs/c4/` 试运行配置没有被本次准备覆盖。`selection_summary.json` 记录选题，`preparation_validation.json` 记录全部离线冻结结果。

## 模型与预算

- 本机已检查到 6 张 24GB RTX 4090。新服务默认使用编号 0,1,2,3 的四张卡，TP=4，BF16，GPU memory utilization=0.90，max-num-seqs=4。
- 模型：现有本地 `models/Qwen3.5-9B`；复用 `.venv-vllm`，不下载或升级依赖。
- 最大上下文：262144 tokens，等于本地模型配置的原生上限，不外推位置编码。
- 每轮最大生成：32768 tokens（包含 reasoning 的服务端生成预算）；上下文长度与单次生成长度不是同一参数。
- 客户端显式声明 262144 上下文。输入与生成仍需共同满足服务端窗口，长上下文不保证任务必然完成。
- 每题最多 80 次 agent iteration、160 次环境工具调用、100 次 LLM 请求；任务最多 14400 秒，单次 LLM 最多 3600 秒，自动重试关闭。
- 远端检索工具预算 30 秒，LCB 工具上限 120 秒（具体调用可要求更短）。预算到达将被记录，不伪装为正常完成。
- 此 TP4/C4 配置尚未进行实际加载验证。服务启动时查看 `GPU KV cache size` 与 `Maximum concurrency`，不能从“max-num-seqs=4”推导四条满 262144 上下文一定同时驻留。

可先仅打印命令，不加载模型：

```bash
cd /root/flowpilot_predictor
bash scripts/serve_qwen_c4.sh --print
```

## 1. 当前服务器：把远端启动包复制过去

本次已只读验证当前容器可通过项目已有 SSH 配置直连目标容器。该配置使用现有密钥与 known_hosts，不需要修改用户工作站上的 ProxyJump 配置。若从工作站登录，仍可使用用户已有的 WHU_4090_docker / WHU_4090_6 别名。

```bash
cd /root/flowpilot_predictor
scp -F configs/ssh/whu4090.conf configs/mixed_c4_v1/remote.tar.gz WHU_4090_6:/home/qiulin/flowpilot_predictor/mixed_c4_v1_remote.tar.gz
ssh -F configs/ssh/whu4090.conf WHU_4090_6
```

包内只有计时和启动脚本，没有数据集、模型、密钥。无需再次复制固定语料库。

## 2. 远端服务器：解压并开启语料服务

SSH 登录目标已经是现有远端容器环境；检查时该环境没有可用 Docker daemon。以下在 SSH 进入后的 shell 中执行，不需要猜测 docker 容器名称或再执行 docker run。

```bash
cd /home/qiulin/flowpilot_predictor
mkdir -p runtime/mixed_c4_v1
tar -xzf mixed_c4_v1_remote.tar.gz -C runtime/mixed_c4_v1 --strip-components=1
python3 - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('runtime/mixed_c4_v1')
for name, expected in json.loads((p / 'sha256.json').read_text()).items():
    assert hashlib.sha256((p / name).read_bytes()).hexdigest() == expected, name
print('remote bundle hashes OK')
PY
mkdir -p logs/mixed_c4_v1
ss -ltn | grep -E ':8123|:8124' || true
```

如果端口被旧服务占用，先确认进程归属并手动处理，不要并排启动另一份索引服务。以下命令确实会开启服务，只在准备开始采集时执行：

```bash
tmux new-session -d -s fp-corpus -n hotpot -c /home/qiulin/flowpilot_predictor 'bash -lc "set -o pipefail; bash runtime/mixed_c4_v1/serve_corpus_c4.sh hotpot 2>&1 | tee logs/mixed_c4_v1/hotpot.log"'
tmux new-window -t fp-corpus -n browsecomp -c /home/qiulin/flowpilot_predictor 'bash -lc "set -o pipefail; bash runtime/mixed_c4_v1/serve_corpus_c4.sh browsecomp 2>&1 | tee logs/mixed_c4_v1/browsecomp.log"'
tmux attach -t fp-corpus
```

看到两个服务 ready 后，按 `Ctrl-b d` 脱离。Hotpot 在 8124，带计时的原生 BrowseComp MCP 在 8123。它们复用现有虚拟环境、SQLite/Lucene 索引与原生检索函数。Hotpot 仍保留原有服务端串行锁，不能把其排队时间归入纯计算。

## 3. 当前服务器：两个 tmux 会话

先从远端退出回到当前服务器，然后执行以下命令。这些命令也尚未由本次准备执行。

```bash
cd /root/flowpilot_predictor
mkdir -p logs/mixed_c4_v1
nvidia-smi
tmux new-session -d -s fp-llm -n server -c /root/flowpilot_predictor 'bash -lc "set -o pipefail; bash scripts/serve_qwen_c4.sh 2>&1 | tee logs/mixed_c4_v1/llm.log"'
tmux attach -t fp-llm
```

确认服务已 ready，按 `Ctrl-b d` 脱离。若编号 0–3 被其他任务使用，可在启动命令前设置 `CUDA_VISIBLE_DEVICES` 为四张获准使用的卡；不得因当前一次空闲检查而抢占以后出现的任务。

第二个会话同时包含隧道窗口和采集窗口：

```bash
cd /root/flowpilot_predictor
tmux new-session -d -s fp-collect -n tunnel -c /root/flowpilot_predictor 'bash scripts/open_corpus_tunnel.sh'
tmux new-window -t fp-collect -n tasks -c /root/flowpilot_predictor
tmux attach -t fp-collect:tasks
```

隧道映射本机 18123→远端 8123，18124→远端 8124。保持该窗口运行。已有隧道占用端口时，先确认可用性，不重复开启。

在 tasks 窗口中先检查服务，然后启动训练集采集：

```bash
cd /root/flowpilot_predictor
bash scripts/collect_mixed_c4.sh check
set -o pipefail
bash scripts/collect_mixed_c4.sh run --split fit 2>&1 | tee logs/mixed_c4_v1/fit.log
```

`check` 只查询模型信息，并对每个检索服务发起一个固定的计时探测查询，不生成 LLM 输出；这会产生轻微索引预热，不进入训练数据。`run` 会再检查服务，执行适配器已有的隔离/参考检查，然后启动真正的任务采集。检测不到 BrowseComp 内部计时会报错，而不是静默采集缺失标签。

fit 是每类 100 题，总共 300 题。fit 完成并检查 `collection_summary.json`、`tool_timing_dataset/audit.json` 后，再按需顺序采集：

```bash
bash scripts/collect_mixed_c4.sh run --split tune 2>&1 | tee logs/mixed_c4_v1/tune.log
bash scripts/collect_mixed_c4.sh run --split calibration 2>&1 | tee logs/mixed_c4_v1/calibration.log
bash scripts/collect_mixed_c4.sh run --split test 2>&1 | tee logs/mixed_c4_v1/test.log
```

如果已启用 `scripts/auto_continue_mixed_c4.py arm`，无需手动输入上面三个命令。watcher 在 `fp-collect:auto` 窗口等待 fit 成功结束、完整收据和 `tool_timing_dataset/audit.json` 审计通过，且 tasks 窗口回到 shell 后，自动在原 tasks 窗口依次运行 tune、calibration、test。任何一步失败都会停止；每个分区日志在 `logs/mixed_c4_v1/<split>.log`，状态用 `python3 scripts/auto_continue_mixed_c4.py status` 查看。当前这次采集已启用 watcher，勿再手动运行后续分区。

不要在多个窗口同时启动这些 split，否则总并发会超过 4。测试集先封存，不能根据 test 的预测误差再调模型。中断后的 run 不会静默重跑；保留原目录与失败记录，检查残留进程后再另行制定补采清单。

## 输出与计时定义

每个 split 的目录为 `runs/campaigns/mixed_c4_v1_<split>/`：

```text
campaign.json                    冻结后的交错队列、四槽配置、源码哈希
inputs/                          controller 私有输入（含代码题私有评估资料，不作为特征）
tasks/<job>/attempt-001/
    events.jsonl                 完整调用事件、参数、观察、耗时、错误
    blobs/                       完整 LLM 请求与回复
    prediction_samples.jsonl     单题特征/未来标签，二者必须分开使用
    result.json                  任务结束状态和预算结果
prediction_dataset/
    inputs.jsonl                 T0 因果输入；按现有 load_t0_input 协议使用
    targets.jsonl                实际下一工具、参数与耗时标签
    samples.jsonl                合并审计记录，不可直接整体喂给 T0 模型
tool_timing_dataset/
    tools.jsonl                  逐调用参数、观察、状态及分层计时
    audit.json                   标签缺失、执行状态和轨迹审计问题
```

时间都以毫秒记录：

| 字段 | 定义 |
|---|---|
| `executor_duration_ms` | 本地代码执行器的测量，或远端服务返回的内部 handler 测量 |
| `round_trip_ms` | 客户端从开始调用到形成工具 observation 的墙钟耗时，包含执行、远端等待、网络、协议及本地处理 |
| `queue_wait_ms` | Hotpot 服务端可观测的锁等待/入队处理时间；BrowseComp 未独立测到的队列时间为 null |
| `non_executor_overhead_ms` | RTT − executor；包括队列及协议开销，不是纯网络 RTT |
| `proposal_to_executor_ms` | 从适配器观察到工具动作提交至 executor 开始的客户端等待 |
| `environment_batch_span_ms` | 同一 LLM 回复的环境工具批次从首个开始到最后完成的跨度，沿用现有导出条件 |
| `normal_completion_right_censored` | 超时等预算截断的标记，不能将其正常完成时间当成已知 |

BrowseComp 的计时通过 MCP `_meta.flowpilot_timing` 返回，不修改原生工具 schema、排名、片段、全文和模型观察。其 `executor_timing_scope` 明确为 MCP handler（包含参数处理及结果编码），不包含请求抵达 handler 前的等待；它不是“纯 Lucene CPU 时间”。Hotpot 沿用现有 RPC 内部计时，包含原有 handler 的索引身份检查。服务器单调时钟只用于服务器内计算，客户端 RTT 在客户端计算，不相减两台机器的绝对时间戳。

失败/超时如果没有收到远端计时，内部时间保留 null，RTT 和失败状态照常记录。原有工具输出截断规则仍保留，代码执行 `outcome.truncated` 可用于识别；“完整轨迹”指完整记录调用与实际可见观察，不宣称恢复被执行器截断的字节。

对方案 B（实际调用已生成）可以使用本轮工具名称和参数；本轮 observation、耗时及结束状态只能做标签。对方案 A（LLM 提交时）这些本轮信息都不能作为特征。

## 后处理与后续扩采

任务正确性评分在采集之后单独执行，避免评分争用资源污染耗时：

```bash
bash scripts/collect_mixed_c4.sh evaluate --split fit
```

Hotpot 和 LCB 沿用适配器评分。BrowseComp 会准备官方 judge 输入，其官方 LLM judge 仍需要独立配置并运行，不应将 pending 当成已评分。

如需导出中断采集的已有轨迹，在停止采集并检查进程后执行（不覆盖既有 export 目录）：

```bash
bash scripts/collect_mixed_c4.sh export --split fit
```

重新选择数量时使用新 plan 路径，原计划默认拒绝覆盖：

```bash
source scripts/openhands_env.sh
python scripts/prepare_mixed_c4.py --output configs/mixed_c4_v2 --fit 150 --tune 20 --calibration 10 --test 20
```

同 seed、同 split 的扩展会包含 v1 已选任务。扩采前必须排除已采集任务或明确标记重复 episode，不能把重复轨迹当成新增独立任务。新模型、硬件、服务并发或工具预算产生的新分布应使用新 run/plan 名称。
