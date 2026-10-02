# FlowPilot 时间预测模块交接方案（2026-09-26 更新）

**交接结论：现有数据已足够启动工具耗时预测。第一阶段在工具类型和参数已知时，优先预测单次工具调用的客户端耗时（RTT），并保留工具内部执行时间作为辅助预测目标；“整批工具完成后下一次 LLM 请求何时就绪”是面向调度的扩展，不替代工具耗时预测。工具种类和参数预测后置。先实现统计基线、无泄漏数据构造和在线反馈，再验证分位数模型及调度收益。**

本次更新核对了四个分区的采集汇总、导出标签、任务分组和最终评测。更新范围是方案、审计脚本和证据快照；没有训练预测器，也没有启动采集、模型或远端工具服务。

信息层级约定：

- **已确认**：来自本地轨迹、配置、导出实现或已注明的论文。
- **合理推测**：数据提示的可能机制，尚未通过干预或留出集实验验证。
- **建议**：供实现采用的设计和初始阈值，不代表已经取得的效果。
- **未验证**：当前缺少标签、遥测、实现或实验的能力，不写成既有结果。

## 1. 范围和模块分工

T0 是 LLM 请求准备/提交阶段；T1 是完整 LLM 回复中的 Tool Call 已解析、参数闭合且可供执行的阶段。**T1 的类型和参数是已知输入，不是预测目标。** 本阶段不依赖 T0 猜工具、生成参数、语义缓存或整任务剩余轮次预测。

| 模块 | 交给预测器的事实 / 本模块责任 |
|---|---|
| Agent / Tool adapter | 实际调用列表、参数、工具 schema / 默认值、批次依赖、预算；发出真实开始、返回、失败、取消事件 |
| Tool Cache | 每个调用的真实 resolution、leader 标识、版本、结果有效性；负责命中与准入判断 |
| 时间预测器 | 当前状态下的就绪/返回时间分布、适用范围、回退原因；接收真实标签并更新 |
| KV 后端 | 可操作的状态、真实占用字节、迁移/恢复/重算成本、操作完成事件与能力标记 |
| 调度器 | 根据预测、真实依赖、KV 成本和 deadline 做 KEEP / OFFLOAD / RESTORE / 请求排序 |

预测器不负责证明缓存结果正确，不代替工具执行，不把概率预测变成完成事件。接入时沿用 [design.md](design.md) 的真实调用和 resolution 权威；其中 T0 forecast 是可选扩展，不能成为本轮实现的前置条件。

## 2. 当前数据、评测和可支持的结论

### 2.1 四个分区均已完成

数据根目录：`runs/campaigns/mixed_c4_v1_<split>/`。下表由新增的审计脚本核对；“工具行”包含控制/未执行动作，“RTT 标签”只含实际执行且返回可观测的调用，“ready 标签”按第 3 节严格规则筛选。

| 分区 | 任务数（每类） | task_group 数 | 工具行 | 有效 RTT 标签 | LLM 请求行 | 有效 ready 标签 |
|---|---:|---:|---:|---:|---:|---:|
| fit | 300（100） | 253 | 4560 | 4056 | 4375 | 3768 |
| tune | 60（20） | 52 | 781 | 694 | 755 | 645 |
| calibration | 30（10） | 28 | 474 | 424 | 456 | 401 |
| test | 60（20） | 52 | 766 | 674 | 716 | 613 |
| 合计 | 450（150） | 385 | 6581 | 5848 | 6302 | 5427 |

四个 `collection_summary.json.status` 均为 `collected`，`tool_timing_dataset/audit.json.trace_issues` 均为空；分区之间的 task_group 交集均为 0。输入/标签关联键和工具键没有重复、丢失关联。ready 标签覆盖的组数依次为 252 / 52 / 28 / 52，调用行数不等于独立任务数。

运行结果：439 completed、7 budget_exhausted、3 agent_error、1 llm_error。fit 对应 291 / 6 / 2 / 1；tune 为 58 completed、1 budget_exhausted、1 agent_error。**保留失败任务中合法的时间标签**，不能只用最终答对或运行完成的任务训练。

已确认的执行方式是：任务最大并发 4、单任务内部工具并发 1，空闲 worker 继续取任务。1:1:1 是总任务数比例，不保证任意时刻的在途组成均衡。fit 中 3983/4056 次工具执行与另一任务的 LLM 客户端请求区间重叠，说明已有跨任务并发；这不是 GPU kernel 并行或 GPU 饱和的证据。

### 2.2 配置与适用范围

当前轨迹对应 Qwen3.5-9B、vLLM `0.29.0+cu129`，BF16、TP=4，服务最大上下文 262144，单次最大输出 32768，actor 开启 thinking；每任务上限 80 iterations / 160 tool calls / 100 LLM requests / 14400 s。常规工具 timeout 为 30 s，代码工具为 120 s，实际还可能被剩余预算截断。

这些属于数据生成条件，不能默认迁移到其他模型、并发、网络或后端。后续模型工件应保存 actor / executor / corpus / tool schema / 配置版本和输入文件哈希。

### 2.3 最终评测已完成，但正确性协议仍有边界

使用 `runs/evaluations/mixed_c4_v1_qwen35_9b_json_v2/<split>/task_evaluation_summary.json`，四个汇总均为 `scored`。采集阶段写入的 `result.evaluation_status=pending` 是历史字段，不能据此认定尚未评测。

| 分区 | Hotpot answer EM | Hotpot answer F1 | Hotpot joint EM | BrowseComp 本地判对 | LiveCodeBench 通过 |
|---|---:|---:|---:|---:|---:|
| fit | 50.0% | 61.21% | 17.0% | 30/100 | 88/100 |
| tune | 50.0% | 53.11% | 25.0% | 9/20 | 15/20 |
| calibration | 30.0% | 36.67% | 0.0% | 1/10 | 8/10 |
| test | 55.0% | 74.83% | 10.0% | 9/20 | 19/20 |
| 全量加权 | 49.33% | 60.31% | 16.0% | 49/150（32.67%） | 130/150（86.67%） |

BrowseComp 的 JSON v2 修复了 `correct: yes/no` 缺失，150 题均有判定、judge_errors=0；**格式可解析不等于裁判语义可靠**。目前是本地 Qwen3.5-9B judge，已有要求精确短语却接受不完整答案的争议样例。建议在质量约束实验前人工复核全部 49 个阳性，并分层抽查至少 20 个阴性，单独版本化人工标签；不静默覆盖当前 v2。无需为了时间预测先下载 32B 模型。

LCB 当前 150 题全来自 AtCoder，75 easy + 75 medium、无 hard，日期 2023-06-17 至 2025-04-05；属于选定的 agent 工具交互子集。86.67% 不能当成完整 LiveCodeBench 官方集合的单次裸生成 pass@1。

时间预测不以任务正确性为训练标签。系统实验需要预先固定二元正确性口径：Hotpot 默认 answer EM=1；若要求证据完整则选 joint EM=1，不能事后切换；BrowseComp 用冻结版本的判定；LCB 用规定测试全部通过。当前尚未固定 arrival / deadline / SLO 协议，因此不能据现有准确率计算“正确且按时”的 goodput。

### 2.4 时间异质性：仅用 fit 作特征设计依据

下表单位 ms；executor 是现有工具内部计时范围，不保证等于纯算法 CPU 时间。

| 后端 / 工具 | 次数 | executor P50 | RTT P50 | RTT P95 |
|---|---:|---:|---:|---:|
| Hotpot / search | 820 | 510.194 | 527.293 | 5789.358 |
| Hotpot / read_document | 223 | 0.546 | 53.824 | 64.610 |
| BrowseComp / search | 1862 | 441.329 | 479.968 | 2627.296 |
| BrowseComp / get_document | 76 | 1.567 | 38.129 | 91.395 |
| LCB / code_terminal | 577 | 33.355 | 87.346 | 586.322 |
| LCB / code_file_editor | 498 | 65.410 | 118.100 | 122.507 |

- 搜索占 2682/4056 次；必须同时报告逐工具、宏平均和微平均，避免只优化样本最多的搜索。
- 232 个回复含多个实际执行的环境工具（182 个双调用、50 个三调用），只预测第一个工具不够。
- 3768 条有效 ready 间隔的 P50=364.431 ms、P95=3619.654 ms。
- code_terminal 有 3 次超时返回，RTT P99=6734.815 ms、最大值 120058.811 ms。其“返回时间”已观测，而无 timeout 的自然完成时间右删失。
- Hotpot/search 查询词数与 RTT 的 Spearman 相关系数为 0.709，BrowseComp/search 为 0.161。词数使用 Python `re.findall(r'\w+', query)`。**合理推测：**参数特征值得测，但不同 search 后端不能混为一种耗时机制；相关性不是因果或泛化证明。
- 文件编辑当前分布集中、get_document 样本少，统计分位数可能比复杂的逐工具模型更稳；不要为每个小类强行训练一套树。

| benchmark（fit） | 工具 RTT / 累计任务时间 | LLM transport / 累计任务时间 |
|---|---:|---:|
| HotpotQA | 6.57% | 93.27% |
| BrowseComp-Plus | 2.88% | 96.94% |
| LiveCodeBench | 0.61% | 96.97% |
| 全体 | 1.91% | 96.57% |

分母为任务时间求和，不是 campaign makespan；LLM transport 含服务排队、推理、传输和异常等待。只看 291 个 completed 任务，工具占比也仅约 2.52%。当前三类整体都以 LLM 请求等待为主，300 题只有 1 题工具占比超过 50%，**尚不支持“三种 benchmark 分别代表 tool-bound 和 LLM-bound”这一前提**。

固定语料只读工具有 288 次历史 exact-key 候选，全部在同一任务内；跨任务和在途同键候选均为 0。在固定轨迹、无限容量、命中读取/校验免费等理想假设下，可移除约 268.56 s，等于工具 RTT 的 8.25%、累计任务时间的 0.157%。这是候选复用估计，既不是实测命中率，也不是系统加速比；代码执行/编辑不能仅凭同参数复用。

**建议的 motivation：**利用阶段性就绪时间、上下文占用和真实复用状态，协调请求与 KV 调度，提高正确且满足 SLO 的任务数和端到端效率。工具占比低限制直接节时空间，但不能排除减少 KV 竞争对 LLM 部分的间接收益；该收益必须由实测 KV 能力与 oracle 对照验证。

## 3. 预测目标、时间端点和训练掩码

| 目标 | 锚点 → 终点 | 现有字段 | 优先级 |
|---|---|---|---|
| 单工具 RTT | 客户端执行器调用前计时 → 返回内容已取得/整理；具体边界见第 12 节 | `round_trip_ms` | **首版主预测目标：工具调用耗时** |
| 工具内部执行时间 | executor 自己的开始 → 结束，范围因后端而异 | `executor_duration_ms` | **辅助预测目标**；若研究纯后端执行成本，单独报告此目标 |
| 工具批次执行跨度 | 首个环境 tool_start → 最后 tool_end | `environment_batch_span_ms` | 诊断，不含全部 continuation 成本 |
| 下一请求就绪间隔 | 当前 llm_response → 下一 llm_request_prepared | `next_request_prepared_gap_ms` | 调度扩展标签，含工具之外的准备开销 |
| 当前剩余就绪时间 | 当前 as_of → 下一 llm_request_prepared | 从就绪标签端点与当前状态计算 | 整批调度接口的扩展输出 |

对调度扩展，定义 `Y_ready = t_next_prepared - t_llm_response`；运行中输出 `R(as_of) = t_next_prepared - t_as_of` 在“此刻尚未就绪及当前状态已知”条件下的分布。

这个 ready 是**客户端请求准备好**，不包括下一请求服务端排队/prefill/decode，也不代表 KV 已恢复。实际推理开始至少受 `max(T_request_ready, T_KV_ready, T_other_dependencies)` 约束，预测器只负责第一项。DCS / 网关机械生成 continuation 是另一条路径，接入后必须单独打点和版本化，不能直接挪用旧 ready 标签声称等价。

使用下列规则构造首版数据：

1. 单工具：`labels.executed=true`，参数可解析，结束状态可观测，RTT 有限非负且开始/结束事件齐全。当前各分区实际执行行均满足，数量见第 2 节。执行错误若有真实返回可用于“何时返回”，须保留 outcome；控制动作、未执行、缺失 END 都不能填成 0。
2. 整批 ready：`masks.environment_batch_duration=true` AND `masks.complete_arguments=true` AND `labels.has_unknown_tool_call=false` AND `labels.next_request_observed=true` AND gap 有限非负。首版计数为 3768 / 645 / 401 / 613。
3. 上述批次规则有意收紧到完整回复、参数完整、环境动作均已结束且下一请求存在的子集。应报告覆盖率，不能声称覆盖所有异常/终止分支；运行时超出范围走回退。
4. 不用 `first_round_trip_duration` 掩码筛整个工具列表，也不用“首个动作是工具”代替“本批含环境工具”。批次的已知调用数来自完整响应及工具 schema，不能用事后“成功执行个数”作特征。
5. 终止/失败后无下一请求，ready=null，不作 0 ms；真实终止直接取消预测。若未来要估计是否 continuation，另立目标，暂不强行给出就绪时间。

计时约束：

- RTT 已含 executor，不能相加；`RTT - executor` 是非 executor 开销，不是纯网络时间。`queue_wait_ms=null` 表示未测量。
- 控制器事件可在同主机同 boot 的 clock_domain 内排序；远端 monotonic 不可与控制器相减。工具 labels 中的 clock_domain 可能属于 executor，不能直接用于解释 controller tool_start/end。
- 串行批次中，后续工具自己的 RTT 不包含前面工具等待；批次需要完整关键路径。Q90 相加不等于总和的 Q90，未来并行也不能直接取各工具 Q90 的最大值冒充 join 分位数。
- 解析、缓存查询和预测计算本身有开销。现有离线 T1 以 llm_response 为逻辑锚点，没有单独的“解析完成”新鲜快照；必须披露这一近似。在线 as_of 在参数/状态准备好后记录，包含已过去时间；不能把锚点预测当作此刻剩余时间原样返回。

## 4. 交接接口（建议版本 3，尚未实现）

这里的 `predictor_schema_version=3` 是新接口版本，与现有 exporter 的 schema 版本无关。统一使用 ms；monotonic 时间戳用 ns，同一 clock_domain 才允许计算差值。以下数值仅为协议示例，不是训练输出。

### 4.1 输入：事实、可用时间与每个调用的真实路径

~~~json
{
  "predictor_schema_version": 3,
  "stage": "tool_call_ready",
  "batch_id": "attemptA:r1",
  "request_id": "r1",
  "context_epoch": 7,
  "resolution_epoch": 2,
  "clock_domain": "controller-monotonic:host:boot",
  "anchor_event": "llm_response",
  "anchor_monotonic_ns": 1000000000,
  "as_of_monotonic_ns": 1002000000,
  "elapsed_since_anchor_ms": 2.0,
  "continuation_path": "agent",
  "execution_mode": "serial",
  "calls": [
    {
      "tool_call_id": "c1",
      "backend_id": "hotpot_rpc:index_revision",
      "tool_schema_version": "search-v1",
      "name": "search",
      "arguments": {"query": "example search words"},
      "configured_timeout_ms": 30000,
      "resolution": "LOCAL_ONLY",
      "state": "PLANNED",
      "leader_id": null,
      "elapsed_running_ms": null
    }
  ],
  "history": {"recent_same_backend_tool_rtt_ms": [480, 620]},
  "load": {
    "tool_inflight": 1,
    "snapshot_monotonic_ns": 900000000,
    "snapshot_age_ms": 102.0,
    "source": "t0_snapshot"
  }
}
~~~

resolution 与 design.md 对齐，且必须**逐调用**提供：同一批次可能混合命中、执行和 follower。

| resolution | 使用的路径 | 当前训练支持 |
|---|---|---|
| HISTORICAL_HIT | 查找 / 校验 / 读取 / continuation | 未采集，不能拿 execute 模型冒充 |
| INFLIGHT_FOLLOWER | leader 的条件剩余时间 + follower 交付 / continuation | 未采集，需 leader 真实进度 |
| LOCAL_LEADER | 实际执行 + continuation | 当前 execute 路径可作为初始参考 |
| LOCAL_ONLY | 实际执行 + continuation | 当前实际执行轨迹 |
| UNKNOWN（接口扩展） | resolution 尚未确认 | 明确回退，不自动假定命中 |

归一化路径 execute / hit / follower 只用于模型分组，不能覆盖缓存模块的权威状态。未知或混合新路径不得伪造 0 ms；允许返回数值为空的 unsupported，由调度器采用无需预测的策略。

### 4.2 输出：剩余分布、来源和失效条件

~~~json
{
  "predictor_schema_version": 3,
  "prediction_id": "p1",
  "batch_id": "attemptA:r1",
  "target": "next_request_prepared",
  "clock_domain": "controller-monotonic:host:boot",
  "as_of_monotonic_ns": 1002000000,
  "remaining_ms": {"q10": 148, "q50": 548, "q90": 5998},
  "per_call": [
    {
      "tool_call_id": "c1",
      "target": "round_trip_from_dispatch",
      "duration_ms": {"q10": 140, "q50": 530, "q90": 5980}
    }
  ],
  "method": "empirical_conditional",
  "base_model_version": "empirical-v0",
  "online_state_version": 12,
  "observed_through_monotonic_ns": 990000000,
  "support": {
    "status": "supported",
    "reference_rows": 820,
    "reference_task_groups": 80
  },
  "fallback": {"used": false, "reason": null},
  "valid_for": {
    "request_id": "r1",
    "context_epoch": 7,
    "resolution_epoch": 2
  }
}
~~~

上面示例展示包含批次就绪预测的完整扩展接口。首版必需交付是 per_call 中的 RTT 分布、身份/时间/版本、support 与 fallback；内部执行时间可增加独立的 `executor_duration_ms` 分布。批次 `target=next_request_prepared` 与 `remaining_ms` 可以后续接入，未实现时应省略或标为 unsupported，不能编造数值。每种输出的端点必须明确，不能把尚未派发的第二个工具的 RTT 当成从 now 开始的剩余等待。

有数值的输出必须有限、非负且 Q10 ≤ Q50 ≤ Q90；unknown 用 null 和回退原因表示，拒绝 NaN/负数。Q10 / Q50 / Q90 是时间分位数，不是确定界或“90% 模型置信度”。支持量同时报告行数和任务组数；schema/backend/version/依赖模式不支持时显式标注，不输出未经验证的 accuracy probability。

调度侧验收预测须核对 request、context_epoch、resolution_epoch、clock_domain 和状态新鲜度。完成/失败/取消、调用列表改变、cache path 改变、上下文替换使旧预测失效；真实就绪事件始终优先。日志记录特征提取和预测延迟，不能用过期输出阻塞真实工具或调度推进。

### 4.3 最小程序接口

- `predict(context)`：校验输入，存档可用特征和版本，产生首次预测。
- `refresh(batch_state)`：未结束时依据已等待时间、已完成子调用和新 resolution 更新；无新真实标签时不训练。
- `observe(feedback)`：先计分后学习，去重并更新在线统计。
- `invalidate(batch_id, reason)`：终止或版本变更后清理 pending，拒绝晚到旧结果。

deadline、KV 字节数和迁移成本是调度器的事实输入，不强制传给首版时间模型。它们可以影响调度策略，但不应混成“工具本身需要多长时间”的标签。

## 5. 数据构造与特征可用性

### 5.1 实际文件与关联键

| 文件 | 用途 |
|---|---|
| `prediction_dataset/inputs.jsonl` | T0 历史、配置和负载快照；**不是现成的 T1 样本** |
| `prediction_dataset/targets.jsonl` | 完整响应中的 actions、批次/ready 标签与 masks |
| `tool_timing_dataset/tools.jsonl` | 单工具参数、执行标签、开始/结束时间 |
| `campaign.json`、`collection_summary.json` | 原始任务清单、attempt 路径与运行结局 |
| 各 attempt 的 `events.jsonl`、`blobs/` | 回查真实事件时序和完整调用；构造在线回放 |
| 最终评测目录的汇总/逐题结果 | 质量分层与后续 goodput；不作为耗时预测输入 |

批次关联键为 `(source_attempt, request_id)`，工具关联键增加 `tool_call_id`；`task_group_id` 用于切分和分组 bootstrap。保留 request_id 与 logical_request_id 的区别，不把重试覆盖成一条记录。

工具行中未来信息很多，**禁止把整行 JSON 直接喂给模型**。T1 导出器应显式建立特征白名单，并为状态保存 observed_at / 来源 / age。

### 5.2 第一轮白名单

| 类别 | 可用特征 | 注意事项 |
|---|---|---|
| 身份与配置 | backend / tool / schema / corpus 版本，configured timeout，串行模式，响应中调用顺序/数量 | 两种 search 必须区分后端；跨版本默认回退 |
| 搜索参数 | 字符数、词数、去重词数、显式 top_k | 不凭空填未声明参数；默认值只能来自冻结 schema/config |
| 文档读取 | 请求长度/行数限制、已知文档大小 | 大小只有在调用前可查时才合法，不用返回字节数 |
| 代码与编辑 | 操作/命令类别、命令或代码长度、管道/重定向/多命令数量 | 文件状态依赖不能只靠字符串解释；首版不推理程序复杂度 |
| 历史 | 此刻以前已结束的同 backend/tool RTT、EWMA、已知失败率 | 历史回放必须先预测再获得当前标签 |
| 状态 | tool_inflight / 队列 / 负载、已等待时间、快照 age 和缺失标记 | 当前现成快照主要是 T0；远端队列没有记录就不要补造 |

关键泄漏点：`effective_arguments` 和 `effective_timeout_s` 主要在 tool_start 时记录。固定默认值可在 T1 从 schema/config 重算；但串行批次后续调用的实际 timeout 可能被那时的剩余预算截断，**不能把未来实际 effective timeout 原样搬回 T1**。首版使用配置上限、T1 已知预算及可用性标记；若在真正 dispatch 再预测，可另设 dispatch-stage 特征。

不得作特征的字段包括当前/未来的 executor、RTT、queue_wait、model_observation、输出长度、timed_out、exit_code、执行成功数、proposal_to_executor_ms、next_request_gap，以及当前任务最终正确性。ID、task_group、research_split 只作关联/评估。T1 可用完整回复，但第一轮无需全文 embedding。

用后续事件重建 T1 在途数是可选增强：只回放截至 as_of 的 START/END；先证明时钟同域和生命周期完整。缺失状态采用 unknown + age，不因离线看到未来就补齐。

## 6. 候选算法和文献定位

**建议顺序：经验分布 / EWMA → 参数特征分位数 GBDT → 在线校正 → 必要时后台重训。** 当前 fit 只有 253 个任务组，别把 4056 次调用当成同等数量独立任务；不建议先微调 LLM。

| 方法 | 必须回答的问题 |
|---|---|
| backend + tool / batch 类别的中位数及经验分位数 | 最便宜的静态基线是否足够 |
| 在线 EWMA + 过去残差/经验分布 | 低成本更新是否已经能适应变化；EWMA 本身不提供分位数 |
| 参数 TF-IDF + KMeans + 簇内经验分布 | 与最贴近的参数聚类方案比较，且支持条件剩余时间 |
| Quantile Regression Forest（QRF） | 分布建模的质量和成本是否优于浅树分位数头 |
| LightGBM quantile | 推荐主候选：工具 + 参数 + 合法历史/状态是否改善尾部预测 |

优先拟合单工具 RTT；内部执行时间若建模，使用独立标签/预测头。整批 ready 另行拟合，作为调度扩展，不要求先完成它才能交付工具耗时模型。批次模型直接使用完整已知调用列表、顺序、数量及参数聚合，避免相加分位数；若以后做单工具预测堆叠，训练阶段必须用按 task_group 划分的 out-of-fold 预测。

GBDT 可从 100–300 棵浅树、较大叶节点最小样本数起步，由 tune 选择；Q10 / Q50 / Q90 分别训练。先比较原始 ms 与 log1p 标签，所有指标还原 ms；分位数交叉须单调整理，再重新测覆盖。小类用共享模型或上级分组回退，不为 76 次 get_document 单独宣称精确尾部分布。LightGBM 支持 quantile / alpha，见[官方参数文档](https://lightgbm.readthedocs.io/en/stable/Parameters.html)。

**三个分位数不足以唯一确定完整分布。** GBDT 三头先用于 T1 锚点预测质量对照；运行中 refresh 先采用完整的经验条件分布，并在输出注明 method。若要同一 GBDT 分布支持任意 elapsed 条件化，需增加分位数网格或经验证的残差 CDF，另测 CPU 成本/覆盖；不能从三个数字凭空推导生存函数。在线 as_of 已晚于锚点时也必须处理这段 elapsed，尚未实现条件化的候选不能直接充当剩余时间服务。

| 参考 | 可借鉴的具体内容 / 不能推出的结论 |
|---|---|
| [Tokencake（2025）](https://arxiv.org/html/2510.18586v2) | 静态初值、EWMA、真实调用事件与 KV 卸载/预恢复结合；§7.1 使用模拟外部函数调用及延迟，不能据此断言真实工具同样可预测 |
| [CacheWise（2026）](https://arxiv.org/html/2606.16824v1) | TF-IDF/KMeans 参数分组、历史时延分布与已等待时间条件化；是直接基线，说明“参数预测工具时间用于 KV”已有先例，引用版本为 arXiv |
| [Tempo（2025）](https://arxiv.org/html/2504.20068v1) | QRF 和不确定性感知调度；目标是 LLM 输出长度，不是工具时间 |
| [ThunderAgent（2026）](https://arxiv.org/html/2602.13692v3) | 在难预测工具等待下用状态/成本和时间衰减决策，作为不依赖精确预测的系统对照 |
| [Cortex，NSDI 2026](https://www.usenix.org/system/files/nsdi26-ruan-cortex.pdf) | 工具远端数据访问/缓存路径的参考；不是本轮选耗时回归器的主要依据 |
| [CQR，NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html) | 分位数区间校准参考；当前调用相关、任务组少且在线漂移，不能直接声称严格逐调用覆盖 |

直接换回归器不构成强系统贡献。差异化应由实验证明：真实执行/hit/follower 路径条件化、延迟反馈下的在线适应、恢复迟到与提前占用的非对称成本、真实混合负载中的质量约束收益。复用基线并注明来源是规范做法，不能把已有方法换名当创新。

## 7. 调度使用规则与必要的 KV 前提

- Q50 用于预计就绪顺序。
- 较低分位数（例如 Q10）用于评估“足够长的空闲窗口”：大于**实测**卸载+恢复成本及余量，同时存在内存压力和其他可运行请求，才可能值得卸载。这不是确定安全界。
- 较高分位数（例如 Q90）用于当前阶段尾部风险；它不等于整个任务剩余时间。
- 预恢复若希望减少迟到，应参考较早就绪分位数减去恢复提前量；按 Q90 才开始“保险恢复”反而容易迟到。过早恢复的显存 byte-seconds 也要计成本。
- 实际提前完成即触发推进/纠偏；预测不能让已经 ready 的任务继续等。

已经等了 e 且仍未就绪，不能长期使用 `max(原 Q50 - e, 0)`。设 S 为总等待时间的条件生存函数：

`P(R > u | Y > e, x) = S(e + u | x) / S(e | x)`。

经验实现是在可比分组历史总时长中保留 `Y > e`，计算 `Y-e` 分位数；有新进度还要更新分组/未完成调用集合。无足够生存样本时回退上级分布或 unsupported，不能输出“马上完成”。新路径、新依赖和混合 cache resolution 不应套旧串行模型。

“SLO 减去卸载和重载时间”不是剩余 slack。调度器应区分 deadline-now、工具等待窗口、后续推理/依赖需求和 KV 迁移成本。只预测当前工具时间不能保证整个任务按时完成。

当前 Qwen3.5-9B 配置是混合注意力（24 层 linear_attention、8 层 full_attention）。不能直接按所有层都是传统全注意力的 KV 公式定价；需要后端实际报告 KV / 其他推理状态占用及可迁移范围。vLLM 有 prefix caching 不代表已经具备本项目所需的逐会话 pin / offload / restore 控制。

**未验证的系统前提：**现有 C4 是否有足够 KV 压力、后端是否支持目标操作、迁移带宽/排队成本、改变策略后的真实收益。它们不阻止先训练时间基线，但阻止提前宣称 KV 调度方案有效。

## 8. 最小实验与验收顺序

### 8.1 固定数据协议

fit 训练，tune 选特征/超参数，calibration 做覆盖/阈值校准，test 做冻结后的最终报告。首轮基础模型只用 fit；如要合并 fit+tune 重训，必须先冻结规则并重新建立校准状态。词表、聚类、标准化都不能在四个分区合并拟合。

本轮跨分区只做清单、完整性、标签覆盖和已有任务评测核对；时间分布/参数选择仍以 fit 为依据。但 test 的任务正确率和部分答案在前序质量分析中已经查看，不能称为“研究全程完全未见的盲测集”。不要再根据 test 误差反复选模型；若后续迭代利用它，应另留新的 untouched holdout。这里的 test 是项目内部切分，不等同官方 benchmark 隐藏测试。

### 8.2 按成本递进，不一次实现全部候选

| 阶段 | 交付 | 继续条件 |
|---|---|---|
| P0：数据与契约 | T1 特征白名单、两个标签集、分组/掩码审计、predict/observe 协议 | 重现第 2 节计数，特征无未来字段；控制/缺失标签不补零 |
| P1：低成本可行性 | 静态经验分布、在线 EWMA、冻结 GBDT；工具身份 → 加参数 → 加历史/状态消融 | tune 上超越强基线，或明确保留基线；不因已有复杂模型就强行部署 |
| P2：在线适应 | 按事件时间回放、残差校正、状态 refresh、漂移/长尾统计 | 优于冻结模式且无不可接受的尾部/开销退化 |
| P3：调度收益 | 无预测 / EWMA / 候选 / oracle，相同资源与正确性协议 | oracle 存在实际机会，候选能覆盖预测与迁移成本 |
| P4：扩展场景 | 真实新负载、cache hit/follower、网关 ready、必要的周期重训 | 前面瓶颈明确后补采，不为完整架构而一次填满所有模块 |

P3 的 oracle / 后端能力检查可在 P1 后尽早并行开展，避免模型迭代到最后才发现没有调度空间；这里不要求另启动服务或实验。

建议首轮报告：

- 时间误差：优先报告 RTT，内部时间单独报告；实现 ready 扩展后再报告该目标，逐工具/benchmark 的 Q50 MAE、Q10/Q50/Q90 pinball loss，Q10/Q90 单侧经验覆盖、中央 80% 区间覆盖/宽度；RTT 报告工具宏平均，ready 报告三个 benchmark 等权宏平均及 batch signature 分层，另报逐行微平均；按 task_group bootstrap。多工具批次不重复算作多条 ready 样本。
- 决策误差：同一时刻共同待调度线路的就绪排序错误，窗口误判超过 KV 迁移阈值的比例，恢复迟到 ms，提前占用 byte-seconds。全局给从不同时间来的所有调用排名没有调度意义。
- 运行成本：特征提取+预测 CPU P50/P95、反馈更新成本、内存/pending 队列、回退比例；不把模型函数调用时间当全部热路径成本。
- 质量分层：任务完成/失败、最终正确/错误分层报告时间误差；不据此删除失败/错误任务的有效工具标签。

**建议的工程起点，尚未实测：**进程内特征+预测 P95 预算先设 1 ms；超限则采用廉价基线并记录。首版复杂模型晋级可先设为 tune 上 RTT 的工具宏平均 pinball 相对最佳静态/EWMA 改善至少 10%，且关键工具覆盖、调度风险和成本无明显退化；门槛在看 test 前冻结。若不满足，交付经验/EWMA 基线也属于有效结论。

固定轨迹重放只能评估机会与决策差异，不能假定改变并发/缓存/KV 后工具队列仍与旧 trace 相同。当前 C4 也不证明 C8/C16 泛化。若 oracle 几乎无收益，先换有真实长工具/上下文压力的目标负载或降低系统复杂度；不要优先换更大的预测网络。人工 sleep/缩放延迟只作为注明注入的敏感性测试。

### 8.3 系统主指标与 SLO

对任务 i，A_i 是提交到系统的到达时刻，F_i 是最终完成时刻，D_i 是预先规定的相对 deadline，C_i 是冻结协议的正确性：

`qualified_count = sum_i 1[C_i = correct AND F_i - A_i <= D_i]`

`quality_slo_goodput = qualified_count / observation_window_seconds`

同时报告所有提交任务的 qualified fraction、拒绝/失败/超时数、各 benchmark 正确率、E2E P50/P95 和 makespan；分母不能只留下被接纳/完成的任务。明确 observation window，SLO 包含排队/admission/工具/KV 等实际等待；现有 result.duration_s 不自动等于完整 arrival-to-finish。

SLO 档位可在 fit/tune 参考运行上制定后冻结，不能用每道 test 题事后真实耗时反推它的 deadline。缓存和调度改变执行轨迹后，应重新评测最终答案，不能沿用原任务的正确性标签当新系统质量。

## 9. 实现交付清单与当前状态

下表中的模块路径是**建议新增，尚未实现**，不能当成已有可运行训练命令。

| 建议模块 / 工件 | 最小职责 |
|---|---|
| `predictor/data.py` | 多文件关联、明确 masks、任务组切分、事件顺序与标签可用时间 |
| `predictor/features.py` | T1 / dispatch 白名单，schema 默认值、版本/缺失/快照 age |
| `predictor/baselines.py` | 分组经验分布、EWMA、条件剩余分布、稀疏/未知回退 |
| `predictor/quantile.py` | 单工具/整批 GBDT，分位数单调修正，可选 QRF 对照 |
| `predictor/online.py` | pending、延迟反馈、先计分后更新、残差缓冲和模型版本 |
| `predictor/service.py` | predict / refresh / observe / invalidate，非阻塞调度接入 |
| `scripts/train_time_predictor.py` | 固定 split、配置/seed、模型工件与离线指标 |
| `scripts/replay_time_predictor.py` | 合并并发事件流、冻结/在线对照、预测/更新耗时报告 |
| 模型 manifest | 特征/目标/单位、适用 backend/schema、数据哈希、split/seed、校准状态、回退规则、依赖版本 |

实现时先完成 P0/P1；在线状态层可以独立于 GBDT 开发。必须覆盖的行为验证是：时间泄漏、重复/晚到反馈、clock/domain 或版本不匹配、超时/缺失标签、分位数交叉、存活样本不足、混合 cache path 和真实事件覆盖预测。这些关系到标签正确性与调度行为，不用为每个纯字段复制编写冗余测试。

| 已完成 | 未完成 / 不能据此宣称完成 |
|---|---|
| 四分区采集、工具计时导出、最终任务评测汇总、当前清单与哈希审计 | T1 白名单训练集、已训练预测模型、在线预测服务 |
| fit 工具/ready/复用/并发统计 | hit/follower 时延、T1 新鲜远端队列、KV 成本和控制能力验证 |
| 方案和接口草案 | 调度接入、SLO 设定与质量约束收益、BrowseComp judge 人工复核 |

当前模型/采集服务已按前序清理要求停止；离线读文件即可开展 P0/P1。本次不重启它们。

## 10. 在线更新：从第一阶段就支持

### 10.1 区分状态、统计和基础模型更新

1. **状态更新**：调用还没结束，elapsed 增大、子调用完成或 resolution 改变。更新剩余时间；没有完整标签时不能拿自己的预测当真值。
2. **在线统计更新**：工具真实返回或下一请求真正准备好，先计分，再更新经验分布/EWMA/校正状态，使之后预测受益。
3. **基础模型更新**：积累足够新数据后后台重训，经后续 shadow 验证再原子切换。逐条学习不等于每次完成都训练 GBDT。

优先实现前两项。维护有界 pending 和反馈队列，不占用推理 GPU；队列满时记录丢弃/退化信息，不阻塞真实工具完成或调度推进。

### 10.2 快速校正及稀疏回退

按 `(target, backend/version, tool或batch类别, execution/cache path, base_model_version)` 维护历史。RTT 和 ready、execute/hit/follower、不同基础模型的残差不能混用；先用粗分组。

一个可测试的方案：对分位数 τ，记录历史真实标签相对当时基础预测的残差
`r_τ = log1p(y_ms) - log1p(q_base,τ_ms)`，用过去残差的经验 τ 分位数校正新预测，再转回 ms；少样本向 0/上级组收缩。它是经验方法，不承诺单调变好或分布无关覆盖。单调整理之后重新测覆盖。基础模型若预测锚点总时长，残差必须配对该总时长和当时存档的未条件化基础预测；refresh 的剩余时间误差单独计分，不能混入总时长残差缓冲。

初始缓冲可取每组 256–512 条、少于 32 条不独立校正；均匀滑窗与时间衰减在 tune 比较。以上是实验起点。长任务反复调用会支配样本，应同时看按任务分层指标，必要时在 tune 检验组平衡权重。

等待中的调用采用第 7 节条件剩余分布，而非把残差校正误当成 survival 更新。短调用先返回会造成暂时的反馈偏差，需保留 pending 数量和年龄；没有返回不等于 0 ms。

### 10.3 反馈协议及计分顺序

首版以每次工具返回作为 RTT / 内部执行时间反馈；下例展示 ready 扩展的反馈，不能要求工具预测器等待下一次 LLM 请求才更新已经观测到的单工具标签。

~~~json
{
  "predictor_schema_version": 3,
  "event_id": "attemptA:ready:101",
  "batch_id": "attemptA:r1",
  "request_id": "r1",
  "context_epoch": 7,
  "resolution_epoch": 2,
  "target": "next_request_prepared",
  "clock_domain": "controller-monotonic:host:boot",
  "anchor_monotonic_ns": 1000000000,
  "event_observed_monotonic_ns": 1612000000,
  "duration_ms": 612.0,
  "duration_from": "llm_response",
  "label_kind": "observed_ready",
  "prediction_ids": ["p1"]
}
~~~

例中 p1 在 1002000000 ns 产生，故其真实 remaining 标签为 610 ms，而总间隔标签为 612 ms。**按 prediction 的 as_of 对齐，不能将总时长误作剩余时长，也不能重复减 elapsed。**

单工具反馈以 controller tool_start 为 anchor，工具 END/ERROR 的真实可观测时刻为终点；ready 反馈必须等下一 llm_request_prepared，不能用最后 tool_end 擅自替代。无 ready 的终止事件仅 invalidate，不生成 ready=0。

处理顺序：校验 domain / identity / version → event_id 去重 → 对存档预测计分 → 更新统计/校正 → 释放对应 pending。同一调用多次 refresh 可分别评价对应时刻的 remaining，但一次终点只计一条完整时长训练样本，不把慢调用刷新 50 次当 50 个独立训练样本。

若真实事件早于某个晚到预测的 as_of，该预测应作无效/迟到记录，不生成负时长训练样本。反馈重放必须按事件可观测时间合并各任务，在同时间戳用源事件 seq 等确定顺序；不能按 task 文件逐一跑完从而提前见到未来标签。

### 10.4 超时、选择偏差与新路径

- 有超时 observation 的实际 RTT 是“到返回”的完整标签；假设无超时的自然执行完成时间才右删失。缺失 END 不作精确标签；取消事件按具体目标单独处理。
- 缓存命中的短耗时只进入 hit 模型。命中后没有实际执行，反事实 miss 时间不可观测，不能用旧预测补造 execute 标签。
- follower 的 leader 状态必须是当时可见的；还需 follower 的结果交付成本，不能把 leader 的总耗时重新算一遍。
- 并发、语料、工具、缓存和调度策略变化会改变分布；记录版本，区分工具成本变化与策略选择偏差。当前旧轨迹不能验证尚未启用路径的在线学习效果。

### 10.5 重训与在线评估

建议慢层触发条件从“新增至少 500 条有效标签且距上次训练至少 10 分钟”起步，在 tune 固定。新模型只用当时已到达反馈，限制 CPU/内存，近期数据混合历史锚点；在之后的新反馈上与旧模型同时做 shadow 预测，达到冻结门槛再切换，否则保留旧版。切换时重建或冷启动校正状态，不直接沿用旧模型残差。

同时报告冻结模型、EWMA、GBDT+在线校正，收益明确后再比较加后台重训。可选 [River ARF](https://riverml.xyz/latest/api/forest/ARFRegressor/) / [ADWIN](https://riverml.xyz/latest/api/drift/ADWIN/) 检验结构更新/漂移告警，但树间预测离散度不自动等于时延分位数。[Gibbs 与 Candès，JMLR 2024](https://www.jmlr.org/papers/v25/22-1218.html) 可作在线区间校正参考，延迟标签、任务相关和策略偏差需另行验证。

评估必须遵循 predict → 等真实反馈 → score → update；初始状态、事件排序、缓冲/触发规则在进入 test 前冻结。在线 test 可使用已到达的过去 test 标签，不能用未来反馈；同时保留完全冻结对照，避免把适应收益与离线拟合混为一谈。每个评估流明确重置状态；若用 calibration 初始化 tune 的回放，这只属于部署模拟，不能冒充历史时间顺序无泄漏的回放，首轮避免如此设置。

现有 calibration 只有 28 个 ready 任务组，即使有 401 次调用也不足以可靠细分许多工具簇，更不能声称严格在线 SLO 保证。先报告经验覆盖和任务组不确定性；确认系统需要更精细校准后再补采。

## 11. 可复现核查与证据

现在可以运行的核查命令如下；只读取现有数据，不启动模型/采集/训练：

~~~bash
cd /root/flowpilot_predictor
python3 scripts/audit_predictor_handoff.py
~~~

如需重新生成本次清单和源文件 SHA256：

~~~bash
python3 scripts/audit_predictor_handoff.py \
  --output evidence/predictor_handoff_20260926/collection_snapshot.json
~~~

主要材料：

- [collection_snapshot.json](evidence/predictor_handoff_20260926/collection_snapshot.json)：四分区清单、掩码计数、task_group 交集、最终评测、LCB 组成和源文件哈希。
- [audit_predictor_handoff.py](scripts/audit_predictor_handoff.py)：上述核查脚本；没有训练逻辑。
- [fit_analysis.json](evidence/predictor_handoff_20260926/fit_analysis.json)、[fit_batch_analysis.json](evidence/predictor_handoff_20260926/fit_batch_analysis.json)：耗时与批次 ready 证据。
- [fit_argument_analysis.json](evidence/predictor_handoff_20260926/fit_argument_analysis.json)、[fit_outcome_analysis.json](evidence/predictor_handoff_20260926/fit_outcome_analysis.json)：参数关联与运行结果分层。
- [workload_reuse_analysis.json](evidence/predictor_handoff_20260926/workload_reuse_analysis.json)、[reuse_outcome_analysis.json](evidence/predictor_handoff_20260926/reuse_outcome_analysis.json)、[concurrency_analysis.json](evidence/predictor_handoff_20260926/concurrency_analysis.json)：复用机会、集中度与跨任务重叠。
- [prediction_export.py](repos/Openhands-software-agent-sdk/benchmarks/flowpilot/src/benchmark_adapters/prediction_export.py)：现有 masks 和时间标签的实际定义。
- [WORKLOAD_REUSE_MOTIVATION_20260926.md](WORKLOAD_REUSE_MOTIVATION_20260926.md)：fit 的历史成本/复用分析；其中“评测 pending”等状态描述已过期，当前状态以本文和 JSON v2 最终评测为准。

交接时应先统一第 3–5 节的时间端点、字段可用性和协议，再分工实现第 9 节；模型精度、在线适应和调度收益分别验收，避免把一个阶段的成功当成整个系统已经成立。


## 12. 交接补充：预测什么，以及时间在哪里采集

### 12.1 当前仍然是工具耗时预测

上一版将“下一请求就绪时间”放在首位，容易让人误以为目标改成了预测 LLM 时间。这里明确实现顺序：

1. **先预测单工具耗时。** 输入是实际工具名称、参数、后端和当时可用状态；主输出为 `tool_round_trip_ms` 的分位数。若“执行时间”严格指后端内部，则对应 `executor_duration_ms`，另设辅助输出，不能将二者混称。
2. **调度优先用 RTT。** Agent 必须等结果返回才能继续，传输、服务等待和包装开销同样占用这一段等待窗口；只预测工具内部时间会漏掉它们。当前 LCB 是本机进程工具，也有进程启动/包装成本，不是所有 RTT 都是远端网络往返。
3. **再扩展批次/ready。** 一个回复有多个工具，或工具返回后还需整理上下文时，单工具耗时不能直接等同于下一次 LLM 请求就绪时间。该扩展服务调度，既不是 LLM 推理时间，也不是整任务完成时间。

例如：工具内部执行 800 ms，调用的其他开销 200 ms，则工具 RTT 为 1000 ms；若返回后整理下一请求又需 50 ms，且调用前没有额外等待，则就绪间隔约为 1050 ms。这只是说明口径的例子，不是本次实测值。

单工具的首版预测字段建议统一命名为：

~~~json
{
  "tool_call_id": "c1",
  "target": "round_trip_from_dispatch",
  "duration_ms": {"q10": 700, "q50": 1000, "q90": 1400},
  "executor_duration_ms": {"q10": 550, "q50": 800, "q90": 1100}
}
~~~

这里 executor 输出是可选的独立目标；不要把其分位数与其他开销分位数简单相加来得到 RTT 分位数。若只实现 RTT，应明确内部时间预测暂未提供，已有内部时间真值仍保留用于分析。

### 12.2 实际已采集的字段与可用范围

以下为对四分区文件的再次核查，区分“字段已有值”与“筛选后可作某个目标的训练标签”。

| 时间 / 字段 | 当前是否采集、何时获得 | 全量可用情况与口径 |
|---|---|---|
| 工具内部 `executor_duration_ms` | 已采集；工具执行结束后从后端返回 | 5848/5848 次实际调用有值；是后端定义的 wall time，不是统一的纯计算时间 |
| 工具客户端 `round_trip_ms` | 已采集；客户端工具包装器返回/报错时记录 | 5848/5848 有值；首版调度用途的工具耗时标签 |
| `non_executor_overhead_ms` | 导出时用 RTT − executor 计算 | 5848/5848 有值；含协议、等待和包装等差额，不能命名为纯网络耗时 |
| `proposal_to_executor_ms` | 已采集；SDK action 提出后到执行器开始时记录 | 5848/5848 有值；是客户端 action 到执行器的间隔，不是服务端队列，也不等于 llm_response 到 tool_start |
| Hotpot `queue_wait_ms` | 已采集；RPC 服务取得执行锁时可确定，随结果返回 | Hotpot 1446/1446 有值；从 handler 进入到取得锁，包含读请求体/解析及锁等待，不能视为纯锁排队 |
| BrowseComp / LCB `queue_wait_ms` | 未独立采集 | 2844 次 BrowseComp、1558 次 LCB 均无该标签，null 不等于 0 |
| `environment_batch_span_ms` | 从本批次 controller tool_start/end 事件推导 | 5434 个请求行有值；不等于完整 ready 间隔 |
| `next_request_prepared_gap_ms` | 下一次请求准备事件到达后推导 | 5852 行有原始间隔，包含非工具轮次；按第 3 节规则仅 5427 条作为有效工具 ready 标签 |
| `llm_transport_duration_ms` | 已采集；完整 LLM 响应返回时记录 | 6302 个请求中 6297 有值；其余保留缺失，不能补 0；含客户端准备/传输、服务等待和推理 |
| `request_to_response_ms` | 从 request_prepared 与 response 事件时间戳推导 | 6297 有值；与上一行起止点不同，不能要求逐条完全相等 |
| 任务 `result.duration_s` | run_task 结束、清理完成后记录 | 450/450 有值，单位 s；含该函数内环境准备、运行、工件导出/清理，不含 campaign 中排队等全部 admission 时间 |

未独立采集的包括：纯网络单向传输时间、BrowseComp/代码后端纯排队时间、LLM 服务端 queue / prefill / decode 的分解、TTFT/逐 token 时延、实际 KV 卸载/恢复/重算时间、cache hit/follower 路径时间。它们需要相应组件新增打点；不能从现有总 RTT 唯一分解出来。当前 LLM 包装器显式仅支持非流式记录。

**在线可见性：**名称/参数在调用前可知，配置 timeout 只是预算；真实 executor 和 RTT 在调用返回后才可作为标签。工具未结束时，只知道已经等待多久及可见状态。就绪间隔要等下一请求准备完成，任务总时长要等任务结束。不能在预测输入里使用未来才得到的实际耗时。

### 12.3 真正负责计时的文件

不是一个独立“采集时间脚本”包办所有计时：**执行路径中的适配器负责打点，工具服务负责内部计时，导出脚本事后拼接标签。** 下列路径均相对项目根目录，本机完整路径前缀为 `/root/flowpilot_predictor/`。

| 组件 | 文件 / 关键入口 | 实际负责的事情 |
|---|---|---|
| 统一事件记录 | [tracing.py](repos/Openhands-software-agent-sdk/benchmarks/flowpilot/src/benchmark_adapters/tracing.py)：`TraceRecorder.emit/request/start_tool` | 写 wall_time、monotonic_ns、seq、身份、request/tool 事件到每个 attempt 的 events.jsonl；计算 proposal_to_executor_ms |
| LLM 调用计时 | [sdk_bridge.py](repos/Openhands-software-agent-sdk/benchmarks/flowpilot/src/benchmark_adapters/sdk_bridge.py)：`RecordedLLM._transport_call` | 用 monotonic_ns 计客户端完整调用，写 llm_response.duration_ms；异常写 llm_error |
| 检索工具 RTT | [retrieval_tools.py](repos/Openhands-software-agent-sdk/benchmarks/flowpilot/src/benchmark_adapters/retrieval_tools.py)：`RetrievalExecutor.__call__` | 包住 Hotpot / BrowseComp 调用，记录 RTT，将服务端内部时间和 queue 字段写入 tool_end/tool_error |
| Hotpot 内部/等待 | [hotpot_rpc.py](repos/Openhands-software-agent-sdk/benchmarks/flowpilot/src/benchmark_adapters/hotpot_rpc.py)：`HotpotHandler.do_POST` | 服务端独立测收到请求、取得 gate 锁、完成检索的 duration；客户端 `HotpotRPCEnvironment` 读取 timing |
| BrowseComp 内部 | [browsecomp_timing.py](scripts/browsecomp_timing.py)：`install_timing` | 包装 MCP CallTool handler，在结果 metadata 的 flowpilot_timing 中返回内部 duration |
| BrowseComp 接入包装 | [serve_browsecomp_timed.py](scripts/serve_browsecomp_timed.py)、[native_browsecomp.py](repos/Openhands-software-agent-sdk/benchmarks/flowpilot/src/benchmark_adapters/native_browsecomp.py)：`call_tool` | 前者安装服务端计时包装，后者读取 metadata 到 last_timing；之后由 RetrievalExecutor 统一落盘 |
| LCB 内部命令计时 | [environment.py](repos/Openhands-software-agent-sdk/benchmarks/flowpilot/src/benchmark_adapters/environment.py)：`_COMMAND_RUNNER`；[local_environment.py](repos/Openhands-software-agent-sdk/benchmarks/flowpilot/src/benchmark_adapters/local_environment.py)：`LocalPilotEnvironment.execute` | runner 在子进程中测执行 wall time；LCB 当前走本地隔离进程执行并读取结果，不应根据类名误称所有代码工具都经 Docker |
| LCB 客户端 RTT | [sdk_bridge.py](repos/Openhands-software-agent-sdk/benchmarks/flowpilot/src/benchmark_adapters/sdk_bridge.py)：`ContainerExecutor.__call__` | 包住 environment.execute，分别写 result.duration_ms 与客户端 RTT；类名不决定实际后端 |
| 任务总耗时 | [runner.py](repos/Openhands-software-agent-sdk/benchmarks/flowpilot/src/benchmark_adapters/runner.py)：`run_task` | 从该函数内计时起点到清理完成写 result.duration_s |

计时边界要保留以下细节：

- 客户端 RTT 的 `start = time.monotonic_ns()` 在 tool_start 事件写出之后，终点在 tool_end 写出之前计算。代码工具还有 observation 构造等微小边界差异。因此事件的 tool_end − tool_start 是生命周期跨度，**不与 round_trip_ms 逐位相等**。训练 RTT 直接用已记录字段；事件时间戳用于状态回放与阶段分析。
- Hotpot executor 包括取得锁后的索引身份检查和检索等，排除取得锁前的阶段；BrowseComp 的 scope 为 `mcp_handler_including_validation_and_result_encoding`，包含 handler 的校验和结果编码，未独立拆排队。
- LCB 内部 runner 包含 shell 启动、命令运行、超时处理、子进程清理及输出读取等；排除父进程启动该 runner 之前的部分开销。它也不等于“用户 Python 函数自身计算时间”。
- BrowseComp 的旧 prepare 元数据可能仍写 `executor_timing_available=false`，但安装 timed wrapper 后，实际 tool_end / 导出行中的 executor_duration_ms 已有值。本次以实际 2844 条标签为依据，不能只看这个旧能力标记。

### 12.4 采集入口、导出入口与两台机器的责任

| 入口 | 文件 | 使用说明 |
|---|---|---|
| 任务采集总入口 | [collect_mixed_c4.sh](scripts/collect_mixed_c4.sh) → [mixed_c4_control.py](scripts/mixed_c4_control.py) | 负责配置环境、检查服务、启动既定 campaign；运行时通过上表适配器自动计时 |
| 并发执行/请求标签导出 | [parallel_collection.py](repos/Openhands-software-agent-sdk/benchmarks/flowpilot/src/benchmark_adapters/parallel_collection.py)：`run_campaign` | C4 执行结束后调用 write_prediction_dataset，形成 prediction_dataset |
| 请求/批次时间标签定义 | [prediction_export.py](repos/Openhands-software-agent-sdk/benchmarks/flowpilot/src/benchmark_adapters/prediction_export.py)：`build_prediction_rows/write_prediction_dataset` | 关联事件，计算 batch span、next-ready gap、非 executor 开销和 masks；不调用工具 |
| 单工具时间导出 | [export_tool_timings.py](scripts/export_tool_timings.py) | 读取既有 events 与响应，导出完整调用、observation 和时间至 tool_timing_dataset；**不是在线计时器** |
| 当前数据核查 | [audit_predictor_handoff.py](scripts/audit_predictor_handoff.py) | 核查已导出数据的计数/关联/切分，不启动实验 |

本机负责 Agent / LLM 客户端 / LCB 工具包装计时；另一台语料服务器负责 Hotpot RPC 和 BrowseComp MCP 内部计时，再把 duration 随响应传回本机。两端各自用单调时钟测持续时间，不跨主机相减时间戳。

远端部署使用 [prepare_mixed_c4.py](scripts/prepare_mixed_c4.py) 生成的 `remote/` 包，其中包含 `serve_corpus_c4.sh`、`serve_browsecomp_timed.py`、`browsecomp_timing.py` 和复制过去的 `hotpot_rpc.py`。远端入口是包内 [serve_corpus_c4.sh](scripts/serve_corpus_c4.sh)，不能只启动未经包装的 BrowseComp 服务便假定有内部时延。源码在上表路径维护，部署副本应同步；本次没有连接远端重启或修改服务。

导出流程已经自动执行过。现有 export 目录存在时，导出器会拒绝覆盖；这次交接直接使用现有文件，无需重跑已完成的 fit 或重新采集。

### 12.5 接手后在哪里读数据

~~~text
runs/campaigns/mixed_c4_v1_<split>/
├── campaign.json                         # jobs[].attempt_dir 定位原始轨迹
├── tasks/<job_id>/attempt-001/
│   ├── events.jsonl                      # 运行过程中追加的原始事件/时间
│   ├── blobs/                            # 完整请求/响应内容
│   └── result.json                       # duration_s、运行状态
├── tool_timing_dataset/
│   ├── tools.jsonl                       # labels.round_trip_ms / executor_duration_ms 等
│   └── audit.json
└── prediction_dataset/
    ├── inputs.jsonl                      # T0 可用特征，不含当前真实耗时
    ├── targets.jsonl                     # LLM / 批次 / ready 标签
    └── manifest.json
~~~

读取示例，不启动任何任务：

~~~bash
cd /root/flowpilot_predictor
python3 scripts/audit_predictor_handoff.py
python3 - <<'PY'
import json
from pathlib import Path
path = Path("runs/campaigns/mixed_c4_v1_fit/tool_timing_dataset/tools.jsonl")
with path.open() as stream:
    row = next(r for line in stream if (r := json.loads(line))["labels"]["executed"])
print(json.dumps({
    "adapter": row["adapter"],
    "tool": row["call"]["tool_name"],
    "arguments": row["call"]["arguments_parsed"],
    "executor_duration_ms": row["labels"]["executor_duration_ms"],
    "round_trip_ms": row["labels"]["round_trip_ms"],
    "queue_wait_ms": row["labels"]["queue_wait_ms"],
    "source_attempt": row["source_attempt"]
}, ensure_ascii=False, indent=2))
PY
~~~

交给同学的一句话约定：**先做“已知工具调用 → RTT 分位数”，内部执行时间保留为独立辅助目标；实际返回后更新模型。需要整批 KV 调度时，再接 next-request-ready 扩展。计时以现有执行器打点和原始 events 为依据，导出脚本只整理标签。**
