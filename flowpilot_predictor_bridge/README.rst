T1 工具 RTT 预测适配层
====================

此包位于预测器仓库，仅实现预测输入、并发推理、在线反馈和耗时先验交付。
不包含工具执行、缓存策略、admission、KV offload/reload 决策。
原 predictor/*.py、正式 LightGBM 工件和用户 SDK 修改保持原样。

交给调度器的具体参数
--------------------

``duration_estimate_ms`` 是选定 Q50 或 Q90 的客户端 RTT 标量，单位毫秒。
配置 ``configs/predictor/runtime.json`` 中 ``selected_quantile`` 为 ``q50``
或 ``q90``。``DurationPrior.duration_p50`` / ``duration_p90`` 始终保留
各自实际分位数，不把 Q90 填进名字叫 p50 的字段。

完整输出保留用户模型的 ``duration_ms.q10/q50/q90/q99``、support、fallback，
另加原始分位数 raw_duration_ms、身份、时间和在线版本。
模型依旧预测本地执行 RTT；命中时的预测是“若执行需要多久”，不是命中交付时间。

``FlowPilotDurationSink`` 将结果送给外部提供的原子 setter 回调，其明确关键字为：

::

    await update(
        identity=prior.identity,
        expected_resolution_version=prior.resolution_version,
        duration_estimate_ms=prior.duration_estimate_ms,
        source="predictor_t1_rtt",
        prediction_id=prior.prediction_id,
        expires_at_monotonic=prior.expires_at_monotonic,
        metadata={...真实 Q50/Q90、选择分位数、版本...},
    )

这份 callback 是需要框架绑定的预测消费边界，不是假定现有框架已有同名方法。
它应在现有 resolution 锁内核验当前调用身份、epoch/tail、local 路径、非终态、
resolution version 和有效期后写入已有耗时字段，并触发现有投影刷新。
成功返回 True，过期/冲突返回 False；错误被记录为 sink_errors，不改变工具结果。
不应在 callback 内引入新的排序、缓存或 KV 决策公式。

2026-09-28 最新融合：``framework.FrameworkPredictor`` 已绑定真实的
``ToolResolutionStore.apply_duration_estimate``，经原子版本校验写入上述字段。
``serve.create_app`` 自动注册 T1 hook、resolution hook 和跨进程 RTT 反馈。
旧 ``FlowPilotDurationSink`` 仍可用于其他消费端；最新版入口无需手工注册它。
详见 ``/root/flowpilot_integration_20260928/LATEST_INTEGRATION.md``。

调用时序
--------

完整 LLM 回复解析后，立即调用 ``submit`` 或 ``on_llm_response``，不等 hit/miss。
每个 Tool Call 一个 ToolPredictionRequest；多个请求并发。缓存分支独立继续。
无工具回复传空列表，记录 no_tool；没有完整参数的调用不能虚构预测。

::

    from flowpilot_predictor_bridge import (
        PredictorRuntime, FlowPilotDurationSink, ToolPredictionRequest,
    )

    # actual_atomic_duration_setter 由目标框架提供，未在本包重写。
    sink = FlowPilotDurationSink(actual_atomic_duration_setter)
    runtime = PredictorRuntime.from_config(
        "/root/flowpilot_predictor/configs/predictor/runtime.json",
        duration_sink=sink,
    )

    async with runtime:
        key = runtime.submit(ToolPredictionRequest(identity, context))
        # 此处独立启动/继续同门已有的 reuse resolve；不 await runtime.wait(key)。
        # 收到权威 resolve 回调后：
        runtime.resolve(key, actual_resolution, version=resolution_version)
        # 收到真实客户端 RTT 后：
        runtime.observe(feedback)

``context`` 使用原 predictor/features.py 和 data.py 的字段：backend_id/version、
tool_name/schema_version、arguments、execution_mode、batch_index/size、配置 timeout、
T0 历史/负载/预算及快照 age。``request_from_tool_call`` 可从已闭合的 Chat 工具调用
取参数并按原 schema 补默认值；工具注册、T0 采集和权威 identity 由运行时提供。
它不接管完整 agent loop，也不猜测不存在的后端版本/历史/负载。

适配器只在独立复制的模型输入中设置 resolution=LOCAL_ONLY，表达本地执行假设，
从不修改缓存权威状态。hit/follower 也运行模型，但不回写本地执行耗时先验。
模型对未知 backend/schema 或请求内部并行工具执行保持 unsupported。
跨 request 并发预测与原有单请求串行工具执行是不同约束。

缓存先返回时可以立即交付，不等待模型；预测先返回则暂存，收到 local 事实再交付。
真实 Tool 结束后即使模型晚到，也不会再写入该调用的时间先验。
同族多工具由 tool_call_id 区分；重试由 attempt、llm_call_id、execution_attempt 区分。
新 tail/epoch 或请求取消时调用 invalidate_line / cancel，拒绝旧结果回写。
单个 wait 协程取消只取消该等待者；生命周期取消需显式调用 cancel/invalidate。

实时反馈更新
------------

默认配置使用新 27B 轨迹训练的 LightGBM，开启 ``online=true``、
``online_method=quantile_residual_v1``，并配置 4 个并发预测副本。
每次真实本地工具返回都会更新校正状态；树权重不在单次返回时重训。
每个精确 backend/version/tool/schema 组合独立维护有界的逐分位残差窗口：

::

    residual_q = log1p(actual_client_rtt_ms) - log1p(raw_q_for_this_call)
    target_q = quantile(past_residuals_q, q) * n / (n + shrinkage_rows)
    target_q = clip(target_q, -log(2), log(2))
    step_q = clip(alpha * (target_q - offset_q), -0.02, 0.02)
    offset_q = clip(offset_q + step_q, -log(2), log(2))
    online_q = max(0, expm1(log1p(raw_q) + offset_q))

每个头独立更新，不使用 Q50 残差同时推动 Q90/Q99。输出以累积最大值保持
Q10 <= Q50 <= Q90 <= Q99，保留头身份。现配置 alpha=0.1、shrinkage_rows=256、
窗口 1,024 条，参数仅在 tune_forward 的 939 调用中选择。

Q10/Q50 至少需要 32 调用/4 任务，Q90 为 100/10，Q99 为 1,000/30。
不足时仍积累反馈和更新版本，该头输出基础模型预测；任务数默认按 job_id 去重，
有明确 task_group_id 时可在请求上下文中提供。支持度门槛不等于覆盖保证。
下一次提交使用更新后的状态；现有预测快照保持原值。

原共同 EWMA 偏差可通过 ``online_method=ewma_log_bias`` 显式选择，主要用于历史对照。
``online=false`` 仅用于冻结实验对照；正式实时方案默认开启校正。

每个请求在提交时固定在线状态快照，真实反馈只能影响以后提交的请求。
误差参考该调用保存的四个原始分位数，不使用其他并发请求的预测代替。
反馈早于预测完成时暂存，原始预测就绪后再更新，不把当前标签泄漏到当前预测。

只接受真实 LOCAL_ONLY/LOCAL_LEADER 执行的完整可观测客户端 RTT。
completed 与 execution_error 的返回 RTT 沿用原数据口径；cancelled/blocked、
缺失标签以及 hit/follower 不更新模型。正在执行的 elapsed 不能当完整 RTT。
ToolFeedback.event_id 和完整调用身份用于去重与冲突检查；实际重试需要新
execution_attempt 的身份与预测记录，不能反复更新同一调用。

``feedback_from_tool_event`` 能映射现有 ToolTelemetryEvent 的 measured_latency_ms，
但调用方必须明确 timing_scope='client_round_trip' 且先通过运行时事件校验。
不能把 executor_duration 或下一请求 gap 冒充训练目标。

在线状态目前仅在进程内保存，重启从原模型重新开始。反馈去重与调用记录有界，
默认反馈窗口 600 秒；窗口之外或无对应预测的反馈明确忽略，不声称跨重启 exactly-once。
逐分位方案已完成事件回放和桥接验证：连续反馈测试中 Q50 误差基本持平，
Q90 覆盖率 89.37% -> 90.89%，Q99 分位损失基本持平。原测试数据已经查看，
本次属于探索性分析，尚未测量真实调度吞吐或 SLO 收益。
报告位于 ``/data1/ql_flowpilot_predictor/predictor_experiments/native27b_realtime_quantile_v1/REPORT.md``。

并发与运行边界
--------------

两个或更多常驻模型副本，各自独占一次 native 调用，CPU 在线程池中执行，
事件循环保持可响应。max_pending 限制排队与计算总数；饱和请求明确返回 overloaded。
即使超时或取消，native 计算结束前也不归还执行槽，避免后台任务无限积累。
默认保留工件每模型 8 个线程；workers 与线程总数应按真实资源测量配置。

仅支持正式 raw LightGBM 的 round_trip_ms 目标。配置中的 ``artifact_code_root``
显式指向 27B 工件对应的代码快照；加载核验完整快照哈希、模型哈希、数据 manifest、
依赖版本、实际模型/特征/事件实现，以及 data.require 的兼容性。
无该参数时沿用原严格 load_artifact。运行时不改磁盘权重，不改 SDK 环境。
关闭使用 async context manager / aclose，等待在途 native 工作安全结束。
单个 runtime 属于一个事件循环，不能在不同线程/循环间直接调用其方法。

expires_at_monotonic 仅能在同进程/时钟域消费；本接口是进程内 Python 接口。
跨主机 RPC 必须另做时钟与 TTL 转换，不可直接比较 monotonic 值。
不计算 ready_at/t_need，不把 RTT 当剩余时间；起止转换仍由已有 resolution 投影负责。
已有设计中实际恢复由 vLLM 执行，本包没有新增外部 reload/RESTORE 命令。

验证
----

::

    cd /root/flowpilot_predictor
    PYTHONDONTWRITEBYTECODE=1 .venv-predictor/bin/python -m pytest -q \
      -p no:cacheprovider tests/test_predictor_bridge.py tests/test_time_predictor.py

测试中的 duration consumer 是明确的 fixture，用来检查接口字段、CAS/终态竞争，
不代表目标调度器已绑定。正式本地工件存在时同时运行 LightGBM 工件兼容与在线更新检查。

OpenHands 实时耗时反馈（2026-09-28）
----------------------------------

已有计时无需重做：sdk_bridge.ContainerExecutor 和 retrieval_tools.RetrievalExecutor
在实际执行前取 monotonic_ns，在 tool_end/tool_error 中写 round_trip_ms；
executor_duration_ms 是另一种口径，不用于本预测器的 RTT 标签。
TraceRecorder 原来实时 flush 到 events.jsonl，本次增加可选 tool_timing_observer。
改动仅位于集成副本：
/root/flowpilot_integration_20260928/openhands/benchmarks/flowpilot/src/benchmark_adapters/tracing.py。
原始用户 SDK 目录及已有适配器修改完整保留。

推荐复用执行器内的计时值。通用 PreToolUse/PostToolUse hook 分别处理 ActionEvent
和 ObservationEvent，边界可能包括确认、其他 hook、调度等时间，直接相减与模型
训练目标不一致。新增的是预测遥测回调，不是阻断工具执行的 hook 命令。

在拥有 PredictorRuntime 的事件循环中创建 observer，每个 recorder 单独一个。
调用方在实际 action 确定、本地执行开始前显式绑定 SDK trace UUID 与框架身份：

::

    from flowpilot_predictor_bridge import OpenHandsTimingObserver

    observer = OpenHandsTimingObserver(runtime.observe)
    recorder.tool_timing_observer = observer

    # T1 complete tool_calls: first submit prediction, independent of reuse lookup.
    runtime.submit(prediction_request)
    # Once the existing resolver chooses local execution and the SDK action exists:
    observer.bind(
        trace_request_id=sdk_request_id,
        tool_call_id=prediction_request.identity.tool_call_id,
        action_event_id=sdk_action_event_id,
        identity=prediction_request.identity,
        tool_name=prediction_request.context["tool_name"],
        resolution="LOCAL_ONLY",  # or authoritative LOCAL_LEADER
    )
    # Existing executors emit tool_end/tool_error automatically.
    # Observer delivers ToolFeedback to runtime.observe on the predictor loop.
    # Drain tool work before teardown; detach then close.
    recorder.tool_timing_observer = None
    observer.close()

不能把 SDK request_id 直接当作框架 request_id。绑定包含 action_event_id，
执行重试需新 action 和 execution_attempt 身份；缺失、过期或不匹配的样本拒绝。
hit/follower 同样预测，但不绑定本地执行耗时；其终态通过原 runtime.observe
接口反馈 executed=False。取消/未执行调用也由拥有生命周期的调用方终结。
本 callback 只覆盖已绑定适配器的真实 tool_end/tool_error，不宣称覆盖全部 SDK 工具。

默认 128 个待投递反馈、4096 个绑定、600 秒绑定有效期；满载丢弃反馈并计数，
JSONL 仍保留实际 RTT。snapshot() 可查看 pending、bindings、unmatched、expired、
dropped_full、delivery_errors 和 delivered/result_updated 等计数；
recorder.tool_timing_observer_errors 记录回调异常。反馈不等待预测，也不等待模型更新。
超时可从 retrieval 的 timed_out、container 的 outcome.timed_out 或 TimeoutError
识别，仅通知终态，不使用被截断的耗时更新模型；非超时执行错误沿用原 RTT 口径。

此通道是进程内线程安全对接。仅导入 OpenHandsTimingObserver 不加载 ML 依赖；
实际同进程运行 runtime 仍需兼容的 SDK/ML 依赖环境。最新版增加 ``PredictorTraceRecorder`` 的独立 HTTP 反馈队列，SDK 无需加载 ML。
使用融合工作区的 ``scripts/serve_predictor.sh``，并给 benchmark 设置
``FLOWPILOT_PREDICTOR_GATEWAY`` 和 ``FLOWPILOT_INGRESS_API_KEY`` 即自动启用。
也支持显式传入 ``run_task(..., flowpilot_config=...)``。
已完成真实 Qwen3.5-9B + 修改版 vLLM + 原调度的 LCB 单任务验证：
6 次预测、6 次真实 RTT 更新、5/5 测试通过。大规模精度/吞吐收益仍未评估。
