# 当前 workload、工具复用与研究 motivation 的证据审查

分析日期：2026-09-26。分析对象仅为已冻结的 `runs/campaigns/mixed_c4_v1_fit`，三类各 100 个任务，任务并发上限 4，单任务工具串行。没有改变正在运行的采集，也没有查看 test 内容进行设计。

## 1. 计时口径与分 benchmark 结果

每个 benchmark 的比例按 `sum(phase_time) / sum(result.duration_s)` 计算，不是各题百分比的算术平均，也不是整个 campaign 的墙钟时间比例。工具使用客户端 `round_trip_ms`，已经包含内部执行耗时；LLM 使用 `llm_transport_duration_ms`，包括客户端观测到的请求排队、推理、往返和异常等待，不能称为纯 GPU 计算时间。其余为没有归入这两段的运行耗时。

| benchmark | 任务数 | 工具内部累计 s | 工具 RTT 累计 s | LLM 请求累计 s | 工具占比 | LLM 占比 | 其余占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 100 | 1169.00 | 1190.74 | 16905.13 | 6.57% | 93.27% | 0.16% |
| BrowseComp-Plus | 100 | 1349.91 | 1441.55 | 48457.79 | 2.88% | 96.94% | 0.18% |
| LiveCodeBench | 100 | 565.11 | 621.68 | 99484.50 | 0.61% | 96.97% | 2.42% |

只保留运行完成的任务：

| benchmark | 运行完成数 | 工具占比 | LLM 占比 |
|---|---:|---:|---:|
| HotpotQA | 96 | 6.27% | 93.57% |
| BrowseComp-Plus | 100 | 2.88% | 96.94% |
| LiveCodeBench | 95 | 0.91% | 98.99% |

运行完成不是答案正确。300 个任务在 collection_summary 中的 `evaluation_status` 均为 `pending`，所以现阶段不能报告正确率或质量约束下的 SLO goodput。

**已确认的事实：当前三类 benchmark 按上述客户端时间口径，整体都由 LLM 请求耗时主导。** HotpotQA 的工具占比明显更高，但不能据此把 HotpotQA/BrowseComp-Plus 直接称为“工具主导型任务”。改变模型速度、后端、并发和语料，会改变这个比例；这里描述的是当前配置。

逐题工具时间比例的分布说明还有任务内部/类别内部差异：

| benchmark | 工具占比中位数 | P90 | 超过 20% 的任务数 | 超过 50% 的任务数 |
|---|---:|---:|---:|---:|
| HotpotQA | 5.42% | 25.94% | 17 | 1 |
| BrowseComp-Plus | 4.66% | 12.82% | 2 | 0 |
| LiveCodeBench | 1.03% | 2.26% | 1 | 0 |

300 题只有 1 题工具占比超过 50%。可以据此讨论异质性，不能把少量尾部样本写成整体任务分成 tool-bound 与 LLM-bound 两群的证明。客户端长等待也不等于已证明 GPU 计算饱和；若主张资源瓶颈互补，应补服务端排队、prefill/decode、GPU 利用率和 KV 压力证据。

## 2. 当前能观测到的工具复用机会

分析仅包含固定语料的只读 search/read/get_document。键由 adapter、tool、有效参数、记录的语料 revision、后端配置和执行 profile 构成；字符串不做语义合并。要求相同键的先前结果在当前调用开始前已经完成；排除未来结果泄漏。相同 controller clock_domain 才比较先后。

这些是 **exact-key 候选复用**，不是已经启用缓存测得的命中率。需满足相同授权 scope、固定语料和工具版本等真实条件。估计假设初始空缓存、容量无限、无淘汰、读取/验证免费且轨迹不变。它是固定轨迹的理想节省估算，不是系统加速比或实际策略的全局收益上限。

| benchmark | 只读工具调用数 | 历史 exact 候选 | 候选比例 | 可免去的原始 RTT 合计 | 占该 benchmark 累计任务时间 |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 1043 | 207 | 19.85% | 201.06 s | 1.109% |
| BrowseComp-Plus | 1938 | 81 | 4.18% | 67.50 s | 0.135% |
| LiveCodeBench | 不适用 | 不估计 | 不适用 | 不估计 | 不适用 |

代码执行、文件编辑依赖工作目录/文件状态或产生副作用，不能仅凭同名同参数复用；不适用不表示它们理论上不存在任何状态感知缓存方法。

分工具：Hotpot/search 为 200/820 次，候选比例 24.39%，对应 200.88 s；Hotpot/read_document 为 7/223 次、0.18 s。BrowseComp/search 为 75/1862 次、66.87 s；get_document 为 6/76 次、0.63 s。当前复用价值基本来自搜索，重复读取的累计直接价值很小。

进一步核查：

- 288 次历史 exact 候选全部来自同一任务内部；跨任务已完成的 exact 候选为 0，在途同键候选也为 0。Hotpot 与 BrowseComp 的 search 访问不同语料，不能因为同名或查询相似就互相复用。
- 候选键的重复观测内容没有发现不一致。这是回顾性诊断，不是未来语义复用正确性的证明，也没有测量语义复用候选覆盖率。
- Hotpot 候选出现在 21 个任务，节省量最高的 5 个任务贡献约 71.94%；BrowseComp 出现在 34 个任务，前 5 个贡献约 61.93%。缓存机会分布并不均匀。
- Hotpot 运行完成任务贡献 123 次候选、123.81 s；其余 84 次、77.25 s 来自未正常完成的任务。重复循环也会抬高候选率，复用相同结果未必能让任务跳出循环或提高正确率。

合计理想移除 268.56 s，相当于所有工具 RTT 的 8.25%、全部任务累计时间的 0.157%。不能把“省了 8.25% 工具时间”写成“端到端加速 8.25%”。单个临近 deadline 的任务仍可能因这点节省而达标，但需要真实 SLO 实验。

**我的判断：** 当前数据支持先做低成本 session-local exact cache，而且它是共享缓存必须比较的强基线；暂不支持把跨任务共享/在途合并作为主要实测收益来源。语义缓存是否明显增加安全复用空间尚未验证，不能从 exact=0 推出 semantic=0，也不能反向假设它一定有效。

可用于后续 cache admission 的成本分解是：对未来调用估计 `P(valid_hit) × (T_execute - T_hit_path) - T_lookup/validation - amortized_storage_cost`，最终还要结合 deadline 和关键路径。正确性先作为约束，不能通过放宽正确性抵换任意延迟收益。当前估算没有 TTL、容量、验证开销，不能直接用于给各条目定价。

## 3. 对 motivation 的修改建议

原说法：“不同类别的任务，有的是 tool 耗时长，有的是 LLM 耗时长，我们需要利用这些信息进行充分的调度，以确保最终满足正确与 SLO 任务的数量以及整体端到端效率。”

其中研究目标合理，但有三处需要修改：

1. “有的类别由工具主导”是待验证假设，当前配置的数据不支持将三类任务如此二分。
2. 时间异质性不会自动产生可兑现的调度收益。需证明工具等待期间有其他可运行工作、GPU/KV 存在资源约束、控制面能采取有效动作，且收益覆盖预测/迁移成本。
3. 时间预测不能确保答案正确，工具正确复用也不能确保 LLM 最终答对。正确且按时完成应作为联合评价目标，而非预测器的保证。

建议修改为：

> Agent 任务在 LLM 推理、外部工具等待与结果复用阶段之间交替推进，其下一次推理就绪时间、上下文占用及复用收益随任务和运行状态变化。我们利用可在线校正的就绪时间估计与真实工具/KV 状态，协调请求调度、上下文驻留和安全结果复用，在保持结果有效性的约束下，提高单位时间内正确且满足 SLO 的任务数，并改善端到端完成效率。

这是待验证的系统假设。若要保留“工具主导 vs LLM 主导”的原动机，应补真实长工具工作负载，例如规模变化明确的测试/编译、较大检索/分析任务，并证明这些任务属于目标应用；不要通过无依据的 sleep 或选择性展示少量尾部样本制造证据。受控延迟可作为敏感性实验，但必须标明注入，不可冒充实际 workload。

## 4. 目标与评估口径

对任务 i，令 A_i 为真实到达时刻、F_i 为完成时刻、D_i 为允许的相对 deadline，C_i 为独立评测得到的正确性指标：

`quality_slo_goodput = sum_i 1[C_i correct AND F_i - A_i <= D_i] / observation_time`。

固定任务集合时，同时报告 `qualified_count`、总任务数与 qualified fraction；不是只有吞吐分母变化。不能通过拒绝/丢弃困难任务后只统计已接受任务来提高表面指标；需要给出所有提交任务上的覆盖、拒绝、失败、超时与分 benchmark 结果。

还应报告：任务 E2E P50/P95、makespan、正确率、SLO miss rate、工具复用错误/过期开销、KV 恢复等待和重算成本。当前没有 SLO 标准和完成的答案评测，因此暂时无法给出 qualified_count 的实测值。

SLO 必须在实验前固定，可按业务标准，或使用训练/调优数据定义的统一 workload 档位；不能按每条测试任务事后观测的运行时间设置其 deadline。也不能自动用本次 `result.duration_s` 代替 arrival 到 finish；应明确包括排队、admission、工具和恢复等待。

实验中逐步比较：

1. 相同资源与并发约束下的无预测调度；
2. 加 session-local exact cache；
3. 加共享/语义缓存（若安全覆盖率足够）；
4. 加离线就绪预测和 KV 策略；
5. 加在线校正；
6. 使用真实未来就绪时间的 oracle 对照。

固定轨迹用于隔离策略与机会分析，真实 Agent 执行用于检验端到端、质量和反馈效应。缓存/调度会改变排队及后续状态，旧 trace 的时延不能当作新策略下永远不变的环境。

当前只有 tool-ready 预测并不足以对整个任务的完成 deadline 作可靠判断；后续 LLM 阶段和剩余轮次也影响 SLO。初期调度可结合 deadline age、当前推理队列、实测 KV 成本和保守的 LLM 阶段估计，而不要求工具预测器承担“整任务时间与正确率预测”。

## 5. 可复现材料

- 统计脚本：`scripts/analyze_predictor_workload.py`。
- 主统计：`evidence/predictor_handoff_20260926/workload_reuse_analysis.json`。
- 复用按任务结果分层与集中度：`evidence/predictor_handoff_20260926/reuse_outcome_analysis.json`。
- 在线方案：`PREDICTOR_HANDOFF_20260926.md` 第 10 节。

复现只读分析：

```bash
cd /root/flowpilot_predictor
python3 scripts/analyze_predictor_workload.py \
  runs/campaigns/mixed_c4_v1_fit \
  --output evidence/predictor_handoff_20260926/workload_reuse_analysis.json
```
