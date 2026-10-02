# 新 27B 轨迹：分组校准与 LightGBM 在线反馈

执行用户要求的两项预测实验，所有产物位于 `/data1/ql_flowpilot_predictor/predictor_experiments/native27b_calibration_online_v1`。

本次不启动 LLM/vLLM 服务，只回放已采集的原生 Qwen3.5-27B 工具 RTT，不部署模型或修改 FlowPilot 调度策略。

## 分组校准

保留原 LightGBM 的权重，使用原 calibration 1,588 条调用拟合残差校正，对相同 test 2,169 条调用比较 raw、原 pooled、直接分组和分位数支持度回退方案。

分组键完整保留后端名称、后端版本、工具名称和工具 schema 版本。Q10/Q50 最少 32 调用/4 题目组，Q90 为 100/10，Q99 为 1,000/30。支持不足依次回退至同后端版本池、全局池和 raw。不同工具的池化回退仅提供总体残差统计，不能解释成每类工具的覆盖保证。

阈值在执行前固定，不针对 test 调参。原 test 的结果已经被查看，因此这次属于补充探索性实验；最终确认仍需新的独立数据。

## 在线反馈

另训练使用 fit_primary 8,941 条调用的 LightGBM，配置保持不变。晚期 QuixBugs 辅助数据排除，保证训练结束早于所有在线回放事件。

直接导入现有 `flowpilot_predictor_bridge/online.py` 的 OnlineBias，alpha=0.2、偏差限幅 ±log(4)。按现有 event_order 回放：预测点保存状态和本调用 raw Q50，返回点先评分再更新；仅接收实际本地执行、completed/execution_error、未超过 600 秒反馈 TTL 的客户端 RTT。

比较三条流，每条流的 frozen/online 调用完全配对：

- tune_forward：939 调用，诊断流；
- 增量 test 冷启动：922 调用，从零校正状态开始；
- 增量 test 连续反馈：先接受 tune_forward 939 + 增量 calibration 655 调用的真实反馈，再在 922 test 调用上评分。这是主要在线比较。

模型权重冻结，在线版本只有桥接校正状态变化；没有叠加离线校准，没有 alpha 搜索。沿用现有任务分组，以完整题目组进行配对 bootstrap，并同时报告 Q10/Q50/Q90/Q99 误差和覆盖、区间宽度、逐工具指标及预测开销。

## 运行与产物

```bash
PLAN=/root/flowpilot_predictor/plans/native27b_calibration_online_20261001
bash "$PLAN/run.sh" check
bash "$PLAN/run.sh" start
bash "$PLAN/run.sh" status
```

运行使用独立 CPU tmux 会话，成功或失败后自动退出。启动拒绝覆盖已有实验。测试临时目录自动删除，字节码缓存禁用；结果、模型、真实轨迹和依赖环境保留。

主要报告为 `reports/REPORT.md`、`reports/overall.csv`、`reports/by_tool.csv`、`reports/paired_comparisons.json`；`grouped_calibrator.json` 保存校准池及支持度，`online/*/*/replay_audit.json` 保存更新和并发回放审计。
