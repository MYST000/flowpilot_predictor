# 正式实验实时校正对接

用户要求实际实验实时校正。默认预测配置现在启用 `online=true` 和 `quantile_residual_v1`，使用新 27B 全 fit LightGBM 的 raw 工件、四个并发预测副本、Q50 调度标量，同时保留 Q10/Q50/Q90/Q99 输出。

每次真实工具返回都会将自身 raw 分位数与实际客户端 RTT 的 log 残差写入该工具/后端的有界窗口。每个分位数独立估计残差分位数，使用样本量收缩、平滑、偏差限幅和单步限幅。支持度不足时输出基础模型，并继续接收反馈。窗口和状态在单个网关进程内保存，进程重启重新积累。

这属于在线校正层更新，LightGBM 树权重不逐调用重训。原 `OnlineBias` 保留为显式历史对照，原 RTT 特征、模型工件、OpenHands 适配器以及调度/缓存策略没有修改。

## 已完成的回放

仅在 tune_forward 939 调用上选择 9 个候选参数，选出 alpha=0.1、shrinkage_rows=256。其他门槛预先固定：窗口 1,024，Q10/Q50 至少 32 调用/4 任务，Q90 为 100/10，Q99 为 1,000/30，log 偏差限幅 ±log(2)，单步最多 0.02。

随后在相同 922 次增量 test 上比较。连续反馈版先接收此前 939 tune + 655 calibration 调用；另做冷启动对照。使用训练时间严格更早、排除晚期 QuixBugs 的 fit_primary 模型，避免在线回放读取未来训练信息。

结果见 `/data1/ql_flowpilot_predictor/predictor_experiments/native27b_realtime_quantile_v1/REPORT.md`。测试数据此前已查看，本次为探索性验证。实际启动默认配置时装载的是全 fit 的正式新 27B 模型；回放模型与部署模型的训练样本范围已明确区分。

## 实际框架启动时使用

在原有 FlowPilot 启动环境中配置：

```bash
export FLOWPILOT_PREDICTOR_CONFIG=/root/flowpilot_predictor/configs/predictor/runtime.json
```

使用 `flowpilot_predictor_bridge.serve:create_app` 入口时会注册 T1 预测、权威 resolution 和真实工具 RTT 反馈；普通 FlowPilot app 入口不会启用这些 hook。

原 `examples.experiments.qwen35_9b_tp4.launch gateway` 直接调用普通 app factory，因此使用正式实验 profile 时，应由预测侧包装入口启动。包装入口仍调用同门的 `load_profile/gateway_settings`，使用同一份 profile、registry、cost model 和环境凭据：

```bash
source /root/flowpilot_integration_20260928/env.sh
export FLOWPILOT_PREDICTOR_CONFIG=/root/flowpilot_predictor/configs/predictor/runtime.json
"$FLOWPILOT_PREDICTOR_PYTHON" -B -m flowpilot_predictor_bridge.launch_gateway \
  --config "$EXPERIMENT_PROFILE" --run-dir "$RUN_DIR" --registry "$TOOL_REGISTRY" --check
```

`EXPERIMENT_PROFILE`、`RUN_DIR`、`TOOL_REGISTRY` 使用本次正式系统实验原定的值，凭据沿用 `FLOWPILOT_INGRESS_API_KEY` 和 `FLOWPILOT_DCS_ENCRYPTION_KEY`。正式启动同一命令时去掉 `--check`。本次仅验证包装入口，不开启服务；包装入口不修改模型服务参数或策略配置。

当前目录不启动 vLLM、LLM 或正式任务采集。仅预测相关代码和配置已准备并验证；原四任务并发、工具后端及其他系统策略由已有实验启动配置决定。

## 检查

```bash
cd /root/flowpilot_predictor
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" .venv-predictor/bin/python -B -m pytest -q -p no:cacheprovider tests/test_predictor_bridge.py tests/test_online_quantile_calibration.py
```

新 27B 模型通过 `artifact_code_root` 指定原部署快照；加载保持代码/模型/数据/依赖校验，不绕过工件身份检查。
