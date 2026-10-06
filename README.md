# FlowPilot 工具 RTT 预测模块

本仓库保存预测器、FlowPilot/OpenHands 桥接代码、实验方法与 benchmark 适配器源码。数据、模型二进制、本机环境、日志和缓存不随 GitHub 发布。当前正式模型为 **Qwen3.5-27B 轨迹训练的 LightGBM，配合逐分位实时校正**；Qwen 本身不是预测器，使用预测器不需要启动 Qwen 或 vLLM。

模型保存在共享磁盘：

```text
/data/ql_flowpilot_predictor/predictor_experiments/native27b_1077_v1/train/lightgbm/model.joblib
```

同门需要读取该文件、同目录的 `manifest.json`、对应代码快照，以及准备数据目录中的 `manifest.json`。仅加载和推理不读取训练 JSONL，不需要重新采集或训练。

快速开始（Linux，已安装 conda）：

```bash
git clone https://github.com/Fate0123/flowpilot_predictor.git
cd flowpilot_predictor
conda create -n flowpilot-predictor-27b python=3.12.9 pip -y
conda activate flowpilot-predictor-27b
python -m pip install -r scripts/requirements-predictor-lock.txt
cp configs/predictor/shared-data1.example.json configs/predictor/runtime.json
python -B scripts/check_shared_model.py
```

检查只加载模型并发起四个合成预测，不启动 benchmark、模型服务或网关，不写共享模型和数据。`status: ready` 表示预测模块可加载；`duration_sink_bound: false` 在此独立检查中正常，正式网关会在启动时绑定调度回调。

默认输出 Q10/Q50/Q90/Q99，单位毫秒；`duration_estimate_ms` 默认交付 Q50。预测发生在 LLM 完整返回工具调用后，和缓存查找并行；跨请求支持四并发预测。实际本地工具执行 RTT 更新在线校正状态，缓存命中/复用结果不作为真实本地执行 RTT 训练。未知工具/后端/schema 明确返回 unsupported，不伪造预测。

**完整上传命令、共享文件权限、输入输出、实时反馈和正式网关接入见 [操作手册](docs/GITHUB_HANDOFF.md)。** 正式接入必须使用预测包装入口或显式注册 `FrameworkPredictor`；直接运行同门原网关入口不会自动开启预测器。

- `predictor/`：经验分位数、EWMA、聚类、QRF、LightGBM 五种算法及训练/评测实现。
- `flowpilot_predictor_bridge/`：并行预测、调度耗时交付、真实耗时反馈、在线校正。
- `adapters/openhands_current/`：发布时导出的当前整合版本适配器，含未提交与新增代码。
- `adapters/openhands_legacy/`：原 `flowpilot_predictor/repos` 中的适配器历史副本。
- `idea/`：框架、design 和接口说明。
- `plans/`、`PREDICTOR_*.md`：实验源码与历史报告；其中路径和旧模型描述代表当时实验，当前入口以本 README 和操作手册为准。
- `DEPENDENCIES.json`、`SOURCE_MANIFEST.json`：发布时生成的兼容仓库提交与源码清单。

本次发布不修改同门的调度、工具复用或 KV 策略，也不发布完整 vLLM/OpenHands 依赖仓库。
