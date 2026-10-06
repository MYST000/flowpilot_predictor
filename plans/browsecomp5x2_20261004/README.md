# BrowseComp-Plus 5 × 2 实机测试入口

完整准备状态、SLO、固定题目、命令和证据位置见 [实机测试交接](../../docs/BROWSECOMP_5X2_HANDOFF_20261004.md)。

```bash
python3 plans/browsecomp5x2_20261004/experiment.py check
# GPU 0–3 可用后，显式执行实机实验：
python3 plans/browsecomp5x2_20261004/experiment.py run
```

`check` 不启动 GPU 推理；`run` 启动完整链路、运行两轮、汇总并清理本机服务。远端 BrowseComp MCP 保持运行。
