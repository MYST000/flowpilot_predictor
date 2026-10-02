"""Validate persisted results independently and write the interpretation."""

import csv
import json
import math
from pathlib import Path

from experiment import (
    NAMES, OnlineBias, digest, event_order, identity_key, load_split,
    require, utc_now, write_json,
)


PLAN = Path(__file__).resolve().parent
PROTOCOL = json.loads((PLAN / "protocol.json").read_text())
OUT = Path(PROTOCOL["output"])
DATA = Path(PROTOCOL["prepared_data"])


def read_records(path):
    return [json.loads(line) for line in path.open()]


def validate():
    state = json.loads((OUT / "status.json").read_text())
    require(state["status"] == "complete" and state["exit_code"] == 0, "Experiment incomplete")
    provenance = json.loads((OUT / "provenance.json").read_text())
    require(all(digest(p) == h for p, h in provenance["source_sha256"].items()), "Source integrity changed")
    frozen = json.loads((OUT / "forward_protocol_frozen.json").read_text())
    require(digest(OUT / "forward_lightgbm" / "model.joblib") == frozen["model_sha256"], "Forward weights changed")
    fit = load_split(DATA, "fit_primary")
    tune = load_split(DATA, "tune_forward")
    cal = [s for s in load_split(DATA, "calibration") if s["source_batch"] == "native27b_c4_increment_v2"]
    test = [s for s in load_split(DATA, "test") if s["source_batch"] == "native27b_c4_increment_v2"]
    groups = [{s["task_group_id"] for s in rows} for rows in [fit, tune, cal, test]]
    require(all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:]), "Partition overlap")
    streams = {"tune_forward_diagnostic": tune, "test_forward_cold": test,
               "test_forward_warm": tune + cal + test}
    audits = {}
    for name, samples in streams.items():
        directory = OUT / "online" / name
        full = name == "test_forward_warm"
        paths = {v: directory / (v + "_full_stream" if full else v) / "predictions.jsonl"
                 for v in ["frozen", "online"]}
        raw = {r["sample_id"]: r for r in read_records(paths["frozen"])}
        records = {r["sample_id"]: r for r in read_records(paths["online"])}
        require(raw.keys() == records.keys() == {s["sample_id"] for s in samples}, "Sample pairing changed")
        bias = OnlineBias(alpha=.2)
        observations = 0
        for ns, _, _, kind, _, i in event_order(samples):
            s = samples[i]; r = records[s["sample_id"]]; c = s["context"]
            if kind == "predict":
                current = bias.snapshot(c)
                stored = r["bias_snapshot"]
                require(current.version == stored["version"] and current.observations == stored["observations"],
                        "Prediction used incorrect feedback version")
                require(math.isclose(current.bias, stored["bias"], abs_tol=1e-12), "Prediction used future/stale bias")
                require(r["raw_duration_ms"] == raw[s["sample_id"]]["prediction"]["duration_ms"],
                        "Frozen/online base predictions differ")
                expected = bias.apply(r["raw_duration_ms"], current)
                require(expected == r["prediction"]["duration_ms"], "Persisted online prediction mismatch")
                if expected is not None:
                    require(list(expected.values()) == sorted(expected.values()), "Crossing quantiles")
            else:
                require(r["y_ms"] == s["labels"]["round_trip_ms"], "RTT target changed")
                eligible = (c["resolution"] in {"LOCAL_ONLY", "LOCAL_LEADER"}
                            and s["labels"]["execution_status"] in {"completed", "execution_error"}
                            and ns - s["as_of_ns"] < 600e9 and r["raw_duration_ms"] is not None)
                if eligible:
                    require(r["feedback_action"] == "updated", "Real feedback was not applied")
                    bias.observe(c, r["raw_duration_ms"]["q50"], r["y_ms"])
                    observations += 1
                else:
                    require(r["feedback_action"] != "updated", "Invalid feedback learned")
        audit = json.loads((directory / "online" / "replay_audit.json").read_text())
        require(audit["counters"].get("updates", 0) == observations, "Update count mismatch")
        audits[name] = {"rows": len(samples), "checked_prediction_snapshots": len(samples),
                        "checked_delayed_updates": observations, "paired_predictions_equal_before_bias": True}
    grouped = json.loads((OUT / "calibration" / "hierarchical_tail_support" / "metrics.json").read_text())
    require(grouped["micro"]["rows"] == 2169, "Calibration test changed")
    require(grouped["calibration_fallback_by_quantile"]["q99"] == {"global_pool": 2169},
            "Unexpected Q99 support/fallback")
    result = {"validated_at": utc_now(), "status": "passed",
              "original_source_files_unchanged": len(provenance["source_sha256"]),
              "forward_weights_unchanged": True, "task_group_overlap": 0,
              "replays": audits, "grouped_q99_all_use_global_fallback": True}
    write_json(OUT / "reports" / "validation.json", result)
    return result


def summarize():
    comparisons = json.loads((OUT / "reports" / "paired_comparisons.json").read_text())
    cal = {s: json.loads((OUT / "calibration" / s / "metrics.json").read_text())
           for s in ["pooled", "grouped_min32", "hierarchical_tail_support"]}
    online = {v: json.loads((OUT / "online" / "test_forward_warm" / v / "metrics.json").read_text())
              for v in ["frozen", "online"]}
    paired = comparisons["test_forward_warm_frozen_vs_online"]["delta"]
    hp_ci = comparisons["calibration_pooled_vs_hierarchical_tail_support/hotpot/search"]["delta"]["q90_coverage"]["ci95"]
    lines = [
        "# 实验结论与运行建议", "",
        "两项实验已成功完成。分组校准有局部收益；现有 LightGBM OnlineBias(alpha=0.2) 未显示整体收益，Q99 损失反而增加。",
        "实验仅回放原生 Qwen3.5-27B 的实际轨迹，没有启动 LLM/vLLM 或更改部署配置和同门调度算法。", "",
        "## 分组校准", "",
        "使用 1,588 calibration 调用，比较同一 2,169 test 调用；原测试结果已查看，因此本次为补充探索性分析。",
        "Q10/Q50 门槛为 32 调用/4 独立题目组，Q90 为 100/10，Q99 为 1,000/30；门槛在运行前固定。", "",
        "|指标|原 pooled|分组与支持度回退|", "|---|---:|---:|",
    ]
    for title, key in [("Q50 MAE（ms）", "q50_mae_ms"), ("Q90 分位损失（ms）", "q90_pinball_ms"),
                       ("Q99 分位损失（ms）", "q99_pinball_ms")]:
        def value(s):
            return s["micro"][key] if key == "q50_mae_ms" else s["micro"]["pinball_ms"][key[:3]]
        lines.append(f"|{title}|{value(cal['pooled']):.3f}|{value(cal['hierarchical_tail_support']):.3f}|")
    lines += [
        f"|工具宏平均分位损失（ms）|{cal['pooled']['tool_macro_mean_pinball_ms']:.3f}|{cal['hierarchical_tail_support']['tool_macro_mean_pinball_ms']:.3f}|",
        "", "Hotpot/search 的 Q90 覆盖从 80.84% 提升到 87.46%，仍低于名义 90%。",
        f"改善为 6.62 个百分点，按题目组配对 bootstrap 的 95% 区间为 [{hp_ci[0]*100:.2f}, {hp_ci[1]*100:.2f}] 个百分点。",
        "其 Q90 损失从 413.90 降至 374.57 ms，但该损失差的置信区间跨 0。",
        "BrowseComp/search 的 Q90 覆盖从 91.41% 变为 90.71%；code_terminal 从 89.08% 变为 92.06%。",
        "文档工具出现欠覆盖：Hotpot/read_document 从 96.90% 变为 84.50%；BrowseComp/get_document 从 96.51% 变为 88.37%。",
        "因此不能统一替换为分组校准并宣称所有工具更准，Q50 的整体误差也几乎没有改善。", "",
        "所有精确工具/后端组的 calibration 样本数都不足 1,000，Q99 全部借用全局校准池。",
        "总体 Q99 覆盖 99.17% 不能解释为每个工具都达到 99%；例如 Hotpot/search 为 98.26%。",
        "直接使用 32 样本的分组 Q99 虽然总分位损失更低，但总体 Q99 覆盖降至 98.80%，尾部证据不足。",
        "Q99 回退维持统计支持度；后续需要更多独立题目组验证工具级尾部。", "",
        "## 现有桥接在线校正", "",
        "另训练 fit_primary 8,941 调用的 LightGBM，排除晚期 QuixBugs；所有回放均晚于训练结束。",
        "连续流先接受 tune_forward 939 + 增量 calibration 655 次真实反馈，再在增量 test 922 调用上评分。",
        "冻结与在线版本使用完全相同的 raw 预测和题目，alpha=0.2 固定，无参数搜索、无离线校准叠加。", "",
        "|指标|冻结|在线校正|相对变化|", "|---|---:|---:|---:|",
    ]
    for title, key in [("Q50 MAE（ms）", "q50_mae_ms"), ("Q90 分位损失（ms）", "q90_pinball_ms"),
                       ("Q99 分位损失（ms）", "q99_pinball_ms")]:
        def value(s):
            return s["micro"][key] if key == "q50_mae_ms" else s["micro"]["pinball_ms"][key[:3]]
        a, b = value(online["frozen"]), value(online["online"])
        lines.append(f"|{title}|{a:.3f}|{b:.3f}|{(b/a-1)*100:+.2f}%|")
    lines += [
        "", "Q50 误差增加的 95% 配对区间跨 0，因此不能确认总体 Q50 恶化具有统计稳定性，但没有观察到整体收益。",
        f"Q99 损失增加 {paired['q99_pinball_ms']['estimate']:.2f} ms，95% 配对区间 [{paired['q99_pinball_ms']['ci95'][0]:.2f}, {paired['q99_pinball_ms']['ci95'][1]:.2f}] ms，均在 0 以上。",
        "Q90 覆盖从 89.37% 变为 88.72%，Q99 从 97.83% 变为 97.94%；Q10–Q90 平均区间宽度却增加约 34.57%。",
        "冷启动测试也出现相同方向：Q50 点估计增加约 3.87%，Q99 损失增加约 5.89%。",
        "tune 诊断流中 Q50 误差增加约 7.07%，说明此前独立 EWMA 的改善不能迁移解释为 LightGBM 桥接校正有效。", "",
        "一种可能原因是该校正器根据按工具汇总的 log RTT 残差，同时平移所有分位数；参数不同、尾部异常和短期波动可能影响下一次调用。",
        "这只是机制解释，当前实验没有单独验证原因。", "",
        "## 当前建议", "",
        "第一版系统比较继续使用冻结 LightGBM Q50，同时保留实际工具耗时采集。",
        "分组校准作为独立候选保留，下一轮只用训练/调优数据确定小样本回退，再以新数据确认，尤其检查文档工具的欠覆盖。",
        "现有 alpha=0.2 的全分位 OnlineBias 暂不作为默认运行方案；仍可积累反馈用于后续校准和周期性训练。",
        "后续若比较更平滑的残差、分位数独立校正或自适应覆盖校准，需要另立协议，当前实验未执行这些方法。", "",
        "脚本验证了延迟反馈、并发状态快照、重复调用、过期反馈和任务/时间划分；独立结果验证重新逐事件核对了保存的在线状态。",
        "所有模型和原始数据均保留，没有执行端到端 Tool/KV 缓存收益实验。", "",
    ]
    (OUT / "reports" / "ANALYSIS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    validation = validate()
    summarize()
    print(json.dumps(validation, ensure_ascii=False, indent=2))
