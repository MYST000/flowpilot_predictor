"""Summarize completed prediction artifacts; no fitting or model selection on test."""

import argparse
import csv
import json
from pathlib import Path

NAMES = ("q10", "q50", "q90", "q99")
TAUS = (0.1, 0.5, 0.9, 0.99)
ALGORITHMS = ("empirical", "ewma", "cluster", "qrf", "lightgbm")


def emit_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_row(meta, metrics):
    row = {
        **meta,
        "rows": metrics["rows"],
        "task_groups": metrics["task_groups"],
        "supported_rows": metrics["supported_rows"],
        "unsupported_rows": metrics["unsupported_rows"],
        "q50_mae_ms": metrics.get("q50_mae_ms"),
        "mean_pinball_ms": metrics.get("mean_pinball_ms"),
        "central_80_coverage": metrics.get("central_80_coverage"),
        "central_80_width_ms": metrics.get("central_80_width_ms"),
        "q99_exceedances": metrics.get("q99_exceedances"),
        "q99_mean_excess_ms": metrics.get("q99_mean_excess_ms"),
        "q99_low_support_fraction": metrics.get("q99_low_support_fraction"),
        "fallback_fraction": metrics["fallback_fraction"],
        "pipeline_predict_p95_ms": (metrics.get("predict_ms") or {}).get("p95"),
    }
    for q, tau in zip(NAMES, TAUS):
        coverage = metrics.get("coverage", {}).get(q)
        row[q + "_pinball_ms"] = metrics.get("pinball_ms", {}).get(q)
        row[q + "_coverage"] = coverage
        row[q + "_coverage_error_pp"] = None if coverage is None else 100 * (coverage - tau)
    return row


def summarize(root):
    expected = (
        [root / "train" / a for a in ALGORITHMS]
        + [
            root / "final" / (a + "_test_" + suffix)
            for a in ALGORITHMS
            for suffix in ("raw", "calibrated")
        ]
        + [root / "online" / s for s in ("ewma_forward_fit", "ewma_forward_online")]
    )
    reports = root / "reports"
    reports.mkdir(exist_ok=False)
    overall = []
    tools = []
    benchmarks = []
    streams = []
    for directory in expected:
        manifest = json.loads((directory / "manifest.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        if manifest["status"] != "complete" or manifest["smoke_only"]:
            raise ValueError("Incomplete/smoke artifact: " + str(directory))
        meta = {
            "experiment": str(directory.relative_to(root)),
            "algorithm": manifest["algorithm"],
            "training_split": manifest["training_split"],
            "evaluation_split": manifest["evaluation_split"],
            "evaluation_mode": manifest["evaluation_mode"],
            "calibration_version": manifest.get("calibration_version"),
        }
        row = metric_row(meta, metrics["micro"])
        row["tool_macro_mean_pinball_ms"] = metrics["tool_macro_mean_pinball_ms"]
        row["tools_total"] = metrics["tools_total"]
        row["tools_scored"] = metrics["tools_scored"]
        row["macro_excludes_unsupported_tools"] = metrics["macro_excludes_unsupported_tools"]
        for q in NAMES:
            row[q + "_tool_macro_pinball_ms"] = (metrics["tool_macro_pinball_ms"] or {}).get(q)
        row["bootstrap"] = json.dumps(metrics.get("bootstrap"))
        overall.append(row)
        for name, m in metrics["by_tool"].items():
            tools.append(metric_row({**meta, "tool_group": name}, m))
        for name, m in metrics["by_adapter"].items():
            benchmarks.append(metric_row({**meta, "benchmark": name}, m))
        streams.append((directory, meta))
    emit_csv(reports / "overall.csv", overall)
    emit_csv(reports / "by_tool.csv", tools)
    emit_csv(reports / "by_benchmark.csv", benchmarks)
    prediction_fields = [
        "experiment",
        "algorithm",
        "training_split",
        "evaluation_split",
        "evaluation_mode",
        "calibration_version",
        "sample_id",
        "task_group_id",
        "benchmark",
        "tool",
        "request_id",
        "tool_call_id",
        "y_ms",
        "q10_ms",
        "q50_ms",
        "q90_ms",
        "q99_ms",
        "supported",
        "fallback_reason",
        "execution_status",
    ]
    with (reports / "predictions.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_fields)
        writer.writeheader()
        for directory, meta in streams:
            with (directory / "predictions.jsonl").open() as predictions:
                for line in predictions:
                    r = json.loads(line)
                    duration = r["prediction"]["duration_ms"] or {}
                    writer.writerow(
                        {
                            **meta,
                            "sample_id": r["sample_id"],
                            "task_group_id": r["task_group_id"],
                            "benchmark": r["adapter"],
                            "tool": r["tool"],
                            "request_id": r["envelope"]["valid_for"]["request_id"],
                            "tool_call_id": r["envelope"]["per_call"][0]["tool_call_id"],
                            "y_ms": r["y_ms"],
                            **{q + "_ms": duration.get(q) for q in NAMES},
                            "supported": r["score"] is not None,
                            "fallback_reason": r["prediction"]["fallback"]["reason"],
                            "execution_status": r["execution_status"],
                        }
                    )
    latencies = [json.loads((root / "latency" / (a + ".json")).read_text()) for a in ALGORITHMS]
    emit_csv(
        reports / "isolated_latency.csv",
        [
            {
                "algorithm": r["algorithm"],
                "threads": r["threads"],
                **r["overall"],
                "scope": r["scope"],
            }
            for r in latencies
        ],
    )
    frozen = json.loads((root / "frozen_before_test.json").read_text())

    def fmt(x, pct=False):
        return "N/A" if x is None else f"{x * 100:.2f}%" if pct else f"{x:.3f}"

    lines = [
        "# 27B 工具 RTT 五算法结果",
        "",
        f"Tune 预先选出的候选：**{frozen['tune_candidate']}**。没有根据 test 改选、重训或自动部署。",
        "",
        "主实验 fit 含 QuixBugs 辅助数据；校准和测试仅包含 Hotpot/BrowseComp/LCB。",
        "以下误差单位为 ms。覆盖率应接近名义 10%/50%/90%/99%，不能视为越高越好。",
        "",
        "| 方法 | 校准 | 宏 pinball↓ | Q50 MAE↓ | Q10覆盖 | Q50覆盖 | Q90覆盖 | Q99覆盖 | 支持行/总行 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        if row["evaluation_split"] != "test":
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    row["algorithm"],
                    "是" if row["calibration_version"] else "否",
                    fmt(row["tool_macro_mean_pinball_ms"]),
                    fmt(row["q50_mae_ms"]),
                    *[fmt(row[q + "_coverage"], True) for q in NAMES],
                    str(row["supported_rows"]) + "/" + str(row["rows"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "逐工具、逐 benchmark、Q99 超出、80% 区间宽度及逐条预测见同目录 CSV；任务组 bootstrap 区间在 overall.csv 和原始 metrics.json。",
        "online 目录是独立 EWMA 对照：只用 fit_primary 训练，在第二轮 tune 顺序回放，先预测再接收真实反馈；不可与合并 tune 总体混比。",
        "isolated_latency.csv 为所有拟合/评估工作退出后逐算法测得的进程内开销；流水线 metrics 中的预测耗时受并行竞争影响。",
        "校准使用与旧实验相同的全局毫秒残差偏移与单调重排，无严格逐工具或有限样本覆盖保证；Q99 需结合每工具样本量/任务组及超出次数解释。",
        "本次是单工具串行执行 RTT 的离线评价，不代表命中复用、在途复用或联合 KV 调度收益。",
    ]
    (reports / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {"reports": str(reports), "streams": len(overall), "model_deployed": False}, indent=2
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    summarize(p.parse_args().root)
