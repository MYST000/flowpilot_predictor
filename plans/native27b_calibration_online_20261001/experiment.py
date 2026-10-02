"""Grouped RTT calibration and exact bridge-bias replay on frozen 27B traces."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import types
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PLAN = Path(__file__).resolve().parent
BASE_PLAN = PLAN.parent / "native27b_rtt_1077_20261001"
sys.path.insert(0, str(BASE_PLAN / "code"))

import numpy as np
from threadpoolctl import threadpool_limits

from predictor import NAMES, QUANTILES
from predictor.cli import load_artifact, save_records, versions
from predictor.data import digest, load_split, require, signature, write_json
from predictor.evaluation import evaluate, metrics, report
from predictor.events import event_order
from predictor.features import backend_key, group_key
from predictor.models import monotone

BRIDGE = Path("/root/flowpilot_predictor/flowpilot_predictor_bridge")


def import_bridge_bias():
    """Load actual source with its relative contracts, avoiding SDK startup imports."""
    package_name = "_native27b_bias_source"
    package = types.ModuleType(package_name)
    package.__path__ = [str(BRIDGE)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(package_name + ".online", BRIDGE / "online.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.OnlineBias


OnlineBias = import_bridge_bias()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def identity_key(context):
    return tuple(group_key(context))


class GroupedCalibrator:
    """Quantile-specific support gates; never accesses evaluation outcomes."""

    def __init__(self, records, samples, *, min_rows, min_groups):
        self.min_rows = dict(min_rows)
        self.min_groups = dict(min_groups)
        contexts = {s["sample_id"]: s["context"] for s in samples}
        self.buckets = defaultdict(list)
        self.calibration_ids = set()
        self.calibration_groups = set()
        for r in records:
            if r["score"] is None:
                continue
            c = contexts[r["sample_id"]]
            residual = [r["y_ms"] - r["prediction"]["duration_ms"][n] for n in NAMES]
            entry = (residual, r["task_group_id"])
            self.buckets[("exact_tool_backend", identity_key(c))].append(entry)
            self.buckets[("same_backend_version_pool", tuple(backend_key(c)))].append(entry)
            self.buckets[("global_pool", ())].append(entry)
            self.calibration_ids.add(r["sample_id"])
            self.calibration_groups.add(r["task_group_id"])
        require(bool(self.calibration_ids), "No supported calibration rows")
        self.stats = {}
        for key, entries in self.buckets.items():
            residuals = np.asarray([e[0] for e in entries], float)
            self.stats[key] = {
                "rows": len(entries),
                "task_groups": len({e[1] for e in entries}),
                "offsets_ms": {
                    n: float(np.quantile(residuals[:, j], q))
                    for j, (n, q) in enumerate(zip(NAMES, QUANTILES))
                },
            }
        self.version = "grouped-calibration:" + signature(self.to_dict())[:16]

    def select(self, context, strategy):
        if strategy == "raw":
            return [0.0] * 4, {n: {"level": "raw", "rows": 0, "task_groups": 0} for n in NAMES}
        candidates = [
            ("exact_tool_backend", identity_key(context)),
            ("same_backend_version_pool", tuple(backend_key(context))),
            ("global_pool", ()),
        ]
        offsets, sources = [], {}
        for n in NAMES:
            rows, groups = self.min_rows[n], self.min_groups[n]
            choices = candidates
            if strategy == "pooled":
                choices, rows, groups = candidates[-1:], 1, 1
            elif strategy == "grouped_min32":
                # Direct grouped calibration is the comparator for support gating.
                rows, groups = 32, 4
            elif strategy != "hierarchical_tail_support":
                raise ValueError(strategy)
            chosen = None
            for key in choices:
                stats = self.stats.get(key)
                if stats and stats["rows"] >= rows and stats["task_groups"] >= groups:
                    chosen = (key, stats)
                    break
            if chosen:
                key, stats = chosen
                offset = stats["offsets_ms"][n]
                sources[n] = {
                    "level": key[0], "rows": stats["rows"], "task_groups": stats["task_groups"],
                    "required_rows": rows, "required_task_groups": groups,
                    "offset_ms": offset,
                    "conditional_coverage_guarantee": False,
                }
            else:
                offset = 0.0
                sources[n] = {"level": "raw", "rows": 0, "task_groups": 0,
                              "required_rows": rows, "required_task_groups": groups,
                              "offset_ms": 0.0, "conditional_coverage_guarantee": False}
            offsets.append(offset)
        return offsets, sources

    def apply(self, quantiles, context, strategy):
        offsets, sources = self.select(context, strategy)
        values = np.asarray([quantiles[n] for n in NAMES]) + offsets
        return dict(zip(NAMES, monotone(values).tolist())), sources

    def to_dict(self):
        return {
            "method": "additive_ms_quantile_residual_with_support_gates",
            "min_rows": self.min_rows, "min_task_groups": self.min_groups,
            "rows": len(self.calibration_ids), "task_groups": len(self.calibration_groups),
            "calibration_sample_ids_sha256": signature(sorted(self.calibration_ids)),
            "buckets": [{"level": k[0], "key": list(k[1]), **v}
                        for k, v in sorted(self.stats.items())],
            "guarantee": "empirical; task dependence and pooled fallback prevent a conditional guarantee",
        }


def rescore(record):
    q = record["prediction"]["duration_ms"]
    if q is None:
        record["score"] = None
        return
    errors = record["y_ms"] - np.asarray([q[n] for n in NAMES])
    loss = np.maximum(np.asarray(QUANTILES) * errors, (np.asarray(QUANTILES) - 1) * errors)
    record["score"] = {
        "pinball_ms": dict(zip(NAMES, loss.tolist())),
        "absolute_q50_error": float(abs(errors[1])),
    }


def calibrated_records(raw_records, samples, calibrator, strategy):
    require(not calibrator.calibration_groups & {s["task_group_id"] for s in samples},
            "Calibration/evaluation task overlap")
    contexts = {s["sample_id"]: s["context"] for s in samples}
    records = copy.deepcopy(raw_records)
    for r in records:
        c = contexts[r["sample_id"]]
        r["backend_identity"] = list(identity_key(c))
        if r["prediction"]["duration_ms"] is not None:
            before = time.perf_counter_ns()
            q, sources = calibrator.apply(r["prediction"]["duration_ms"], c, strategy)
            correction_ms = (time.perf_counter_ns() - before) / 1e6
            r["predict_ms"] += correction_ms
            r["prediction"]["duration_ms"] = q
            r["prediction"]["calibration"] = strategy
            r["prediction"]["calibration_version"] = calibrator.version
            r["calibration_sources"] = sources
            r["envelope"]["per_call"][0]["duration_ms"] = q
            rescore(r)
    return records


def replay_bridge(model, samples, *, online, alpha=0.2, feedback_ttl_seconds=600):
    """Predict before observe; overlapping calls retain their own raw Q50."""
    bias = OnlineBias(alpha=alpha)
    pending, records = {}, []
    counters = Counter()
    last_feedback_ns = {}
    peak_pending = 0
    for ns, _, _, kind, _, i in event_order(samples):
        s = samples[i]
        c = s["context"]
        k = identity_key(c)
        if kind == "predict":
            # No label lookup occurs on this branch.
            start = time.perf_counter_ns()
            snapshot = bias.snapshot(c)
            require(last_feedback_ns.get(k, -1) <= ns, "Future feedback reached prediction")
            raw = model.predict(c)
            prediction = copy.deepcopy(raw)
            if online:
                prediction["duration_ms"] = bias.apply(raw["duration_ms"], snapshot)
                prediction["online_state_version"] = snapshot.version
            elapsed = (time.perf_counter_ns() - start) / 1e6
            pending[i] = (raw, prediction, elapsed, snapshot, last_feedback_ns.get(k))
            peak_pending = max(peak_pending, len(pending))
            continue
        require(i in pending, "Feedback before prediction")
        raw, prediction, cost, snapshot, observed_before_prediction = pending.pop(i)
        y = s["labels"]["round_trip_ms"]
        require(type(y) in (int, float) and math.isfinite(y) and y >= 0, "Invalid feedback")
        record = {
            "sample_id": s["sample_id"], "task_group_id": s["task_group_id"],
            "task_id": s.get("task_id"), "adapter": s["adapter"], "tool": c["tool_name"],
            "backend_identity": list(k), "split": s["split"], "source_batch": s["source_batch"],
            "as_of_ns": s["as_of_ns"], "observed_ns": ns, "y_ms": y,
            "execution_status": s["labels"]["execution_status"], "prediction": prediction,
            "raw_duration_ms": raw["duration_ms"], "predict_ms": cost,
            "bias_snapshot": {"bias": snapshot.bias, "version": snapshot.version,
                              "observations": snapshot.observations,
                              "last_observed_ns": observed_before_prediction},
            "update_ms": 0.0,
        }
        rescore(record)  # Score this invocation before it teaches the corrector.
        reason = None
        if c.get("resolution") not in {"LOCAL_ONLY", "LOCAL_LEADER"}:
            reason = "not_local_execution"
        elif record["execution_status"] not in {"completed", "execution_error"}:
            reason = "no_observed_rtt"
        elif ns - s["as_of_ns"] >= feedback_ttl_seconds * 1e9:
            reason = "feedback_expired"
        elif raw["duration_ms"] is None:
            reason = "raw_prediction_not_available"
        if online and reason is None:
            start = time.perf_counter_ns()
            bias.observe(c, raw["duration_ms"]["q50"], y)
            record["update_ms"] = (time.perf_counter_ns() - start) / 1e6
            last_feedback_ns[k] = ns
            counters["updates"] += 1
            record["feedback_action"] = "updated"
        elif online:
            counters["ignored_" + reason] += 1
            record["feedback_action"] = reason
        else:
            record["feedback_action"] = "frozen_mode"
        records.append(record)
    require(not pending, "Unresolved calls")
    audit = {"online": online, "alpha": alpha, "peak_pending_predictions": peak_pending,
             "counters": dict(counters), "rows": len(records),
             "final_state": [{"key": list(k), "bias": v.bias, "version": v.version,
                              "observations": v.observations} for k, v in sorted(bias._state.items())],
             "prediction_model_weights_unchanged": True,
             "feedback_reference": "this invocation's raw Q50 from prediction time"}
    return records, audit


def extended_report(records, repeats, seed):
    result = report(records, repeats=repeats, seed=seed)
    grouped = defaultdict(list)
    for r in records:
        grouped[json.dumps(r["backend_identity"], ensure_ascii=False)].append(r)
    result["by_tool_backend"] = {k: metrics(v) for k, v in grouped.items()}
    result["calibration_fallback_by_quantile"] = {
        n: dict(Counter(r.get("calibration_sources", {}).get(n, {}).get("level", "none")
                        for r in records)) for n in NAMES
    }
    return result


def paired_comparison(before, after, *, repeats=500, seed=20260927):
    left, right = {r["sample_id"]: r for r in before}, {r["sample_id"]: r for r in after}
    require(left.keys() == right.keys(), "Unpaired samples")
    names = ["q50_mae_ms", *[n + "_pinball_ms" for n in NAMES],
             *[n + "_coverage" for n in NAMES]]
    groups = sorted({r["task_group_id"] for r in before})
    gi = {g: j for j, g in enumerate(groups)}
    sums, counts = np.zeros((len(groups), len(names))), np.zeros(len(groups))
    for sid, b in left.items():
        a = right[sid]
        require(a["task_group_id"] == b["task_group_id"], "Group mismatch")
        if a["score"] is None or b["score"] is None:
            continue
        def values(r):
            return np.asarray([r["score"]["absolute_q50_error"],
                               *[r["score"]["pinball_ms"][n] for n in NAMES],
                               *[float(r["y_ms"] <= r["prediction"]["duration_ms"][n]) for n in NAMES]])
        j = gi[b["task_group_id"]]
        sums[j] += values(a) - values(b)
        counts[j] += 1
    require(counts.sum() > 0, "No paired supported rows")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        ids = rng.integers(0, len(groups), len(groups))
        if counts[ids].sum():
            draws.append(sums[ids].sum(axis=0) / counts[ids].sum())
    bounds = np.quantile(draws, [0.025, 0.975], axis=0)
    point = sums.sum(axis=0) / counts.sum()
    return {"direction": "after minus before; negative error/loss is better",
            "rows": int(counts.sum()), "task_groups": len(groups),
            "bootstrap_unit": "task_group", "repeats": repeats,
            "delta": {n: {"estimate": float(point[j]), "ci95": bounds[:, j].tolist()}
                      for j, n in enumerate(names)}}


def write_variant(root, name, records, *, repeats=500, seed=20260927, extra=None):
    out = root / name
    out.mkdir(parents=True, exist_ok=False)
    save_records(out, records)
    summary = extended_report(records, repeats, seed)
    if extra:
        summary["experiment"] = extra
    write_json(out / "metrics.json", summary)
    return summary


def disjoint_groups(*streams):
    groups = [{s["task_group_id"] for s in stream} for stream in streams]
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            require(not a & b, "Task-group overlap between experiment partitions")


def ensure_forward(fit, stream):
    require(len({s["clock_domain"] for s in fit + stream}) == 1, "Clock mismatch")
    end = max(s["observed_ns"] for s in fit)
    require(all(s["as_of_ns"] > end for s in stream), "Future training data in online replay")
    disjoint_groups(fit, stream)


def isolated_latency(model, calibrator, samples, output):
    contexts = defaultdict(list)
    for s in sorted(samples, key=lambda s: s["sample_id"]):
        key = s["adapter"] + "/" + s["context"]["tool_name"]
        if len(contexts[key]) < 64:
            contexts[key].append(s["context"])
    rows = []
    for strategy in ["raw", "pooled", "grouped_min32", "hierarchical_tail_support", "bridge_online_bias"]:
        bias = OnlineBias(alpha=0.2)
        # Stable nonzero state exercises the transform; synthetic values never train a model.
        if strategy == "bridge_online_bias":
            for cs in contexts.values():
                c = cs[0]
                q50 = model.predict(c)["duration_ms"]["q50"]
                bias.observe(c, q50, q50 * 1.2)
        def predict(c):
            p = model.predict(c)
            if p["duration_ms"] is None:
                return
            if strategy == "bridge_online_bias":
                bias.apply(p["duration_ms"], bias.snapshot(c))
            elif strategy != "raw":
                calibrator.apply(p["duration_ms"], c, strategy)
        for cs in contexts.values():
            for c in cs[:3]:
                predict(c)
        times = []
        for _ in range(3):
            for cs in contexts.values():
                for c in cs:
                    start = time.perf_counter_ns()
                    predict(c)
                    times.append((time.perf_counter_ns() - start) / 1e6)
        rows.append({"strategy": strategy, "n": len(times),
                     "p50_ms": float(np.quantile(times, .5)),
                     "p95_ms": float(np.quantile(times, .95)),
                     "p99_ms": float(np.quantile(times, .99))})
    write_json(output, {"scope": "feature extraction + model + correction; excludes RPC/scheduler/replay bookkeeping",
                        "threads": model.config["threads"], "source": "tune contexts; no outcomes used",
                        "results": rows})
    return rows


def write_summary(output, calibration, online, comparisons, latency):
    fields = ["experiment", "variant", "rows", "task_groups", "q50_mae_ms", "tool_macro_mean_pinball_ms",
              *[n + "_coverage" for n in NAMES], *[n + "_pinball_ms" for n in NAMES],
              "central_80_width_ms", "q99_exceedances"]
    rows, tool_rows, backend_rows = [], [], []
    for experiment, variants in [("grouped_calibration", calibration), *online.items()]:
        for variant, summary in variants.items():
            m = summary["micro"]
            row = {"experiment": experiment, "variant": variant,
                   **{k: m.get(k) for k in ["rows", "task_groups", "q50_mae_ms", "central_80_width_ms", "q99_exceedances"]},
                   "tool_macro_mean_pinball_ms": summary["tool_macro_mean_pinball_ms"],
                   **{n + "_coverage": m["coverage"][n] for n in NAMES},
                   **{n + "_pinball_ms": m["pinball_ms"][n] for n in NAMES}}
            rows.append(row)
            for tool, tm in summary["by_tool"].items():
                tool_rows.append({**row, "tool": tool,
                                  **{k: tm.get(k) for k in ["rows", "task_groups", "q50_mae_ms", "central_80_width_ms", "q99_exceedances"]},
                                  "tool_macro_mean_pinball_ms": tm["mean_pinball_ms"],
                                  **{n + "_coverage": tm["coverage"][n] for n in NAMES},
                                  **{n + "_pinball_ms": tm["pinball_ms"][n] for n in NAMES}})
            for key, tm in summary["by_tool_backend"].items():
                backend_id, backend_version, tool_name, tool_schema_version = json.loads(key)
                backend_rows.append({**row, "backend_id": backend_id, "backend_version": backend_version,
                    "tool_name": tool_name, "tool_schema_version": tool_schema_version,
                    **{k: tm.get(k) for k in ["rows", "task_groups", "q50_mae_ms", "central_80_width_ms", "q99_exceedances"]},
                    "tool_macro_mean_pinball_ms": tm["mean_pinball_ms"],
                    **{n + "_coverage": tm["coverage"][n] for n in NAMES},
                    **{n + "_pinball_ms": tm["pinball_ms"][n] for n in NAMES}})
    for filename, columns, content in [("overall.csv", fields, rows), ("by_tool.csv", [*fields, "tool"], tool_rows)]:
        with (output / "reports" / filename).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader(); writer.writerows(content)
    with (output / "reports" / "by_tool_backend.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[*fields, "backend_id", "backend_version", "tool_name", "tool_schema_version"])
        writer.writeheader(); writer.writerows(backend_rows)
    write_json(output / "reports" / "paired_comparisons.json", comparisons)
    lines = ["# 27B 分组校准与 LightGBM 在线反馈实验", "",
             "本实验只回放实际工具 RTT，不启动 LLM/vLLM，不更改或部署调度策略。",
             "原测试集此前已查看；本次为预先固定补充协议下的探索性比较，不能视为新的盲测。", "",
             "## 分组校准", "",
             "冻结原全量 fit 模型，在 1,588 条 calibration 调用上校准，在同一 2,169 条 test 调用上比较。",
             "Q10/Q50 支持门槛 32 调用/4 题目组，Q90 为 100/10，Q99 为 1,000/30。",
             "回退顺序：精确工具与后端版本 → 同后端版本池 → 全局池 → 原始预测。",
             "使用池化回退的尾部分位数仅有总体统计支持，不代表工具级覆盖保证。", "",
             "|方案|Q50 MAE ms|工具宏平均分位损失 ms|Q90覆盖|Q99覆盖|", "|---|---:|---:|---:|---:|"]
    for name, s in calibration.items():
        m = s["micro"]
        lines.append(f"|{name}|{m['q50_mae_ms']:.3f}|{s['tool_macro_mean_pinball_ms']:.3f}|{m['coverage']['q90']:.2%}|{m['coverage']['q99']:.2%}|")
    lines += ["", "### 逐工具 Q90/Q99", "", "|工具|pooled Q90|hierarchical Q90|pooled Q99|hierarchical Q99|", "|---|---:|---:|---:|---:|"]
    for tool, old in calibration["pooled"]["by_tool"].items():
        new = calibration["hierarchical_tail_support"]["by_tool"][tool]
        lines.append(f"|{tool}|{old['coverage']['q90']:.2%}|{new['coverage']['q90']:.2%}|{old['coverage']['q99']:.2%}|{new['coverage']['q99']:.2%}|")
    lines += ["", "## 实时反馈", "",
              "另训练 fit_primary 的 LightGBM，排除晚期 QuixBugs；全部评测事件均晚于训练结束。",
              "直接使用现有 OnlineBias，alpha=0.2，log 偏差限幅 ±log(4)，反馈 TTL=600s。",
              "预测时间点保存状态与本次 raw Q50，工具返回后评分再更新；并发调用不会读到未来反馈。",
              "主比较在增量 test 上，在线状态连续接收此前 tune_forward 和增量 calibration 的真实返回；另列冷启动与 tune 诊断。",
              "模型权重冻结，没有叠加离线校准，没有挑选 alpha。", "",
              "|流|方案|调用数|Q50 MAE ms|Q90损失 ms|Q99损失 ms|Q90覆盖|Q99覆盖|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for stream, variants in online.items():
        for name, s in variants.items():
            m = s["micro"]
            lines.append(f"|{stream}|{name}|{m['rows']}|{m['q50_mae_ms']:.3f}|{m['pinball_ms']['q90']:.3f}|{m['pinball_ms']['q99']:.3f}|{m['coverage']['q90']:.2%}|{m['coverage']['q99']:.2%}|")
    lines += ["", "逐工具结果见 by_tool.csv；以题目组进行配对 bootstrap 的差值区间见 paired_comparisons.json。",
              "覆盖更高也可能伴随区间变宽或尾部损失变大，应同时查看误差、分位损失、区间宽度和置信区间。", "",
              "## 单进程预测开销", "", "|方案|P50 ms|P95 ms|P99 ms|", "|---|---:|---:|---:|"]
    for r in latency:
        lines.append(f"|{r['strategy']}|{r['p50_ms']:.3f}|{r['p95_ms']:.3f}|{r['p99_ms']:.3f}|")
    lines += ["", "开销不包括 RPC、调度和回放记录写入；不是在线服务负载测试。", "",
              "完整协议、代码快照、源文件和模型哈希以及事件回放审计保存在本实验目录。", ""]
    (output / "reports" / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((PLAN / "protocol.json").read_text())
    if not args.execute:
        print(json.dumps(protocol, ensure_ascii=False, indent=2)); return
    output = Path(protocol["output"])
    output.mkdir(parents=True, exist_ok=False)
    status = {"started_at": utc_now(), "status": "running", "stage": "preflight", "pid": os.getpid()}
    def stage(name):
        status["stage"] = name; status["updated_at"] = utc_now()
        write_json(output / "status.json", status)
        print(name, flush=True)
    try:
        (output / "reports").mkdir()
        snapshot = output / "deployment_snapshot"; snapshot.mkdir()
        for name in ["protocol.json", "experiment.py", "test_experiment.py", "run.sh"]:
            (snapshot / name).write_bytes((PLAN / name).read_bytes())
        data, old_run = Path(protocol["prepared_data"]), Path(protocol["base_run"])
        # Preserve the imported implementation for future reproducibility.
        base_code = snapshot / "base_code" / "predictor"; base_code.mkdir(parents=True)
        bridge_code = snapshot / "bridge_code"; bridge_code.mkdir()
        for p in sorted((BASE_PLAN / "code" / "predictor").glob("*.py")):
            (base_code / p.name).write_bytes(p.read_bytes())
        for name in ["online.py", "contracts.py", "runtime.py"]:
            (bridge_code / name).write_bytes((BRIDGE / name).read_bytes())
        (snapshot / "model_config.json").write_bytes((BASE_PLAN / "model_config.json").read_bytes())
        provenance_paths = [*sorted((BASE_PLAN / "code" / "predictor").glob("*.py")),
                            BRIDGE / "online.py", BRIDGE / "contracts.py", BRIDGE / "runtime.py",
                            BASE_PLAN / "model_config.json", data / "manifest.json",
                            *sorted(data.glob("*.jsonl")),
                            old_run / "train" / "lightgbm" / "model.joblib",
                            old_run / "final" / "lightgbm_calibrated" / "model.joblib",
                            *sorted(p for p in snapshot.rglob("*") if p.is_file())]
        hashes = {str(p): digest(p) for p in provenance_paths}
        write_json(output / "provenance.json", {"frozen_at": utc_now(), "source_sha256": hashes,
                                                "dependencies": versions(), "protocol": protocol})
        splits = {n: load_split(data, n) for n in ["fit", "fit_primary", "tune_forward", "calibration", "test", "tune"]}
        disjoint_groups(splits["fit"], splits["calibration"], splits["test"])
        raw_artifact = load_artifact(old_run / "train" / "lightgbm" / "model.joblib", data)
        pooled_artifact = load_artifact(old_run / "final" / "lightgbm_calibrated" / "model.joblib", data)
        require(not raw_artifact["smoke"] and not raw_artifact["calibrated"], "Full raw model required")
        model = raw_artifact["model"]
        config = protocol["grouped_calibration"]
        threads = model.config["threads"]
        with threadpool_limits(limits=threads):
            stage("grouped_calibration")
            cal_records = evaluate(model, splits["calibration"])
            cal_contexts = {s["sample_id"]: s["context"] for s in splits["calibration"]}
            for r in cal_records:
                r["backend_identity"] = list(identity_key(cal_contexts[r["sample_id"]]))
            calibrator = GroupedCalibrator(cal_records, splits["calibration"],
                                          min_rows=config["min_rows"], min_groups=config["min_task_groups"])
            # Establish exact equivalence to the existing pooled baseline.
            require(np.allclose([calibrator.stats[("global_pool", ())]["offsets_ms"][n] for n in NAMES],
                                pooled_artifact["offsets"], rtol=0, atol=1e-9), "Pooled calibration mismatch")
            write_json(output / "grouped_calibrator.json", {"version": calibrator.version, **calibrator.to_dict()})
            save_records(output, cal_records)
            (output / "predictions.jsonl").rename(output / "calibration_predictions.jsonl")
            raw_test = evaluate(model, splits["test"])
            calibration, calibration_records, comparisons = {}, {}, {}
            for strategy in config["strategies"]:
                records = calibrated_records(raw_test, splits["test"], calibrator, strategy)
                calibration_records[strategy] = records
                calibration[strategy] = write_variant(output / "calibration", strategy, records,
                    extra={"role": "post-hoc test follow-up", "model_sha256": hashes[str(old_run / "train" / "lightgbm" / "model.joblib")]})
            # Verify each baseline prediction against the previously saved test output.
            prior_records = [json.loads(line) for line in (old_run / "final" / "lightgbm_test_calibrated" / "predictions.jsonl").open()]
            previous = {r["sample_id"]: r["prediction"]["duration_ms"] for r in prior_records}
            require(all(r["prediction"]["duration_ms"] == previous[r["sample_id"]] for r in calibration_records["pooled"]),
                    "Existing pooled test baseline changed")
            for variant in ["grouped_min32", "hierarchical_tail_support"]:
                comparisons["calibration_pooled_vs_" + variant] = paired_comparison(calibration_records["pooled"], calibration_records[variant])
                for tool in calibration["pooled"]["by_tool"]:
                    def selected(records):
                        return [r for r in records if r["adapter"] + "/" + r["tool"] == tool]
                    comparisons["calibration_pooled_vs_" + variant + "/" + tool] = paired_comparison(
                        selected(calibration_records["pooled"]), selected(calibration_records[variant]))
            stage("forward_lightgbm_training")
            forward_dir = output / "forward_lightgbm"
            command = [sys.executable, "-B", "-u", "-m", "predictor.cli", "train", "--algorithm", "lightgbm",
                       "--data", str(data), "--config", str(BASE_PLAN / "model_config.json"),
                       "--fit-split", "fit_primary", "--tune-split", "tune_forward", "--output", str(forward_dir)]
            env = dict(os.environ, PYTHONPATH=str(BASE_PLAN / "code"), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
            with (output / "training.log").open("w") as log:
                child = subprocess.Popen(command, cwd=BASE_PLAN / "code", env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    require(child.wait() == 0, "Forward training failed; inspect training.log")
                except BaseException:
                    child.terminate(); child.wait(); raise
            forward = load_artifact(forward_dir / "model.joblib", data)
            require(forward["training_split"] == "fit_primary" and not forward["calibrated"], "Wrong forward model")
            require(all(s["adapter"] != "quixbugs" for s in forward["model"].samples), "Late auxiliary leaked")
            cal_future = [s for s in splits["calibration"] if s["source_batch"] == "native27b_c4_increment_v2"]
            test_future = [s for s in splits["test"] if s["source_batch"] == "native27b_c4_increment_v2"]
            tune_future = splits["tune_forward"]
            disjoint_groups(splits["fit_primary"], tune_future, cal_future, test_future)
            require(max(s["observed_ns"] for s in tune_future) < min(s["as_of_ns"] for s in cal_future), "Warmup chronology")
            require(max(s["observed_ns"] for s in cal_future) < min(s["as_of_ns"] for s in test_future), "Test chronology")
            for stream in [tune_future, cal_future, test_future]:
                ensure_forward(splits["fit_primary"], stream)
            write_json(output / "forward_protocol_frozen.json", {
                "frozen_at": utc_now(), "model_sha256": digest(forward_dir / "model.joblib"),
                "alpha": .2, "fit_rows": len(splits["fit_primary"]),
                "fit_end_ns": forward["fit_end_ns"], "diagnostic_rows": len(tune_future),
                "warmup_rows": len(tune_future) + len(cal_future), "test_rows": len(test_future),
                "all_evaluation_after_fit": True, "group_overlap": 0, "alpha_selection": "existing bridge default, no search",
                "test_has_previously_been_inspected": True})
            stage("lightgbm_online_replay")
            online_results = {}
            for stream_name, stream, score_ids in [
                ("tune_forward_diagnostic", tune_future, None),
                ("test_forward_cold", test_future, None),
                ("test_forward_warm", tune_future + cal_future + test_future, {s["sample_id"] for s in test_future}),
            ]:
                ensure_forward(splits["fit_primary"], stream)
                variants, stream_records = {}, {}
                for enabled in [False, True]:
                    name = "online" if enabled else "frozen"
                    records, audit = replay_bridge(forward["model"], stream, online=enabled)
                    if score_ids is not None:
                        replay_dir = output / "online" / stream_name / (name + "_full_stream")
                        replay_dir.mkdir(parents=True)
                        save_records(replay_dir, records)
                        records = [r for r in records if r["sample_id"] in score_ids]
                    stream_records[name] = records
                    variants[name] = write_variant(output / "online" / stream_name, name, records,
                                                   extra={"stream": stream_name, "score_only_test": score_ids is not None})
                    write_json(output / "online" / stream_name / name / "replay_audit.json", audit)
                online_results[stream_name] = variants
                comparisons[stream_name + "_frozen_vs_online"] = paired_comparison(stream_records["frozen"], stream_records["online"])
                for tool in variants["frozen"]["by_tool"]:
                    def selected(records):
                        return [r for r in records if r["adapter"] + "/" + r["tool"] == tool]
                    comparisons[stream_name + "/" + tool] = paired_comparison(selected(stream_records["frozen"]), selected(stream_records["online"]))
            stage("isolated_latency")
            latency = isolated_latency(model, calibrator, splits["tune"], output / "reports" / "isolated_latency.json")
            stage("report")
            write_summary(output, calibration, online_results, comparisons, latency)
        require(all(digest(p) == h for p, h in hashes.items()), "Source/model/data changed during experiment")
        status.update(status="complete", stage="complete", exit_code=0, finished_at=utc_now())
        write_json(output / "status.json", status)
        (output / "exit_code").write_text("0\n")
        print(json.dumps({"status": "complete", "output": str(output)}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        status.update(status="failed", exit_code=1, error=repr(exc), finished_at=utc_now())
        write_json(output / "status.json", status)
        (output / "exit_code").write_text("1\n")
        raise


if __name__ == "__main__":
    main()
