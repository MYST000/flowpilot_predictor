"""Select a bounded live calibrator on tune, then evaluate delayed test feedback."""

import copy
import importlib.util
import itertools
import json
import math
import os
import sys
import time
import types
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PLAN = Path(__file__).resolve().parent
PREVIOUS_PLAN = PLAN.parent / "native27b_calibration_online_20261001"
BASE_PLAN = PLAN.parent / "native27b_rtt_1077_20261001"
sys.path.insert(0, str(BASE_PLAN / "code"))

from predictor import NAMES
from predictor.data import digest, load_split, require, signature, write_json
from predictor.events import event_order
from predictor.evaluation import report

# Reuse the previously reviewed scoring/bootstrap helpers, not its live source loader.
spec = importlib.util.spec_from_file_location("_prior_calibration_experiment", PREVIOUS_PLAN / "experiment.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

bridge = Path("/root/flowpilot_predictor/flowpilot_predictor_bridge")
package = types.ModuleType("_realtime_quantile_source")
package.__path__ = [str(bridge)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(package.__name__ + ".quantile_online", bridge / "quantile_online.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Corrector = module.OnlineQuantileCalibration


def now():
    return datetime.now(timezone.utc).isoformat()


def cached_records(path):
    return [json.loads(line) for line in path.open()]


def replay(samples, raw_records, config):
    source = {r["sample_id"]: r for r in raw_records}
    require(set(source) == {s["sample_id"] for s in samples}, "Cached prediction pairing mismatch")
    corrector = Corrector(**config)
    pending, output = {}, []
    updates, peak = 0, 0
    for ns, _, _, kind, _, i in event_order(samples):
        s = samples[i]
        c = s["context"]
        if kind == "predict":
            start = time.perf_counter_ns()
            snapshot = corrector.snapshot(c)
            raw = source[s["sample_id"]]["prediction"]["duration_ms"]
            corrected = corrector.apply(raw, snapshot)
            apply_ms = (time.perf_counter_ns() - start) / 1e6
            r = copy.deepcopy(source[s["sample_id"]])
            r["raw_duration_ms"] = raw
            r["prediction"]["duration_ms"] = corrected
            r["prediction"]["online_state_version"] = snapshot.version
            r["online_calibration"] = corrector.metadata(snapshot)
            r["online_snapshot"] = asdict(snapshot)
            r["predict_ms"] += apply_ms
            pending[i] = r
            peak = max(peak, len(pending))
        else:
            r = pending.pop(i)
            require(r["y_ms"] == s["labels"]["round_trip_ms"], "Feedback target changed")
            helpers.rescore(r)
            eligible = (c["resolution"] in {"LOCAL_ONLY", "LOCAL_LEADER"}
                        and s["labels"]["execution_status"] in {"completed", "execution_error"}
                        and ns - s["as_of_ns"] < 600e9 and r["raw_duration_ms"] is not None)
            start = time.perf_counter_ns()
            if eligible:
                corrector.observe(c, r["raw_duration_ms"], r["y_ms"], s["task_group_id"])
                r["feedback_action"] = "updated"
                updates += 1
            else:
                r["feedback_action"] = "ignored"
            r["update_ms"] = (time.perf_counter_ns() - start) / 1e6
            output.append(r)
    require(not pending, "Unresolved feedback")
    return output, {
        "updates": updates, "rows": len(samples), "peak_pending_predictions": peak,
        "active_at_prediction": {n: sum(n in r["online_calibration"]["active_quantiles"] for r in output) for n in NAMES},
        "final_states": [{"key": list(k), **asdict(v)} for k, v in sorted(corrector._state.items())],
    }


def main():
    protocol = json.loads((PLAN / "protocol.json").read_text())
    out, prior, data = (Path(protocol[k]) for k in ["output", "prior_experiment", "prepared_data"])
    out.mkdir(parents=True, exist_ok=False)
    state = {"started_at": now(), "status": "running", "stage": "selection", "pid": os.getpid()}
    write_json(out / "status.json", state)
    try:
        source_paths = [PLAN / "protocol.json", Path(__file__), bridge / "quantile_online.py",
                        prior / "forward_lightgbm" / "model.joblib", data / "manifest.json"]
        source_hashes = {str(p): digest(p) for p in source_paths}
        snapshot_dir = out / "deployment_snapshot"; snapshot_dir.mkdir()
        for p in source_paths[:3]:
            (snapshot_dir / p.name).write_bytes(p.read_bytes())
        write_json(out / "protocol_frozen.json", {"frozen_at": now(), "protocol": protocol, "source_sha256": source_hashes})
        fit = load_split(data, "fit_primary")
        tune = load_split(data, "tune_forward")
        baseline_tune = cached_records(prior / "online" / "tune_forward_diagnostic" / "frozen" / "predictions.jsonl")
        helpers.ensure_forward(fit, tune)
        candidates = []
        for alpha, shrink in itertools.product(protocol["candidate_grid"]["alpha"], protocol["candidate_grid"]["shrinkage_rows"]):
            config = dict(protocol["fixed"], alpha=alpha, shrinkage_rows=shrink)
            records, audit = replay(tune, baseline_tune, config)
            summary = report(records, repeats=0)
            candidates.append({"config": config, "tool_macro_mean_pinball_ms": summary["tool_macro_mean_pinball_ms"],
                               "micro": summary["micro"], "audit": audit})
        best = min(candidates, key=lambda c: (c["tool_macro_mean_pinball_ms"], c["config"]["alpha"], c["config"]["shrinkage_rows"]))
        write_json(out / "selection.json", {"selected_at": now(), "selected": best,
                     "candidates": candidates, "selection_split": "tune_forward", "test_labels_used": False,
                     "frozen_comparator": report(baseline_tune, repeats=0)})
        write_json(out / "runtime_options.json", {"online": True, "online_method": "quantile_residual_v1", "online_config": best["config"]})
        print(json.dumps({"selected": best["config"], "tune_macro_loss": best["tool_macro_mean_pinball_ms"]}), flush=True)
        state["stage"] = "test"; write_json(out / "status.json", state)
        cal = [s for s in load_split(data, "calibration") if s["source_batch"] == "native27b_c4_increment_v2"]
        test = [s for s in load_split(data, "test") if s["source_batch"] == "native27b_c4_increment_v2"]
        helpers.disjoint_groups(fit, tune, cal, test)
        all_samples = tune + cal + test
        helpers.ensure_forward(fit, all_samples)
        raw_all = cached_records(prior / "online" / "test_forward_warm" / "frozen_full_stream" / "predictions.jsonl")
        raw_test = cached_records(prior / "online" / "test_forward_cold" / "frozen" / "predictions.jsonl")
        ids = {s["sample_id"] for s in test}
        results, comparisons = {}, {}
        for name, samples, raw in [("warm", all_samples, raw_all), ("cold", test, raw_test)]:
            records, audit = replay(samples, raw, best["config"])
            if name == "warm":
                full_dir = out / "warm_full_stream"; full_dir.mkdir()
                helpers.save_records(full_dir, records)
                records = [r for r in records if r["sample_id"] in ids]
            results[name] = helpers.write_variant(out, name, records, extra={"online_enabled": True, "config": best["config"]})
            write_json(out / name / "replay_audit.json", audit)
            comparisons[name] = helpers.paired_comparison(raw_test, records)
            for tool in results[name]["by_tool"]:
                def selected(rows):
                    return [r for r in rows if r["adapter"] + "/" + r["tool"] == tool]
                comparisons[name + "/" + tool] = helpers.paired_comparison(selected(raw_test), selected(records))
        write_json(out / "paired_comparisons.json", comparisons)
        baseline = report(raw_test, repeats=500, seed=20260927)
        write_json(out / "baseline.json", baseline)
        legacy = json.loads((prior / "online" / "test_forward_warm" / "online" / "metrics.json").read_text())
        lines = ["# 实时逐分位校正实验", "",
            "实时反馈为正式方案的要求。每次真实工具返回都会更新有界残差窗口与状态版本；样本不足的头使用原模型预测。",
            "Q10/Q50/Q90/Q99 分别使用自身的 log 残差分位数；支持度门槛、收缩、平滑和更新限幅在协议中固定。",
            "9 个候选仅在 939 条 tune_forward 上选择，然后在后续 922 条增量 test 上比较；该测试数据此前已查看，本次为探索性分析。",
            "", "选定配置：", "", "```json", json.dumps(best["config"], ensure_ascii=False, indent=2), "```", "",
            "|方案|Q50 MAE ms|Q90损失 ms|Q99损失 ms|Q90覆盖|Q99覆盖|", "|---|---:|---:|---:|---:|---:|"]
        for name, summary in [("冻结对照", baseline), ("原 OnlineBias", legacy), ("新校正连续反馈", results["warm"]), ("新校正冷启动", results["cold"])]:
            m = summary["micro"]
            lines.append(f"|{name}|{m['q50_mae_ms']:.3f}|{m['pinball_ms']['q90']:.3f}|{m['pinball_ms']['q99']:.3f}|{m['coverage']['q90']:.2%}|{m['coverage']['q99']:.2%}|")
        lines += ["", "## 逐工具连续反馈", "", "|工具|冻结Q50 MAE|新校正Q50 MAE|冻结Q90覆盖|新校正Q90覆盖|", "|---|---:|---:|---:|---:|"]
        for tool, b in baseline["by_tool"].items():
            a = results["warm"]["by_tool"][tool]
            lines.append(f"|{tool}|{b['q50_mae_ms']:.3f}|{a['q50_mae_ms']:.3f}|{b['coverage']['q90']:.2%}|{a['coverage']['q90']:.2%}|")
        lines += ["", "校正算法更新每个工具/后端的校正层，LightGBM 树权重不在单次反馈时重训。",
                  "Q99 在本次回放中样本不足时使用基础模型；它仍被记录和评分，不能声称工具级99%保证。",
                  "预测点状态固定，当前反馈只影响后续预测；hit/follower 不进入本地执行 RTT 的校正窗口。",
                  "配对 bootstrap 结果见 paired_comparisons.json；实时配置见 runtime_options.json。", ""]
        (out / "REPORT.md").write_text("\n".join(lines))
        require(all(digest(p) == h for p, h in source_hashes.items()), "Source/model changed during replay")
        state.update(status="complete", stage="complete", exit_code=0, finished_at=now())
        write_json(out / "status.json", state); (out / "exit_code").write_text("0\n")
        print(json.dumps({"status": "complete", "output": str(out)}), flush=True)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), exit_code=1, finished_at=now())
        write_json(out / "status.json", state); (out / "exit_code").write_text("1\n")
        raise


if __name__ == "__main__":
    main()
