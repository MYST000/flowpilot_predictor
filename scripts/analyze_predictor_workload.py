"""Describe frozen campaign timings and exact read-only reuse opportunities.

This does not execute tools or simulate a changed scheduler. Reuse totals are
trace-accounting opportunities under an initially empty, unbounded cache.
"""

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


READ_ONLY = {("hotpot", "search"), ("hotpot", "read_document"),
             ("browsecomp", "search"), ("browsecomp", "get_document")}


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def percentile(values, probability):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * probability
    i = int(position)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (position - i)


def timing_summary(tasks):
    total = sum(t["task_s"] for t in tasks)
    tool = sum(t["tool_s"] for t in tasks)
    llm = sum(t["llm_s"] for t in tasks)
    executor = sum(t["executor_s"] for t in tasks)
    shares = [100 * t["tool_s"] / t["task_s"] for t in tasks if t["task_s"] > 0]
    return {
        "tasks": len(tasks), "tool_calls": sum(t["tool_calls"] for t in tasks),
        "llm_requests": sum(t["llm_requests"] for t in tasks),
        "task_sum_s": total, "tool_rtt_sum_s": tool,
        "tool_executor_sum_s": executor, "llm_transport_sum_s": llm,
        "unattributed_sum_s": total - tool - llm,
        "tool_pct_of_task_sum": 100 * tool / total if total else None,
        "llm_pct_of_task_sum": 100 * llm / total if total else None,
        "tool_pct_of_tool_plus_llm": 100 * tool / (tool + llm) if tool + llm else None,
        "task_tool_share_pct": {
            "median": statistics.median(shares) if shares else None,
            "p90": percentile(shares, .9), "p95": percentile(shares, .95),
            "max": max(shares) if shares else None,
            "above_10pct": sum(v > 10 for v in shares),
            "above_20pct": sum(v > 20 for v in shares),
            "above_50pct": sum(v > 50 for v in shares),
        },
        "execution_status": dict(Counter(t["status"] for t in tasks)),
    }


def analyze(campaign):
    summary = json.loads((campaign / "collection_summary.json").read_text())
    if summary.get("status") != "collected":
        raise ValueError("Only frozen, collected campaigns are supported")
    tasks = {}
    for record in summary["records"]:
        result = record["result"]
        job = record["job_id"]
        tasks[job] = {
            "adapter": job.split("--", 1)[0], "status": result["execution_status"],
            "task_s": result["duration_s"], "tool_s": 0., "executor_s": 0.,
            "llm_s": 0., "tool_calls": 0, "llm_requests": 0,
        }
    request_context = {}
    for row in rows(campaign / "prediction_dataset/inputs.jsonl"):
        features = row["features"]
        request_context[(row["source_attempt"], row["request_id"])] = {
            "clock_domain": row.get("clock_domain"),
            "dataset_revision": row.get("dataset_revision"),
            "environment": features.get("environment_known_at_t0"),
            "execution_profile": features.get("tool_execution_profile"),
        }
    for row in rows(campaign / "prediction_dataset/targets.jsonl"):
        task = tasks[Path(row["source_attempt"]).parent.name]
        task["llm_requests"] += 1
        duration = row["labels"].get("llm_transport_duration_ms")
        if duration is not None:
            task["llm_s"] += duration / 1000
    tools = []
    for row in rows(campaign / "tool_timing_dataset/tools.jsonl"):
        label = row["labels"]
        if not label.get("executed"):
            continue
        task = tasks[row["job_id"]]
        task["tool_calls"] += 1
        if label.get("round_trip_ms") is not None:
            task["tool_s"] += label["round_trip_ms"] / 1000
        if label.get("executor_duration_ms") is not None:
            task["executor_s"] += label["executor_duration_ms"] / 1000
        if (row["adapter"], row["call"]["tool_name"]) in READ_ONLY:
            if label.get("execution_status") != "completed":
                continue
            if row.get("tool_start_monotonic_ns") is None or row.get("tool_end_monotonic_ns") is None:
                continue
            context = request_context[(row["source_attempt"], row["request_id"])]
            args = row["call"].get("effective_arguments")
            if args is None:
                args = row["call"].get("arguments_parsed")
            row["reuse_key"] = canonical([row["adapter"], row["call"]["tool_name"], context, args])
            row["observation_digest"] = hashlib.sha256(canonical(row.get("model_observation")).encode()).hexdigest()
            tools.append(row)
    domains = {v["clock_domain"] for v in request_context.values()}
    if len(domains) != 1 or None in domains:
        raise ValueError("Chronological comparison requires one controller clock domain")
    previous = defaultdict(list)
    reuse = defaultdict(lambda: {
        "calls": 0, "historical_exact_candidates": 0,
        "same_task_only_candidates": 0, "cross_task_available_candidates": 0,
        "historical_candidate_rtt_s": 0., "historical_candidate_executor_s": 0.,
        "same_task_only_rtt_s": 0., "cross_task_available_rtt_s": 0.,
        "historical_observation_mismatches": 0,
        "inflight_only_candidates": 0, "inflight_ideal_saved_rtt_s": 0.,
    })
    observations = defaultdict(set)
    key_groups = {}
    for row in sorted(tools, key=lambda r: r["tool_start_monotonic_ns"]):
        key = row["reuse_key"]
        group = row["adapter"] + "/" + row["call"]["tool_name"]
        key_groups[key] = group
        observations[key].add(row["observation_digest"])
        metric = reuse[group]
        metric["calls"] += 1
        start = row["tool_start_monotonic_ns"]
        ended = [p for p in previous[key] if p["tool_end_monotonic_ns"] <= start]
        if ended:
            metric["historical_exact_candidates"] += 1
            rtt = row["labels"]["round_trip_ms"] / 1000
            metric["historical_candidate_rtt_s"] += rtt
            metric["historical_candidate_executor_s"] += row["labels"]["executor_duration_ms"] / 1000
            cross = any(p["job_id"] != row["job_id"] for p in ended)
            prefix = "cross_task_available" if cross else "same_task_only"
            metric[prefix + "_candidates"] += 1
            metric[prefix + "_rtt_s"] += rtt
            last = max(ended, key=lambda p: p["tool_end_monotonic_ns"])
            metric["historical_observation_mismatches"] += last["observation_digest"] != row["observation_digest"]
        elif previous[key]:
            leader = min(previous[key], key=lambda p: p["tool_start_monotonic_ns"])
            metric["inflight_only_candidates"] += 1
            original = row["labels"]["round_trip_ms"] / 1000
            follower_wait = (leader["tool_end_monotonic_ns"] - start) / 1e9
            metric["inflight_ideal_saved_rtt_s"] += original - follower_wait
        previous[key].append(row)
    for group, metric in reuse.items():
        keys = [k for k, g in key_groups.items() if g == group]
        metric["distinct_exact_keys"] = len(keys)
        metric["keys_with_different_observations"] = sum(len(observations[k]) > 1 for k in keys)
        metric["historical_exact_candidate_rate_pct"] = 100 * metric["historical_exact_candidates"] / metric["calls"]
    benchmarks = {}
    for adapter in sorted({t["adapter"] for t in tasks.values()}):
        subset = [t for t in tasks.values() if t["adapter"] == adapter]
        benchmarks[adapter] = {"all": timing_summary(subset),
                               "completed_only": timing_summary([t for t in subset if t["status"] == "completed"])}
        matching = [v for k, v in reuse.items() if k.startswith(adapter + "/")]
        candidate_s = sum(v["historical_candidate_rtt_s"] for v in matching)
        benchmarks[adapter]["exact_readonly_reuse"] = {
            "eligible_calls": sum(v["calls"] for v in matching),
            "historical_candidates": sum(v["historical_exact_candidates"] for v in matching),
            "historical_candidate_rtt_s": candidate_s,
            "ideal_direct_task_sum_reduction_pct": 100 * candidate_s / benchmarks[adapter]["all"]["task_sum_s"],
            "cross_task_available_candidates": sum(v["cross_task_available_candidates"] for v in matching),
            "cross_task_available_rtt_s": sum(v["cross_task_available_rtt_s"] for v in matching),
            "inflight_only_candidates": sum(v["inflight_only_candidates"] for v in matching),
        }
    return {
        "campaign": str(campaign),
        "method": {
            "ratio_denominator": "sum of per-task duration_s, not campaign wall-clock",
            "llm": "llm_transport_duration_ms; includes queue/inference/transport/error wait",
            "tool": "round_trip_ms; executor is a subset, not an additional term",
            "reuse": "exact effective args + adapter/tool + recorded environment/revision/profile; prior result ended before current call started",
            "reuse_assumptions": "same permitted scope and fixed corpus/tool versions; empty unbounded cache; zero lookup/hit/validation cost; fixed recorded trajectory",
            "limitations": "Not observed cache hits or measured speedup. No semantic matching. Code execution/edit excluded. Equal observations are a diagnostic, not a proof of correctness.",
        },
        "benchmarks": benchmarks, "reuse_by_tool": dict(reuse),
        "overall": timing_summary(list(tasks.values())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.campaign.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
