"""Prepare audited 27B RTT samples. Does not fit, calibrate or score any model."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PLAN = Path(__file__).resolve().parent
sys.path.insert(0, str(PLAN / "code"))
from predictor.data import (
    ENV_TOOLS,
    digest,
    finite,
    index_unique,
    key,
    require,
    rows,
    schema_signature,
    signature,
    write_json,
)

PRIMARY = ("fit", "tune", "calibration", "test")


def assert_disjoint(groups):
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            require(not groups[a] & groups[b], f"task-group leakage: {a}/{b}")


def append_sample(row, inp, target, attempt_data, split, batch):
    label, call = row["labels"], row["call"]
    require(label["execution_status"] in ("completed", "execution_error"), "unobservable end")
    require(finite(label["round_trip_ms"]), "invalid RTT")
    require(
        label["arguments_json_valid"] and isinstance(call["arguments_parsed"], dict),
        "invalid arguments",
    )
    responses, starts, ends, schemas = attempt_data
    response, start, end = (
        responses[row["request_id"]],
        starts[call["tool_call_id"]],
        ends[call["tool_call_id"]],
    )
    domain = inp["clock_domain"]
    require(
        all(e["clock_domain"] == domain for e in (response, start, end)), "clock domain mismatch"
    )
    t1, begin, finish = (e["monotonic_ns"] for e in (response, start, end))
    require(t1 <= begin <= finish, "invalid lifecycle")
    require(
        begin == row["tool_start_monotonic_ns"] and finish == row["tool_end_monotonic_ns"],
        "event/export mismatch",
    )
    # RTT remains the exporter-defined client endpoint; it is not recomputed from other clocks.
    f = inp["features"]
    t0 = f["monotonic_ns"]
    require(t0 <= t1, "future snapshot")
    adapter, tool = row["adapter"], call["tool_name"]
    require(
        adapter in ENV_TOOLS and tool in ENV_TOOLS[adapter] and tool in schemas,
        "unsupported tool schema",
    )
    profile, env = f["tool_execution_profile"], f["environment_known_at_t0"]
    require(profile["intra_task_tool_concurrency"] == 1, "only serial supported")
    history = []
    for h in f["prior_tool_executions"]:
        prior_end = ends[h["tool_call_id"]]
        require(
            prior_end["clock_domain"] == domain and prior_end["monotonic_ns"] <= t0,
            "future history",
        )
        if h["tool_name"] == tool and finite(h.get("round_trip_ms")):
            history.append(
                {"rtt_ms": h["round_trip_ms"], "failed": h["execution_status"] != "completed"}
            )
    load = f.get("t0_features", {}).get("client_load", {})
    sampled = load.get("sampled_monotonic_ns")
    require(sampled is None or sampled <= t0, "future load")
    arguments = dict(call["arguments_parsed"])
    for name, prop in schemas[tool].get("parameters", {}).get("properties", {}).items():
        if name not in arguments and "default" in prop:
            arguments[name] = prop["default"]
    # Same contract as v2: full proposed response actions, never successful-action count.
    batch_count = sum(a["tool_name"] in ENV_TOOLS[adapter] for a in target["labels"]["actions"])
    context = {
        "backend_id": env.get("backend", adapter),
        "backend_version": signature(
            {
                "environment": env,
                "dataset_revision": inp["dataset_revision"],
                "actor": inp["replica_id"],
                "profile": profile,
            }
        ),
        "tool_schema_version": schema_signature(schemas[tool]),
        "tool_name": tool,
        "arguments": arguments,
        "configured_timeout_ms": profile["tool_timeout_seconds"] * 1000,
        "batch_index": call["batch_index"],
        "batch_size": batch_count,
        "execution_mode": "serial",
        "resolution": "LOCAL_ONLY",
        "history": history[-64:],
        "history_snapshot_age_ms": (t1 - t0) / 1e6,
        "load": {
            k: load.get(k)
            for k in ("tool_inflight", "llm_inflight", "active_sessions", "max_sessions")
        },
        "load_snapshot_age_ms": None if sampled is None else (t1 - sampled) / 1e6,
        "remaining_budget_t0_s": f["budget_at_t0"].get("remaining_seconds"),
    }
    return {
        "sample_id": signature([*key(row), call["tool_call_id"]]),
        "source_attempt": row["source_attempt"],
        "request_id": row["request_id"],
        "tool_call_id": call["tool_call_id"],
        "task_group_id": row["task_group_id"],
        "task_id": row["task_id"],
        "split": split,
        "source_research_split": row["research_split"],
        "source_batch": batch,
        "adapter": adapter,
        "clock_domain": domain,
        "as_of_ns": t1,
        "predict_seq": response["seq"],
        "dispatch_ns": begin,
        "observed_ns": finish,
        "observe_seq": end["seq"],
        "context": context,
        "labels": {
            "round_trip_ms": label["round_trip_ms"],
            "executor_duration_ms": label.get("executor_duration_ms"),
            "execution_status": label["execution_status"],
            "normal_completion_right_censored": label["normal_completion_right_censored"],
        },
    }


def prepare(output, protocol_path):
    protocol = json.loads(protocol_path.read_text())
    require(
        protocol["quixbugs_role"] in ("auxiliary_fit", "audit_only"),
        "resolve QuixBugs protocol before preparation",
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 2,
        "stage": "T1_llm_response_proxy",
        "splits": {},
        "protocol_sha256": digest(protocol_path),
        "source_sha256": {},
        "quantiles": protocol["quantiles"],
        "quality_filter": False,
        "quixbugs_role": protocol["quixbugs_role"],
        "source_batches": [],
        "limitations": [
            "T1 is llm_response proxy; load/history observed at T0 with age",
            "RTT from dispatch to client return; execute/serial only; no cache-hit/follower labels",
            "environment/dataset/profile identity preserved; backend versions never force-merged",
            "QuixBugs historical_dev is auxiliary only; absent from calibration/test",
            "Group-disjoint retrospective test is not a chronological deployment backtest",
            "No task-quality labels enter features or sample selection",
        ],
    }
    source_hashes = manifest["source_sha256"]

    def record(path):
        path = Path(path)
        source_hashes[str(path)] = digest(path)

    record(protocol_path)
    buckets = defaultdict(list)
    groups = defaultdict(set)
    tasks = set()
    task_audit = []
    seen_samples = set()
    for source in protocol["sources"]:
        root = Path(source["root"])
        split = source["research_split"]
        batch = source["batch"]
        campaign_path = root / "campaign.json"
        record(campaign_path)
        campaign = json.loads(campaign_path.read_text())
        require(len(campaign["jobs"]) == source["tasks"], "task count changed")
        require(
            campaign["concurrency"] == 4 and campaign["intra_task_tool_concurrency"] == 1,
            "concurrency changed",
        )
        record(root / "collection_summary.json")
        require(
            json.loads((root / "collection_summary.json").read_text())["status"] == "collected",
            "incomplete collection",
        )
        jobs = {}
        for job in campaign["jobs"]:
            tc = job["trace_context"]
            job_input = Path(job["input_path"])
            record(job_input)
            require(digest(job_input) == job["input_sha256"], "job input changed")
            identity = tuple(job["job_id"].split("--", 1))
            require(identity not in tasks, "duplicate task across batches")
            tasks.add(identity)
            require(
                tc["research_split"] == split and tc["replica_id"] == protocol["replica_id"],
                "source identity mismatch",
            )
            groups[split].add(tc["task_group_id"])
            jobs[job["attempt_dir"]] = job
            result_path = Path(job["attempt_dir"]) / "result.json"
            record(result_path)
            result = json.loads(result_path.read_text())
            task_audit.append(
                {
                    "benchmark": identity[0],
                    "task_id": identity[1],
                    "research_split": split,
                    "source_batch": batch,
                    "task_group_id": tc["task_group_id"],
                    "source_attempt": job["attempt_dir"],
                    "execution_status": result["execution_status"],
                    "valid_rtt_rows": 0,
                }
            )
        paths = [
            root / p
            for p in (
                "prediction_dataset/inputs.jsonl",
                "prediction_dataset/targets.jsonl",
                "tool_timing_dataset/tools.jsonl",
            )
        ]
        for p in paths:
            record(p)
        inputs = index_unique(rows(paths[0]), key)
        targets = index_unique(rows(paths[1]), key)
        require(inputs.keys() == targets.keys(), "unmatched prediction input/target")
        tools = list(rows(paths[2]))
        index_unique(tools, lambda r: (*key(r), r["call"]["tool_call_id"]))
        attempts = {}
        for inp in inputs.values():
            attempt = inp["source_attempt"]
            require(attempt in jobs, "unknown source attempt")
            require(
                inp["research_split"] == split and inp["replica_id"] == protocol["replica_id"],
                "input identity mismatch",
            )
            require(
                inp["task_group_id"] == jobs[attempt]["trace_context"]["task_group_id"],
                "input group mismatch",
            )
            if attempt in attempts:
                continue
            event_path = Path(attempt) / "events.jsonl"
            record(event_path)
            events = [
                e
                for e in rows(event_path)
                if e["event"] in ("llm_response", "tool_start", "tool_end", "tool_error")
            ]
            responses = index_unique(
                (e for e in events if e["event"] == "llm_response"), lambda e: e["request_id"]
            )
            starts = index_unique(
                (e for e in events if e["event"] == "tool_start"), lambda e: e["tool_call_id"]
            )
            ends = index_unique(
                (e for e in events if e["event"] in ("tool_end", "tool_error")),
                lambda e: e["tool_call_id"],
            )
            snapshot = Path(attempt) / inp["features"]["request_snapshot"]["path"]
            record(snapshot)
            require(
                digest(snapshot) == inp["features"]["request_snapshot"]["sha256"],
                "snapshot hash mismatch",
            )
            blob = json.loads(snapshot.read_text())
            require(
                blob["model"] == protocol["model"]
                and blob["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False,
                "not the intended 27B actor",
            )
            require(
                blob["timeout"] == 900 and blob["max_completion_tokens"] == 4096,
                "actor configuration drift",
            )
            schemas = {t["function"]["name"]: t["function"] for t in blob["tools"]}
            attempts[attempt] = responses, starts, ends, schemas
        count = Counter()
        skipped = Counter()
        per_attempt = Counter()
        for row in tools:
            require(row["research_split"] == split, "tool split mismatch")
            inp, target = inputs[key(row)], targets[key(row)]
            require(
                row["task_group_id"] == inp["task_group_id"] == target["task_group_id"],
                "group mismatch",
            )
            if not row["labels"]["executed"]:
                skipped["not_executed_control_or_unexecuted"] += 1
                continue
            sample = append_sample(row, inp, target, attempts[row["source_attempt"]], split, batch)
            require(sample["sample_id"] not in seen_samples, "duplicate sample across sources")
            seen_samples.add(sample["sample_id"])
            buckets[split].append(sample)
            count[row["adapter"] + "/" + row["call"]["tool_name"]] += 1
            per_attempt[row["source_attempt"]] += 1
        for row in task_audit:
            if row["source_attempt"] in jobs:
                row["valid_rtt_rows"] = per_attempt[row["source_attempt"]]
        manifest["source_batches"].append(
            {
                **source,
                "valid_rtt": sum(count.values()),
                "by_tool": dict(count),
                "skipped": dict(skipped),
            }
        )
        print(root.name, dict(count), "skipped", dict(skipped), flush=True)
    require(len(tasks) == 1077, "expected 1077 unique tasks")
    assert_disjoint(groups)
    primary_fit = list(buckets["fit"])
    buckets["fit_primary"] = primary_fit
    buckets["quixbugs_auxiliary"] = buckets.pop("historical_dev")
    if protocol["quixbugs_role"] == "auxiliary_fit":
        buckets["fit"] = primary_fit + [
            {**s, "split": "fit", "training_role": "historical_dev_auxiliary"}
            for s in buckets["quixbugs_auxiliary"]
        ]
    buckets["tune_forward"] = [
        {**s, "split": "tune_forward"}
        for s in buckets["tune"]
        if s["source_batch"] == "native27b_c4_increment_v2"
    ]
    require(
        max(s["observed_ns"] for s in primary_fit)
        < min(s["as_of_ns"] for s in buckets["tune_forward"]),
        "online split time reversal",
    )
    domains = {s["clock_domain"] for k in PRIMARY for s in buckets[k]}
    require(len(domains) == 1, "merged data crosses incomparable clock domains")
    for split, samples in sorted(buckets.items()):
        path = output / (split + ".jsonl")
        with path.open("w") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
        manifest["splits"][split] = {
            "rows": len(samples),
            "task_groups": len({s["task_group_id"] for s in samples}),
            "by_tool": dict(
                Counter(s["adapter"] + "/" + s["context"]["tool_name"] for s in samples)
            ),
            "source_research_splits": dict(Counter(s["source_research_split"] for s in samples)),
            "min_as_of_ns": min(s["as_of_ns"] for s in samples),
            "max_observed_ns": max(s["observed_ns"] for s in samples),
            "sha256": digest(path),
        }
    audit_path = output / "task_audit.jsonl"
    audit_path.write_text("".join(json.dumps(r) + "\n" for r in task_audit))
    manifest["task_audit_sha256"] = digest(audit_path)
    manifest["unique_tasks"] = len(tasks)
    manifest["unique_valid_rtt_rows"] = len(seen_samples)
    manifest["tasks_without_valid_rtt"] = [r for r in task_audit if r["valid_rtt_rows"] == 0]
    manifest["group_overlap"] = 0
    manifest["clock_domain"] = next(iter(domains))
    manifest["online_comparison"] = {
        "fit_split": "fit_primary",
        "evaluation_split": "tune_forward",
        "exclude_late_auxiliary_fit": True,
    }
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "prepared": str(output),
                "unique_tasks": len(tasks),
                "unique_valid_rtt_rows": len(seen_samples),
                "split_rows": {s: m["rows"] for s, m in manifest["splits"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PLAN / "protocol.json")
    args = parser.parse_args()
    prepare(args.output, args.protocol)
