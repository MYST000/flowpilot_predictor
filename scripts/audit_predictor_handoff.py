"""Audit frozen mixed-C4 data for the predictor handoff; no services or training."""
import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("fit", "tune", "calibration", "test")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit():
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All-split inventory and task evaluation only; timing feature analysis remains fit-only.",
        "ready_rule": "environment_batch_duration AND complete_arguments AND no unknown tool AND next request observed AND finite gap >= 0",
        "splits": {},
        "sha256": {},
    }
    groups = {}
    all_selected_code = []
    code_manifest = {r["question_id"]: r for r in rows(ROOT / "data/livecodebench/protocol/livecodebench_v1.jsonl")}
    for split in SPLITS:
        root = ROOT / "runs/campaigns" / ("mixed_c4_v1_" + split)
        paths = {name: root / name for name in (
            "campaign.json", "collection_summary.json", "prediction_dataset/manifest.json",
            "prediction_dataset/inputs.jsonl", "prediction_dataset/targets.jsonl",
            "tool_timing_dataset/tools.jsonl", "tool_timing_dataset/audit.json",
        )}
        manifest, collection = read(paths["campaign.json"]), read(paths["collection_summary.json"])
        tools, targets = rows(paths["tool_timing_dataset/tools.jsonl"]), rows(paths["prediction_dataset/targets.jsonl"])
        inputs = rows(paths["prediction_dataset/inputs.jsonl"])
        report = read(paths["tool_timing_dataset/audit.json"])
        key = lambda r: (r["source_attempt"], r["request_id"])
        input_keys, target_keys = [key(r) for r in inputs], [key(r) for r in targets]
        target_key_set = set(target_keys)
        assert len(set(input_keys)) == len(inputs), (split, "duplicate input key")
        assert len(target_key_set) == len(targets), (split, "duplicate target key")
        assert set(input_keys) == target_key_set, (split, "unmatched inputs/targets")
        tool_keys = [(r["source_attempt"], r["request_id"], r["call"]["tool_call_id"]) for r in tools]
        assert len(set(tool_keys)) == len(tools), (split, "duplicate tool key")
        assert all(key(r) in target_key_set for r in tools), (split, "unmatched tool")
        assert collection["status"] == "collected", split
        assert not report["trace_issues"], (split, report["trace_issues"])
        assert len(collection["records"]) == len(manifest["jobs"]), (split, "missing job")
        groups[split] = {r["task_group_id"] for r in targets}
        assert all(r["research_split"] == split for r in targets + inputs + tools)
        actual = [r for r in tools if r["labels"]["executed"]]
        usable = [r for r in actual if
                  r["labels"]["execution_status"] in ("completed", "execution_error")
                  and finite_nonnegative(r["labels"]["round_trip_ms"])
                  and r["tool_start_monotonic_ns"] is not None
                  and r["tool_end_monotonic_ns"] is not None]
        ready = [r for r in targets if r["masks"]["environment_batch_duration"]
                 and r["masks"]["complete_arguments"]
                 and r["labels"]["has_unknown_tool_call"] is False
                 and r["labels"]["next_request_observed"]
                 and finite_nonnegative(r["labels"]["next_request_prepared_gap_ms"])]
        eval_path = ROOT / "runs/evaluations/mixed_c4_v1_qwen35_9b_json_v2" / split / "task_evaluation_summary.json"
        evaluation = read(eval_path)
        assert evaluation["status"] == "scored", split
        assert evaluation["browsecomp"]["judge_errors"] == 0, split
        selected_code = [code_manifest[j["job_id"].removeprefix("livecodebench--")]
                         for j in manifest["jobs"] if j["adapter"] == "livecodebench"]
        all_selected_code.extend(selected_code)
        result["splits"][split] = {
            "tasks": len(manifest["jobs"]),
            "task_counts": dict(Counter(j["adapter"] for j in manifest["jobs"])),
            "collection_status": collection["status"],
            "peak_active_workers": collection["peak_active_workers"],
            "outcomes": dict(Counter(r["result"]["execution_status"] for r in collection["records"])),
            "tool_rows": len(tools), "tool_status": dict(Counter(r["labels"]["execution_status"] for r in tools)),
            "executed": len(actual), "observed_rtt_labels": len(usable),
            "llm_requests": len(targets), "ready_labels": len(ready),
            "task_groups": len(groups[split]), "ready_task_groups": len({r["task_group_id"] for r in ready}),
            "normal_completion_right_censored": sum(bool(r["labels"]["normal_completion_right_censored"]) for r in actual),
            "trace_issues": report["trace_issues"],
            "controller_clock_domains": sorted({r["clock_domain"] for r in targets}),
            "evaluation": evaluation,
        }
        for path in list(paths.values()) + [eval_path]:
            result["sha256"][str(path.relative_to(ROOT))] = file_hash(path)
    result["cross_split_group_overlap"] = {
        a + "/" + b: len(groups[a] & groups[b]) for a, b in itertools.combinations(SPLITS, 2)
    }
    assert not any(result["cross_split_group_overlap"].values()), "split leakage"
    result["totals"] = {field: sum(s[field] for s in result["splits"].values()) for field in (
        "tasks", "tool_rows", "executed", "observed_rtt_labels", "llm_requests", "ready_labels",
    )}
    result["livecodebench_selection"] = {
        "difficulty": dict(Counter(r["difficulty"] for r in all_selected_code)),
        "platform": dict(Counter(r["platform"] for r in all_selected_code)),
        "contest_date_min": min(r["contest_date"] for r in all_selected_code),
        "contest_date_max": max(r["contest_date"] for r in all_selected_code),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)
    print(json.dumps({"totals": report["totals"], "cross_split_group_overlap": report["cross_split_group_overlap"],
                      "output": str(args.output) if args.output else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
