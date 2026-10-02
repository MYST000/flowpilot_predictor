"""Export complete tool observations and timing labels, including partial runs."""

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmark_adapters.prediction_export import build_prediction_rows


def export(root):
    manifest = json.loads((root / "campaign.json").read_text())
    destination = root / "tool_timing_dataset"
    destination.mkdir(exist_ok=False)
    counts = Counter()
    issues = []
    with (destination / "tools.jsonl").open("w") as output:
        for job in manifest["jobs"]:
            attempt = Path(job["attempt_dir"])
            if not (attempt / "events.jsonl").exists():
                issues.append({"job_id": job["job_id"], "reason": "no_trace"})
                continue
            try:
                samples = build_prediction_rows(attempt)
            except Exception as exc:
                issues.append({"job_id": job["job_id"], "reason": str(exc)})
                continue
            events = [
                json.loads(line)
                for line in (attempt / "events.jsonl").read_text().splitlines()
            ]
            starts = {
                e["tool_call_id"]: e for e in events if e["event"] == "tool_start"
            }
            ends = {
                e["tool_call_id"]: e
                for e in events
                if e["event"] in {"tool_end", "tool_error"}
            }
            for sample in samples:
                for action in sample["labels"]["actions"]:
                    cid = action["tool_call_id"]
                    start, end = starts.get(cid, {}), ends.get(cid, {})
                    row = {
                        "schema_version": 1,
                        "job_id": job["job_id"],
                        "adapter": job["adapter"],
                        "request_id": sample["request_id"],
                        "task_id": sample["task_id"],
                        "source_attempt": str(attempt),
                        "research_split": sample["research_split"],
                        "task_group_id": sample["task_group_id"],
                        "call": {
                            k: action[k]
                            for k in (
                                "tool_call_id",
                                "tool_name",
                                "arguments_raw",
                                "arguments_parsed",
                                "effective_arguments",
                                "batch_index",
                            )
                        },
                        "labels": action,
                        "tool_start_monotonic_ns": start.get("monotonic_ns"),
                        "tool_end_monotonic_ns": end.get("monotonic_ns"),
                        "proposal_to_executor_ms": start.get("proposal_to_executor_ms"),
                        "model_observation": end.get("model_observation"),
                        "execution_outcome": end.get("outcome"),
                        "error_type": end.get("error_type"),
                        "environment_batch_span_ms": sample["labels"].get(
                            "environment_batch_span_ms"
                        ),
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[f"{job['adapter']}/{action['execution_status']}"] += 1
                    if action["executed"] and action["executor_duration_ms"] is None:
                        counts[f"{job['adapter']}/missing_executor_timing"] += 1
    report = {
        "counts": dict(counts),
        "trace_issues": issues,
        "training_rule": "T0 features come only from prediction_dataset/inputs.jsonl. This export contains future labels and observations, not a T0 feature file.",
        "timing_rule": "RTT includes executor. RTT minus executor is non-executor overhead, not pure network latency. Missing values remain null.",
    }
    (destination / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.run_dir.resolve()), ensure_ascii=False, indent=2))
