"""Read one adapter attempt and print measured client timing as JSON.

This script starts no model or benchmark and never writes into the attempt.
Only complete, successful C1 serial traces receive an additive decomposition.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def milliseconds(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) and value >= 0 else None
    return None


def operation_key(event, kind):
    if kind == "llm":
        return (event.get("request_id"),)
    return (
        event.get("request_id"),
        event.get("action_event_id") or event.get("tool_call_id"),
        event.get("tool_name"),
    )


def paired_operations(events, kind, issues):
    start_name = "llm_request_prepared" if kind == "llm" else "tool_start"
    terminal_names = (
        {"llm_response", "llm_error"} if kind == "llm" else {"tool_end", "tool_error"}
    )
    terminals = defaultdict(list)
    starts = [event for event in events if event.get("event") == start_name]
    for event in events:
        if event.get("event") in terminal_names:
            terminals[operation_key(event, kind)].append(event)
    rows = []
    seen = set()
    for start in starts:
        key = operation_key(start, kind)
        matches = terminals.pop(key, [])
        if key in seen or len(matches) > 1:
            issues.append(f"Ambiguous {kind} operation identity: {key}")
        seen.add(key)
        terminal = matches[0] if len(matches) == 1 else {}
        rows.append((start, terminal))
    for unmatched in terminals.values():
        issues.append(f"{len(unmatched)} orphan {kind} terminal event(s)")
        rows.extend(({}, terminal) for terminal in unmatched)
    return rows


def aggregate(rows, field):
    values = [row[field] for row in rows]
    observed = [value for value in values if value is not None]
    return {
        "observed_sum_ms": sum(observed),
        "observed_count": len(observed),
        "missing_count": len(values) - len(observed),
        "complete_sum_ms": sum(observed) if len(observed) == len(values) else None,
    }


def campaign_identity(attempt):
    for directory in attempt.parents:
        path = directory / "campaign.json"
        if path.is_file():
            campaign = json.loads(path.read_text())
            if any(
                Path(job["attempt_dir"]).resolve() == attempt
                for job in campaign.get("jobs", [])
            ):
                return {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "concurrency": campaign.get("concurrency"),
                    "intra_task_tool_concurrency": campaign.get(
                        "intra_task_tool_concurrency"
                    ),
                }
    return None


def serial_order(events):
    active = set()
    starts = {"llm_request_prepared": "llm", "tool_start": "tool"}
    ends = {
        "llm_response": "llm",
        "llm_error": "llm",
        "tool_end": "tool",
        "tool_error": "tool",
    }
    for event in events:
        name = event.get("event")
        if name in starts:
            key = (starts[name], operation_key(event, starts[name]))
            if active:
                return False
            active.add(key)
        elif name in ends:
            key = (ends[name], operation_key(event, ends[name]))
            if key not in active:
                return False
            active.remove(key)
    return not active


def summarize(attempt):
    attempt = Path(attempt).resolve(strict=True)
    result_path, event_path = attempt / "result.json", attempt / "events.jsonl"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    events, issues = [], []
    for number, line in enumerate(event_path.read_text().splitlines(), 1):
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("Event is not an object")
            events.append(event)
        except ValueError:
            issues.append(f"Unreadable event at line {number}")
    llm_rows = []
    for index, (start, end) in enumerate(paired_operations(events, "llm", issues), 1):
        duration = (
            milliseconds(end.get("duration_ms"))
            if end.get("event") == "llm_response"
            else None
        )
        llm_rows.append(
            {
                "index": index,
                "request_id": start.get("request_id", end.get("request_id")),
                "terminal_event": end.get("event"),
                "error_type": end.get("error_type"),
                "client_transport_duration_ms": duration,
                "duration_observed": duration is not None,
                "response_received": end.get("event") == "llm_response",
                "start_event_present": bool(start),
                "terminal_event_present": bool(end),
            }
        )
    tool_rows = []
    for index, (start, end) in enumerate(paired_operations(events, "tool", issues), 1):
        name = start.get("tool_name", end.get("tool_name"))
        category = (
            "search"
            if name == "search"
            else "read"
            if name in {"read_document", "get_document"}
            else "other"
        )
        executor = milliseconds(end.get("executor_duration_ms"))
        roundtrip = milliseconds(end.get("round_trip_ms"))
        outcome = end.get("outcome") or {}
        timed_out = bool(end.get("timed_out") or outcome.get("timed_out"))
        failed = (
            end.get("event") == "tool_error"
            or timed_out
            or outcome.get("exit_code", 0) != 0
        )
        tool_rows.append(
            {
                "index": index,
                "tool_name": name,
                "category": category,
                "request_id": start.get("request_id", end.get("request_id")),
                "tool_call_id": start.get("tool_call_id", end.get("tool_call_id")),
                "terminal_event": end.get("event"),
                "executor_duration_ms": executor,
                "round_trip_ms": roundtrip,
                "executor_duration_observed": executor is not None,
                "round_trip_observed": roundtrip is not None,
                "timed_out": timed_out,
                "failed": failed,
                "error_type": end.get("error_type"),
                "start_event_present": bool(start),
                "terminal_event_present": bool(end),
            }
        )
    llm = aggregate(llm_rows, "client_transport_duration_ms")
    tools = {
        category: {
            "calls": len(
                selected := [
                    row
                    for row in tool_rows
                    if category == "all" or row["category"] == category
                ]
            ),
            "executor": aggregate(selected, "executor_duration_ms"),
            "round_trip": aggregate(selected, "round_trip_ms"),
        }
        for category in ("all", "search", "read", "other")
    }
    requests = [
        event for event in events if event.get("event") == "llm_request_prepared"
    ]
    loads = [
        (event.get("t0_features") or {}).get("client_load") or {} for event in requests
    ]
    campaign = campaign_identity(attempt)
    c1 = bool(requests) and all(
        load.get("max_sessions") == 1 and load.get("active_sessions") == 1
        for load in loads
    )
    if campaign is not None:
        c1 = c1 and campaign["concurrency"] == 1
    tool_serial = bool(requests) and all(
        (event.get("tool_execution_profile") or {}).get("intra_task_tool_concurrency")
        == 1
        for event in requests
    )
    serial = serial_order(events)
    total_seconds = milliseconds(result.get("duration_s"))
    total_ms = total_seconds * 1000 if total_seconds is not None else None
    not_sent = [
        event for event in events if event.get("event") == "llm_request_not_sent"
    ]
    reasons = list(issues)
    if not c1:
        reasons.append(
            "Campaign C1 is not confirmed by every request's client load metadata"
        )
    if not tool_serial or not serial:
        reasons.append("Serial non-overlapping LLM/tool execution is not confirmed")
    if total_ms is None:
        reasons.append("Task total duration is missing")
    if (
        result.get("execution_status") != "completed"
        or result.get("artifact_status") != "valid"
        or result.get("trace_quality") != "recorded"
    ):
        reasons.append("Task did not complete with a valid artifact and recorded trace")
    if not any(event.get("event") == "task_start" for event in events) or not any(
        event.get("event") == "task_end" for event in events
    ):
        reasons.append("Task start/end boundary is missing")
    if (
        any(
            event.get("event")
            in {
                "task_error",
                "llm_error",
                "llm_request_not_sent",
                "tool_error",
                "artifact_export_error",
                "tool_not_executed",
            }
            for event in events
        )
        or result.get("errors")
        or result.get("environment_cleanup_error")
        or result.get("conversation_close_error")
    ):
        reasons.append(
            "Failure or unexecuted operation prevents a complete successful decomposition"
        )
    if any(row["failed"] for row in tool_rows):
        reasons.append("A tool timed out or returned a failure")
    if (
        llm["complete_sum_ms"] is None
        or tools["all"]["round_trip"]["complete_sum_ms"] is None
    ):
        reasons.append("LLM duration or tool round-trip duration is missing")
    if any(
        not row["start_event_present"] or not row["terminal_event_present"]
        for row in llm_rows + tool_rows
    ):
        reasons.append("Operation start/terminal pairing is incomplete")
    if result.get("llm_requests") != len(requests) or result.get("tool_calls") != len(
        tool_rows
    ):
        reasons.append("Result operation counts do not match trace events")
    other = None
    if not reasons:
        other = (
            total_ms
            - llm["complete_sum_ms"]
            - tools["all"]["round_trip"]["complete_sum_ms"]
        )
        if other < 0:
            reasons.append("Observed serial components exceed the task total")
            other = None
    return {
        "schema_version": 1,
        "attempt_dir": str(attempt),
        "identity": {
            key: result.get(key)
            for key in (
                "task_id",
                "dataset_id",
                "dataset_revision",
                "run_id",
                "attempt_id",
            )
        },
        "execution_status": result.get("execution_status"),
        "termination_reason": result.get("termination_reason"),
        "task_total_ms": total_ms,
        "task_total_s": total_seconds,
        "counts": {
            "llm_requests_prepared": len(requests),
            "llm_responses": sum(row["response_received"] for row in llm_rows),
            "llm_errors": sum(event.get("event") == "llm_error" for event in events),
            "llm_requests_not_sent": len(not_sent),
            "tools": len(tool_rows),
            "tools_by_name": dict(Counter(row["tool_name"] for row in tool_rows)),
        },
        "llm_client_transport": {"summary": llm, "calls": llm_rows},
        "tools": {"summary": tools, "calls": tool_rows},
        "llm_not_sent_errors": [
            {key: event.get(key) for key in ("logical_request_id", "error_type")}
            for event in not_sent
        ],
        "serial_evidence": {
            "campaign": campaign,
            "c1_confirmed": c1,
            "intra_task_tool_concurrency_one": tool_serial,
            "nonoverlapping_event_order": serial,
        },
        "additive_decomposition": {
            "available": not reasons,
            "unavailable_reasons": reasons,
            "llm_client_transport_ms": llm["complete_sum_ms"] if not reasons else None,
            "tool_client_round_trip_ms": tools["all"]["round_trip"]["complete_sum_ms"]
            if not reasons
            else None,
            "other_ms": other,
        },
        "definitions": {
            "task_total": "run_task duration through artifact export and environment cleanup; excludes model startup, campaign validation, scheduling wait and worker import/setup before run_task",
            "llm_client_transport": "Recorded client _transport_call duration including request preparation, client/network overhead and server waiting/inference; not pure GPU compute time or an exact HTTP-only interval",
            "tool_round_trip": "Client-observed tool invocation duration; executor duration is its internal detail and must not be added again",
            "executor_missing": "null means unobserved, not zero; the native BrowseComp MCP bridge cannot observe server execution time",
            "other": "For a complete successful C1 serial trace only: task total minus summed LLM client transport and tool round trips; includes setup, SDK orchestration, tracing, export and cleanup, not isolated scheduling overhead",
            "failure": "Failed or missing LLM calls have null duration; available measured partial sums are reported but never substituted for a complete additive decomposition",
        },
        "source_sha256": {
            "events.jsonl": hashlib.sha256(event_path.read_bytes()).hexdigest(),
            "result.json": hashlib.sha256(result_path.read_bytes()).hexdigest()
            if result_path.exists()
            else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "attempt",
        type=Path,
        help="One attempt directory containing events.jsonl and result.json",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(args.attempt), ensure_ascii=False, indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
