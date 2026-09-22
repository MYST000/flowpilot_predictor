import hashlib
import json

import pytest

from benchmark_adapters.tracing import TraceRecorder, write_json


def make_trace(path, *, capped=False, malformed=False):
    identity = dict(
        task_id="q1",
        dataset_id="livecodebench/code_generation_lite",
        dataset_revision="v",
        split="fit",
        run_id="run",
        attempt_id="attempt-001",
        conversation_id="conv",
        environment_id="env",
    )
    recorder = TraceRecorder(path, identity)
    recorder.environment_tool_names = {"code_terminal", "code_file_editor"}
    write_json(
        path / "public_task.json",
        {
            "task_id": "q1",
            "dataset_id": identity["dataset_id"],
            "revision": "v",
            "split": "fit",
            "public_metadata": {"task_group_id": "lcb/family", "research_split": "fit"},
        },
    )
    recorder.emit("environment_ready", metadata={"backend": "fixture"})
    recorder.request(
        "r1",
        {
            "messages": [{"role": "user", "content": "solve"}],
            "tools": [{"type": "function", "function": {"name": "code_terminal"}}],
            "max_tokens": 16,
        },
        logical_request_id="logical1",
        request_role="actor",
        snapshot_stage="litellm_transport_input",
        budget_at_t0={"remaining_s": 20},
    )
    raw_args = "{bad" if malformed else '{"command":"echo hello", "timeout":120}'
    calls = [
        dict(id="think1", function=dict(name="think", arguments='{"thought":"inspect"}')),
        dict(id="call1", function=dict(name="code_terminal", arguments=raw_args)),
    ]
    response = dict(
        id="response1",
        choices=[dict(finish_reason="tool_calls", message={"tool_calls": calls})],
        usage=dict(prompt_tokens=8, completion_tokens=16 if capped else 8),
    )
    recorder.emit(
        "llm_response",
        request_id="r1",
        llm_response_id="response1",
        duration_ms=125.5,
        response=recorder.blob(response),
    )
    recorder.response_decision("r1", response)
    link = dict(
        request_id="r1",
        llm_response_id="response1",
        tool_call_id="call1",
        action_event_id="action1",
    )
    if malformed:
        recorder.emit(
            "tool_not_executed", tool_name="code_terminal", reason="invalid_action", **link
        )
    else:
        recorder.emit(
            "tool_start",
            tool_name="code_terminal",
            arguments={"command": "echo hello", "timeout": 120, "kind": "ShellAction"},
            effective_timeout_s=3.0,
            **link,
        )
        recorder.emit(
            "tool_end",
            tool_name="code_terminal",
            executor_duration_ms=3010,
            round_trip_ms=3020,
            outcome={
                "clock_domain": "local-process",
                "exit_code": -9,
                "timed_out": True,
                "truncated": False,
            },
            **link,
        )
    recorder.request(
        "r2",
        {"messages": [{"role": "user", "content": "continue"}], "tools": []},
        logical_request_id="logical2",
        request_role="actor",
        snapshot_stage="litellm_transport_input",
    )
    recorder.emit("llm_error", request_id="r2", error_type="Timeout")
    recorder.close()
    return path


def test_export_preserves_every_request_batch_arguments_and_censoring(tmp_path):
    from benchmark_adapters.prediction_export import build_prediction_rows, load_t0_input

    attempt = make_trace(tmp_path / "attempt")
    rows = build_prediction_rows(attempt)
    assert len(rows) == 2
    first, missing = rows
    assert first["labels"]["next_tool_name"] == "think"
    assert first["labels"]["first_is_environment"] is False
    assert first["labels"]["has_environment_tool_call"] is True
    assert first["task_group_id"] == "lcb/family"
    assert first["research_split"] == "fit"
    action = first["labels"]["actions"][1]
    assert action["arguments_parsed"]["timeout"] == 120
    assert action["validated_executor_arguments"]["timeout"] == 120
    assert action["effective_timeout_s"] == 3
    assert action["executor_duration_ms"] == 3010
    assert action["normal_completion_right_censored"] is True
    assert first["labels"]["actions"][0]["executor_duration_ms"] is None
    assert missing["labels"]["next_action_kind"] == "unavailable"
    assert missing["labels"]["has_environment_tool_call"] is None
    assert missing["masks"]["next_action"] is False
    x = load_t0_input(attempt, first)
    assert "labels" not in x and "response_usage" not in x
    assert x["snapshot"]["messages"][0]["content"] == "solve"


def test_output_cap_masks_labels_even_if_provider_reports_tool_calls(tmp_path):
    from benchmark_adapters.prediction_export import build_prediction_rows

    rows = build_prediction_rows(make_trace(tmp_path / "attempt", capped=True))
    assert rows[0]["labels"]["response_complete"] is True
    assert rows[0]["labels"]["output_token_cap_reached"] is True
    assert not rows[0]["masks"]["next_action"]
    assert not rows[0]["masks"]["complete_arguments"]


def test_invalid_arguments_remain_proposals_without_invented_execution(tmp_path):
    from benchmark_adapters.prediction_export import build_prediction_rows

    row = build_prediction_rows(make_trace(tmp_path / "attempt", malformed=True))[0]
    action = row["labels"]["actions"][1]
    assert action["arguments_raw"] == "{bad"
    assert action["arguments_json_valid"] is False
    assert action["execution_status"] == "not_executed"
    assert action["executor_duration_ms"] is None
    assert not row["masks"]["complete_arguments"]


def test_future_changes_do_not_change_t0_features(tmp_path):
    from benchmark_adapters.prediction_export import build_prediction_rows, load_t0_input

    attempt = make_trace(tmp_path / "attempt")
    before = build_prediction_rows(attempt)[0]
    x_before = load_t0_input(attempt, before)
    path = attempt / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events:
        if event["event"] == "tool_end":
            event["executor_duration_ms"] = 2
            event["outcome"]["timed_out"] = False
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    write_json(attempt / "result.json", {"future_secret": "DO_NOT_USE"})
    write_json(attempt / "evaluation.json", {"passed": True})
    after = build_prediction_rows(attempt)[0]
    assert after["features"] == before["features"]
    assert load_t0_input(attempt, after) == x_before


@pytest.mark.parametrize("corruption", ["duplicate_request", "orphan_end", "cross_request", "blob"])
def test_corrupted_linkages_or_blobs_are_rejected(tmp_path, corruption):
    from benchmark_adapters.prediction_export import TraceIntegrityError, build_prediction_rows

    attempt = make_trace(tmp_path / "attempt")
    path = attempt / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == "blob":
        req = next(e for e in events if e["event"] == "llm_request_prepared")
        (attempt / req["request"]["path"]).write_text("{}")
    elif corruption == "duplicate_request":
        event = dict(next(e for e in events if e["event"] == "llm_request_prepared"))
        event["seq"] = events[-1]["seq"] + 1
        event["monotonic_ns"] = events[-1]["monotonic_ns"] + 1
        events.append(event)
    elif corruption == "orphan_end":
        next(e for e in events if e["event"] == "tool_end")["tool_call_id"] = "missing"
    else:
        next(e for e in events if e["event"] == "tool_end")["request_id"] = "r2"
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    with pytest.raises(TraceIntegrityError):
        build_prediction_rows(attempt)


def test_export_writes_separate_inputs_targets_and_hashes(tmp_path):
    from benchmark_adapters.prediction_export import write_prediction_dataset

    attempt = make_trace(tmp_path / "attempt")
    report = write_prediction_dataset([attempt], tmp_path / "export")
    assert report["requests"] == 2
    assert report["eligible_next_action"] == 1
    inputs = [
        json.loads(line) for line in (tmp_path / "export/inputs.jsonl").read_text().splitlines()
    ]
    targets = [
        json.loads(line) for line in (tmp_path / "export/targets.jsonl").read_text().splitlines()
    ]
    assert len(inputs) == len(targets) == 2
    assert all("labels" not in r and "response_usage" not in r for r in inputs)
    assert targets[1]["labels"]["response_available"] is False
    assert report["files"]["inputs.jsonl"]["sha256"]
    assert (
        report["sources"][0]["public_task_sha256"]
        == hashlib.sha256((attempt / "public_task.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "stale_complete",
        "stale_finish_reason",
        "response_id",
        "response_blob_id",
        "mixed_attempt",
        "duplicate_rejection",
        "foreign_rejection",
        "early_rejection",
    ],
)
def test_response_summary_rejection_and_trace_identity_are_canonical(tmp_path, corruption):
    from benchmark_adapters.prediction_export import TraceIntegrityError, build_prediction_rows

    attempt = make_trace(
        tmp_path / "attempt",
        malformed=corruption in {"duplicate_rejection", "foreign_rejection", "early_rejection"},
    )
    path = attempt / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    decision = next(e for e in events if e["event"] == "response_action_summary")
    response_event = next(e for e in events if e["event"] == "llm_response")
    if corruption == "stale_complete":
        decision["response_complete"] = False
    elif corruption == "stale_finish_reason":
        decision["finish_reason"] = "length"
    elif corruption == "response_id":
        response_event["llm_response_id"] = "foreign-response"
    elif corruption == "response_blob_id":
        for event in events:
            if "llm_response_id" in event:
                event["llm_response_id"] = "foreign-response"
    elif corruption == "mixed_attempt":
        next(e for e in events if e["event"] == "llm_request_prepared" and e["request_id"] == "r2")[
            "attempt_id"
        ] = "attempt-999"
    else:
        rejected = next(e for e in events if e["event"] == "tool_not_executed")
        if corruption == "duplicate_rejection":
            duplicate = dict(
                rejected, seq=events[-1]["seq"] + 1, monotonic_ns=events[-1]["monotonic_ns"] + 1
            )
            events.append(duplicate)
        elif corruption == "foreign_rejection":
            rejected["llm_response_id"] = "foreign-response"
        else:
            events.remove(rejected)
            events.insert(events.index(decision), rejected)
            for index, event in enumerate(events, 1):
                event["seq"] = index
                event["monotonic_ns"] = index * 1_000_000
    events.sort(key=lambda e: e["seq"])
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    with pytest.raises(TraceIntegrityError):
        build_prediction_rows(attempt)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_undeclared_tool_and_nonfinite_arguments_are_not_trainable_targets(tmp_path, constant):
    from benchmark_adapters.prediction_export import build_prediction_rows

    attempt = make_trace(tmp_path / "attempt", malformed=True)
    path = attempt / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    rejected = next(e for e in events if e["event"] == "tool_not_executed")
    rejected["tool_name"] = "swe_terminal"
    decision = next(e for e in events if e["event"] == "response_action_summary")
    decision["tool_calls"] = [
        {
            "id": "call1",
            "function": {"name": "swe_terminal", "arguments": '{"timeout": ' + constant + "}"},
        }
    ]
    decision["next_tool_name"] = "swe_terminal"
    response_event = next(e for e in events if e["event"] == "llm_response")
    response_path = attempt / response_event["response"]["path"]
    response = json.loads(response_path.read_text())
    response["choices"][0]["message"]["tool_calls"] = decision["tool_calls"]
    raw = json.dumps(response, allow_nan=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    new_path = attempt / "blobs" / f"{digest}.json"
    new_path.write_bytes(raw)
    response_event["response"] = {"path": f"blobs/{digest}.json", "sha256": digest}
    path.write_text("".join(json.dumps(e) + "\n" for e in events))

    row = build_prediction_rows(attempt)[0]
    action = row["labels"]["actions"][0]
    assert row["labels"]["next_action_kind"] == "unknown_tool"
    assert row["labels"]["has_environment_tool_call"] is None
    assert action["arguments_json_valid"] is False
    assert not row["masks"]["complete_arguments"]


def test_export_rejects_duplicate_sources_and_duplicate_request_identity(tmp_path):
    from benchmark_adapters.prediction_export import TraceIntegrityError, write_prediction_dataset

    first = make_trace(tmp_path / "first")
    with pytest.raises(TraceIntegrityError, match="Duplicate source"):
        write_prediction_dataset([first, first], tmp_path / "duplicate-source")

    second = make_trace(tmp_path / "second")
    with pytest.raises(TraceIntegrityError, match="Duplicate request identity"):
        write_prediction_dataset([first, second], tmp_path / "duplicate-key")


@pytest.mark.parametrize("field", ["task_id", "dataset_id", "revision", "split", "research_split"])
def test_public_task_identity_must_match_trace(tmp_path, field):
    from benchmark_adapters.prediction_export import TraceIntegrityError, build_prediction_rows

    attempt = make_trace(tmp_path / "attempt")
    path = attempt / "public_task.json"
    task = json.loads(path.read_text())
    if field == "research_split":
        task["public_metadata"][field] = "test"
    else:
        task[field] = "foreign"
    write_json(path, task)
    with pytest.raises(TraceIntegrityError, match="public_task"):
        build_prediction_rows(attempt)


def test_llm_duration_labels_are_distinct_and_missing_response_is_null(tmp_path):
    from benchmark_adapters.prediction_export import build_prediction_rows, load_t0_input

    attempt = make_trace(tmp_path / "attempt")
    events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
    request = next(event for event in events if event["event"] == "llm_request_prepared")
    response = next(event for event in events if event["event"] == "llm_response")
    first, missing = build_prediction_rows(attempt)
    assert first["labels"]["llm_transport_duration_ms"] == 125.5
    assert (
        first["labels"]["request_to_response_ms"]
        == (response["monotonic_ns"] - request["monotonic_ns"]) / 1e6
    )
    assert first["labels"]["llm_duration_clock_domain"] == "controller-process-monotonic"
    assert missing["labels"]["llm_transport_duration_ms"] is None
    assert missing["labels"]["request_to_response_ms"] is None
    assert "llm_transport_duration_ms" not in load_t0_input(attempt, first)


@pytest.mark.parametrize("ending", ["tool_end", "tool_error"])
def test_t0_history_contains_only_preceding_tool_measurements(tmp_path, ending):
    from benchmark_adapters.prediction_export import build_prediction_rows, load_t0_input

    attempt = make_trace(tmp_path / "attempt")
    path = attempt / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    end = next(event for event in events if event["event"] == "tool_end")
    if ending == "tool_error":
        end["event"] = ending
        end["error_type"] = "RuntimeError"
        end.pop("outcome")
        end.pop("executor_duration_ms")
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    first, second = build_prediction_rows(attempt)
    assert load_t0_input(attempt, first)["prior_tool_executions"] == []
    history = load_t0_input(attempt, second)["prior_tool_executions"]
    assert len(history) == 1
    previous = history[0]
    assert previous["tool_name"] == "code_terminal"
    assert previous["tool_call_id"] == "call1"
    assert previous["request_id"] == "r1"
    assert previous["round_trip_ms"] == 3020
    if ending == "tool_end":
        assert previous["executor_duration_ms"] == 3010
        assert previous["execution_outcome"]["timed_out"] is True
        assert previous["execution_outcome"]["exit_code"] == -9
    else:
        assert previous["executor_duration_ms"] is None
        assert previous["execution_outcome"] is None
        assert previous["error_type"] == "RuntimeError"
    before = load_t0_input(attempt, first)
    end["round_trip_ms"] = 4000
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    changed_first, changed_second = build_prediction_rows(attempt)
    assert load_t0_input(attempt, changed_first) == before
    assert (
        load_t0_input(attempt, changed_second)["prior_tool_executions"][0]["round_trip_ms"] == 4000
    )


@pytest.mark.parametrize("duration", [-1, "125.5", True])
def test_invalid_llm_transport_durations_are_rejected(tmp_path, duration):
    from benchmark_adapters.prediction_export import TraceIntegrityError, build_prediction_rows

    attempt = make_trace(tmp_path / "attempt")
    path = attempt / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    next(event for event in events if event["event"] == "llm_response")["duration_ms"] = duration
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(TraceIntegrityError, match="duration"):
        build_prediction_rows(attempt)


def test_future_response_and_usage_changes_do_not_enter_t0(tmp_path):
    from benchmark_adapters.prediction_export import build_prediction_rows, load_t0_input

    attempt = make_trace(tmp_path / "attempt")
    before = build_prediction_rows(attempt)[0]
    before_x = load_t0_input(attempt, before)
    path = attempt / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    response_event = next(event for event in events if event["event"] == "llm_response")
    response = json.loads((attempt / response_event["response"]["path"]).read_text())
    response["choices"][0]["message"]["content"] = "FUTURE_RESPONSE_SECRET"
    response["usage"]["completion_tokens"] = 16
    raw = json.dumps(response).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (attempt / "blobs" / f"{digest}.json").write_bytes(raw)
    response_event["response"] = {"path": f"blobs/{digest}.json", "sha256": digest}
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    after = build_prediction_rows(attempt)[0]
    assert before["masks"]["next_action"] is True
    assert after["masks"]["next_action"] is False
    assert load_t0_input(attempt, after) == before_x


def test_environment_and_conversation_identity_can_appear_after_start(tmp_path):
    from benchmark_adapters.prediction_export import build_prediction_rows

    attempt = make_trace(tmp_path / "attempt")
    path = attempt / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    start = dict(events[0], event="task_start")
    start.pop("environment_id")
    start.pop("conversation_id")
    start.pop("metadata")
    events[0].pop("conversation_id")
    events.insert(0, start)
    for index, event in enumerate(events, 1):
        event["seq"] = index
        event["monotonic_ns"] = index * 1_000_000
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    assert len(build_prediction_rows(attempt)) == 2
