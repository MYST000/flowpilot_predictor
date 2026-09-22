import json

import pytest

from benchmark_adapters.contracts import CommandResult
from benchmark_adapters.tracing import Budget, BudgetExceeded, TraceRecorder


def test_effective_timeout_and_rejected_call_identity_are_recorded(tmp_path):
    from benchmark_adapters.sdk_bridge import Binding, ContainerExecutor, ShellAction

    class Environment:
        def execute(self, command, *, timeout, max_bytes):
            self.timeout = timeout
            return CommandResult("ok", "", 0, 0)

    env = Environment()
    recorder = TraceRecorder(tmp_path, {"task_id": "q"})
    recorder.pending["code_terminal"].extend(
        [
            dict(
                tool_call_id="c1",
                request_id="r1",
                llm_response_id="response1",
                action_event_id="a1",
            ),
            dict(
                tool_call_id="c2",
                request_id="r1",
                llm_response_id="response1",
                action_event_id="a2",
            ),
        ]
    )
    executor = ContainerExecutor(
        Binding(env, recorder, Budget(1, 2, 100), 3, 1000), "code_terminal"
    )
    executor(ShellAction(command="true", timeout=120))
    with pytest.raises(BudgetExceeded):
        executor(ShellAction(command="false", timeout=120))
    recorder.close()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    start = next(e for e in events if e["event"] == "tool_start")
    assert start["arguments"]["timeout"] == 120
    assert start["effective_timeout_s"] == env.timeout == 3
    rejected = next(e for e in events if e["event"] == "tool_not_executed")
    assert rejected["request_id"] == "r1" and rejected["tool_call_id"] == "c2"
    assert not [e for e in events if e["event"] == "tool_start" and e["tool_call_id"] == "c2"]


def test_budget_snapshot_is_taken_at_the_request_not_after_completion():
    budget = Budget(4, 3, 100)
    budget.request()
    snapshot = budget.snapshot()
    budget.tool()
    assert snapshot["llm_requests_used"] == 1
    assert snapshot["tool_calls_used"] == 0
    assert 0 < snapshot["remaining_s"] <= 100
    assert budget.snapshot()["tool_calls_used"] == 1
