"""Prediction boundary tests; sinks here are fixtures, not FlowPilot policies."""

import asyncio
import copy
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from flowpilot_predictor_bridge import (
    CallIdentity,
    PredictorRuntime,
    ToolFeedback,
    ToolPredictionRequest,
)
from flowpilot_predictor_bridge.context import request_from_tool_call
from predictor.data import schema_signature


class Probe:
    def __init__(self):
        self.lock = threading.Lock()
        self.entered = threading.Event()
        self.gate = None
        self.barrier = None
        self.inputs = []
        self.active = 0
        self.maximum = 0
        self.fail = False


class Model:
    name = "fixture"
    version = "fixture-v1"

    def __init__(self, probe):
        self.probe = probe

    def __deepcopy__(self, memo):
        return Model(self.probe)

    def predict(self, context):
        p = self.probe
        with p.lock:
            p.inputs.append((id(self), copy.deepcopy(context)))
            p.active += 1
            p.maximum = max(p.maximum, p.active)
            p.entered.set()
        try:
            if p.barrier:
                p.barrier.wait(timeout=3)
            if p.gate:
                assert p.gate.wait(timeout=5)
            if p.fail:
                raise ValueError("fixture failure")
            if (
                context["backend_version"] != "v1"
                or context["execution_mode"] != "serial"
            ):
                return {
                    "duration_ms": None,
                    "support": {"status": "unsupported"},
                    "fallback": {"used": True, "reason": "unknown_domain"},
                }
            k = context["arguments"].get("scale", 1)
            return {
                "duration_ms": {
                    n: v * k
                    for n, v in zip(
                        ["q10", "q50", "q90", "q99"], [10.0, 50.0, 90.0, 99.0]
                    )
                },
                "support": {"status": "supported"},
                "fallback": {"used": False, "reason": None},
            }
        finally:
            with p.lock:
                p.active -= 1


def request(name="a", *, job="job", tool="search", **changes):
    identity = CallIdentity(job, "line", "request", "tail", "llm-call", name, 1, 1, 1)
    c = {
        "backend_id": "backend",
        "backend_version": "v1",
        "tool_name": tool,
        "tool_schema_version": "s1",
        "arguments": {},
        "history": [],
        "load": {},
        "batch_index": 0,
        "batch_size": 1,
        "execution_mode": "serial",
        "resolution": "UNKNOWN",
    }
    c.update(changes)
    return ToolPredictionRequest(identity, c)


def feedback(req, value=200.0, *, resolution="LOCAL_ONLY", executed=True, **kwargs):
    return ToolFeedback(
        req.identity,
        "event:" + req.identity.job_id + ":" + req.identity.tool_call_id,
        1,
        resolution,
        executed,
        value,
        **kwargs,
    )


@pytest.mark.parametrize("quantile,expected", [("q50", 50.0), ("q90", 90.0)])
def test_parallel_requests_and_selected_scalar(quantile, expected):
    async def run():
        probe = Probe()
        probe.barrier = threading.Barrier(2)
        accepted = []

        async def sink(prior):
            accepted.append(prior)
            return True

        async with PredictorRuntime(
            Model(probe),
            selected_quantile=quantile,
            duration_sink=sink,
            timeout_seconds=3,
        ) as runtime:
            a, b = request("a"), request("b", arguments={"scale": 2})
            runtime.on_llm_response([a, b])
            runtime.resolve(a.identity, "LOCAL_ONLY", version=1)
            runtime.resolve(b.identity, "LOCAL_LEADER", version=2)
            pa, pb = await asyncio.gather(
                runtime.wait(a.identity), runtime.wait(b.identity)
            )
            await runtime.drain_deliveries()
            assert probe.maximum == 2
            assert len({x[0] for x in probe.inputs}) == 2
            assert pa["duration_estimate_ms"] == expected
            assert pb["duration_estimate_ms"] == expected * 2
            priors = {p.identity.tool_call_id: p for p in accepted}
            assert priors["a"].duration_estimate_ms == expected
            assert priors["a"].duration_p50 == 50 and priors["a"].duration_p90 == 90
            assert priors["b"].duration_estimate_ms == expected * 2

    asyncio.run(run())


@pytest.mark.parametrize("resolution", ["HISTORICAL_HIT", "INFLIGHT_FOLLOWER"])
def test_hit_and_follower_still_predict_but_never_publish_local_prior(resolution):
    async def run():
        probe = Probe()
        probe.gate = threading.Event()

        async def sink(_):
            pytest.fail("hit/follower must not receive a local duration prior")

        async with PredictorRuntime(
            Model(probe), duration_sink=sink, timeout_seconds=2
        ) as runtime:
            req = request(resolution=resolution)
            runtime.submit(req)
            runtime.resolve(req.identity, resolution, version=1)
            # Cache handling has returned while the model is still computing.
            assert runtime.snapshot()["native_or_queued"] == 1
            probe.gate.set()
            result = await runtime.wait(req.identity)
            assert result["duration_ms"]["q50"] == 50
            assert req.context["resolution"] == resolution
            assert probe.inputs[0][1]["resolution"] == "LOCAL_ONLY"
            assert (
                runtime.observe(
                    feedback(req, 1, resolution=resolution, executed=False)
                )["status"]
                == "ignored"
            )
            await runtime.drain_deliveries()
            assert runtime.snapshot()["metrics"].get("online_updates", 0) == 0

    asyncio.run(run())


def test_prediction_before_resolution_hands_off_only_when_local_fact_arrives():
    async def run():
        priors = []

        async def sink(prior):
            priors.append(prior)
            return True

        async with PredictorRuntime(Model(Probe()), duration_sink=sink) as runtime:
            req = request()
            await runtime.predict(req)
            assert priors == []
            runtime.resolve(req.identity, "LOCAL_ONLY", version=4)
            await runtime.drain_deliveries()
            assert len(priors) == 1 and priors[0].resolution_version == 4
            runtime.resolve(req.identity, "LOCAL_ONLY", version=4)
            await runtime.drain_deliveries()
            assert len(priors) == 1
            assert runtime.resolve(req.identity, "HISTORICAL_HIT", version=3) is False
            with pytest.raises(ValueError, match="conflicting"):
                runtime.resolve(req.identity, "HISTORICAL_HIT", version=4)

    asyncio.run(run())


def test_feedback_updates_next_prediction_not_original_and_not_other_backend_tool():
    async def run():
        async with PredictorRuntime(Model(Probe()), alpha=0.2, online_method="ewma_log_bias") as runtime:
            a = request("a")
            original = await runtime.predict(a)
            update = runtime.observe(feedback(a, 500))
            assert update == {"status": "updated", "online_state_version": 1}
            assert runtime.observe(feedback(a, 500))["status"] == "duplicate"
            b = await runtime.predict(request("b"))
            assert b["duration_ms"]["q50"] > original["duration_ms"]["q50"]
            assert b["raw_duration_ms"] == original["raw_duration_ms"]
            assert b["online_state_version"] == 1
            assert (await runtime.wait(a.identity)) == original
            other = await runtime.predict(request("c", tool="read"))
            assert other["duration_ms"]["q50"] == 50
            assert other["online_state_version"] == 0
            with pytest.raises(ValueError, match="conflicting"):
                runtime.observe(feedback(a, 600))

    asyncio.run(run())


def test_feedback_before_native_prediction_is_buffered_without_label_leakage():
    async def run():
        probe = Probe()
        probe.gate = threading.Event()
        async with PredictorRuntime(Model(probe), timeout_seconds=2, alpha=0.2, online_method="ewma_log_bias") as runtime:
            req = request()
            runtime.submit(req)
            assert runtime.observe(feedback(req, 500))["status"] == "pending_prediction"
            probe.gate.set()
            first = await runtime.wait(req.identity)
            assert (
                first["online_state_version"] == 0 and first["duration_ms"]["q50"] == 50
            )
            second = await runtime.predict(request("next"))
            assert (
                second["online_state_version"] == 1
                and second["duration_ms"]["q50"] > 50
            )

    asyncio.run(run())


def test_delayed_feedback_uses_each_calls_own_raw_prediction():
    async def run():
        async with PredictorRuntime(Model(Probe()), alpha=1, online_method="ewma_log_bias") as runtime:
            slow = request("slow", arguments={"scale": 10})
            fast = request("fast")
            runtime.on_llm_response([slow, fast])
            await asyncio.gather(
                runtime.wait(slow.identity), runtime.wait(fast.identity)
            )
            runtime.observe(feedback(fast, 500))
            runtime.observe(feedback(slow, 500))
            result = await runtime.predict(request("next"))
            # The slow call was predicted at raw Q50=500, so its residual is zero.
            assert result["duration_ms"]["q50"] == 50
            assert result["online_state_version"] == 2

    asyncio.run(run())


def test_timeout_does_not_release_native_capacity_early():
    async def run():
        probe = Probe()
        probe.gate = threading.Event()
        async with PredictorRuntime(
            Model(probe), max_pending=2, timeout_seconds=0.05
        ) as runtime:
            a, b = request("a"), request("b")
            runtime.on_llm_response([a, b])
            pa, pb = await asyncio.gather(
                runtime.wait(a.identity), runtime.wait(b.identity)
            )
            assert (
                pa["fallback"]["reason"]
                == pb["fallback"]["reason"]
                == "prediction_timeout"
            )
            assert runtime.snapshot()["native_or_queued"] == 2
            c = await runtime.predict(request("c"))
            assert c["fallback"]["reason"] == "overloaded"
            probe.gate.set()
        assert runtime.snapshot()["native_or_queued"] == 0

    asyncio.run(run())


def test_invalidated_call_cannot_write_or_learn_from_late_result():
    async def run():
        probe = Probe()
        probe.gate = threading.Event()
        priors = []

        async def sink(prior):
            priors.append(prior)
            return True

        async with PredictorRuntime(
            Model(probe), duration_sink=sink, timeout_seconds=2
        ) as runtime:
            req = request()
            runtime.submit(req)
            runtime.resolve(req.identity, "LOCAL_ONLY", version=1)
            await asyncio.to_thread(probe.entered.wait, 1)
            runtime.invalidate_line("job", "line")
            assert runtime.observe(feedback(req))["status"] == "ignored"
            probe.gate.set()
            assert (await runtime.wait(req.identity))["duration_ms"] is None
        assert priors == []
        assert runtime.snapshot()["metrics"].get("online_updates", 0) == 0

    asyncio.run(run())


def test_terminal_before_prediction_prevents_handoff_but_allows_feedback():
    async def run():
        probe = Probe()
        probe.gate = threading.Event()

        async def sink(_):
            pytest.fail("terminal tool must not be overwritten")

        async with PredictorRuntime(
            Model(probe), duration_sink=sink, timeout_seconds=2
        ) as runtime:
            req = request()
            runtime.submit(req)
            runtime.resolve(req.identity, "LOCAL_ONLY", version=1)
            runtime.observe(feedback(req))
            probe.gate.set()
            await runtime.wait(req.identity)
            await runtime.drain_deliveries()
            assert runtime.snapshot()["metrics"]["online_updates"] == 1

    asyncio.run(run())


def test_prediction_snapshots_are_isolated_and_input_duplicates_are_checked():
    async def run():
        async with PredictorRuntime(Model(Probe())) as runtime:
            req = request()
            key = runtime.submit(req)
            assert runtime.submit(copy.deepcopy(req)) == key
            changed = replace(req, context={**req.context, "arguments": {"scale": 2}})
            with pytest.raises(ValueError, match="conflicting input"):
                runtime.submit(changed)
            req.context["arguments"]["scale"] = 99
            result = await runtime.wait(key)
            assert result["duration_ms"]["q50"] == 50
            result["duration_ms"]["q50"] = 10000
            assert (await runtime.wait(key))["duration_ms"]["q50"] == 50

    asyncio.run(run())


def test_job_and_attempt_isolation_and_scoped_cancellation():
    async def run():
        async with PredictorRuntime(Model(Probe())) as runtime:
            a, b = request("same", job="A"), request("same", job="B")
            runtime.on_llm_response([a, b])
            await runtime.cancel("request", job_id="A", line_id="line")
            assert (await runtime.wait(a.identity))["duration_ms"] is None
            assert (await runtime.wait(b.identity))["duration_ms"] is not None
            with pytest.raises(ValueError, match="attempt"):
                runtime.observe(replace(feedback(b), execution_attempt=2))

    asyncio.run(run())


def test_quantile_ttl_prevents_stale_handoff_and_bad_sink_is_observable():
    async def run():
        count = []

        async def sink(prior):
            count.append(prior)
            raise RuntimeError("control plane unavailable")

        async with PredictorRuntime(
            Model(Probe()), duration_sink=sink, prediction_ttl_seconds=0.02
        ) as runtime:
            req = request()
            await runtime.predict(req)
            await asyncio.sleep(0.03)
            runtime.resolve(req.identity, "LOCAL_ONLY", version=1)
            await runtime.drain_deliveries()
            assert count == []
        async with PredictorRuntime(Model(Probe()), duration_sink=sink) as runtime:
            req = request()
            await runtime.predict(req)
            runtime.resolve(req.identity, "LOCAL_ONLY", version=1)
            await runtime.drain_deliveries()
            assert runtime.snapshot()["metrics"]["sink_errors"] == 1

    asyncio.run(run())


def test_unknown_model_errors_and_frozen_mode():
    async def run():
        probe = Probe()
        async with PredictorRuntime(Model(probe), online=False) as runtime:
            unsupported = await runtime.predict(
                request("unknown", backend_version="new")
            )
            assert unsupported["duration_ms"] is None
            req = request("known")
            await runtime.predict(req)
            assert runtime.observe(feedback(req))["status"] == "recorded"
            unchanged = await runtime.predict(request("next"))
            assert unchanged["duration_ms"]["q50"] == 50
            probe.fail = True
            failed = await runtime.predict(request("failure"))
            assert failed["fallback"]["reason"] == "model_error:ValueError"
            assert runtime.on_llm_response([]) == []

    asyncio.run(run())


def test_tool_call_builder_uses_actual_arguments_and_original_schema_defaults():
    schema = {"name": "search", "parameters": {"properties": {"top_k": {"default": 5}}}}
    req = request(tool_schema_version=schema_signature(schema))
    call = {
        "id": req.identity.tool_call_id,
        "function": {"name": "search", "arguments": '{"query":"abc"}'},
    }
    built = request_from_tool_call(req.identity, call, req.context, tool_schema=schema)
    assert built.context["arguments"] == {"query": "abc", "top_k": 5}
    assert req.context["arguments"] == {}
    with pytest.raises(ValueError, match="schema version"):
        request_from_tool_call(
            req.identity,
            call,
            req.context,
            tool_schema={**schema, "description": "changed"},
        )
    with pytest.raises(ValueError, match="stage|T1"):
        asyncio.run(
            PredictorRuntime(Model(Probe())).predict(replace(req, stage="llm_request"))
        )


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_feedback_cannot_update_model(value):
    with pytest.raises(ValueError):
        feedback(request(), value)


def test_frozen_real_lightgbm_artifact_and_online_wrapper():
    root = Path(__file__).resolve().parents[1]
    path = root / "runs/predictor_experiments/20260927T170043Z/lightgbm/model.joblib"
    if not path.exists():
        pytest.skip("formal artifact not installed")

    async def run():
        runtime = PredictorRuntime.from_artifact(
            path, root / "runs/predictor_prepared/v1", timeout_seconds=3, online_method="ewma_log_bias"
        )
        async with runtime:
            context = copy.deepcopy(runtime.models[0].contexts[0])
            req = replace(request(), context=context)
            result = await runtime.predict(req)
            assert result["duration_ms"] == result["raw_duration_ms"]
            runtime.observe(feedback(req, result["duration_ms"]["q50"] * 2 + 1))
            next_req = replace(req, identity=replace(req.identity, tool_call_id="next"))
            updated = await runtime.predict(next_req)
            assert updated["duration_ms"]["q50"] > result["duration_ms"]["q50"]
            assert updated["raw_duration_ms"] == result["raw_duration_ms"]
            assert (
                runtime.models[0].state_version == runtime.models[1].state_version == 0
            )

    asyncio.run(run())


def test_actual_handoff_keyword_and_quantile_metadata_are_not_confused():
    from flowpilot_predictor_bridge.flowpilot import FlowPilotDurationSink

    async def run():
        received = []

        async def framework_duration_setter(**kwargs):
            received.append(kwargs)
            return True

        sink = FlowPilotDurationSink(framework_duration_setter)
        async with PredictorRuntime(
            Model(Probe()), selected_quantile="q90", duration_sink=sink
        ) as runtime:
            req = request()
            runtime.submit(req)
            runtime.resolve(req.identity, "LOCAL_ONLY", version=7)
            await runtime.wait(req.identity)
            await runtime.drain_deliveries()
            assert received[0]["duration_estimate_ms"] == 90
            assert received[0]["expected_resolution_version"] == 7
            assert received[0]["metadata"]["duration_p50"] == 50
            assert received[0]["metadata"]["selected_quantile"] == "q90"
            assert received[0]["identity"] == req.identity

    asyncio.run(run())


def test_telemetry_feedback_requires_matching_call_and_rtt_measurement():
    from flowpilot_predictor_bridge.flowpilot import feedback_from_tool_event

    req = request()
    event = {
        name: getattr(req.identity, name)
        for name in (
            "job_id",
            "line_id",
            "tail_request_id",
            "llm_call_id",
            "tool_call_id",
        )
    }
    event.update(
        event_id="finish-a",
        event_kind="finish",
        execution_attempt=1,
        measured_latency_ms=123.0,
    )
    f = feedback_from_tool_event(
        req.identity, event, resolution="LOCAL_LEADER", timing_scope="client_round_trip"
    )
    assert f.round_trip_ms == 123 and f.executed
    with pytest.raises(ValueError, match="RTT"):
        feedback_from_tool_event(
            req.identity,
            event,
            resolution="LOCAL_LEADER",
            timing_scope="executor_duration",
        )
    with pytest.raises(ValueError, match="identity"):
        feedback_from_tool_event(
            req.identity,
            {**event, "tool_call_id": "other"},
            resolution="LOCAL_LEADER",
            timing_scope="client_round_trip",
        )


def test_cancelled_native_runner_keeps_capacity_until_the_real_call_finishes():
    async def run():
        probe = Probe()
        probe.gate = threading.Event()
        async with PredictorRuntime(
            Model(probe), max_pending=2, timeout_seconds=2
        ) as runtime:
            a, b = request("a"), request("b")
            runtime.on_llm_response([a, b])
            await asyncio.to_thread(probe.entered.wait, 1)
            runtime.calls[a.identity].task.cancel()
            assert (await runtime.wait(a.identity))["duration_ms"] is None
            assert runtime.snapshot()["native_or_queued"] == 2
            overloaded = await runtime.predict(request("c"))
            assert overloaded["fallback"]["reason"] == "overloaded"
            probe.gate.set()

    asyncio.run(run())


def test_consumer_rechecks_version_after_async_handoff_starts():
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()
        current = {"version": 1, "terminal": False, "written": None}

        async def sink(prior):
            entered.set()
            await release.wait()
            if current["terminal"] or prior.resolution_version != current["version"]:
                return False
            current["written"] = prior.duration_estimate_ms
            return True

        async with PredictorRuntime(Model(Probe()), duration_sink=sink) as runtime:
            req = request()
            runtime.submit(req)
            runtime.resolve(req.identity, "LOCAL_ONLY", version=1)
            await entered.wait()
            current["terminal"] = True
            runtime.observe(feedback(req))
            release.set()
            await runtime.drain_deliveries()
            assert current["written"] is None
            assert runtime.snapshot()["metrics"]["priors_rejected"] == 1

    asyncio.run(run())


def test_config_selects_q90_without_changing_the_original_model(tmp_path):
    root = Path(__file__).resolve().parents[1]
    model_path = (
        root / "runs/predictor_experiments/20260927T170043Z/lightgbm/model.joblib"
    )
    if not model_path.exists():
        pytest.skip("formal artifact not installed")
    import json

    config = json.loads((root / "configs/predictor/runtime.json").read_text())
    config.update(selected_quantile="q90", timeout_seconds=3)
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(config))

    async def run():
        async with PredictorRuntime.from_config(path) as runtime:
            req = replace(
                request(), context=copy.deepcopy(runtime.models[0].contexts[0])
            )
            result = await runtime.predict(req)
            assert result["duration_estimate_ms"] == result["duration_ms"]["q90"]
            assert result["online_state_version"] == 0

    asyncio.run(run())


def sdk_recorder_class():
    # Import the actual integration recorder without importing heavyweight SDK deps.
    import importlib.util
    import os

    path = (
        Path(
            os.environ.get(
                "FLOWPILOT_SDK_REPO",
                str(
                    Path(__file__).resolve().parents[2]
                    / "flowpilot_integration_20260928"
                    / "openhands"
                ),
            )
        )
        / "benchmarks/flowpilot/src/benchmark_adapters/tracing.py"
    )
    spec = importlib.util.spec_from_file_location("integration_tracing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TraceRecorder


def bind_timing(observer, req, action="action", resolution="LOCAL_ONLY"):
    observer.bind(
        trace_request_id="sdk-request",
        tool_call_id=req.identity.tool_call_id,
        action_event_id=action,
        identity=req.identity,
        tool_name=req.context["tool_name"],
        resolution=resolution,
    )


def timing_event(req, action="action", **kwargs):
    return dict(
        request_id="sdk-request",
        tool_call_id=req.identity.tool_call_id,
        action_event_id=action,
        tool_name=req.context["tool_name"],
        round_trip_ms=200.0,
        **kwargs,
    )


def test_sdk_live_recorder_updates_next_prediction_from_tool_thread(tmp_path):
    import json
    from flowpilot_predictor_bridge import OpenHandsTimingObserver

    async def run():
        async with PredictorRuntime(
            Model(Probe()), timeout_seconds=3,
            online_config={
                "min_rows": {n: 1 for n in ("q10", "q50", "q90", "q99")},
                "min_task_groups": {n: 1 for n in ("q10", "q50", "q90", "q99")},
            },
        ) as runtime:
            req = request()
            original = await runtime.predict(req)
            observer = OpenHandsTimingObserver(runtime.observe)
            bind_timing(observer, req)
            recorder = sdk_recorder_class()(tmp_path, {}, tool_timing_observer=observer)
            try:
                await asyncio.to_thread(recorder.emit, "tool_end", **timing_event(req))
                await asyncio.sleep(0)
                assert observer.snapshot()["metrics"]["result_updated"] == 1
                assert runtime.snapshot()["metrics"]["online_updates"] == 1
                after = await runtime.predict(request("next"))
                assert after["duration_ms"]["q50"] > original["duration_ms"]["q50"]
                assert original["duration_ms"]["q50"] == 50.0
                rows = [
                    json.loads(s)
                    for s in (tmp_path / "events.jsonl").read_text().splitlines()
                ]
                assert rows[0]["round_trip_ms"] == 200.0
                # Replayed trace event has no remaining binding; cannot learn twice.
                recorder.emit("tool_end", **timing_event(req))
                await asyncio.sleep(0)
                assert runtime.snapshot()["metrics"]["online_updates"] == 1
            finally:
                observer.close()
                recorder._file.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "event,extra,status,rtt",
    [
        ("tool_end", {}, "completed", 200.0),
        ("tool_error", {}, "execution_error", 200.0),
        ("tool_end", {"outcome": {"exit_code": 1}}, "execution_error", 200.0),
        ("tool_end", {"outcome": {"timed_out": True}}, "cancelled", None),
        ("tool_error", {"timed_out": True}, "cancelled", None),
        ("tool_error", {"error_type": "TimeoutError"}, "cancelled", None),
    ],
)
def test_sdk_timing_scope_and_censored_feedback(tmp_path, event, extra, status, rtt):
    from flowpilot_predictor_bridge import OpenHandsTimingObserver

    async def run():
        received = []

        def receive(f):
            received.append(f)
            return {"status": "recorded"}

        observer = OpenHandsTimingObserver(receive)
        req = request()
        bind_timing(observer, req)
        recorder = sdk_recorder_class()(tmp_path, {}, tool_timing_observer=observer)
        try:
            recorder.emit(
                event, **timing_event(req), executor_duration_ms=9999, **extra
            )
            await asyncio.sleep(0)
            assert len(received) == 1
            assert received[0].status == status
            assert received[0].round_trip_ms == rtt
            assert received[0].identity == req.identity
        finally:
            observer.close()
            recorder._file.close()

    asyncio.run(run())


def test_sdk_feedback_is_bounded_and_rejects_wrong_identity():
    from flowpilot_predictor_bridge import OpenHandsTimingObserver

    async def run():
        received = []
        observer = OpenHandsTimingObserver(
            lambda f: received.append(f) or {"status": "recorded"}, max_pending=1
        )
        a, b = request("a"), request("b", job="another")
        with pytest.raises(ValueError):
            bind_timing(observer, a, resolution="HISTORICAL_HIT")
        bind_timing(observer, a)
        bind_timing(observer, b)
        with pytest.raises(ValueError):
            bind_timing(observer, a)
        observer(dict(event="tool_end", **timing_event(a, action="wrong-action")))
        observer(dict(event="tool_end", **timing_event(a)))
        observer(dict(event="tool_end", **timing_event(b)))
        assert observer.snapshot()["pending"] == 1
        assert observer.snapshot()["metrics"]["dropped_full"] == 1
        await asyncio.sleep(0)
        assert [f.identity for f in received] == [a.identity]
        assert observer.snapshot()["pending"] == 0
        observer.close()

    asyncio.run(run())


def test_sdk_observer_failures_do_not_break_tool_tracing(tmp_path):
    import json

    def fail(event):
        raise RuntimeError("prediction unavailable")

    recorder = sdk_recorder_class()(tmp_path, {}, tool_timing_observer=fail)
    try:
        recorder.emit("tool_end", **timing_event(request()))
        assert recorder.tool_timing_observer_errors == 1
        assert (
            json.loads((tmp_path / "events.jsonl").read_text())["event"] == "tool_end"
        )
    finally:
        recorder._file.close()
