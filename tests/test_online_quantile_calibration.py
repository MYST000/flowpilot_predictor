"""Real-feedback calibration must remain bounded and respect T1 snapshots."""

import math
import asyncio
import copy
import threading
from pathlib import Path

import pytest

from flowpilot_predictor_bridge.quantile_online import OnlineQuantileCalibration
from predictor import NAMES
from flowpilot_predictor_bridge import CallIdentity, ToolFeedback, ToolPredictionRequest
from flowpilot_predictor_bridge.runtime import PredictorRuntime


def context(tool="search", version="v1"):
    return {"backend_id": "b", "backend_version": version,
            "tool_name": tool, "tool_schema_version": "s1"}


def raw():
    return dict(zip(NAMES, [10., 50., 90., 99.]))


def fixture(**options):
    return OnlineQuantileCalibration(buffer_size=8, shrinkage_rows=1,
        min_rows=dict(zip(NAMES, [4, 4, 6, 8])),
        min_task_groups=dict(zip(NAMES, [2, 2, 2, 3])), **options)


def test_every_feedback_updates_state_and_sparse_heads_use_base_prediction():
    corrector = fixture()
    for i in range(3):
        s = corrector.observe(context(), raw(), 100, "task-" + str(i % 2))
        assert s.version == i + 1 and s.observations == i + 1
        assert corrector.apply(raw(), s) == raw()
    s = corrector.observe(context(), raw(), 100, "task-0")
    assert s.active == (True, True, False, False)
    updated = corrector.apply(raw(), s)
    assert updated["q50"] > raw()["q50"]
    assert updated["q90"] == raw()["q90"]
    assert updated["q99"] == raw()["q99"]
    assert corrector.metadata(s)["fallback_to_base"] == ["q90", "q99"]


def test_snapshot_before_feedback_is_immutable_and_tools_are_isolated():
    corrector = fixture()
    before = corrector.snapshot(context())
    for i in range(8):
        corrector.observe(context(), raw(), 100, "task-" + str(i % 3))
    assert corrector.apply(raw(), before) == raw()
    assert before.version == 0
    assert corrector.snapshot(context(tool="read")).version == 0
    assert corrector.snapshot(context(version="v2")).version == 0


def test_updates_and_window_are_bounded_and_output_is_monotone():
    corrector = fixture(alpha=1, max_step_log=.02)
    old = corrector.snapshot(context())
    for i in range(40):
        new = corrector.observe(context(), raw(), 10**9, "task-" + str(i % 3))
        assert all(abs(a - b) <= .02000000001 for a, b in zip(old.offsets, new.offsets))
        assert all(abs(x) <= math.log(2) for x in new.offsets)
        q = list(corrector.apply(raw(), new).values())
        assert q == sorted(q) and all(x >= 0 and math.isfinite(x) for x in q)
        old = new
    assert new.window_rows == 8 and new.observations == 40


def test_single_extreme_return_does_not_pull_q50_away_from_residual_median():
    corrector = fixture(alpha=1)
    for i in range(7):
        corrector.observe(context(), raw(), 50, "task-" + str(i % 3))
    corrector.observe(context(), raw(), 10**9, "task-0")
    assert corrector.snapshot(context()).offsets[1] == 0
    assert corrector.apply(raw(), corrector.snapshot(context()))["q50"] == 50


def test_distinct_task_gate_rejects_many_calls_from_one_task():
    corrector = fixture()
    for _ in range(20):
        corrector.observe(context(), raw(), 100, "one-task")
    s = corrector.snapshot(context())
    assert not any(s.active) and s.task_groups == 1
    assert corrector.apply(raw(), s) == raw()


@pytest.mark.parametrize("label", [-1, float("nan"), float("inf"), True])
def test_invalid_feedback_never_enters_the_window(label):
    corrector = fixture()
    with pytest.raises(ValueError):
        corrector.observe(context(), raw(), label, "task")
    assert corrector.snapshot(context()).observations == 0


def test_configuration_must_allow_q99_threshold():
    with pytest.raises(ValueError, match="largest quantile"):
        OnlineQuantileCalibration(buffer_size=512)


def test_crossing_raw_prediction_is_rejected_before_learning():
    corrector = fixture()
    prediction = dict(zip(NAMES, [10., 100., 90., 99.]))
    with pytest.raises(ValueError, match="crossing"):
        corrector.observe(context(), prediction, 100, "task")
    assert corrector.snapshot(context()).observations == 0


class RuntimeProbe:
    def __init__(self, parallel=0):
        self.barrier = threading.Barrier(parallel) if parallel else None
        self.gate = None
        self.lock = threading.Lock()
        self.active = self.maximum = 0


class RuntimeModel:
    name = "fixture"
    version = "fixture-v1"

    def __init__(self, probe):
        self.probe = probe

    def __deepcopy__(self, memo):
        return RuntimeModel(self.probe)

    def predict(self, c):
        p = self.probe
        with p.lock:
            p.active += 1
            p.maximum = max(p.maximum, p.active)
        try:
            if p.barrier:
                p.barrier.wait(timeout=3)
            if p.gate:
                assert p.gate.wait(timeout=3)
            return {"duration_ms": raw(), "support": {"status": "supported"},
                    "fallback": {"used": False, "reason": None}}
        finally:
            with p.lock:
                p.active -= 1


def request(number, *, resolution="UNKNOWN"):
    identity = CallIdentity("task-" + str(number), "line", "request", "tail", "llm", "tool", 1, 0, 1)
    return ToolPredictionRequest(identity, {**context(), "arguments": {}, "history": [], "load": {},
                                "execution_mode": "serial", "resolution": resolution})


def feedback(req, *, resolution="LOCAL_ONLY", executed=True):
    return ToolFeedback(req.identity, req.identity.job_id + ":feedback", 1, resolution, executed, 100)


def test_runtime_four_parallel_requests_learn_then_deliver_correct_q50_and_ignore_hits():
    async def run():
        probe = RuntimeProbe(parallel=4)
        delivered = []
        async def sink(prior):
            delivered.append(prior)
            return True
        options = {"buffer_size": 8, "shrinkage_rows": 1,
                   "min_rows": dict(zip(NAMES, [4, 4, 6, 8])),
                   "min_task_groups": dict(zip(NAMES, [2, 2, 2, 3]))}
        async with PredictorRuntime(RuntimeModel(probe), workers=4, timeout_seconds=3,
                                    online_config=options, duration_sink=sink) as runtime:
            first = [request(i) for i in range(4)]
            runtime.on_llm_response(first)
            predictions = await asyncio.gather(*(runtime.wait(r.identity) for r in first))
            assert probe.maximum == 4
            assert all(p["duration_ms"] == raw() for p in predictions)
            probe.barrier = None
            for req in first:
                runtime.resolve(req.identity, "LOCAL_ONLY", version=1)
            await runtime.drain_deliveries()
            for req in first:
                assert runtime.observe(feedback(req))["status"] == "updated"
            assert runtime.observe(feedback(first[0]))["status"] == "duplicate"
            next_req = request(4)
            result = await runtime.predict(next_req)
            assert result["online_state_version"] == 4
            assert result["online_calibration"]["active_quantiles"] == ["q10", "q50"]
            assert result["duration_ms"]["q50"] > 50
            assert result["duration_ms"]["q99"] == 99
            runtime.resolve(next_req.identity, "LOCAL_ONLY", version=1)
            await runtime.drain_deliveries()
            assert delivered[-1].duration_estimate_ms == result["duration_ms"]["q50"]
            assert delivered[-1].duration_p90 == result["duration_ms"]["q90"]
            hit = request(5, resolution="HISTORICAL_HIT")
            await runtime.predict(hit)
            runtime.resolve(hit.identity, "HISTORICAL_HIT", version=1)
            assert runtime.observe(feedback(hit, resolution="HISTORICAL_HIT", executed=False))["status"] == "ignored"
            await runtime.drain_deliveries()
            assert len(delivered) == 5
            assert runtime.snapshot()["metrics"]["online_updates"] == 4
            assert await runtime.wait(first[0].identity) == predictions[0]
    asyncio.run(run())


def test_quantile_feedback_before_native_prediction_does_not_leak_into_it():
    async def run():
        probe = RuntimeProbe()
        probe.gate = threading.Event()
        options = {"min_rows": {n: 1 for n in NAMES}, "min_task_groups": {n: 1 for n in NAMES}}
        async with PredictorRuntime(RuntimeModel(probe), timeout_seconds=3, online_config=options) as runtime:
            req = request(0)
            runtime.submit(req)
            assert runtime.observe(feedback(req))["status"] == "pending_prediction"
            probe.gate.set()
            first = await runtime.wait(req.identity)
            assert first["online_state_version"] == 0 and first["duration_ms"] == raw()
            second = await runtime.predict(request(1))
            assert second["online_state_version"] == 1
            assert second["duration_ms"]["q50"] > 50
    asyncio.run(run())


def test_native27b_default_config_loads_snapshot_and_realtime_corrector():
    root = Path(__file__).resolve().parents[1]
    model = Path('/data/ql_flowpilot_predictor/predictor_experiments/native27b_1077_v1/train/lightgbm/model.joblib')
    if not model.exists():
        pytest.skip("27B artifact not installed")
    async def run():
        async with PredictorRuntime.from_config(root / "configs/predictor/runtime.json") as runtime:
            assert len(runtime.models) == 4
            assert runtime.online and runtime.online_method == "quantile_residual_v1"
            c = copy.deepcopy(runtime.models[0].contexts[0])
            req = ToolPredictionRequest(request(0).identity, c)
            prediction = await runtime.predict(req)
            assert prediction["duration_ms"] is not None
            assert prediction["duration_estimate_ms"] == prediction["duration_ms"]["q50"]
            runtime.observe(feedback(req))
            assert runtime.snapshot()["metrics"]["online_updates"] == 1
            assert all(m.state_version == 0 for m in runtime.models)
    asyncio.run(run())


def test_saved_artifact_snapshot_mismatch_is_rejected(tmp_path):
    from flowpilot_predictor_bridge.artifact import load_runtime_model
    import shutil
    root = Path('/data/ql_flowpilot_predictor/predictor_experiments/native27b_1077_v1')
    model = root / "train/lightgbm/model.joblib"
    if not model.exists():
        pytest.skip("27B artifact not installed")
    shutil.copytree(
        root / "deployment_snapshot/code/predictor", tmp_path / "code/predictor"
    )
    p = tmp_path / "code/predictor/models.py"
    p.write_text(p.read_text() + "\n# deliberate mismatch\n")
    with pytest.raises(ValueError, match="saved model/code snapshot mismatch"):
        load_runtime_model(model, '/data/ql_flowpilot_predictor/predictor_prepared/native27b_1077_v1', tmp_path / "code")


def test_gateway_factory_preserves_settings_and_registers_realtime_adapter(tmp_path, monkeypatch):
    config = pytest.importorskip("flowpilot.config")
    import flowpilot.app
    from flowpilot_predictor_bridge.serve import create_app
    passed = {}
    original_factory = flowpilot.app.create_app
    def capture(settings, **kwargs):
        passed["settings"] = settings
        return original_factory(settings, **kwargs)
    monkeypatch.setattr(flowpilot.app, "create_app", capture)
    settings = config.Settings(
        instances=(config.InferenceInstance("fixture", "http://127.0.0.1:9"),),
        trace_path=tmp_path / "gateway.jsonl", reuse_enabled=False, dcs_enabled=False,
        ingress_api_key="offline-test-fixture",
    )
    app = create_app(settings)
    adapter = app.state.tool_duration_adapter
    assert adapter is not None
    assert adapter.runtime.online and adapter.runtime.online_method == "quantile_residual_v1"
    assert len(adapter.runtime.models) == 4
    # The wrapper forwards the same Settings object instead of reconstructing strategies.
    assert passed["settings"] is settings
    async def check_binding():
        async with app.router.lifespan_context(app):
            assert adapter.runtime.snapshot()["duration_sink_bound"]
        assert adapter.runtime._closed
    asyncio.run(check_binding())
