"""Response collection waits for applicable writes, not just native inference."""

import asyncio
import threading

import pytest
from flowpilot_predictor_bridge import PredictorRuntime
from test_predictor_bridge import Model, Probe, feedback, request


def test_completion_waits_for_resolution_and_sink_without_waiting_for_other_calls():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def sink(prior):
            entered.set()
            await release.wait()
            return True

        async with PredictorRuntime(Model(Probe()), duration_sink=sink) as runtime:
            req = request()
            await runtime.predict(req)
            other = request("unrelated")
            await runtime.predict(other)
            pending = asyncio.create_task(runtime.wait_for_delivery(req.identity))
            await asyncio.sleep(0)
            assert not pending.done()  # Native result is not a cache miss.
            runtime.resolve(req.identity, "LOCAL_LEADER", version=2)
            await entered.wait()
            assert not pending.done()  # Sink still owns the atomic version check.
            release.set()
            await asyncio.wait_for(pending, 1)
            assert runtime.metrics["priors_accepted"] == 1
            assert runtime.calls[other.identity].resolution == "UNKNOWN"

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["model", "sink", "timeout"])
def test_unavailable_prediction_and_failed_write_fail_response_collection(failure):
    async def run():
        probe = Probe()
        probe.fail = failure == "model"
        if failure == "timeout":
            probe.gate = threading.Event()

        async def sink(prior):
            raise ValueError("fixture write failure")

        async with PredictorRuntime(
            Model(probe), duration_sink=sink, timeout_seconds=0.03
        ) as runtime:
            req = request()
            runtime.submit(req)
            runtime.resolve(req.identity, "LOCAL_ONLY", version=1)
            try:
                with pytest.raises(RuntimeError, match="unavailable|handoff failed"):
                    await asyncio.wait_for(runtime.wait_for_delivery(req.identity), 1)
            finally:
                if probe.gate:
                    probe.gate.set()

    asyncio.run(run())


@pytest.mark.parametrize(
    "outcome", ["HISTORICAL_HIT", "INFLIGHT_FOLLOWER", "terminal", "invalidated"]
)
def test_irrelevant_prediction_does_not_hold_response_collection(outcome):
    async def run():
        probe = Probe()
        probe.gate = threading.Event()
        async with PredictorRuntime(Model(probe), timeout_seconds=3) as runtime:
            req = request()
            runtime.submit(req)
            pending = asyncio.create_task(runtime.wait_for_delivery(req.identity))
            try:
                await asyncio.sleep(0)
                assert not pending.done()
                if outcome == "terminal":
                    runtime.mark_terminal(req.identity)
                elif outcome == "invalidated":
                    runtime.invalidate(req.identity)
                else:
                    runtime.resolve(req.identity, outcome, version=2)
                await asyncio.wait_for(pending, 1)
                assert not probe.gate.is_set()
            finally:
                probe.gate.set()

    asyncio.run(run())


def test_cancelling_collection_preserves_late_rtt_learning():
    async def run():
        probe = Probe()
        probe.gate = threading.Event()
        async with PredictorRuntime(Model(probe), timeout_seconds=3) as runtime:
            req = request()
            runtime.submit(req)
            runtime.resolve(req.identity, "LOCAL_ONLY", version=1)
            pending = asyncio.create_task(runtime.wait_for_delivery(req.identity))
            try:
                await asyncio.sleep(0)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                assert runtime.observe(feedback(req))["status"] == "pending_prediction"
            finally:
                probe.gate.set()
            await runtime.wait(req.identity)
            assert runtime.metrics["online_updates"] == 1
            assert not runtime.calls[req.identity].invalid

    asyncio.run(run())


def test_failed_counterfactual_prediction_is_irrelevant_after_a_cache_hit():
    async def run():
        probe = Probe()
        probe.fail = True
        async with PredictorRuntime(Model(probe)) as runtime:
            req = request()
            await runtime.predict(req)
            pending = asyncio.create_task(runtime.wait_for_delivery(req.identity))
            await asyncio.sleep(0)
            assert not pending.done()
            runtime.resolve(req.identity, "HISTORICAL_HIT", version=2)
            await asyncio.wait_for(pending, 1)

    asyncio.run(run())
