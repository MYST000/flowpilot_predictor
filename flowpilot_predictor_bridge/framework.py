"""T1 RTT integration with FlowPilot's existing resolution and retention consumers."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter, OrderedDict

from predictor.data import backend_signature, schema_signature

from .contracts import CallIdentity, ToolFeedback, ToolPredictionRequest
from .runtime import PredictorRuntime


class FrameworkPredictor:
    def __init__(self, runtime: PredictorRuntime):
        self.runtime = runtime
        runtime.sink = self._deliver
        self.app = None
        self.calls = OrderedDict()
        self.tasks = set()
        self.metrics = Counter()

    def bind(self, app):
        self.app = app

    def _task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self._done)
        return task

    def _done(self, task):
        self.tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self.metrics["background_errors"] += 1

    @staticmethod
    def key(value):
        return tuple(
            getattr(value, k)
            for k in (
                "job_id",
                "line_id",
                "tail_request_id",
                "llm_call_id",
                "tool_call_id",
            )
        )

    def on_response(
        self,
        identity,
        version,
        api_kind,
        request,
        response,
        context_header,
        *,
        has_reuse_policy,
        elapsed_ms=0.0,
    ):
        from flowpilot.gateway.service import _provider_tool_calls

        self.metrics["responses"] += 1
        calls = _provider_tool_calls(response, api_kind)
        if not calls:
            self.metrics["no_tool"] += 1
            return
        if not context_header:
            self.metrics["missing_context"] += 1
            return
        context = json.loads(context_header)
        if context.get("schema_version") != 1:
            raise ValueError("unsupported predictor context")
        tools = {
            f["name"]: f
            for t in request.get("tools", [])
            if (f := t.get("function", t)).get("name")
        }
        names = set(context["environment_tools"])
        environment_calls = [c for c in calls if c["name"] in names]
        submitted = []
        for index, call in enumerate(environment_calls):
            name = call["name"]
            arguments = call["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            arguments = dict(arguments)
            schema = tools[name]
            for k, prop in schema.get("parameters", {}).get("properties", {}).items():
                if k not in arguments and "default" in prop:
                    arguments[k] = prop["default"]
            f = context["features"]
            profile, env = f["tool_execution_profile"], f["environment_known_at_t0"]
            load = (f.get("t0_features") or {}).get("client_load") or {}
            # Sender and gateway can be on separate hosts: ages are measured by
            # sender before transport; add this gateway's request elapsed below.
            age = max(0.0, context.get("snapshot_age_ms", 0.0))
            gateway_age = max(0.0, elapsed_ms)
            history = [
                {
                    "rtt_ms": h["round_trip_ms"],
                    "failed": h["execution_status"] != "completed",
                }
                for h in f["prior_tool_executions"]
                if h["tool_name"] == name and h.get("round_trip_ms") is not None
            ][-64:]
            model_context = dict(
                backend_id=env.get("backend", context["benchmark"]),
                backend_version=backend_signature(
                    env, context["dataset_revision"], profile
                ),
                tool_schema_version=schema_signature(schema),
                tool_name=name,
                arguments=arguments,
                configured_timeout_ms=profile["tool_timeout_seconds"] * 1000,
                batch_index=index,
                batch_size=len(environment_calls),
                execution_mode="serial",
                resolution="UNKNOWN",
                history=history,
                history_snapshot_age_ms=age + gateway_age,
                load={
                    k: load.get(k)
                    for k in (
                        "tool_inflight",
                        "llm_inflight",
                        "active_sessions",
                        "max_sessions",
                    )
                },
                load_snapshot_age_ms=(context.get("load_age_ms", 0.0) + gateway_age)
                if load.get("sampled_monotonic_ns") is not None
                else None,
                remaining_budget_t0_s=f["budget_at_t0"].get("remaining_seconds"),
            )
            call_id = CallIdentity(
                identity.job_id,
                identity.line_id,
                identity.request_id,
                identity.tail_request_id,
                identity.llm_call_id,
                call["id"],
                identity.attempt,
                identity.context_epoch,
                version,
            )
            self.calls[self.key(call_id)] = call_id
            while len(self.calls) > self.runtime.max_records:
                self.calls.popitem(last=False)
            self.runtime.submit(ToolPredictionRequest(call_id, model_context))
            submitted.append(call_id)
            self.metrics["submitted"] += 1
            self._task(self._record_prediction(call_id))
        if submitted:
            return self._task(self._wait_response(submitted))

    async def _wait_response(self, identities):
        # Initial Tool records are placeholders, not authoritative cache misses.
        # on_resolution supplies gateway reuse / SDK execution facts separately.
        outcomes = await asyncio.gather(
            *(self.runtime.wait_for_delivery(identity) for identity in identities),
            return_exceptions=True,
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome

    def on_resolution(self, record):
        identity = self.calls.get(self.key(record))
        if identity is None:
            return
        resolution = record.resolution.value.upper()
        self.runtime.resolve(identity, resolution, version=record.version)
        if record.status.value in {"ready", "failed", "cancelled"}:
            self.runtime.mark_terminal(identity)

    async def _record_prediction(self, identity):
        result = await self.runtime.wait(identity)
        await self.app.state.recorder.emit(
            "tool_duration_prediction",
            identity={
                k: getattr(identity, k)
                for k in (
                    "job_id",
                    "line_id",
                    "tail_request_id",
                    "llm_call_id",
                    "tool_call_id",
                )
            },
            fields={
                k: result.get(k)
                for k in (
                    "duration_estimate_ms",
                    "duration_ms",
                    "raw_duration_ms",
                    "selected_quantile",
                    "online_state_version",
                    "predictor_version",
                    "fallback",
                )
            },
        )

    async def _deliver(self, prior):
        identity = prior.identity

        async def current():
            if time.monotonic() >= prior.expires_at_monotonic:
                return False
            line = await self.app.state.frontier.line_snapshot(
                identity.job_id, identity.line_id
            )
            return (
                line.get("tail_request_id") == identity.tail_request_id
                and line.get("llm_call_id") == identity.llm_call_id
                and line.get("version") == identity.tail_version
                and line.get("context_epoch") == identity.context_epoch
                and line.get("phase") in {"BLOCKED", "READY"}
            )

        accepted = await self.app.state.tool_resolutions.apply_duration_estimate(
            identity,
            expected_version=prior.resolution_version,
            duration_ms=prior.duration_estimate_ms,
            quantile=prior.selected_quantile,
            is_current=current,
        )
        if accepted:
            self.metrics["scheduler_handoffs"] += 1
            retention = self.app.state.scheduling.retention
            if retention is not None:
                retention.line_changed(identity.job_id, identity.line_id)
            projection = await self.app.state.projection_calculator.for_line(
                identity.job_id, identity.line_id
            )
            await self.app.state.recorder.emit(
                "tool_duration_applied",
                identity={
                    k: getattr(identity, k)
                    for k in ("job_id", "line_id", "tail_request_id", "tool_call_id")
                },
                fields={
                    "duration_estimate_ms": prior.duration_estimate_ms,
                    "selected_quantile": prior.selected_quantile,
                    "projection": projection.model_dump(mode="json"),
                },
            )
        return accepted

    async def feedback(self, payload):
        key = tuple(
            payload[k]
            for k in (
                "job_id",
                "line_id",
                "tail_request_id",
                "llm_call_id",
                "tool_call_id",
            )
        )
        identity = self.calls.get(key)
        if identity is None:
            return {"status": "ignored", "reason": "unknown_call"}
        if any(
            payload[k] != getattr(identity, k)
            for k in ("request_id", "attempt", "context_epoch", "execution_attempt")
        ):
            raise ValueError("feedback invocation mismatch")
        call = self.runtime.calls.get(identity)
        if call is None:
            return {"status": "ignored", "reason": "expired_call"}
        result = self.runtime.observe(
            ToolFeedback(
                identity,
                payload["event_id"],
                payload["execution_attempt"],
                call.resolution,
                True,
                payload.get("round_trip_ms"),
                status=payload["status"],
                timing_scope=payload["timing_scope"],
            )
        )
        await self.app.state.recorder.emit(
            "tool_duration_feedback",
            identity={
                k: getattr(identity, k) for k in ("job_id", "line_id", "tool_call_id")
            },
            fields={"round_trip_ms": payload.get("round_trip_ms"), **result},
        )
        return result

    def snapshot(self):
        return {
            "enabled": True,
            "bridge": dict(self.metrics),
            "runtime": self.runtime.snapshot(),
        }

    async def close(self):
        await self.runtime.aclose()
        await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
