"""Prediction-only translation at FlowPilot's duration_estimate_ms boundary.

This module does not import/reimplement a scheduler or guess its private state.
The framework owns the atomic write and invokes its existing refresh mechanism.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from .contracts import LOCAL, CallIdentity, DurationPrior, ToolFeedback


class FlowPilotDurationSink:
    """Bind the framework's atomic duration setter, not a dispatch/KV policy.

    ``update`` must return bool and atomically check expected identity/version,
    expiry, local resolution and nonterminal status before setting the scalar.
    The actual framework callback cannot be bound until its source is available.
    """

    def __init__(self, update: Callable[..., Awaitable[bool]]):
        self.update = update

    async def __call__(self, prior: DurationPrior) -> bool:
        return await self.update(
            identity=prior.identity,
            expected_resolution_version=prior.resolution_version,
            duration_estimate_ms=prior.duration_estimate_ms,
            source="predictor_t1_rtt",
            prediction_id=prior.prediction_id,
            expires_at_monotonic=prior.expires_at_monotonic,
            metadata={
                "target": prior.target,
                "selected_quantile": prior.selected_quantile,
                "duration_p50": prior.duration_p50,
                "duration_p90": prior.duration_p90,
                "predictor_version": prior.predictor_version,
                "online_state_version": prior.online_state_version,
            },
        )


def feedback_from_tool_event(
    identity: CallIdentity, event: dict, *, resolution: str, timing_scope: str
) -> ToolFeedback | None:
    """Translate tool terminal telemetry only with an explicit RTT scope.

    Call after the runtime has authenticated/validated its real lifecycle event.
    The caller must establish measured_latency_ms is client RTT. A provider's
    internal executor time cannot be silently renamed to that training target.
    """
    for name in ("job_id", "line_id", "tail_request_id", "llm_call_id", "tool_call_id"):
        if event.get(name) != getattr(identity, name):
            raise ValueError(f"tool event identity mismatch: {name}")
    if event.get("execution_attempt") != identity.execution_attempt:
        raise ValueError("tool event execution attempt mismatch")
    kind = event.get("event_kind")
    if kind == "start":
        return None
    status = {
        "finish": "completed",
        "fail": "execution_error",
        "cancel": "cancelled",
        "blocked": "blocked",
    }.get(kind)
    if status is None:
        raise ValueError("unknown tool event kind")
    return ToolFeedback(
        identity=identity,
        event_id=event["event_id"],
        execution_attempt=identity.execution_attempt,
        resolution=resolution,
        executed=resolution in LOCAL and kind in ("finish", "fail", "cancel"),
        round_trip_ms=event.get("measured_latency_ms"),
        status=status,
        timing_scope=timing_scope,
    )
