"""T1 per-call contracts; durations are client RTT in milliseconds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

Quantile = Literal["q50", "q90"]
LOCAL = frozenset({"LOCAL_ONLY", "LOCAL_LEADER"})
RESOLUTIONS = LOCAL | {"UNKNOWN", "HISTORICAL_HIT", "INFLIGHT_FOLLOWER"}


def nonnegative(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class CallIdentity:
    job_id: str
    line_id: str
    request_id: str
    tail_request_id: str
    llm_call_id: str
    tool_call_id: str
    attempt: int
    context_epoch: int
    tail_version: int
    execution_attempt: int = 1

    def __post_init__(self):
        for name in (
            "job_id",
            "line_id",
            "request_id",
            "tail_request_id",
            "llm_call_id",
            "tool_call_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} is required")
        for name, lower in [
            ("attempt", 1),
            ("context_epoch", 0),
            ("tail_version", 0),
            ("execution_attempt", 1),
        ]:
            value = getattr(self, name)
            if type(value) is not int or value < lower:
                raise ValueError(f"invalid {name}")


@dataclass(frozen=True)
class ToolPredictionRequest:
    identity: CallIdentity
    context: dict[str, Any]
    stage: str = "tool_call_ready"


@dataclass(frozen=True)
class ToolFeedback:
    identity: CallIdentity
    event_id: str
    execution_attempt: int
    resolution: str
    executed: bool
    round_trip_ms: float | None
    status: str = "completed"
    timing_scope: str = "client_round_trip"

    def __post_init__(self):
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("feedback event_id is required")
        if type(self.execution_attempt) is not int or self.execution_attempt < 1:
            raise ValueError("invalid execution_attempt")
        if self.resolution not in RESOLUTIONS or type(self.executed) is not bool:
            raise ValueError("invalid execution resolution")
        if self.status not in {"completed", "execution_error", "cancelled", "blocked"}:
            raise ValueError("invalid feedback status")
        if self.round_trip_ms is not None:
            nonnegative(self.round_trip_ms, "round_trip_ms")
        if self.timing_scope != "client_round_trip":
            raise ValueError(
                "feedback must measure client RTT, not executor/ready time"
            )
        if self.executed and self.resolution not in LOCAL:
            raise ValueError("a reused result is not a local execution")


@dataclass(frozen=True)
class DurationPrior:
    identity: CallIdentity
    resolution_version: int
    prediction_id: str
    duration_estimate_ms: float
    selected_quantile: Quantile
    duration_p50: float
    duration_p90: float
    predictor_version: str
    online_state_version: int
    expires_at_monotonic: float
    target: str = "local_execution_round_trip_from_dispatch"


# The framework supplies an atomic per-call write + existing refresh callback.
# True means accepted, False means stale. It must revalidate identity, local
# resolution/version, nonterminal status and expiry at its actual write boundary.
DurationSink = Callable[[DurationPrior], Awaitable[bool]]
