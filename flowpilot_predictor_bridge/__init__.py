"""Online T1 RTT adapter; no scheduler policy or tool execution implementation."""

from .contracts import CallIdentity, DurationPrior, ToolFeedback, ToolPredictionRequest
from .context import request_from_tool_call
from .flowpilot import FlowPilotDurationSink, feedback_from_tool_event
from .openhands import OpenHandsTimingObserver

__all__ = [
    "OpenHandsTimingObserver",
    "CallIdentity",
    "DurationPrior",
    "ToolFeedback",
    "ToolPredictionRequest",
    "PredictorRuntime",
    "request_from_tool_call",
    "FlowPilotDurationSink",
    "feedback_from_tool_event",
]


def __getattr__(name):
    # Timing collection can run in an SDK environment without ML dependencies.
    if name == "PredictorRuntime":
        from .runtime import PredictorRuntime

        return PredictorRuntime
    raise AttributeError(name)
