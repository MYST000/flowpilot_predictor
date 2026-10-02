"""Validate and copy the existing predictor context without cache-state gating."""

from __future__ import annotations

import copy
import json

from .contracts import ToolPredictionRequest, nonnegative


def model_context(request: ToolPredictionRequest):
    if request.stage != "tool_call_ready":
        raise ValueError("prediction requires complete T1 tool calls")
    c = copy.deepcopy(request.context)
    for name in ("backend_id", "backend_version", "tool_name", "tool_schema_version"):
        if not isinstance(c.get(name), str) or not c[name]:
            raise ValueError(f"missing {name}")
    if not isinstance(c.get("arguments"), dict):
        raise ValueError("complete parsed arguments are required")
    json.dumps(c["arguments"], allow_nan=False)
    if c.get("execution_mode") not in ("serial", "parallel"):
        raise ValueError("execution_mode is required")
    history = c.get("history", [])
    if not isinstance(history, list) or len(history) > 64:
        raise ValueError("history must contain at most 64 completed local RTTs")
    for row in history:
        nonnegative(row["rtt_ms"], "history RTT")
        if type(row.get("failed")) is not bool:
            raise ValueError("history.failed must be a boolean")
    if not isinstance(c.get("load", {}), dict):
        raise ValueError("load must be a mapping")
    # This is a COUNTERFACTUAL model input only. Never modify the framework's
    # real resolution. A hit/follower/unknown call still runs the RTT model.
    c["resolution"] = "LOCAL_ONLY"
    return c


def request_from_tool_call(
    identity, tool_call: dict, context: dict, *, tool_schema=None
):
    """Bind a closed Chat Completions tool call to the user's T1 context.

    ``context`` supplies the existing backend/version/history/load/batch fields.
    No prediction or cache lookup is performed while preparing this request.
    """
    from predictor.data import schema_signature

    if tool_call.get("id") != identity.tool_call_id:
        raise ValueError("tool_call_id mismatch")
    function = tool_call.get("function", {})
    name = function.get("name")
    if name != context.get("tool_name"):
        raise ValueError("tool_name mismatch")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("closed object arguments are required")
    arguments = copy.deepcopy(arguments)
    if tool_schema is not None:
        if tool_schema.get("name") != name:
            raise ValueError("tool schema name mismatch")
        if schema_signature(tool_schema) != context.get("tool_schema_version"):
            raise ValueError("tool schema version mismatch")
        for key, prop in (
            tool_schema.get("parameters", {}).get("properties", {}).items()
        ):
            if key not in arguments and "default" in prop:
                arguments[key] = copy.deepcopy(prop["default"])
    request = ToolPredictionRequest(
        identity, {**copy.deepcopy(context), "arguments": arguments}
    )
    model_context(request)  # Validate before enqueueing native work.
    return request
