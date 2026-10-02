"""Attach timing to MCP metadata, preserving the official tool result content."""

import os
import socket
import time
import uuid

from mcp import types


def install_timing(server):
    handlers = server._mcp_server.request_handlers
    original = handlers[types.CallToolRequest]
    clock = (
        f"browsecomp-handler:{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
    )

    async def timed(request):
        started = time.monotonic_ns()
        result = await original(request)
        elapsed = (time.monotonic_ns() - started) / 1e6
        payload = result.root
        if not isinstance(payload, types.CallToolResult):
            raise TypeError("Unexpected MCP tool result")
        payload.meta = {
            **(payload.meta or {}),
            "flowpilot_timing": {
                "schema_version": 1,
                "tool_name": request.params.name,
                "executor_duration_ms": elapsed,
                "queue_wait_ms": None,
                "executor_clock_domain": clock,
                "executor_timing_scope": "mcp_handler_including_validation_and_result_encoding",
            },
        }
        return result

    handlers[types.CallToolRequest] = timed
    return server
