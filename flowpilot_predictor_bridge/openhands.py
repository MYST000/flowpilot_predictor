"""Nonblocking bridge from one SDK TraceRecorder to the predictor's event loop.

No SDK/ML dependencies. Explicit bindings prevent trace UUIDs, reused tool-call
IDs and execution retries from being mistaken for framework request identities.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .contracts import LOCAL, CallIdentity, ToolFeedback


@dataclass(frozen=True)
class _Binding:
    identity: CallIdentity
    tool_name: str
    resolution: str
    expires: float


class OpenHandsTimingObserver:
    """Attach one instance to one recorder; bind each local invocation explicitly.

    Construct on the predictor loop and pass ``runtime.observe`` as ``observe``.
    ``bind`` may run on SDK threads, before execution starts. ``__call__`` never
    waits for prediction. Full queues drop feedback with counters; JSONL remains
    the durable source. Timeout samples are censored and not used as exact RTT.
    """

    def __init__(
        self,
        observe: Callable[[ToolFeedback], dict],
        *,
        max_pending=128,
        max_bindings=4096,
        binding_ttl=600.0,
    ):
        if max_pending < 1 or max_bindings < 1 or binding_ttl <= 0:
            raise ValueError("positive feedback bounds are required")
        self.loop = asyncio.get_running_loop()
        self.observe = observe
        self.max_pending = max_pending
        self.max_bindings = max_bindings
        self.binding_ttl = binding_ttl
        self._lock = threading.Lock()
        self._bindings = {}
        self._pending = 0
        self._closed = False
        self._scope = uuid.uuid4().hex
        self._metrics = Counter()

    def bind(
        self,
        *,
        trace_request_id: str,
        tool_call_id: str,
        action_event_id: str,
        identity: CallIdentity,
        tool_name: str,
        resolution: str,
    ):
        """Bind authoritative local resolution, including its execution attempt.

        Use a new action_event_id for a retry; ambiguous rebinding is rejected.
        Cache hits/followers do not execute and must not get a timing binding.
        """
        key = (trace_request_id, tool_call_id, action_event_id)
        if any(not isinstance(v, str) or not v for v in (*key, tool_name)):
            raise ValueError("explicit trace request, call, action and tool required")
        if tool_call_id != identity.tool_call_id or resolution not in LOCAL:
            raise ValueError("binding requires the matching local tool invocation")
        with self._lock:
            if self._closed:
                raise RuntimeError("timing observer is closed")
            now = time.monotonic()
            expired = [k for k, v in self._bindings.items() if v.expires <= now]
            for k in expired:
                del self._bindings[k]
                self._metrics["expired"] += 1
            if key in self._bindings:
                raise ValueError("trace invocation already bound")
            if len(self._bindings) >= self.max_bindings:
                raise RuntimeError("timing binding capacity reached")
            self._bindings[key] = _Binding(
                identity, tool_name, resolution, now + self.binding_ttl
            )

    def __call__(self, event: dict):
        if event.get("event") not in {"tool_end", "tool_error"}:
            return
        key = tuple(
            event.get(k) for k in ("request_id", "tool_call_id", "action_event_id")
        )
        with self._lock:
            if self._closed:
                self._metrics["closed"] += 1
                return
            binding = self._bindings.get(key)
            if binding is None or binding.tool_name != event.get("tool_name"):
                self._metrics["unmatched"] += 1
                return
            if binding.expires <= time.monotonic():
                del self._bindings[key]
                self._metrics["expired"] += 1
                return
            try:
                censored = event.get("timed_out") is True
                feedback = ToolFeedback(
                    identity=binding.identity,
                    event_id=f"openhands:{self._scope}:{event.get('seq')}:{event.get('monotonic_ns')}",
                    execution_attempt=binding.identity.execution_attempt,
                    resolution=binding.resolution,
                    executed=True,
                    round_trip_ms=None if censored else event.get("round_trip_ms"),
                    status="cancelled"
                    if censored
                    else (
                        "execution_error"
                        if event["event"] == "tool_error"
                        or event.get("execution_error")
                        else "completed"
                    ),
                )
                if not censored and feedback.round_trip_ms is None:
                    raise ValueError("missing observed RTT")
            except (ValueError, TypeError):
                self._metrics["invalid"] += 1
                return
            del self._bindings[key]
            if self._pending >= self.max_pending:
                self._metrics["dropped_full"] += 1
                return
            self._pending += 1
            try:
                self.loop.call_soon_threadsafe(self._deliver, feedback)
            except RuntimeError:
                self._pending -= 1
                self._metrics["loop_closed"] += 1

    def _deliver(self, feedback):
        try:
            with self._lock:
                if self._closed:
                    self._metrics["closed"] += 1
                    return
            result = self.observe(feedback)
            with self._lock:
                self._metrics["delivered"] += 1
                self._metrics["result_" + result.get("status", "unknown")] += 1
        except Exception:
            with self._lock:
                self._metrics["delivery_errors"] += 1
        finally:
            with self._lock:
                self._pending -= 1

    def close(self):
        """Detach logically; pending callbacks are ignored, retained bindings cleared."""
        with self._lock:
            self._closed = True
            self._bindings.clear()

    def snapshot(self):
        with self._lock:
            return {
                "pending": self._pending,
                "bindings": len(self._bindings),
                "metrics": dict(self._metrics),
            }
