"""Concurrent T1 inference, delayed RTT feedback and per-call duration handoff.

All public operations run on one asyncio event loop. Each native inference has
an exclusive model replica; online bias and lifecycle state stay on the loop.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import time
import uuid
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from .artifact import load_runtime_model
from .context import model_context
from .contracts import (
    LOCAL,
    RESOLUTIONS,
    CallIdentity,
    DurationPrior,
    DurationSink,
    Quantile,
    ToolFeedback,
    ToolPredictionRequest,
)
from .online import BiasSnapshot, OnlineBias
from .quantile_online import OnlineQuantileCalibration, QuantileSnapshot


@dataclass
class _Call:
    identity: CallIdentity
    context: dict
    fingerprint: str
    prediction_id: str
    created: float
    snapshot: BiasSnapshot | QuantileSnapshot
    future: asyncio.Future
    expires: float
    result: dict | None = None
    raw: dict | None = None
    feedback: ToolFeedback | None = None
    learned: bool = False
    resolution: str = "UNKNOWN"
    resolution_version: int = -1
    terminal: bool = False
    invalid: bool = False
    running: bool = False
    attempted_version: int = -1
    delivery: asyncio.Task | None = None
    delivery_error: Exception | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    timer: asyncio.TimerHandle | None = None


class PredictorRuntime:
    def __init__(
        self,
        model,
        *,
        selected_quantile: Quantile = "q50",
        workers: int = 2,
        max_pending: int = 128,
        max_records: int = 4096,
        timeout_seconds: float = 0.25,
        prediction_ttl_seconds: float = 30,
        feedback_ttl_seconds: float = 600,
        online: bool = True,
        alpha: float = 0.1,
        online_method: str = "quantile_residual_v1",
        online_config: dict | None = None,
        duration_sink: DurationSink | None = None,
    ):
        if type(online) is not bool:
            raise ValueError("online must be a boolean")
        if duration_sink is not None and not callable(duration_sink):
            raise ValueError("duration_sink must be callable")
        if selected_quantile not in ("q50", "q90"):
            raise ValueError("select q50 or q90")
        for name, value in [
            ("workers", workers),
            ("max_pending", max_pending),
            ("max_records", max_records),
        ]:
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if workers < 2 or max_pending < workers or max_records < max_pending:
            raise ValueError("require 2 <= workers <= max_pending <= max_records")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (timeout_seconds, prediction_ttl_seconds, feedback_ttl_seconds)
        ):
            raise ValueError("timeouts and TTLs must be finite and positive")
        self.quantile = selected_quantile
        self.timeout = timeout_seconds
        self.prediction_ttl = prediction_ttl_seconds
        self.feedback_ttl = feedback_ttl_seconds
        self.max_pending = max_pending
        self.max_records = max_records
        self.online = online
        if online_method not in {"ewma_log_bias", "quantile_residual_v1"}:
            raise ValueError("unsupported online_method")
        if online_config is not None and not isinstance(online_config, dict):
            raise ValueError("online_config must be an object")
        self.online_method = online_method
        if online_method == "ewma_log_bias":
            if online_config:
                raise ValueError("legacy bias does not accept quantile online_config")
            self.bias = OnlineBias(alpha)
            self.online_config = {"alpha": alpha}
        else:
            self.online_config = {
                "alpha": alpha,
                "shrinkage_rows": 256,
                **(online_config or {}),
            }
            self.bias = OnlineQuantileCalibration(**self.online_config)
        self.sink = duration_sink
        self.model_version = getattr(model, "version", model.name)
        config_hash = hashlib.sha256(
            json.dumps(self.online_config, sort_keys=True).encode()
        ).hexdigest()[:12]
        mode = f"{online_method}:{config_hash}" if online else "frozen"
        self.predictor_version = (
            f"{self.model_version}:t1-rtt-v1:{selected_quantile}:{mode}"
        )
        self.models = [copy.deepcopy(model) for _ in range(workers)]
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="tool-rtt"
        )
        self.slots = asyncio.Queue()
        for i in range(workers):
            self.slots.put_nowait(i)
        self.calls: OrderedDict[CallIdentity, _Call] = OrderedDict()
        self.feedback_events: OrderedDict[str, str] = OrderedDict()
        self.metrics = Counter()
        self._active = 0
        self._closed = False
        self._loop = None
        self.artifact_sha256: str | None = None

    @classmethod
    def from_artifact(
        cls, model_path, prepared_data, *, artifact_code_root=None, **kwargs
    ):
        # Startup-only, outside the serving hot path.
        model, digest = load_runtime_model(
            model_path, prepared_data, artifact_code_root
        )
        runtime = cls(model, **kwargs)
        runtime.artifact_sha256 = digest
        return runtime

    @classmethod
    def from_config(cls, config_path, *, duration_sink=None, project_root=None):
        from pathlib import Path

        root = (
            Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        )
        config = json.loads(Path(config_path).read_text())
        if config.get("artifact_code_root"):
            config["artifact_code_root"] = str(root / config["artifact_code_root"])
        model_path = root / config.pop("model_path")
        prepared_data = root / config.pop("prepared_data")
        return cls.from_artifact(
            model_path, prepared_data, duration_sink=duration_sink, **config
        )

    def _check_loop(self):
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("PredictorRuntime belongs to a different event loop")
        return loop

    def _prune(self):
        now = time.monotonic()
        for key, call in list(self.calls.items()):
            busy = call.running or (call.delivery and not call.delivery.done())
            if not busy and now - call.created >= self.feedback_ttl:
                del self.calls[key]
        while len(self.calls) >= self.max_records:
            victim = next(
                (
                    key
                    for key, call in self.calls.items()
                    if not call.running
                    and (call.terminal or call.invalid)
                    and (call.delivery is None or call.delivery.done())
                ),
                None,
            )
            if victim is None:
                break
            del self.calls[victim]

    def submit(self, request: ToolPredictionRequest) -> CallIdentity:
        """Start T1 prediction immediately, without consulting cache resolution."""
        loop = self._check_loop()
        if self._closed:
            raise RuntimeError("predictor is closed")
        context = model_context(request)
        fingerprint = hashlib.sha256(
            json.dumps(context, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        self._prune()
        if request.identity in self.calls:
            old = self.calls[request.identity]
            if old.fingerprint != fingerprint:
                raise ValueError("conflicting input for the same tool invocation")
            return request.identity
        if len(self.calls) >= self.max_records:
            self.metrics["record_capacity_rejected"] += 1
            raise RuntimeError("prediction record capacity exhausted")
        now = time.monotonic()
        call = _Call(
            request.identity,
            context,
            fingerprint,
            uuid.uuid4().hex,
            now,
            self.bias.snapshot(context),
            loop.create_future(),
            now + self.prediction_ttl,
        )
        self.calls[request.identity] = call
        self.metrics["submitted"] += 1
        if self._active >= self.max_pending:
            self._complete(call, self._unavailable(call, "overloaded"))
            self.metrics["overloaded"] += 1
        else:
            call.running = True
            self._active += 1
            call.timer = loop.call_later(self.timeout, self._timeout, call)
            call.task = loop.create_task(self._run(call))
        return request.identity

    def on_llm_response(self, requests: list[ToolPredictionRequest]):
        """One complete response; an empty list is a factual no-tool response."""
        if not requests:
            self.metrics["no_tool_responses"] += 1
        return [self.submit(request) for request in requests]

    async def predict(self, request: ToolPredictionRequest) -> dict:
        return await self.wait(self.submit(request))

    async def wait(self, identity: CallIdentity) -> dict:
        self._check_loop()
        # Cancelling one waiter never cancels the native worker or another waiter.
        return copy.deepcopy(await asyncio.shield(self.calls[identity].future))

    async def wait_for_delivery(self, identity: CallIdentity) -> None:
        """Wait for this call's factual resolution and version-checked handoff.

        Cancelling a response collector does not cancel native inference or RTT
        learning. Hits, followers and terminal calls need no local duration.
        """
        self._check_loop()
        call = self.calls[identity]
        while True:
            call.changed.clear()
            if call.invalid or call.terminal or self._closed:
                return
            if call.resolution != "UNKNOWN" and call.resolution not in LOCAL:
                return
            remaining = call.expires - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("prediction expired before duration handoff")
            if call.resolution in LOCAL and call.result is not None:
                if call.result["duration_ms"] is None:
                    reason = call.result["fallback"]["reason"]
                    raise RuntimeError(f"prediction unavailable: {reason}")
                if (
                    call.delivery is None
                    and call.attempted_version == call.resolution_version
                ):
                    if call.delivery_error is not None:
                        raise RuntimeError(
                            "duration handoff failed"
                        ) from call.delivery_error
                    return
            await asyncio.wait_for(call.changed.wait(), timeout=remaining)

    def _unavailable(self, call, reason):
        return {
            "prediction_id": call.prediction_id,
            "identity": asdict(call.identity),
            "duration_ms": None,
            "duration_estimate_ms": None,
            "selected_quantile": self.quantile,
            "support": {"status": "unsupported"},
            "fallback": {"used": True, "reason": reason},
            "predictor_version": self.predictor_version,
            "online_state_version": call.snapshot.version,
        }

    def _complete(self, call, result):
        if not call.future.done():
            call.result = copy.deepcopy(result)
            call.future.set_result(call.result)
            call.changed.set()

    def _timeout(self, call):
        if not call.future.done():
            self.metrics["timeouts"] += 1
            self._complete(call, self._unavailable(call, "prediction_timeout"))
        # Keep its native slot and active accounting until _run really ends.

    async def _run(self, call):
        slot = None
        native = None
        try:
            slot = await self.slots.get()
            if call.invalid or self._closed:
                self._complete(call, self._unavailable(call, "invalidated"))
                return
            if call.future.done():  # Timed out while still queued.
                return
            native = self._loop.run_in_executor(
                self.executor, self.models[slot].predict, call.context
            )
            raw = await asyncio.shield(native)
            call.raw = copy.deepcopy(raw)
            if not call.invalid:
                quantiles = self.bias.apply(raw["duration_ms"], call.snapshot)
                result = {
                    **raw,
                    "prediction_id": call.prediction_id,
                    "identity": asdict(call.identity),
                    "raw_duration_ms": copy.deepcopy(raw["duration_ms"]),
                    "duration_ms": quantiles,
                    "duration_estimate_ms": quantiles[self.quantile]
                    if quantiles
                    else None,
                    "selected_quantile": self.quantile,
                    "target": "local_execution_round_trip_from_dispatch",
                    "predictor_version": self.predictor_version,
                    "online_method": self.online_method if self.online else "frozen",
                    "online_state_version": call.snapshot.version,
                    "online_observations": call.snapshot.observations,
                    "online_calibration": self.bias.metadata(call.snapshot)
                    if isinstance(self.bias, OnlineQuantileCalibration)
                    else {"method": "ewma_log_bias", "bias": call.snapshot.bias},
                    "as_of_monotonic": call.created,
                    "expires_at_monotonic": call.expires,
                }
                self._complete(call, result)
                self._learn(call)
                self._schedule_delivery(call)
            self.metrics["model_completed"] += 1
        except asyncio.CancelledError:
            call.invalid = True
            self._complete(call, self._unavailable(call, "prediction_cancelled"))
            if native is not None and not native.done():
                await asyncio.gather(asyncio.shield(native), return_exceptions=True)
            raise
        except Exception as exc:
            self.metrics["model_errors"] += 1
            self._complete(
                call, self._unavailable(call, f"model_error:{type(exc).__name__}")
            )
        finally:
            if slot is not None:
                self.slots.put_nowait(slot)
            call.running = False
            self._active -= 1
            if call.timer:
                call.timer.cancel()

    def resolve(self, identity: CallIdentity, resolution: str, *, version: int):
        """Supply authoritative resolution; this never gates prediction startup."""
        self._check_loop()
        if resolution not in RESOLUTIONS or type(version) is not int or version < 0:
            raise ValueError("invalid resolution/version")
        call = self.calls[identity]
        if version < call.resolution_version:
            return False
        if version == call.resolution_version and resolution != call.resolution:
            raise ValueError("conflicting resolution at the same version")
        call.resolution, call.resolution_version = resolution, version
        self._schedule_delivery(call)
        call.changed.set()
        return True

    def _can_deliver(self, call):
        return (
            self.sink is not None
            and not self._closed
            and not call.invalid
            and not call.terminal
            and call.resolution in LOCAL
            and time.monotonic() < call.expires
            and call.result is not None
            and call.result["duration_ms"] is not None
            and call.resolution_version != call.attempted_version
        )

    def _schedule_delivery(self, call):
        if self._can_deliver(call) and (call.delivery is None or call.delivery.done()):
            call.delivery = self._loop.create_task(self._deliver(call))

    async def _deliver(self, call):
        try:
            if not self._can_deliver(call):
                return
            call.attempted_version = call.resolution_version
            call.delivery_error = None
            q = call.result["duration_ms"]
            prior = DurationPrior(
                call.identity,
                call.resolution_version,
                call.prediction_id,
                q[self.quantile],
                self.quantile,
                q["q50"],
                q["q90"],
                self.predictor_version,
                call.snapshot.version,
                call.expires,
            )
            accepted = await self.sink(prior)
            if type(accepted) is not bool:
                raise TypeError("duration sink must report accepted/stale as bool")
            self.metrics["priors_accepted" if accepted else "priors_rejected"] += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            call.delivery_error = exc
            self.metrics["sink_errors"] += 1
        finally:
            call.delivery = None
            # Only a newer authoritative version permits a new handoff attempt.
            self._schedule_delivery(call)
            call.changed.set()

    def observe(self, feedback: ToolFeedback) -> dict:
        """Accept actual client RTT; a hit/follower is never an execution label."""
        self._check_loop()
        signature = json.dumps(asdict(feedback), sort_keys=True, allow_nan=False)
        previous = self.feedback_events.get(feedback.event_id)
        if previous is not None:
            if previous != signature:
                raise ValueError("conflicting feedback event_id")
            return {"status": "duplicate"}
        call = self.calls.get(feedback.identity)
        if call is None or call.invalid or self._closed:
            return {"status": "ignored", "reason": "unknown_or_invalidated_call"}
        if time.monotonic() - call.created >= self.feedback_ttl:
            return {"status": "ignored", "reason": "feedback_expired"}
        if feedback.execution_attempt != call.identity.execution_attempt:
            raise ValueError("execution attempt mismatch")
        if call.feedback is not None:
            if call.feedback != feedback:
                raise ValueError("conflicting terminal feedback for tool invocation")
            return {"status": "duplicate"}
        if (
            call.resolution != "UNKNOWN"
            and call.resolution != feedback.resolution
            and not {call.resolution, feedback.resolution} <= LOCAL
        ):
            raise ValueError("feedback conflicts with authoritative resolution")
        self.feedback_events[feedback.event_id] = signature
        while len(self.feedback_events) > self.max_records:
            self.feedback_events.popitem(last=False)
        call.feedback = feedback
        call.terminal = True
        call.changed.set()
        if not feedback.executed or feedback.resolution not in LOCAL:
            return {"status": "ignored", "reason": "not_local_execution"}
        if feedback.round_trip_ms is None or feedback.status not in (
            "completed",
            "execution_error",
        ):
            return {"status": "ignored", "reason": "no_observed_rtt"}
        if not self.online:
            return {"status": "recorded", "reason": "frozen_mode"}
        self._learn(call)
        if call.learned:
            return {
                "status": "updated",
                "online_state_version": self.bias.snapshot(call.context).version,
            }
        return {
            "status": "pending_prediction" if call.running else "ignored",
            "reason": "raw_prediction_not_available",
        }

    def _learn(self, call):
        f = call.feedback
        if (
            not self.online
            or call.learned
            or call.invalid
            or f is None
            or not f.executed
            or f.resolution not in LOCAL
            or f.status not in ("completed", "execution_error")
            or f.round_trip_ms is None
            or call.raw is None
            or call.raw["duration_ms"] is None
            or time.monotonic() - call.created >= self.feedback_ttl
        ):
            return
        # Always use this invocation's frozen base prediction. Other tool
        # completions must not replace its residual reference with a newer one.
        if isinstance(self.bias, OnlineQuantileCalibration):
            self.bias.observe(
                call.context,
                call.raw["duration_ms"],
                f.round_trip_ms,
                call.context.get("task_group_id", call.identity.job_id),
            )
        else:
            self.bias.observe(
                call.context, call.raw["duration_ms"]["q50"], f.round_trip_ms
            )
        call.learned = True
        self.metrics["online_updates"] += 1

    def mark_terminal(self, identity: CallIdentity):
        """Stop scheduling delivery while allowing delayed measured RTT feedback."""
        self._check_loop()
        call = self.calls.get(identity)
        if call is not None:
            call.terminal = True
            call.changed.set()

    def invalidate(self, identity: CallIdentity, reason: str = "invalidated"):
        self._check_loop()
        call = self.calls.get(identity)
        if call is not None:
            call.invalid = True
            call.changed.set()
            self._complete(call, self._unavailable(call, reason))
            if call.delivery:
                call.delivery.cancel()
            self.metrics["invalidated"] += 1

    def invalidate_line(self, job_id: str, line_id: str):
        for key in list(self.calls):
            if key.job_id == job_id and key.line_id == line_id:
                self.invalidate(key, "line_replaced_or_cancelled")

    async def cancel(self, request_id: str, *, job_id: str, line_id: str):
        for key in list(self.calls):
            if (key.request_id, key.job_id, key.line_id) == (
                request_id,
                job_id,
                line_id,
            ):
                self.invalidate(key, "request_cancelled")

    def snapshot(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "native_or_queued": self._active,
            "records": len(self.calls),
            "workers": len(self.models),
            "duration_sink_bound": self.sink is not None,
            "selected_quantile": self.quantile,
            "online_enabled": self.online,
            "online_method": self.online_method if self.online else "frozen",
            "online_config": copy.deepcopy(self.online_config),
            "model_version": self.model_version,
            "predictor_version": self.predictor_version,
            "artifact_sha256": self.artifact_sha256,
        }

    async def drain_deliveries(self):
        """Diagnostic/test barrier; never use this to gate cache delivery."""
        while tasks := [c.delivery for c in self.calls.values() if c.delivery]:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self):
        self._check_loop()
        self._closed = True
        for key in list(self.calls):
            self.invalidate(key, "shutdown")
        await asyncio.gather(
            *(c.task for c in self.calls.values() if c.task), return_exceptions=True
        )
        await self.drain_deliveries()
        await asyncio.to_thread(self.executor.shutdown, wait=True)

    async def __aenter__(self):
        self._check_loop()
        return self

    async def __aexit__(self, *_):
        await self.aclose()
