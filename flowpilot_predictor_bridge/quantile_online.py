"""Bounded, per-quantile online calibration of frozen tool RTT predictions."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from predictor import NAMES, QUANTILES
from predictor.features import group_key

from .contracts import nonnegative


@dataclass(frozen=True)
class QuantileSnapshot:
    offsets: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    active: tuple[bool, ...] = (False, False, False, False)
    version: int = 0
    observations: int = 0
    window_rows: int = 0
    task_groups: int = 0


class OnlineQuantileCalibration:
    """Estimate each head's log residual quantile from past completed calls.

    New residuals always enter a bounded window. A head is applied only when
    its window has enough calls and distinct tasks; otherwise it falls back to
    the base model. Thresholds are evidence gates, not coverage guarantees.
    """

    method = "quantile_residual_v1"

    def __init__(
        self,
        alpha: float = 0.05,
        buffer_size: int = 1024,
        shrinkage_rows: int = 128,
        max_abs_log_offset: float = math.log(2),
        max_step_log: float = 0.02,
        min_rows: dict | None = None,
        min_task_groups: dict | None = None,
    ):
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        for name, value in [("buffer_size", buffer_size), ("shrinkage_rows", shrinkage_rows)]:
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in [("max_abs_log_offset", max_abs_log_offset), ("max_step_log", max_step_log)]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.min_rows = dict(min_rows or dict(zip(NAMES, [32, 32, 100, 1000])))
        self.min_groups = dict(min_task_groups or dict(zip(NAMES, [4, 4, 10, 30])))
        for thresholds in [self.min_rows, self.min_groups]:
            if set(thresholds) != set(NAMES) or any(type(v) is not int or v < 1 for v in thresholds.values()):
                raise ValueError("all four quantiles need positive integer support thresholds")
        if buffer_size < max(self.min_rows.values()):
            raise ValueError("buffer_size must allow the largest quantile support threshold")
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.shrinkage_rows = shrinkage_rows
        self.limit = max_abs_log_offset
        self.max_step = max_step_log
        self._windows: dict[tuple, deque] = {}
        self._state: dict[tuple, QuantileSnapshot] = {}

    def snapshot(self, context):
        return self._state.get(group_key(context), QuantileSnapshot())

    @staticmethod
    def _values(quantiles):
        values = [nonnegative(quantiles[n], n) for n in NAMES]
        if values != sorted(values):
            raise ValueError("model returned crossing quantiles")
        return values

    def apply(self, quantiles, snapshot: QuantileSnapshot):
        if quantiles is None:
            return None
        values = self._values(quantiles)
        corrected = []
        for raw, offset, active in zip(values, snapshot.offsets, snapshot.active):
            value = max(0.0, math.expm1(math.log1p(raw) + offset)) if active else raw
            # Preserve head identities; later heads cannot be below earlier ones.
            corrected.append(max(value, corrected[-1] if corrected else 0.0))
        return dict(zip(NAMES, corrected))

    def observe(self, context, raw_quantiles, actual_ms, task_group_id):
        values = self._values(raw_quantiles)
        actual = nonnegative(actual_ms, "actual RTT")
        if not isinstance(task_group_id, str) or not task_group_id:
            raise ValueError("task_group_id is required")
        key = group_key(context)
        residuals = tuple(math.log1p(actual) - math.log1p(q) for q in values)
        window = self._windows.setdefault(key, deque(maxlen=self.buffer_size))
        window.append((residuals, task_group_id))
        n, groups = len(window), len({entry[1] for entry in window})
        old = self.snapshot(context)
        active = tuple(n >= self.min_rows[name] and groups >= self.min_groups[name] for name in NAMES)
        offsets = list(old.offsets)
        if any(active):
            residual_matrix = np.asarray([entry[0] for entry in window], dtype=float)
            shrink = n / (n + self.shrinkage_rows)
            for j, (q, enabled) in enumerate(zip(QUANTILES, active)):
                if not enabled:
                    continue
                target = float(np.quantile(residual_matrix[:, j], q)) * shrink
                target = max(-self.limit, min(self.limit, target))
                step = self.alpha * (target - offsets[j])
                step = max(-self.max_step, min(self.max_step, step))
                offsets[j] = max(-self.limit, min(self.limit, offsets[j] + step))
        new = QuantileSnapshot(tuple(offsets), active, old.version + 1, old.observations + 1, n, groups)
        self._state[key] = new
        return new

    def metadata(self, snapshot):
        return {
            "method": self.method,
            "window_rows": snapshot.window_rows,
            "task_groups": snapshot.task_groups,
            "active_quantiles": [n for n, enabled in zip(NAMES, snapshot.active) if enabled],
            "fallback_to_base": [n for n, enabled in zip(NAMES, snapshot.active) if not enabled],
            "log_offsets": dict(zip(NAMES, snapshot.offsets)),
            "min_rows": self.min_rows,
            "min_task_groups": self.min_groups,
            "strict_coverage_guarantee": False,
        }
