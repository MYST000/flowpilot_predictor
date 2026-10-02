"""Per-tool EWMA correction of a frozen model's log RTT residual."""

from __future__ import annotations

import math
from dataclasses import dataclass

from predictor import NAMES
from predictor.features import group_key

from .contracts import nonnegative


@dataclass(frozen=True)
class BiasSnapshot:
    bias: float = 0.0
    version: int = 0
    observations: int = 0


class OnlineBias:
    def __init__(self, alpha: float = 0.2, max_abs_log_bias: float = math.log(4)):
        if not 0 < alpha <= 1 or not math.isfinite(alpha):
            raise ValueError("alpha must be in (0, 1]")
        if not math.isfinite(max_abs_log_bias) or max_abs_log_bias <= 0:
            raise ValueError("max_abs_log_bias must be positive")
        self.alpha = alpha
        self.limit = max_abs_log_bias
        self._state: dict[tuple, BiasSnapshot] = {}

    def snapshot(self, context) -> BiasSnapshot:
        return self._state.get(group_key(context), BiasSnapshot())

    def apply(self, quantiles, snapshot: BiasSnapshot):
        if quantiles is None:
            return None
        values = [nonnegative(quantiles[name], name) for name in NAMES]
        if values != sorted(values):
            raise ValueError("model returned crossing quantiles")
        if snapshot.bias == 0:
            return dict(zip(NAMES, values))
        # A common increasing transform preserves order; this is an online
        # point-bias correction, not a new quantile coverage guarantee.
        return {
            name: max(0.0, math.expm1(math.log1p(value) + snapshot.bias))
            for name, value in zip(NAMES, values)
        }

    def observe(self, context, raw_q50: float, actual_ms: float) -> BiasSnapshot:
        residual = math.log1p(nonnegative(actual_ms, "actual RTT")) - math.log1p(
            nonnegative(raw_q50, "raw Q50")
        )
        old = self.snapshot(context)
        bias = (1 - self.alpha) * old.bias + self.alpha * residual
        new = BiasSnapshot(
            max(-self.limit, min(self.limit, bias)),
            old.version + 1,
            old.observations + 1,
        )
        self._state[group_key(context)] = new
        return new
