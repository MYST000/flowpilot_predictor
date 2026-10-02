"""Guard against label leakage, stale feedback and unsupported tail calibration."""

import copy
import math

import pytest

from experiment import GroupedCalibrator, OnlineBias, NAMES, ensure_forward, replay_bridge


def context(backend="b", tool="search"):
    return {"backend_id": backend, "backend_version": "v1", "tool_name": tool,
            "tool_schema_version": "s1", "execution_mode": "serial", "resolution": "LOCAL_ONLY"}


class FixedModel:
    def predict(self, c):
        return {"duration_ms": dict(zip(NAMES, [10., 50., 90., 99.])),
                "method": "fixture", "online_state_version": 0,
                "support": {"q99_low_support": True}, "fallback": {"used": False}}


def sample(sid, predict, observe, y, *, c=None):
    return {"sample_id": sid, "source_attempt": "attempt", "task_group_id": "task-" + sid,
            "clock_domain": "clock", "as_of_ns": predict, "dispatch_ns": predict,
            "observed_ns": observe, "predict_seq": 1, "observe_seq": 2,
            "context": c or context(), "labels": {"round_trip_ms": y, "execution_status": "completed"},
            "adapter": "fixture", "split": "test", "source_batch": "fixture"}


def test_overlapping_calls_only_see_returned_feedback_and_keep_own_raw_q50():
    samples = [sample("a", 0, 30, 100), sample("b", 10, 20, 200),
               sample("c", 15, 40, 300), sample("d", 25, 50, 400)]
    records, audit = replay_bridge(FixedModel(), samples, online=True)
    rows = {r["sample_id"]: r for r in records}
    for sid in ["a", "b", "c"]:
        assert rows[sid]["bias_snapshot"]["observations"] == 0
        assert rows[sid]["prediction"]["duration_ms"]["q50"] == 50
    assert rows["d"]["bias_snapshot"]["observations"] == 1
    expected = .2 * (math.log1p(200) - math.log1p(50))
    assert rows["d"]["bias_snapshot"]["bias"] == pytest.approx(expected)
    assert rows["d"]["bias_snapshot"]["last_observed_ns"] == 20
    bias = OnlineBias()
    for sid in ["b", "a", "c", "d"]:
        s = next(s for s in samples if s["sample_id"] == sid)
        bias.observe(s["context"], 50, s["labels"]["round_trip_ms"])
    assert audit["final_state"][0]["bias"] == pytest.approx(bias.snapshot(context()).bias)
    assert audit["peak_pending_predictions"] == 3


def test_changing_future_label_cannot_change_earlier_predictions():
    original = [sample("a", 0, 20, 100), sample("b", 10, 30, 200), sample("c", 25, 40, 300)]
    changed = copy.deepcopy(original)
    changed[1]["labels"]["round_trip_ms"] = 99999
    first, _ = replay_bridge(FixedModel(), original, online=True)
    second, _ = replay_bridge(FixedModel(), changed, online=True)
    for a, b in zip(first, second):
        assert a["sample_id"] == b["sample_id"]
        assert a["prediction"]["duration_ms"] == b["prediction"]["duration_ms"]


def test_frozen_mode_never_updates_and_expired_or_nonlocal_feedback_is_ignored():
    nonlocal_context = context(); nonlocal_context["resolution"] = "HISTORICAL_HIT"
    rows = [sample("hit", 0, 10, 1, c=nonlocal_context),
            sample("expired", 1, 600_000_000_002, 100)]
    _, frozen = replay_bridge(FixedModel(), rows, online=False)
    assert frozen["counters"] == {}
    _, online = replay_bridge(FixedModel(), rows, online=True)
    assert online["counters"] == {"ignored_not_local_execution": 1, "ignored_feedback_expired": 1}
    assert online["final_state"] == []


def test_duplicate_samples_fail_before_duplicate_feedback_can_learn():
    s = sample("a", 0, 10, 100)
    with pytest.raises(ValueError, match="duplicate"):
        replay_bridge(FixedModel(), [s, copy.deepcopy(s)], online=True)


def calibration_fixture():
    samples, records = [], []
    for j in range(1100):
        # Tiny backend has 40 calls; the larger unrelated backend enables global fallback.
        s = sample(str(j), j, j + 1, 100, c=context("small" if j < 40 else "large"))
        s["task_group_id"] = "cal-group-" + str(j % 40)
        samples.append(s)
        records.append({"sample_id": s["sample_id"], "task_group_id": s["task_group_id"],
                        "score": {}, "y_ms": 100, "prediction": FixedModel().predict(s["context"])})
    return records, samples


def test_tail_support_gates_use_global_fallback_for_sparse_tool():
    records, samples = calibration_fixture()
    cal = GroupedCalibrator(records, samples,
        min_rows=dict(zip(NAMES, [32, 32, 100, 1000])),
        min_groups=dict(zip(NAMES, [4, 4, 10, 30])))
    q, sources = cal.apply(FixedModel().predict(context())["duration_ms"], context("small"), "hierarchical_tail_support")
    assert sources["q50"]["level"] == "exact_tool_backend"
    assert sources["q90"]["level"] == "global_pool"
    assert sources["q99"]["level"] == "global_pool"
    assert sources["q99"]["rows"] == 1100
    assert list(q.values()) == sorted(q.values())
    _, direct = cal.apply(q, context("small"), "grouped_min32")
    assert direct["q99"]["level"] == "exact_tool_backend"


def test_empty_calibration_and_future_training_fail():
    with pytest.raises(ValueError, match="No supported"):
        GroupedCalibrator([], [], min_rows={n: 32 for n in NAMES}, min_groups={n: 4 for n in NAMES})
    with pytest.raises(ValueError, match="Future training"):
        ensure_forward([sample("fit", 0, 50, 10)], [sample("eval", 40, 60, 10)])


def test_forward_guard_rejects_same_task_in_fit_and_eval():
    train = sample("a", 0, 10, 10)
    evaluate = sample("b", 20, 30, 10)
    evaluate["task_group_id"] = train["task_group_id"]
    with pytest.raises(ValueError, match="Task-group overlap"):
        ensure_forward([train], [evaluate])
