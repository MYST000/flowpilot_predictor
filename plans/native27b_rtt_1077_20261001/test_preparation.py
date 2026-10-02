"""Synthetic contract checks only: no estimator fitting or real-data prediction."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PLAN = Path(__file__).resolve().parent
sys.path.insert(0, str(PLAN / "code"))
from predictor.cli import validate_mode
from predictor.evaluation import report

from prepare_data import append_sample, assert_disjoint
from summarize import summarize
from workflow import ALGORITHMS, Runner, commands


def fixture():
    row = {
        "source_attempt": "/synthetic/attempt",
        "request_id": "r",
        "task_id": "t",
        "task_group_id": "g",
        "research_split": "fit",
        "adapter": "hotpot",
        "tool_start_monotonic_ns": 20,
        "tool_end_monotonic_ns": 40,
        "call": {
            "tool_call_id": "c",
            "tool_name": "search",
            "arguments_parsed": {"query": "hello"},
            "batch_index": 0,
        },
        "labels": {
            "execution_status": "completed",
            "round_trip_ms": 2.0,
            "arguments_json_valid": True,
            "normal_completion_right_censored": False,
        },
    }
    inp = {
        "clock_domain": "synthetic",
        "dataset_revision": "v1",
        "replica_id": "actor",
        "features": {
            "monotonic_ns": 5,
            "tool_execution_profile": {
                "intra_task_tool_concurrency": 1,
                "tool_timeout_seconds": 30,
            },
            "environment_known_at_t0": {"backend": "hotpot_rpc"},
            "prior_tool_executions": [],
            "t0_features": {"client_load": {"sampled_monotonic_ns": 4}},
            "budget_at_t0": {"remaining_seconds": 100},
        },
    }
    target = {"labels": {"actions": [{"tool_name": "search"}]}}

    def event(ns, seq):
        return {"clock_domain": "synthetic", "monotonic_ns": ns, "seq": seq}

    events = (
        {"r": event(10, 1)},
        {"c": event(20, 2)},
        {"c": event(40, 3)},
        {"search": {"name": "search", "parameters": {"properties": {"top_k": {"default": 5}}}}},
    )
    return row, inp, target, events


class Contracts(unittest.TestCase):
    def test_current_label_and_result_cannot_change_features(self):
        row, inp, target, events = fixture()
        a = append_sample(row, inp, target, events, "fit", "batch")
        row["labels"]["round_trip_ms"] = 999.0
        row["labels"]["executor_duration_ms"] = 998.0
        row["labels"]["execution_status"] = "execution_error"
        b = append_sample(row, inp, target, events, "fit", "batch")
        self.assertEqual(a["context"], b["context"])
        self.assertEqual(a["context"]["arguments"]["top_k"], 5)
        self.assertNotIn("qualified", a["context"])
        self.assertEqual(b["labels"]["round_trip_ms"], 999.0)

    def test_future_history_is_rejected(self):
        row, inp, target, events = fixture()
        inp["features"]["prior_tool_executions"] = [
            {
                "tool_call_id": "c",
                "tool_name": "search",
                "round_trip_ms": 2.0,
                "execution_status": "completed",
            }
        ]
        with self.assertRaisesRegex(ValueError, "future history"):
            append_sample(row, inp, target, events, "fit", "batch")

    def test_cross_clock_label_is_rejected(self):
        row, inp, target, events = fixture()
        events[2]["c"]["clock_domain"] = "other"
        with self.assertRaisesRegex(ValueError, "clock domain"):
            append_sample(row, inp, target, events, "fit", "batch")

    def test_group_overlap_and_calibrated_online_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_disjoint({"fit": {"g"}, "test": {"g"}})
        with self.assertRaisesRegex(ValueError, "frozen calibration"):
            validate_mode({"model": SimpleNamespace(name="ewma"), "calibrated": True}, "online")

    def test_pipeline_stage_order_and_bounded_parallelism(self):
        stages = commands(Path("/data1/ql_flowpilot_predictor/predictor_experiments/synthetic"))
        self.assertEqual(
            [s[0] for s in stages],
            ["train", "calibration", "test", "forward_online", "isolated_latency", "summary"],
        )
        self.assertEqual(len(stages[0][1]), 5)
        self.assertTrue(all(s[2] <= 5 for s in stages))
        for _, cmd in stages[2][1]:
            self.assertIn("--allow-test", cmd)
        self.assertIn("fit_primary", stages[3][1][0][1])
        self.assertIn("tune_forward", stages[3][1][1][1])
        self.assertTrue(all(s[2] == 1 for s in stages[4:]))

    def test_cancellation_only_stops_owned_child_groups(self):
        with tempfile.TemporaryDirectory() as name:
            runner = Runner(Path(name))
            owned = subprocess.Popen(["/usr/bin/sleep", "20"], start_new_session=True)
            control = subprocess.Popen(["/usr/bin/sleep", "20"], start_new_session=True)
            try:
                runner.children = [owned]
                runner.cancel_children()
                self.assertIsNotNone(owned.poll())
                self.assertIsNone(control.poll())
            finally:
                for child in (owned, control):
                    if child.poll() is None:
                        child.terminate()
                    child.wait(timeout=5)

    def test_complete_report_from_synthetic_predictions_only(self):
        pred = {
            "sample_id": "sample",
            "task_group_id": "g",
            "adapter": "hotpot",
            "tool": "search",
            "y_ms": 10.0,
            "predict_ms": 0.1,
            "update_ms": 0.0,
            "execution_status": "completed",
            "prediction": {
                "duration_ms": {"q10": 1.0, "q50": 8.0, "q90": 20.0, "q99": 30.0},
                "fallback": {"used": False, "reason": None},
                "support": {"q99_low_support": True},
            },
            "score": {
                "pinball_ms": {"q10": 0.9, "q50": 1.0, "q90": 1.0, "q99": 0.2},
                "absolute_q50_error": 2.0,
            },
            "envelope": {"valid_for": {"request_id": "r"}, "per_call": [{"tool_call_id": "c"}]},
        }
        metrics = report([pred], repeats=0)
        self.assertEqual(
            metrics["micro"]["coverage"], {"q10": 0.0, "q50": 0.0, "q90": 1.0, "q99": 1.0}
        )
        with tempfile.TemporaryDirectory(prefix="predictor-prep-test-") as name:
            root = Path(name)
            streams = (
                [("train/" + a, a, "tune", False) for a in ALGORITHMS]
                + [
                    ("final/" + a + "_test_" + s, a, "test", s == "calibrated")
                    for a in ALGORITHMS
                    for s in ("raw", "calibrated")
                ]
                + [
                    ("online/ewma_forward_fit", "ewma", "tune_forward", False),
                    ("online/ewma_forward_online", "ewma", "tune_forward", False),
                ]
            )
            for sub, a, split, calibrated in streams:
                path = root / sub
                path.mkdir(parents=True)
                manifest = {
                    "status": "complete",
                    "smoke_only": False,
                    "algorithm": a,
                    "training_split": "fit_primary" if split == "tune_forward" else "fit",
                    "evaluation_split": split,
                    "evaluation_mode": "online" if sub.endswith("_online") else "frozen",
                    "calibration_version": "synthetic" if calibrated else None,
                }
                (path / "manifest.json").write_text(json.dumps(manifest))
                (path / "metrics.json").write_text(json.dumps(metrics))
                (path / "predictions.jsonl").write_text(json.dumps(pred) + "\n")
            (root / "latency").mkdir()
            for a in ALGORITHMS:
                (root / "latency" / (a + ".json")).write_text(
                    json.dumps(
                        {
                            "algorithm": a,
                            "threads": 8,
                            "overall": {"n": 1, "p95_ms": 0.1},
                            "scope": "synthetic",
                        }
                    )
                )
            (root / "frozen_before_test.json").write_text(
                json.dumps({"tune_candidate": "lightgbm"})
            )
            summarize(root)
            self.assertIn("Q99覆盖", (root / "reports/REPORT.md").read_text())
            self.assertEqual(len((root / "reports/predictions.csv").read_text().splitlines()), 18)


if __name__ == "__main__":
    unittest.main()
