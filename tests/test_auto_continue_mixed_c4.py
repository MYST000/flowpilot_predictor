"""Safety gates for unattended mixed C4 continuation."""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/auto_continue_mixed_c4.py"
SPEC = importlib.util.spec_from_file_location("auto_continue_mixed_c4", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_split(tmp_path, *, status="collected", trace_issues=None):
    root = tmp_path / "fit"
    (root / "receipts").mkdir(parents=True)
    (root / "prediction_dataset").mkdir()
    (root / "tool_timing_dataset").mkdir()
    adapters = ("hotpot", "browsecomp", "livecodebench")
    (root / "campaign.json").write_text(
        json.dumps({"jobs": [{"adapter": adapter} for adapter in adapters]})
    )
    (root / "collection_summary.json").write_text(json.dumps({"status": status}))
    (root / "tool_timing_dataset/audit.json").write_text(
        json.dumps({"trace_issues": trace_issues or []})
    )
    for name in ("inputs.jsonl", "targets.jsonl"):
        (root / "prediction_dataset" / name).write_text("{}\n")
    for adapter in adapters:
        (root / "receipts" / f"{adapter}.json").write_text(
            json.dumps({"status": "returned"})
        )
    with (root / "tool_timing_dataset/tools.jsonl").open("w") as output:
        for adapter in adapters:
            output.write(
                json.dumps(
                    {
                        "adapter": adapter,
                        "labels": {
                            "executed": True,
                            "executor_duration_ms": 4.0,
                            "round_trip_ms": 6.0,
                        },
                    }
                )
                + "\n"
            )
    return root


def test_verify_accepts_complete_split(tmp_path, monkeypatch):
    root = make_split(tmp_path)
    monkeypatch.setattr(MODULE, "split_root", lambda split: root)
    assert MODULE.verify_split("fit") == {
        "receipts": 3,
        "measured_calls": {"hotpot": 1, "browsecomp": 1, "livecodebench": 1},
    }


@pytest.mark.parametrize(
    "change,match",
    [
        ("incomplete", "collection status"),
        ("trace_issue", "trace issues"),
        ("missing_receipt", "receipts"),
        ("missing_timing", "no measured tool time"),
    ],
)
def test_verify_stops_on_incomplete_or_unusable_data(
    tmp_path, monkeypatch, change, match
):
    root = make_split(tmp_path)
    monkeypatch.setattr(MODULE, "split_root", lambda split: root)
    if change == "incomplete":
        (root / "collection_summary.json").write_text(
            json.dumps({"status": "interrupted_or_failed"})
        )
    elif change == "trace_issue":
        (root / "tool_timing_dataset/audit.json").write_text(
            json.dumps({"trace_issues": [{"reason": "no_trace"}]})
        )
    elif change == "missing_receipt":
        (root / "receipts/browsecomp.json").unlink()
    else:
        rows = [
            json.loads(line)
            for line in (root / "tool_timing_dataset/tools.jsonl")
            .read_text()
            .splitlines()
        ]
        rows[1]["labels"]["executor_duration_ms"] = None
        (root / "tool_timing_dataset/tools.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )
    with pytest.raises(ValueError, match=match):
        MODULE.verify_split("fit")


def test_watcher_dispatches_only_after_fit_audit_and_idle_shell(tmp_path, monkeypatch):
    root = make_split(tmp_path)
    monkeypatch.setattr(MODULE, "split_root", lambda split: root)
    monkeypatch.setattr(MODULE, "write_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(MODULE, "active_collectors", lambda: [])
    commands = iter(("python", "bash"))
    calls = []

    def fake_tmux(*args):
        calls.append(args)
        if args[0] == "display-message":
            return next(commands)
        return ""

    monkeypatch.setattr(MODULE, "tmux", fake_tmux)
    monkeypatch.setattr(MODULE.time, "sleep", lambda seconds: None)
    MODULE.watch("%9")
    assert sum(args[0] == "send-keys" for args in calls) == 2
    assert calls.index(next(args for args in calls if args[0] == "send-keys")) > 0


def test_watcher_does_not_dispatch_if_export_failed(tmp_path, monkeypatch):
    root = make_split(tmp_path)
    (root / "tool_timing_dataset/audit.json").unlink()
    monkeypatch.setattr(MODULE, "split_root", lambda split: root)
    monkeypatch.setattr(MODULE, "write_state", lambda *args, **kwargs: None)
    calls = []

    def fake_tmux(*args):
        calls.append(args)
        return "bash"

    monkeypatch.setattr(MODULE, "tmux", fake_tmux)
    with pytest.raises(ValueError, match="without tool_timing_dataset/audit.json"):
        MODULE.watch("%9")
    assert not any(args[0] == "send-keys" for args in calls)
