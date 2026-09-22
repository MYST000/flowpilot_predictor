import json
from types import SimpleNamespace

import pytest

from benchmark_adapters.code_campaign import freeze_selection, sha
from benchmark_adapters.code_tasks import CodeBundle
from benchmark_adapters.config import Config
from benchmark_adapters.contracts import Task
from benchmark_adapters.tracing import write_json


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    from benchmark_adapters import collection_runner as runner

    bundles = [
        CodeBundle("quixbugs", Task("quix", "rev", "dev", name, "fix"), {}, "a.py", {})
        for name in ("one", "two")
    ]
    config = Config()
    root = freeze_selection(tmp_path, "quixbugs", bundles, {}, config)
    monkeypatch.setattr(runner, "adapter_source_digest", lambda: "source")
    monkeypatch.setattr(runner, "runtime_fingerprint", lambda _: "runtime")
    monkeypatch.setattr(runner, "check_sdk", lambda _: {"checkout_commit": "commit"})
    write_json(
        root / "validation.json",
        {
            "all_checks_passed": True,
            "selection_sha256": sha(root / "selection.json"),
            "adapter_source_sha256": "source",
            "runtime_fingerprint": "runtime",
        },
    )
    calls = []

    def execute(bundle, cfg, attempt, python_bin, run_id):
        calls.append((bundle.task.task_id, attempt.name))
        attempt.mkdir(parents=True, exist_ok=True)
        write_json(
            attempt / "result.json",
            dict(task_id=bundle.task.task_id, execution_status="completed", duration_s=1),
        )
        (attempt / "events.jsonl").write_text("")
        record = {
            "actor": json.loads((attempt / "result.json").read_text()),
            "audit": {"audit_passed": True},
            "evaluation": {"status": "evaluated", "passed": False},
            "attempt": str(attempt),
        }
        write_json(attempt / "attempt_record.json", record)
        return record

    monkeypatch.setattr(runner, "execute_attempt", execute)
    return SimpleNamespace(
        runner=runner, bundles=bundles, config=config, root=root, calls=calls, execute=execute
    )


def test_resume_never_reruns_valid_model_failures(campaign):
    c = campaign
    result = c.runner.run_collection(c.bundles, c.config, c.root, "/usr/bin", run_id="r")
    assert result["stage_complete"] and result["benchmark_passed"] == 0
    before = list(c.calls)
    c.runner.run_collection(c.bundles, c.config, c.root, "/usr/bin", run_id="r", resume=True)
    assert c.calls == before
    with pytest.raises(ValueError, match="resume"):
        c.runner.run_collection(c.bundles, c.config, c.root, "/usr/bin", run_id="r")


def test_interruption_retains_old_attempt_requires_explicit_infrastructure_retry(
    campaign, monkeypatch
):
    c = campaign

    def interrupted(bundle, cfg, attempt, python_bin, run_id):
        if bundle.task.task_id == "two" and attempt.name == "attempt-001":
            attempt.mkdir(parents=True, exist_ok=True)
            (attempt / "events.jsonl").write_text("preserved raw evidence\n")
            raise RuntimeError("fixture infrastructure failure")
        return c.execute(bundle, cfg, attempt, python_bin, run_id)

    monkeypatch.setattr(c.runner, "execute_attempt", interrupted)
    with pytest.raises(RuntimeError):
        c.runner.run_collection(c.bundles, c.config, c.root, "/usr/bin", run_id="r")
    old = c.root / "tasks/two/attempt-001/events.jsonl"
    assert old.read_text() == "preserved raw evidence\n"
    with pytest.raises(ValueError, match="retry-infrastructure"):
        c.runner.run_collection(c.bundles, c.config, c.root, "/usr/bin", run_id="r", resume=True)
    result = c.runner.run_collection(
        c.bundles, c.config, c.root, "/usr/bin", run_id="r", resume=True, retry_infrastructure=True
    )
    assert result["stage_complete"] and result["physical_attempts"] == 3
    assert c.calls.count(("one", "attempt-001")) == 1
    assert ("two", "attempt-002") in c.calls
    assert old.read_text() == "preserved raw evidence\n"


def test_changed_source_cannot_resume_an_old_frozen_run(campaign, monkeypatch):
    c = campaign
    c.runner.run_collection(c.bundles, c.config, c.root, "/usr/bin", run_id="r")
    monkeypatch.setattr(c.runner, "adapter_source_digest", lambda: "changed")
    with pytest.raises(ValueError, match="validation|source"):
        c.runner.run_collection(c.bundles, c.config, c.root, "/usr/bin", run_id="r", resume=True)


def test_collector_lock_rejects_a_second_collector(tmp_path, monkeypatch):
    import subprocess
    import sys

    from benchmark_adapters import collection_runner as runner

    lock_path = tmp_path / "collector.lock"
    monkeypatch.setattr(runner, "COLLECTOR_LOCK_PATH", lock_path)
    probe = "import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
    with runner.collector_lock():
        result = subprocess.run([sys.executable, "-c", probe, str(lock_path)], capture_output=True)
        assert result.returncode != 0
    result = subprocess.run([sys.executable, "-c", probe, str(lock_path)], capture_output=True)
    assert result.returncode == 0


def test_registry_paths_and_accepted_hashes_are_checked_on_export(campaign):
    c = campaign
    c.runner.run_collection(c.bundles, c.config, c.root, "/usr/bin", run_id="r")
    assert len(c.runner.accepted_attempts(c.root)) == 2
    path = c.root / "attempt_registry.json"
    original = json.loads(path.read_text())
    corrupted = json.loads(path.read_text())
    corrupted["tasks"]["one"][0]["path"] = "../../outside"
    write_json(path, corrupted)
    with pytest.raises(ValueError, match="path"):
        c.runner.accepted_attempts(c.root)
    write_json(path, original)
    (c.root / original["tasks"]["one"][0]["path"] / "events.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="modified"):
        c.runner.accepted_attempts(c.root)


@pytest.mark.parametrize("validated_marker", [True, False])
def test_postprocessing_failure_resumes_same_actor(campaign, monkeypatch, validated_marker):
    c = campaign

    def execute(bundle, cfg, attempt, python_bin, run_id):
        record = c.execute(bundle, cfg, attempt, python_bin, run_id)
        (attempt / "events.jsonl").write_text('{"event":"task_end"}\n')
        if validated_marker:
            write_json(
                attempt / "actor_validated.json", {"events_sha256": sha(attempt / "events.jsonl")}
            )
        if bundle.task.task_id == "one":
            raise RuntimeError("evaluator unavailable after valid actor")
        return record

    monkeypatch.setattr(c.runner, "execute_attempt", execute)
    with pytest.raises(RuntimeError):
        c.runner.run_collection(c.bundles, c.config, c.root, "/usr/bin", run_id="r")

    def finalize(bundle, cfg, attempt, python_bin):
        return json.loads((attempt / "attempt_record.json").read_text())

    monkeypatch.setattr(c.runner, "_finalize", finalize)
    result = c.runner.run_collection(
        c.bundles, c.config, c.root, "/usr/bin", run_id="r", resume=True
    )
    assert result["stage_complete"] and result["physical_attempts"] == 2
    assert c.calls == [("one", "attempt-001"), ("two", "attempt-001")]


def test_finalize_retries_only_derived_outputs_and_accepts_invalid_submission(
    tmp_path, monkeypatch
):
    from benchmark_adapters import collection_runner as runner

    attempt = tmp_path / "attempt"
    attempt.mkdir()
    write_json(attempt / "result.json", {"task_id": "one", "artifact_status": "invalid_submission"})
    (attempt / "events.jsonl").write_text('{"event":"task_end"}\n')
    (attempt / "prediction_dataset").mkdir()
    (attempt / "prediction_dataset" / "FAILED.json").write_text("preserve me")
    monkeypatch.setattr(runner, "audit_attempt", lambda _: {"audit_passed": True})
    monkeypatch.setattr(runner, "build_prediction_rows", lambda _: [])

    def export(attempts, path):
        path.mkdir()
        write_json(path / "manifest.json", {"sources": []})

    monkeypatch.setattr(runner, "write_prediction_dataset", export)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid submission must not execute an evaluator or rerun actor")

    monkeypatch.setattr(runner, "evaluate_code", forbidden)
    bundle = CodeBundle("quixbugs", Task("quix", "rev", "dev", "one", "fix"), {}, "a.py", {})
    record = runner._finalize(bundle, Config(), attempt, "/usr/bin")
    assert record["evaluation"]["status"] == "evaluated"
    assert record["evaluation"]["passed"] is False
    assert record["evaluation"]["details"]["failure_kind"] == "invalid_submission"
    assert (attempt / "prediction_dataset" / "FAILED.json").read_text() == "preserve me"
    assert record["prediction_dataset"] != "prediction_dataset"
