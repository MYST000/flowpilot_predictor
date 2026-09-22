"""Offline lifecycle checks; actor fixture emits genuine v2-compatible trace blobs."""

import json
import sqlite3
from pathlib import Path

import pytest


def collection_module():
    from benchmark_adapters import browsecomp_collection
    return browsecomp_collection


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def fixture_data(tmp_path, count=30):
    questions = jsonl(tmp_path / "questions.jsonl", [
        {"query_id": str(i), "query": f"Question {i}", "answer": "SECRET_ANSWER",
         "evidence_docs": [{"docid": f"d{i}", "text": "SECRET_EVIDENCE"}],
         "gold_docs": [{"docid": f"d{i}"}]} for i in range(count)
    ])
    corpus = jsonl(tmp_path / "corpus.jsonl", [
        {"docid": f"d{i}", "text": f"Public document {i}", "url": f"https://example.test/{i}"}
        for i in range(count)
    ])
    history = tmp_path / "history.txt"
    history.write_text("")
    return questions, corpus, history


def prepare_fixture(tmp_path):
    bc = collection_module()
    questions, corpus, history = fixture_data(tmp_path)
    prepared = tmp_path / "prepared"
    bc.prepare_data(questions, corpus, prepared, "fixture-revision", history)
    return prepared


def test_prepare_strips_private_data_and_builds_readonly_complete_index(tmp_path):
    bc = collection_module()
    prepared = prepare_fixture(tmp_path)
    public = (prepared / "public_questions.jsonl").read_text()
    selection = (prepared / "selection.jsonl").read_text()
    assert "SECRET" not in public + selection
    assert "evidence_docs" not in public + selection
    assert all(set(json.loads(line)) == {"query_id", "query"} for line in public.splitlines())
    db = prepared / "corpus.sqlite3"
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT count(*) FROM docs").fetchone()[0] == 30
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert db.stat().st_mode & 0o222 == 0
    assert (prepared / "private/questions.jsonl").is_file()
    assert not list(prepared.glob("normalized-corpus-*"))
    manifest = json.loads((prepared / "manifest.json").read_text())
    assert manifest["question_count"] == 30
    assert manifest["document_count"] == 30
    assert sum(manifest["split_counts"].values()) == 30
    with pytest.raises(ValueError, match="exists"):
        bc.prepare_data(tmp_path / "questions.jsonl", tmp_path / "corpus.jsonl", prepared,
                        "fixture-revision", tmp_path / "history.txt")


def test_group_splits_are_transitive_deterministic_and_history_quarantined(tmp_path):
    bc = collection_module()
    rows = [
        {"query_id": "a", "query": " Same   QUERY ", "evidence_docs": [{"docid": "x"}]},
        {"query_id": "b", "query": "same query", "gold_docs": [{"docid": "y"}]},
        {"query_id": "c", "query": "third", "evidence_docs": [{"docid": "y"}]},
        *[{"query_id": str(i), "query": f"Independent {i}"} for i in range(30)],
    ]
    left = bc.assign_splits(rows, {"a"})
    right = bc.assign_splits(list(reversed(rows)), {"a"})
    assert left == right
    assert {left[i]["research_split"] for i in ("a", "b", "c")} == {"historical_dev"}
    assert len({left[i]["task_group_id"] for i in ("a", "b", "c")}) == 1
    assert any(value["research_split"] == "dev" for value in left.values())
    group_splits = {}
    for value in left.values():
        group_splits.setdefault(value["task_group_id"], set()).add(value["research_split"])
    assert all(len(splits) == 1 for splits in group_splits.values())


def test_arrow_stream_corpus_and_required_history(tmp_path):
    import pyarrow as pa
    bc = collection_module()
    questions, _, history = fixture_data(tmp_path, count=2)
    arrow_dir = tmp_path / "arrow"
    arrow_dir.mkdir()
    for i in range(2):
        table = pa.table({"docid": [f"d{i}"], "text": [f"Body {i}"], "url": ["https://example.test"]})
        with pa.OSFile(str(arrow_dir / f"data-{i}.arrow"), "wb") as sink:
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
    bc.prepare_data(questions, arrow_dir, tmp_path / "prepared", "arrow-revision", history)
    manifest = json.loads((tmp_path / "prepared/manifest.json").read_text())
    assert manifest["document_count"] == 2
    with pytest.raises((ValueError, FileNotFoundError)):
        bc.prepare_data(questions, arrow_dir, tmp_path / "missing", "rev", tmp_path / "not-known.txt")


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    bc = collection_module()
    prepared = prepare_fixture(tmp_path)
    run_root = tmp_path / "run"
    profile_path = tmp_path / "config.json"
    service = tmp_path / "service.json"
    stable = {"service_model_name": "qwen3.5-9b", "max_model_len": 32768, "dtype": "bfloat16"}
    service.write_text(json.dumps({"fingerprint_payload": stable, "fingerprint": bc.digest_json(stable), "pid": 1}))
    bc.create_config(prepared, tmp_path / "sdk", run_root, profile_path,
                     "http://127.0.0.1:8000/v1", service)
    monkeypatch.setattr(bc, "check_sdk", lambda config: {
        "package_version": "1.31.1", "runtime_matches_baseline": True,
        "runtime_base_commit": config.sdk_commit, "editable_url": (Path(config.sdk_path) / "openhands-sdk").as_uri(),
    })
    monkeypatch.setattr(bc, "_check_sdk_import", lambda runtime: None)
    calls = []

    def actor(config, task, attempt, *, run_id):
        from benchmark_adapters.tracing import TraceRecorder, write_json
        attempt = Path(attempt)
        attempt.mkdir(parents=True)
        calls.append(attempt)
        identity = {"task_id": task.task_id, "dataset_id": task.dataset_id,
                    "dataset_revision": task.revision, "split": task.split,
                    "attempt_id": attempt.name, "run_id": run_id,
                    "conversation_id": "fixture-conversation", "environment_id": "fixture-index"}
        recorder = TraceRecorder(attempt, identity)
        write_json(attempt / "public_task.json", task.to_dict())
        write_json(attempt / "profile.json", config.to_dict())
        recorder.emit("task_start")
        recorder.request("request-1", {"model": config.llm.model, "messages": [
            {"role": "user", "content": task.instruction}], "tools": [], "max_tokens": 8192})
        response = {"id": "response-1", "choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": "Candidate answer"}}], "usage": {"completion_tokens": 3}}
        recorder.emit("llm_response", request_id="request-1", llm_response_id="response-1",
                      response=recorder.blob(response), duration_ms=1)
        recorder.response_decision("request-1", response)
        result = {**identity, "execution_status": "completed", "artifact_status": "valid",
                  "evaluation_status": "pending", "termination_reason": "finished", "errors": []}
        write_json(attempt / "submission.json", {"query_id": task.task_id, "result": [{"output": "Candidate answer"}]})
        write_json(attempt / "result.json", result)
        recorder.emit("task_end", result=result)
        recorder.close()
        return result

    monkeypatch.setattr(bc, "run_task", actor)
    return bc, profile_path, run_root, service, calls, actor


def test_profile_resume_skips_accepted_and_exports_v2(campaign, tmp_path):
    bc, profile, root, service, calls, _ = campaign
    config = json.loads(profile.read_text())
    assert config["llm"]["temperature"] == .6
    assert config["llm"]["top_k"] == 20
    assert config["runtime"]["max_iterations"] == 24
    first = bc.collect(profile, "dev", limit=1)
    assert first["accepted_tasks"] == 1
    raw_hash = bc.sha(calls[0] / "events.jsonl")
    bc.collect(profile, "dev", resume=True, limit=1)
    assert len(calls) == 1
    assert bc.sha(calls[0] / "events.jsonl") == raw_hash
    bc.export_run(root, tmp_path / "export")
    sample = json.loads((tmp_path / "export/samples.jsonl").read_text().splitlines()[0])
    assert sample["research_split"] == "dev"
    assert sample["task_group_id"]
    assert "SECRET" not in (tmp_path / "export/inputs.jsonl").read_text()
    assert first["evaluation_status"] == "pending"
    with pytest.raises((ValueError, FileExistsError)):
        bc.export_run(root, tmp_path / "export")


def test_resume_rejects_config_source_service_and_blob_drift(campaign, monkeypatch):
    bc, profile, root, service, calls, _ = campaign
    bc.collect(profile, "dev", limit=1)
    original = profile.read_text()
    changed = json.loads(original)
    changed["llm"]["temperature"] = .7
    profile.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="drift|changed|frozen"):
        bc.collect(profile, "dev", resume=True, limit=1)
    profile.write_text(original)
    original_service = service.read_text()
    manifest = json.loads(original_service)
    manifest["pid"] = 999
    service.write_text(json.dumps(manifest))
    bc.collect(profile, "dev", resume=True, limit=1)
    manifest["fingerprint_payload"]["dtype"] = "float16"
    manifest["fingerprint"] = bc.digest_json(manifest["fingerprint_payload"])
    service.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="drift|changed|frozen"):
        bc.collect(profile, "dev", resume=True, limit=1)
    service.write_text(original_service)
    with monkeypatch.context() as patch:
        patch.setattr(bc, "source_fingerprint", lambda: {"changed.py": "changed"})
        with pytest.raises(ValueError, match="drift|changed|frozen"):
            bc.collect(profile, "dev", resume=True, limit=1)
    blob = next((calls[0] / "blobs").glob("*.json"))
    blob.write_text("{}")
    with pytest.raises(ValueError, match="modified|hash|integrity"):
        bc.collect(profile, "dev", resume=True, limit=1)
    assert len(calls) == 1


def test_test_gate_and_explicit_infrastructure_retry(campaign, monkeypatch):
    bc, profile, root, _, calls, actor = campaign
    with pytest.raises(ValueError, match="release-test"):
        bc.collect(profile, "test", limit=1)
    def outage(config, task, attempt, *, run_id):
        actor(config, task, attempt, run_id=run_id)
        path = Path(attempt) / "result.json"
        result = json.loads(path.read_text())
        result.update(execution_status="llm_error", error_type="APIConnectionError")
        path.write_text(json.dumps(result))
        events = Path(attempt) / "events.jsonl"
        records = [json.loads(line) for line in events.read_text().splitlines()]
        records[-1]["result"] = result
        jsonl(events, records)
        return result
    monkeypatch.setattr(bc, "run_task", outage)
    with pytest.raises(RuntimeError, match="infrastructure"):
        bc.collect(profile, "dev", limit=1)
    assert len(calls) == 1
    with pytest.raises((ValueError, RuntimeError), match="retry-infrastructure"):
        bc.collect(profile, "dev", resume=True, limit=1)
    monkeypatch.setattr(bc, "run_task", actor)
    bc.collect(profile, "dev", resume=True, retry_infrastructure=True, limit=1)
    assert len(calls) == 2
    assert (calls[0] / "events.jsonl").exists()


def test_postprocess_failure_resumes_without_repeating_actor(campaign, monkeypatch):
    bc, profile, _, _, calls, _ = campaign
    original = bc.write_prediction_dataset
    def broken_export(*args, **kwargs):
        raise OSError("disk failure")
    monkeypatch.setattr(bc, "write_prediction_dataset", broken_export)
    with pytest.raises(RuntimeError, match="postprocess"):
        bc.collect(profile, "dev", limit=1)
    monkeypatch.setattr(bc, "write_prediction_dataset", original)
    bc.collect(profile, "dev", resume=True, limit=1)
    assert len(calls) == 1


@pytest.mark.parametrize("status, reason", [
    ("budget_exhausted", "max_llm_requests"), ("llm_error", "ContextWindowExceededError"),
    ("agent_error", "normal_bad_answer"),
])
def test_genuine_failures_are_accepted_without_actor_retry(campaign, monkeypatch, status, reason):
    bc, profile, _, _, calls, actor = campaign
    def unsuccessful(config, task, attempt, *, run_id):
        result = actor(config, task, attempt, run_id=run_id)
        result.update(execution_status=status, termination_reason=reason, error_type=reason)
        (Path(attempt) / "result.json").write_text(json.dumps(result))
        events = Path(attempt) / "events.jsonl"
        records = [json.loads(line) for line in events.read_text().splitlines()]
        records[-1]["result"] = result
        jsonl(events, records)
        return result
    monkeypatch.setattr(bc, "run_task", unsuccessful)
    bc.collect(profile, "dev", limit=1)
    bc.collect(profile, "dev", resume=True, retry_infrastructure=True, limit=1)
    assert len(calls) == 1


def test_actor_logs_are_preserved_and_locks_exclude_concurrent_collection(campaign, monkeypatch):
    bc, profile, root, _, calls, actor = campaign
    def noisy_actor(*args, **kwargs):
        print("actor diagnostic")
        return actor(*args, **kwargs)
    monkeypatch.setattr(bc, "run_task", noisy_actor)
    with bc.collector_lock(root):
        with pytest.raises(ValueError, match="lock"):
            bc.collect(profile, "dev", limit=1)
    bc.collect(profile, "dev", limit=1)
    assert "actor diagnostic" in (calls[0] / "actor.log").read_text()


def test_interrupted_actor_requires_explicit_retry_and_preserves_raw(campaign, monkeypatch):
    bc, profile, root, _, calls, actor = campaign
    def interrupted(config, task, attempt, *, run_id):
        Path(attempt).mkdir(parents=True)
        (Path(attempt) / "events.jsonl").write_text('{"event":"task_start"}\n')
        print("interrupted diagnostic")
        raise KeyboardInterrupt()
    monkeypatch.setattr(bc, "run_task", interrupted)
    with pytest.raises(KeyboardInterrupt):
        bc.collect(profile, "dev", limit=1)
    old = next((root / "tasks").glob("*/attempt-001"))
    old_hash = bc.sha(old / "events.jsonl")
    with pytest.raises(ValueError, match="retry-infrastructure"):
        bc.collect(profile, "dev", resume=True, limit=1)
    monkeypatch.setattr(bc, "run_task", actor)
    bc.collect(profile, "dev", resume=True, retry_infrastructure=True, limit=1)
    assert len(calls) == 1
    assert bc.sha(old / "events.jsonl") == old_hash
    assert "interrupted diagnostic" in (old / "actor.log").read_text()


def test_crash_after_actor_end_recovers_postprocessing_only(campaign, monkeypatch):
    bc, profile, root, _, calls, _ = campaign
    original = bc._finalize
    monkeypatch.setattr(bc, "_finalize", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        bc.collect(profile, "dev", limit=1)
    registry_path = root / "attempt_registry.json"
    registry = json.loads(registry_path.read_text())
    entry = next(entries[0] for entries in registry["tasks"].values() if entries)
    entry["status"] = "running"
    entry.pop("raw_hashes")
    registry_path.write_text(json.dumps(registry))
    monkeypatch.setattr(bc, "_finalize", original)
    bc.collect(profile, "dev", resume=True, limit=1)
    assert len(calls) == 1


def test_export_cannot_write_into_raw_attempt(campaign):
    bc, profile, root, _, calls, _ = campaign
    bc.collect(profile, "dev", limit=1)
    with pytest.raises(ValueError, match="raw|attempt"):
        bc.export_run(root, calls[0] / "nested-export")


def test_source_data_drift_prevents_model_calls(campaign):
    bc, profile, _, _, calls, _ = campaign
    config = json.loads(profile.read_text())
    public = Path(config["prepared_data"]) / "public_questions.jsonl"
    public.chmod(0o644)
    public.write_text(public.read_text() + "\n")
    with pytest.raises(ValueError, match="hash drift"):
        bc.collect(profile, "dev", limit=1)
    assert not calls


@pytest.mark.parametrize("transport_error, budget_reason, infrastructure", [
    ("APIConnectionError", None, True),
    ("InternalServerError", None, True),
    ("UnexpectedProviderError", None, True),
    ("ReadTimeout", None, True),
    ("ContextWindowExceededError", None, False),
    ("ReadTimeout", "task_timeout", False),
])
def test_wrapped_sdk_transport_errors_use_trace_root_cause(
        campaign, monkeypatch, transport_error, budget_reason, infrastructure):
    bc, profile, root, _, calls, actor = campaign
    def wrapped_error(config, task, attempt, *, run_id):
        result = actor(config, task, attempt, run_id=run_id)
        result.update(execution_status="budget_exhausted" if budget_reason else "llm_error",
                      termination_reason=budget_reason or "ConversationRunError",
                      error_type="ConversationRunError", artifact_status="valid")
        events_path = Path(attempt) / "events.jsonl"
        records = [json.loads(line) for line in events_path.read_text().splitlines()]
        prefix = records[:2]
        root_cause = {**records[-1], "event": "llm_error", "seq": prefix[-1]["seq"] + 1,
                      "error_type": transport_error, "request_id": "request-1"}
        root_cause.pop("result")
        final = {**records[-1], "seq": root_cause["seq"] + 1, "result": result}
        jsonl(events_path, [*prefix, root_cause, final])
        (Path(attempt) / "result.json").write_text(json.dumps(result))
        return result
    monkeypatch.setattr(bc, "run_task", wrapped_error)
    if infrastructure:
        with pytest.raises(RuntimeError, match="infrastructure"):
            bc.collect(profile, "dev", limit=1)
        summary = json.loads((root / "summary.json").read_text())
        assert summary["accepted_tasks"] == 0
        with pytest.raises(ValueError, match="retry-infrastructure"):
            bc.collect(profile, "dev", resume=True, limit=1)
    else:
        bc.collect(profile, "dev", limit=1)
        bc.collect(profile, "dev", resume=True, retry_infrastructure=True, limit=1)
    assert len(calls) == 1


def test_quarantined_raw_attempt_is_still_audited_after_successful_retry(campaign, monkeypatch):
    bc, profile, root, _, calls, actor = campaign
    def interrupted(config, task, attempt, *, run_id):
        Path(attempt).mkdir(parents=True)
        (Path(attempt) / "partial.txt").write_text("preserve me")
        raise KeyboardInterrupt()
    monkeypatch.setattr(bc, "run_task", interrupted)
    with pytest.raises(KeyboardInterrupt):
        bc.collect(profile, "dev", limit=1)
    old = next((root / "tasks").glob("*/attempt-001"))
    monkeypatch.setattr(bc, "run_task", actor)
    bc.collect(profile, "dev", resume=True, retry_infrastructure=True, limit=1)
    (old / "partial.txt").write_text("changed")
    with pytest.raises(ValueError, match="modified|integrity"):
        bc.collect(profile, "dev", resume=True, limit=1)
    assert len(calls) == 1


def test_host_lock_excludes_distinct_run_roots(tmp_path):
    bc = collection_module()
    with bc.collector_lock(tmp_path / "run-one"):
        with pytest.raises(ValueError, match="lock"):
            with bc.collector_lock(tmp_path / "run-two"):
                pytest.fail("Concurrent same-host collector entered")
