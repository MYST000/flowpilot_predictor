"""Serial, resumable collection with immutable physical attempts and explicit retry policy."""

import fcntl
import importlib.metadata
import json
import shutil
import tarfile
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .cli import adapter_source_digest, task_key
from .code_audit import audit_attempt
from .code_campaign import runtime_fingerprint, sha
from .code_evaluation import evaluate_code
from .code_tasks import materialize_workspace
from .local_environment import LocalPilotEnvironment
from .prediction_export import build_prediction_rows, write_prediction_dataset
from .runner import run_task
from .sdk_provenance import check_sdk
from .tracing import write_json

COLLECTOR_LOCK_PATH = Path("/root/flowpilot_predictor/.collector.lock")


@contextmanager
def collector_lock():
    """All local code campaigns share fixed UIDs, so they must be serialized."""
    with COLLECTOR_LOCK_PATH.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another local collector holds the fixed-UID lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _read(path):
    return json.loads(Path(path).read_text())


def _derived_directory(attempt, name, complete):
    candidates = [attempt / name, *sorted(attempt.glob(name + "-retry-*"))]
    for path in candidates:
        if path.exists() and complete(path):
            return path, True
    path = attempt / name
    number = 2
    while path.exists():
        path = attempt / f"{name}-retry-{number:03d}"
        number += 1
    return path, False


def _finalize(bundle, config, attempt, python_bin):
    actor = _read(attempt / "result.json")
    invalid = actor.get("artifact_status") == "invalid_submission"
    solution = attempt / "artifacts/solution.py"
    if not invalid and not solution.is_file():
        raise RuntimeError(
            "No exported source; retained attempt requires infrastructure inspection"
        )
    marker = attempt / "actor_validated.json"
    if marker.exists():
        saved = _read(marker)
        if saved["events_sha256"] != sha(attempt / "events.jsonl"):
            raise ValueError("Validated actor trace was modified")
        if saved.get("solution_sha256") != (sha(solution) if solution.is_file() else None):
            raise ValueError("Validated actor submission was modified")
    audit = audit_attempt(attempt)
    if not audit["audit_passed"]:
        raise RuntimeError("Trace audit failed; raw attempt retained")
    build_prediction_rows(attempt)
    write_json(
        marker,
        {
            "events_sha256": sha(attempt / "events.jsonl"),
            "solution_sha256": sha(solution) if solution.is_file() else None,
        },
    )
    export, exported = _derived_directory(
        attempt, "prediction_dataset", lambda path: (path / "manifest.json").is_file()
    )
    if not exported:
        write_prediction_dataset([attempt], export)
    else:
        manifest = _read(export / "manifest.json")
        for name, info in manifest["files"].items():
            if sha(export / name) != info["sha256"]:
                raise ValueError("Completed prediction export was modified")
        for source in manifest["sources"]:
            if source["events_sha256"] != sha(attempt / "events.jsonl"):
                raise ValueError("Completed prediction export has stale source")
            if source.get("public_task_sha256") != sha(attempt / "public_task.json"):
                raise ValueError("Completed prediction export has stale task metadata")
    if invalid:
        evaluation = {
            "task_id": bundle.task.task_id,
            "benchmark": bundle.kind,
            "status": "evaluated",
            "passed": False,
            "actor_feedback": False,
            "details": {"failure_kind": "invalid_submission"},
        }
        evaluation_dir = attempt / "evaluation-invalid-submission"
        write_json(evaluation_dir / "evaluation.json", evaluation)
    else:
        evaluation_dir, evaluated = _derived_directory(
            attempt,
            "evaluation",
            lambda path: (
                (path / "evaluation.json").is_file()
                and _read(path / "evaluation.json").get("status") == "evaluated"
            ),
        )
        if evaluated:
            evaluation = _read(evaluation_dir / "evaluation.json")
            if evaluation.get("code_sha256") != sha(solution):
                raise ValueError("Completed evaluation has stale submission")
        else:
            evaluation = evaluate_code(
                bundle, solution.read_text(), config, evaluation_dir, python_bin=python_bin
            )
        if evaluation["status"] != "evaluated":
            raise RuntimeError(
                "Independent evaluator failed; raw actor retained for postprocessing retry"
            )
    record = {
        "actor": actor,
        "audit": audit,
        "evaluation": evaluation,
        "attempt": str(attempt),
        "prediction_dataset": export.name,
        "evaluation_directory": evaluation_dir.name,
    }
    write_json(attempt / "attempt_record.json", record)
    return record


def execute_attempt(bundle, config, attempt, python_bin, run_id):
    task_root = Path(tempfile.mkdtemp(prefix="flowpilot-code-local-actor-", dir="/tmp"))
    try:
        task = materialize_workspace(bundle, task_root / "repo")
        environment = LocalPilotEnvironment(
            config,
            task,
            attempt / "artifacts",
            task_root=task_root,
            python_bin=python_bin,
            uid=63111,
        )
        print("ACTOR_START", bundle.kind, task.task_id, attempt.name, flush=True)
        run_task(config, task, attempt, environment=environment, run_id=run_id)
        write_json(attempt / "public_files.json", bundle.public_files)
        with tarfile.open(
            attempt / "artifacts/final_workspace.tar.gz", "w:gz", dereference=False
        ) as archive:
            archive.add(
                task_root / "repo",
                arcname="repo",
                filter=lambda info: (
                    None
                    if any(
                        part in {".git", "__pycache__", ".pytest_cache"}
                        for part in info.name.split("/")
                    )
                    or info.size > 10_000_000
                    else info
                ),
            )
        return _finalize(bundle, config, attempt, python_bin)
    finally:
        shutil.rmtree(task_root)


def _frozen_run(bundles, config, root, python_bin, run_id, resume):
    validation = _read(root / "validation.json")
    source = adapter_source_digest()
    runtime = runtime_fingerprint(python_bin)
    selection_hash = sha(root / "selection.json")
    packages = {p: importlib.metadata.version(p) for p in ("openhands-sdk", "litellm", "pydantic")}
    if (
        not validation["all_checks_passed"]
        or validation["selection_sha256"] != selection_hash
        or validation.get("adapter_source_sha256") != source
        or validation.get("runtime_fingerprint") != runtime
    ):
        raise ValueError(
            "Frozen validation/data/source/runtime gate failed; use a new run and validate"
        )
    path = root / "manifest.json"
    if path.exists():
        if not resume:
            raise ValueError("Run already exists; use --resume, never overwrite attempts")
        manifest = _read(path)
        if manifest.get("collection_schema_version") != 2:
            raise ValueError("Legacy run has no resumable registry; retain it and use a new run")
        if any(
            manifest.get(key) != value
            for key, value in {
                "selection_sha256": selection_hash,
                "adapter_source_sha256": source,
                "runtime_fingerprint": runtime,
                "run_id": run_id,
                "config": config.to_dict(),
                "packages": packages,
            }.items()
        ):
            raise ValueError(
                "Cannot resume with changed source/config/selection/runtime/run identity"
            )
        if not (root / "attempt_registry.json").exists():
            raise ValueError("Missing attempt registry; do not infer completed work")
        return source, _read(root / "attempt_registry.json")
    if resume:
        raise ValueError("Cannot --resume a run that has not started")
    sdk = check_sdk(config.runtime)
    manifest = {
        "collection_schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "selection_sha256": selection_hash,
        "adapter_source_sha256": source,
        "runtime_fingerprint": runtime,
        "sdk_provenance": sdk,
        "config": config.to_dict(),
        "python_bin": str(python_bin),
        "single_attempt": False,
        "attempt_policy": "first protocol-valid attempt; ordinary model failures never retried",
        "model_service": {"model": config.llm.model, "base_url": config.llm.base_url},
        "packages": packages,
    }
    registry = {
        "schema_version": 1,
        "selection_sha256": selection_hash,
        "tasks": {b.task.task_id: [] for b in bundles},
    }
    write_json(root / "attempt_registry.json", registry)
    write_json(path, manifest)
    with tarfile.open(root / "adapter_source_snapshot.tar.gz", "w:gz") as archive:
        archive.add(
            Path(__file__).parent,
            arcname="benchmark_adapters",
            filter=lambda info: None if "__pycache__" in info.name else info,
        )
    return source, registry


def _validate_registry(root, registry):
    selection = _read(root / "selection.json")
    task_ids = [task["task_id"] for task in selection["tasks"]]
    if (
        registry.get("schema_version") != 1
        or registry.get("selection_sha256") != sha(root / "selection.json")
        or list(registry.get("tasks", {})) != task_ids
    ):
        raise ValueError("Attempt registry does not match frozen selection")
    for task_id, entries in registry["tasks"].items():
        for number, entry in enumerate(entries, 1):
            expected = Path("tasks") / task_key(task_id) / f"attempt-{number:03d}"
            path = root / entry["path"]
            if entry["path"] != str(expected) or not path.resolve().is_relative_to(
                root.resolve() / "tasks"
            ):
                raise ValueError("Invalid attempt registry path")
            if entry.get("status") not in {
                "running",
                "accepted",
                "quarantined",
                "postprocess_pending",
            }:
                raise ValueError("Unknown attempt registry status")


def _check_accepted(root, entry):
    attempt = root / entry["path"]
    for name, expected in entry.get("files_sha256", {}).items():
        path = attempt / name
        if not path.resolve().is_relative_to(attempt.resolve()):
            raise ValueError("Invalid accepted artifact path")
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Accepted attempt was modified; stop for inspection")
    if (
        sha(attempt / "attempt_record.json") != entry["record_sha256"]
        or sha(attempt / "events.jsonl") != entry["events_sha256"]
    ):
        raise ValueError("Accepted attempt was modified; stop for inspection")
    return attempt


def accepted_attempts(root):
    root = Path(root)
    registry = _read(root / "attempt_registry.json")
    _validate_registry(root, registry)
    result = []
    for attempts in registry["tasks"].values():
        accepted = [entry for entry in attempts if entry["status"] == "accepted"]
        if len(accepted) > 1:
            raise ValueError("Multiple primary attempts for one task")
        if accepted:
            result.append(_check_accepted(root, accepted[0]))
    return result


def _summarize(root, registry):
    results, physical, requests, tools, wall_time = [], 0, 0, 0, 0.0
    missing_result = 0
    unreadable_events = []
    for attempts in registry["tasks"].values():
        for entry in attempts:
            physical += 1
            path = root / entry["path"]
            result_path = path / "result.json"
            if result_path.is_file():
                try:
                    duration = _read(result_path).get("duration_s")
                    if isinstance(duration, (int, float)):
                        wall_time += duration
                    else:
                        missing_result += 1
                except (ValueError, OSError):
                    missing_result += 1
            else:
                missing_result += 1
            if (path / "events.jsonl").exists():
                for line in (path / "events.jsonl").read_text().splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        unreadable_events.append(entry["path"])
                        continue
                    if not isinstance(event, dict):
                        unreadable_events.append(entry["path"])
                        continue
                    requests += event.get("event") == "llm_request_prepared"
                    tools += event.get("event") == "tool_start"
            if entry["status"] == "accepted":
                record = _read(path / "attempt_record.json")
                results.append(
                    {
                        "task_id": record["actor"]["task_id"],
                        "attempt": entry["path"],
                        "execution_status": record["actor"]["execution_status"],
                        "passed": record["evaluation"]["passed"],
                        "audit_passed": record["audit"]["audit_passed"],
                    }
                )
    summary = {
        "planned_tasks": len(registry["tasks"]),
        "attempted": len(results),
        "stage_complete": len(results) == len(registry["tasks"]),
        "results": results,
        "benchmark_passed": sum(r["passed"] for r in results),
        "physical_attempts": physical,
        "all_attempt_llm_requests": requests,
        "all_attempt_tool_starts": tools,
        "recorded_actor_duration_s": wall_time,
        "attempts_without_final_wall_time": missing_result,
        "unreadable_event_attempts": sorted(set(unreadable_events)),
        "cost_note": "Counts include quarantined attempts; missing final wall time is unknown, not zero.",
    }
    write_json(root / "summary.json", summary)
    write_json(root / "primary_attempts.json", {r["task_id"]: r["attempt"] for r in results})
    return summary


def _accept(entry, attempt):
    files = (
        "public_task.json",
        "profile.json",
        "artifacts/solution.py",
        "evaluation/evaluation.json",
        "prediction_dataset/manifest.json",
    )
    record = _read(attempt / "attempt_record.json")
    files += tuple(
        str(Path(record[key]) / filename)
        for key, filename in (
            ("prediction_dataset", "manifest.json"),
            ("evaluation_directory", "evaluation.json"),
        )
        if key in record
    )
    entry["files_sha256"] = {
        name: sha(attempt / name) for name in files if (attempt / name).is_file()
    }
    entry.update(
        status="accepted",
        completed_at=datetime.now(UTC).isoformat(),
        record_sha256=sha(attempt / "attempt_record.json"),
        events_sha256=sha(attempt / "events.jsonl"),
    )


def _actor_ended(attempt):
    try:
        lines = (attempt / "events.jsonl").read_text().splitlines()
        return bool(lines) and json.loads(lines[-1]).get("event") == "task_end"
    except (OSError, ValueError, AttributeError):
        return False


def run_collection(
    bundles, config, root, python_bin, *, run_id, resume=False, retry_infrastructure=False
):
    root = Path(root)
    if retry_infrastructure and not resume:
        raise ValueError("--retry-infrastructure requires --resume")
    source_hash, registry = _frozen_run(bundles, config, root, python_bin, run_id, resume)
    _validate_registry(root, registry)
    for bundle in bundles:
        if adapter_source_digest() != source_hash:
            raise ValueError("Adapter source changed mid-collection; stopping")
        entries = registry["tasks"][bundle.task.task_id]
        accepted = [entry for entry in entries if entry["status"] == "accepted"]
        if len(accepted) > 1:
            raise ValueError("Multiple accepted attempts")
        if accepted:
            _check_accepted(root, accepted[0])
            continue
        if entries and (
            entries[-1]["status"] in {"running", "postprocess_pending"}
            or _actor_ended(root / entries[-1]["path"])
        ):
            entry = entries[-1]
            attempt = root / entry["path"]
            try:
                # A crash after recording actor completion can resume independent evaluation.
                events = [
                    json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()
                ]
                if not events or events[-1]["event"] != "task_end":
                    raise RuntimeError("Actor did not finish recording task_end")
                _finalize(bundle, config, attempt, python_bin)
                _accept(entry, attempt)
                write_json(root / "attempt_registry.json", registry)
                continue
            except Exception as exc:
                pending = (attempt / "actor_validated.json").is_file()
                entry.update(
                    status="postprocess_pending" if pending else "quarantined",
                    reason=f"Interrupted attempt: {type(exc).__name__}: {exc}",
                )
                write_json(root / "attempt_registry.json", registry)
                if pending:
                    _summarize(root, registry)
                    raise RuntimeError(
                        "Valid actor retained; --resume retries only postprocessing"
                    ) from exc
        if entries and not retry_infrastructure:
            _summarize(root, registry)
            raise ValueError(
                "Retained infrastructure attempt; inspect it, then use --resume --retry-infrastructure"
            )
        attempt = root / "tasks" / task_key(bundle.task.task_id) / f"attempt-{len(entries) + 1:03d}"
        if attempt.exists():
            raise ValueError("Unregistered attempt directory exists; do not overwrite")
        entry = {
            "path": str(attempt.relative_to(root)),
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "retry_reason": entries[-1].get("reason") if entries else None,
        }
        entries.append(entry)
        write_json(root / "attempt_registry.json", registry)
        try:
            record = execute_attempt(bundle, config, attempt, python_bin, run_id)
            if not record["audit"]["audit_passed"] or record["evaluation"]["status"] != "evaluated":
                raise RuntimeError("Trace/evaluator gate failed")
            _accept(entry, attempt)
            print("ACTOR_RESULT", bundle.task.task_id, record["evaluation"]["passed"], flush=True)
        except BaseException as exc:
            entry.update(
                status="postprocess_pending"
                if (attempt / "actor_validated.json").is_file()
                else "quarantined",
                reason=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            write_json(root / "attempt_registry.json", registry)
            _summarize(root, registry)
    return _summarize(root, registry)
