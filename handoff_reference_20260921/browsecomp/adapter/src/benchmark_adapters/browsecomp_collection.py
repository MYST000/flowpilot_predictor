"""Frozen-corpus BrowseComp research collection, independent of the LCB collector.

Gold answers and document texts stay in private preparation artifacts. Research
splits are group-disjoint subdivisions of the upstream test set, not leaderboard
splits. Collection never invokes a judge or treats an answer as benchmark success.
"""

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import shutil
import signal
import sqlite3
import sys
import tempfile
import unicodedata
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path

from .config import Config, DatasetConfig, LLMConfig, RetrievalConfig, RuntimeConfig, SDK_COMMIT
from .prediction_export import build_prediction_rows, checked_blob, write_prediction_dataset
from .retrieval import BrowseCompAdapter, build_index
from .runner import run_task
from .sdk_provenance import check_sdk
from .tracing import write_json

SEED = 20260920
FRACTIONS = {"dev": .05, "fit": .60, "tune": .15, "calibration": .10, "test": .10}
SPLITS = ("historical_dev", *FRACTIONS)
DATASET_ID = "Tevatron/browsecomp-plus"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _jsonl(path):
    # Real question rows contain multi-megabyte private document text: never
    # read the entire question file or retain unprojected records in memory.
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object in {path}")
                yield row


def _write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _docids(row):
    ids = set()
    for field in ("evidence_docs", "gold_docs"):
        for document in row.get(field) or []:
            value = document.get("docid") if isinstance(document, dict) else document
            if value is not None:
                ids.add(str(value))
    return ids


def _project_questions(path):
    rows, seen = [], set()
    for row in _jsonl(path):
        task_id = str(row["query_id"])
        query = row.get("query")
        if not task_id or task_id in seen or not isinstance(query, str) or not query.strip():
            raise ValueError("Question IDs must be unique and queries nonempty")
        seen.add(task_id)
        rows.append({"query_id": task_id, "query": query, "gold_docs": sorted(_docids(row))})
    if not rows:
        raise ValueError("Empty question data")
    return rows


def _history_ids(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("A known --history-ids file is required, even if confirmed empty")
    text = path.read_text().strip()
    if not text:
        return set()
    values = json.loads(text) if text.startswith("[") else text.splitlines()
    if not isinstance(values, list) or any(not isinstance(value, (str, int)) for value in values):
        raise ValueError("History IDs must be a JSON list or one ID per line")
    return {str(value).strip() for value in values if str(value).strip()}


def assign_splits(rows, history_ids, seed=SEED):
    """Union duplicate queries/shared evidence transitively, then split whole groups."""
    ids = [str(row["query_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate query IDs")
    if unknown := set(history_ids) - set(ids):
        raise ValueError(f"History IDs absent from this question revision: {sorted(unknown)}")
    parent = {value: value for value in ids}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    owners = {}
    for row in sorted(rows, key=lambda value: str(value["query_id"])):
        task_id = str(row["query_id"])
        query = " ".join(unicodedata.normalize("NFKC", row["query"]).casefold().split())
        for key in [("query", query), *[("doc", value) for value in sorted(_docids(row))]]:
            if key in owners:
                union(task_id, owners[key])
            else:
                owners[key] = task_id
    components = defaultdict(list)
    for task_id in sorted(ids):
        components[find(task_id)].append(task_id)
    groups = sorted(components.values())
    result, remaining = {}, []
    for group in groups:
        group_id = "browsecomp/" + digest_json(group)[:24]
        if set(group) & set(history_ids):
            for task_id in group:
                result[task_id] = {"task_group_id": group_id, "research_split": "historical_dev"}
        else:
            remaining.append((group_id, group))
    random.Random(seed).shuffle(remaining)
    total = sum(len(group) for _, group in remaining)
    counts = Counter()
    # Give development one whole group when possible; all remaining assignments
    # greedily fill the largest row deficit. Component sizes prevent exact ratios.
    for number, (group_id, group) in enumerate(remaining):
        split = "dev" if number == 0 and len(remaining) >= 2 else max(
            FRACTIONS, key=lambda name: FRACTIONS[name] * total - counts[name])
        counts[split] += len(group)
        for task_id in group:
            result[task_id] = {"task_group_id": group_id, "research_split": split}
    return {task_id: result[task_id] for task_id in sorted(result)}


def _source_manifest(path):
    path = Path(path).resolve()
    paths = sorted(file for file in path.rglob("*") if file.is_file()) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"Empty input source: {path}")
    files = {str(file.relative_to(path)) if path.is_dir() else file.name:
             {"sha256": sha(file), "size": file.stat().st_size} for file in paths}
    return {"path": str(path), "files": files, "sha256": digest_json(files)}


def _corpus_rows(path):
    path = Path(path)
    if path.is_file():
        yield from _jsonl(path)
        return
    import pyarrow as pa
    state = _read(path / "state.json") if (path / "state.json").is_file() else {}
    shards = ([path / item["filename"] for item in state["_data_files"]]
              if state.get("_data_files") else sorted(path.glob("*.arrow")))
    if not shards:
        raise ValueError("Corpus directory contains no Arrow data shards")
    for shard in shards:
        if not shard.resolve().is_relative_to(path.resolve()):
            raise ValueError("Arrow shard escapes corpus directory")
        with pa.memory_map(str(shard), "r") as source:
            try:
                batches = pa.ipc.open_stream(source)
            except pa.ArrowInvalid:
                source.seek(0)
                reader = pa.ipc.open_file(source)
                batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
            for batch in batches:
                if not {"docid", "text"} <= set(batch.schema.names):
                    raise ValueError("Arrow corpus requires docid and text")
                yield from batch.to_pylist()


def prepare_data(questions, corpus, output, revision, history_ids):
    output, questions, corpus = Path(output).resolve(), Path(questions).resolve(), Path(corpus).resolve()
    if output.exists():
        raise ValueError("Prepared output already exists; use a new directory")
    if not revision or not questions.is_file() or not corpus.exists():
        raise ValueError("Existing questions/corpus and a frozen revision are required")
    history = _history_ids(history_ids)
    projected = _project_questions(questions)
    split_map = assign_splits(projected, history)
    sources = {"questions": _source_manifest(questions), "corpus": _source_manifest(corpus),
               "history_ids": _source_manifest(history_ids)}
    output.mkdir(parents=True, exist_ok=False)
    private = output / "private"
    private.mkdir(mode=0o700)
    shutil.copyfile(questions, private / "questions.jsonl")
    shutil.copyfile(history_ids, private / "history_ids.txt")
    public = [{"query_id": row["query_id"], "query": row["query"]} for row in projected]
    public.sort(key=lambda row: row["query_id"])
    _write_jsonl(output / "public_questions.jsonl", public)
    _write_jsonl(output / "selection.jsonl", [
        {**row, **split_map[row["query_id"]], "upstream_split": "test"} for row in public])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output,
                                         prefix="normalized-corpus-", suffix=".jsonl", delete=False) as handle:
            temporary = Path(handle.name)
            for row in _corpus_rows(corpus):
                normalized = {"docid": str(row["docid"]), "text": row["text"],
                              "url": str(row.get("url") or "")}
                if not normalized["docid"] or not isinstance(normalized["text"], str) or not normalized["text"]:
                    raise ValueError("Corpus requires nonempty docid and text")
                handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
        build_index(temporary, output / "corpus.sqlite3", revision)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    index = _read(output / "corpus.sqlite3.manifest.json")
    counts = Counter(value["research_split"] for value in split_map.values())
    artifact_names = ("public_questions.jsonl", "selection.jsonl", "corpus.sqlite3",
                      "corpus.sqlite3.manifest.json", "private/questions.jsonl", "private/history_ids.txt")
    artifacts = {name: sha(output / name) for name in artifact_names}
    if artifacts["private/questions.jsonl"] != sources["questions"]["files"][questions.name]["sha256"]:
        raise ValueError("Question source changed during preparation")
    if _source_manifest(corpus) != sources["corpus"]:
        raise ValueError("Corpus source changed during preparation")
    manifest = {"schema_version": 1, "dataset_id": DATASET_ID, "revision": revision,
                "upstream_split": "test", "question_count": len(public),
                "document_count": index["document_count"], "sources": sources, "artifacts": artifacts,
                "split_seed": SEED, "split_target_fractions": FRACTIONS,
                "split_counts": {name: counts[name] for name in SPLITS},
                "group_count": len({value["task_group_id"] for value in split_map.values()}),
                "split_policy": "normalized-query/shared-evidence-or-gold-docid connected components; historical closure",
                "evaluation_status": "pending", "domain_profile": "research; not an official BrowseComp leaderboard run"}
    write_json(output / "manifest.json", manifest)
    for name in (*artifact_names, "manifest.json"):
        (output / name).chmod(0o400 if name.startswith("private/") else 0o444)
    return manifest


def _prepared(path):
    root = Path(path).resolve()
    manifest = _read(root / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("Unsupported prepared data manifest")
    for name, expected in manifest["artifacts"].items():
        artifact = root / name
        if not artifact.resolve().is_relative_to(root) or sha(artifact) != expected:
            raise ValueError("Prepared data hash drift: " + name)
    index = _read(root / "corpus.sqlite3.manifest.json")
    if index["index_sha256"] != manifest["artifacts"]["corpus.sqlite3"] or index["corpus_revision"] != manifest["revision"]:
        raise ValueError("Prepared index does not match frozen revision/hash")
    selection = list(_jsonl(root / "selection.jsonl"))
    if len(selection) != manifest["question_count"] or len({row["query_id"] for row in selection}) != len(selection):
        raise ValueError("Prepared selection count or task IDs mismatch")
    if any(set(row) != {"query_id", "query", "task_group_id", "research_split", "upstream_split"}
           or row["research_split"] not in SPLITS or row["upstream_split"] != "test" for row in selection):
        raise ValueError("Prepared selection contains unexpected/private fields")
    return root, manifest, selection


def create_config(prepared_data, sdk_path, run_root, output, base_url, service_manifest=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Config output already exists")
    root, manifest, _ = _prepared(prepared_data)
    run_root, sdk_path = Path(run_root).resolve(), Path(sdk_path).resolve()
    config = Config(
        dataset=DatasetConfig(kind="browsecomp", id=DATASET_ID, path=str(root / "public_questions.jsonl"),
                              revision=manifest["revision"], split="test",
                              sha256=manifest["artifacts"]["public_questions.jsonl"], setting="openhands-search-read-v1"),
        llm=LLMConfig(model="openai/qwen3.5-9b", base_url=base_url, temperature=.6, top_p=.95, top_k=20,
                      presence_penalty=0.0, min_p=0.0, repetition_penalty=1.0, seed=SEED,
                      enable_thinking=True, num_retries=0, max_output_tokens=8192, timeout=600),
        runtime=RuntimeConfig(sdk_path=str(sdk_path), sdk_commit=SDK_COMMIT, runs_dir=str(run_root),
                              max_iterations=24, max_tool_calls=36, max_llm_requests=28,
                              task_timeout=1800, tool_timeout=30),
        retrieval=RetrievalConfig(index_path=str(root / "corpus.sqlite3"), corpus_revision=manifest["revision"],
                                  top_k=5, snippet_chars=1200, read_chars=6000),
    )
    value = {"profile_name": "browsecomp-plus-qwen35-9b-4090-research-v1", "run_root": str(run_root),
             "sdk_path": str(sdk_path), "prepared_data": str(root),
             "service_manifest": str(Path(service_manifest).resolve() if service_manifest else run_root / "service_manifest.json"),
             "service_model_name": "qwen3.5-9b", **config.to_dict()}
    write_json(output, value)
    return value


def _config(path):
    profile = _read(path)
    allowed = {"profile_name", "run_root", "sdk_path", "prepared_data", "service_manifest", "service_model_name",
               "dataset", "llm", "runtime", "retrieval", "docker", "evaluation"}
    if set(profile) - allowed:
        raise ValueError("Unknown profile fields")
    from .config import DockerConfig, EvaluationConfig, _validate_types
    classes = {"dataset": DatasetConfig, "llm": LLMConfig, "runtime": RuntimeConfig,
               "retrieval": RetrievalConfig, "docker": DockerConfig, "evaluation": EvaluationConfig}
    values = {}
    for key, cls in classes.items():
        section = profile.get(key, {})
        if not isinstance(section, dict) or set(section) - {item.name for item in fields(cls)}:
            raise ValueError(f"Invalid Config section {key}")
        _validate_types(cls, section, key)
        values[key] = cls(**section)
    config = Config(**values)
    if config.runtime.sdk_commit != SDK_COMMIT or config.runtime.sdk_path != profile["sdk_path"]:
        raise ValueError("Profile SDK identity does not match pinned runtime")
    if config.dataset.kind != "browsecomp" or config.dataset.id != DATASET_ID or config.dataset.setting != "openhands-search-read-v1":
        raise ValueError("Profile must use the BrowseComp fixed-corpus adapter")
    if config.llm.model != "openai/" + profile["service_model_name"] or config.llm.num_retries != 0:
        raise ValueError("Model identity mismatch or hidden LLM retries enabled")
    if not config.llm.native_tool_calling:
        raise ValueError("Native tool calling is required")
    for section, names in ((config.runtime, ("max_iterations", "max_tool_calls", "max_llm_requests", "task_timeout", "tool_timeout")),
                           (config.llm, ("max_output_tokens", "timeout")),
                           (config.retrieval, ("top_k", "snippet_chars", "read_chars"))):
        if any(getattr(section, name) <= 0 for name in names):
            raise ValueError("Collection budgets must be positive")
    return profile, config


def source_fingerprint():
    root = Path(__file__).parent
    return {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob("*.py"))
            if "__pycache__" not in path.parts}


def _runtime_fingerprint():
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                if dist.metadata.get("Name")}
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "sqlite": sqlite3.sqlite_version, "packages": packages}


def _check_sdk_import(runtime):
    spec = importlib.util.find_spec("openhands.sdk")
    if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(
            Path(runtime.sdk_path).resolve() / "openhands-sdk"):
        raise ValueError("Imported SDK does not belong to the configured editable checkout")


def _service(profile):
    manifest = _read(profile["service_manifest"])
    payload = manifest.get("fingerprint_payload")
    if not isinstance(payload, dict) or manifest.get("fingerprint") != digest_json(payload):
        raise ValueError("Service manifest stable fingerprint mismatch")
    if payload.get("service_model_name") != profile["service_model_name"] or payload.get("max_model_len") != 32768:
        raise ValueError("Service model/context do not match the frozen profile")
    if payload.get("base_url", profile["llm"]["base_url"]).rstrip("/") != profile["llm"]["base_url"].rstrip("/"):
        raise ValueError("Service base URL does not match profile")
    return {"fingerprint_payload": payload, "fingerprint": manifest["fingerprint"]}


@contextmanager
def collector_lock(run_root):
    """Serialize both this run and all same-UID BrowseComp collectors on the host."""
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    host_lock = Path(tempfile.gettempdir()) / f"flowpilot-browsecomp-collector-{os.getuid()}.lock"
    handles = []
    try:
        for path in (host_lock, root / ".browsecomp-collector.lock"):
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            handle = os.fdopen(descriptor, "a")
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another BrowseComp collector owns the host/run-root lock") from exc
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def _task_key(task_id):
    return hashlib.sha256(task_id.encode()).hexdigest()[:24]


def _adopt_actor_log(attempt):
    pending = attempt.with_name(attempt.name + ".actor.log")
    if pending.exists():
        attempt.mkdir(parents=True, exist_ok=True)
        target = attempt / "actor.log"
        if target.exists():
            raise ValueError("Both pending and completed actor logs exist; refusing overwrite")
        pending.rename(target)


def _run_actor(config, task, attempt, run_id):
    # The runner requires an empty attempt directory, so capture console output
    # beside it, then attach the closed log before freezing any raw checksums.
    attempt.parent.mkdir(parents=True, exist_ok=True)
    pending = attempt.with_name(attempt.name + ".actor.log")
    try:
        with pending.open("x", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                return run_task(config, task, attempt, run_id=run_id)
    finally:
        _adopt_actor_log(attempt)


def _raw_hashes(attempt):
    # Everything produced by the actor, including logs and SDK state, is raw.
    # Derived exports are siblings, never mixed into this immutable snapshot.
    return {str(path.relative_to(attempt)): sha(path) for path in sorted(attempt.rglob("*"))
            if path.is_file()}


def _check_raw(attempt, hashes):
    if _raw_hashes(attempt) != hashes:
        raise ValueError("Immutable raw attempt was modified (file/hash integrity)")


def _actor_ended(attempt):
    try:
        last = None
        for event in _jsonl(attempt / "events.jsonl"):
            last = event
        return bool(last and last.get("event") == "task_end" and (attempt / "result.json").is_file())
    except (OSError, ValueError):
        return False


def _audit(attempt):
    rows = build_prediction_rows(attempt)
    events = list(_jsonl(attempt / "events.jsonl"))
    if not events or events[0].get("event") != "task_start" or events[-1].get("event") != "task_end":
        raise ValueError("Incomplete actor trace")
    if events[-1].get("result") != _read(attempt / "result.json"):
        raise ValueError("Actor result does not match terminal trace event")
    def audit_refs(value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= value.keys() and str(value["path"]).startswith("blobs/"):
                checked_blob(attempt, value)
            for child in value.values():
                audit_refs(child)
        elif isinstance(value, list):
            for child in value:
                audit_refs(child)
    for event in events:
        audit_refs(event)
    return rows


def _failure_class(result, attempt=None):
    status = result.get("execution_status", "")
    reasons = [str(result.get(key, "")) for key in ("error_type", "termination_reason", "errors")]
    transport_errors = []
    if attempt is not None:
        transport_errors = [str(event.get("error_type", "")) for event in _jsonl(attempt / "events.jsonl")
                            if event.get("event") in {"llm_error", "llm_request_not_sent"}]
    text = " ".join([*reasons, *transport_errors]).lower()
    if status == "cancelled":
        return "interrupted"
    # Controller task-budget exhaustion takes precedence over a transport timeout
    # observed at the same deadline. Context exhaustion is also a retained outcome.
    if status == "budget_exhausted" or result.get("termination_reason") in {
            "task_timeout", "max_iterations", "max_tool_calls", "max_llm_requests"} or any(
            word in text for word in ("contextwindow", "context_length", "maximum context",
                                      "contextlengthexceeded", "max_tokens")):
        return "genuine_failure"
    if status == "environment_error" or any(word in text for word in (
            "connection", "connecterror", "connecttimeout", "apitimeout", "readtimeout", "ratelimit",
            "internalserver", "serviceunavailable", "authentication", "permissiondenied")):
        return "infrastructure"
    # SDK ConversationRunError can hide the provider error type. Unclassified
    # LLM/transport failures require inspection rather than silently entering data.
    if status == "llm_error" or transport_errors:
        return "infrastructure"
    return "completed" if status == "completed" else "genuine_failure"


def _validate_registry(root, registry, selection):
    if registry.get("schema_version") != 1 or set(registry.get("tasks", {})) != {row["query_id"] for row in selection}:
        raise ValueError("Registry does not match frozen task selection")
    for task_id, entries in registry["tasks"].items():
        accepted = 0
        for number, entry in enumerate(entries, 1):
            expected = f"tasks/{_task_key(task_id)}/attempt-{number:03d}"
            if entry.get("path") != expected or not (root / expected).resolve().is_relative_to(root / "tasks"):
                raise ValueError("Invalid registry attempt path")
            if entry.get("status") not in {"running", "infrastructure_failed", "postprocess_pending", "accepted"}:
                raise ValueError("Unknown attempt state")
            if entry.get("raw_hashes") is not None:
                _check_raw(root / expected, entry["raw_hashes"])
            accepted += entry["status"] == "accepted"
            if accepted and number != len(entries):
                raise ValueError("Actor rerun exists after a primary accepted attempt")
        if accepted > 1:
            raise ValueError("Multiple accepted attempts for one task")


def _check_accepted(root, entry):
    attempt = root / entry["path"]
    _check_raw(attempt, entry["raw_hashes"])
    _audit(attempt)
    export = root / entry["export_path"]
    if sha(export / "manifest.json") != entry["export_manifest_sha256"]:
        raise ValueError("Accepted export manifest modified")
    for name, info in _read(export / "manifest.json")["files"].items():
        if sha(export / name) != info["sha256"]:
            raise ValueError("Accepted export was modified")
    return attempt


def _finalize(root, entry):
    attempt = root / entry["path"]
    _check_raw(attempt, entry["raw_hashes"])
    _audit(attempt)
    parent = root / "derived" / Path(entry["path"]).parent.name / Path(entry["path"]).name
    number = 1
    export = parent / f"prediction-{number:03d}"
    while export.exists():
        number += 1
        export = parent / f"prediction-{number:03d}"
    write_prediction_dataset([attempt], export)
    entry.update(status="accepted", export_path=str(export.relative_to(root)),
                 export_manifest_sha256=sha(export / "manifest.json"), evaluation_status="pending")


def _summary(root, registry, split=None):
    primary = {task_id: entries[-1]["path"] for task_id, entries in registry["tasks"].items()
               if entries and entries[-1]["status"] == "accepted"}
    summary = {"accepted_tasks": len(primary), "physical_attempts": sum(map(len, registry["tasks"].values())),
               "planned_tasks": len(registry["tasks"]), "requested_split": split,
               "evaluation_status": "pending", "primary_attempts": primary,
               "attempt_policy": "first infrastructure-valid actor; genuine failures retained; judge pending"}
    write_json(root / "primary_attempts.json", primary)
    write_json(root / "summary.json", summary)
    return summary


def collect(config_path, split, resume=False, retry_infrastructure=False, limit=0, release_test=False):
    if split not in SPLITS or limit < 0:
        raise ValueError("Unknown split or negative limit")
    if split == "test" and not release_test:
        raise ValueError("Held-out test requires explicit --release-test")
    if retry_infrastructure and not resume:
        raise ValueError("--retry-infrastructure requires --resume")
    profile, config = _config(config_path)
    prepared, data, selection = _prepared(profile["prepared_data"])
    if (config.dataset.path != str(prepared / "public_questions.jsonl")
            or config.dataset.sha256 != data["artifacts"]["public_questions.jsonl"]
            or config.dataset.revision != data["revision"]
            or config.retrieval.index_path != str(prepared / "corpus.sqlite3")
            or config.retrieval.corpus_revision != data["revision"]):
        raise ValueError("Config does not match frozen prepared data")
    root = Path(profile["run_root"]).resolve()
    with collector_lock(root):
        sdk = check_sdk(config.runtime)
        _check_sdk_import(config.runtime)
        if sdk.get("package_version") != "1.31.1" or not sdk.get("runtime_matches_baseline"):
            raise ValueError("SDK must match baseline a6db5dc and installed version 1.31.1")
        frozen = {"profile": profile, "prepared_manifest_sha256": sha(prepared / "manifest.json"),
                  "source": source_fingerprint(), "runtime": _runtime_fingerprint(), "sdk": sdk,
                  "service": _service(profile)}
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise ValueError("Run already exists; use --resume")
            manifest = _read(manifest_path)
            if manifest.get("frozen") != frozen:
                raise ValueError("Frozen config/data/source/SDK/runtime/service drift; use a new run")
            registry = _read(root / "attempt_registry.json")
        else:
            if resume:
                raise ValueError("Cannot --resume a run that has not started")
            if (root / "tasks").exists() or (root / "attempt_registry.json").exists():
                raise ValueError("Unregistered run artifacts exist; do not overwrite")
            manifest = {"schema_version": 1, "run_id": uuid.uuid4().hex,
                        "created_at": datetime.now(UTC).isoformat(), "frozen": frozen,
                        "host_t0": {"hostname": platform.node(), "machine": platform.machine()},
                        "evaluation_status": "pending"}
            registry = {"schema_version": 1, "tasks": {row["query_id"]: [] for row in selection}}
            write_json(root / "profile_snapshot.json", profile)
            write_json(root / "data_manifest_snapshot.json", data)
            write_json(root / "service_snapshot.json", frozen["service"])
            _write_jsonl(root / "selection.jsonl", selection)
            snapshot = root / "source_snapshot"
            for name in frozen["source"]:
                destination = snapshot / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(__file__).parent / name, destination)
            manifest["snapshot_hashes"] = {str(path.relative_to(root)): sha(path) for path in
                [root / "profile_snapshot.json", root / "data_manifest_snapshot.json", root / "service_snapshot.json",
                 root / "selection.jsonl", *sorted(snapshot.rglob("*.py"))]}
            write_json(root / "attempt_registry.json", registry)
            write_json(manifest_path, manifest)
        for name, expected in manifest["snapshot_hashes"].items():
            if sha(root / name) != expected:
                raise ValueError("Frozen run snapshot modified")
        _validate_registry(root, registry, selection)
        # All accepted actors are audited before any new request, even if this
        # command selects a different research split.
        for entries in registry["tasks"].values():
            if entries and entries[-1]["status"] == "accepted":
                _check_accepted(root, entries[-1])
        selected = [row for row in selection if row["research_split"] == split]
        if limit:
            selected = selected[:limit]
        for row in selected:
            if source_fingerprint() != frozen["source"] or _read(config_path) != profile or _service(profile) != frozen["service"]:
                raise ValueError("Frozen source/config/service changed mid-collection")
            entries = registry["tasks"][row["query_id"]]
            if entries and entries[-1]["status"] == "accepted":
                continue
            if entries:
                entry = entries[-1]
                attempt = root / entry["path"]
                if entry["status"] == "running" and entry.get("raw_hashes") is None:
                    _adopt_actor_log(attempt)
                if entry.get("raw_hashes") is not None:
                    _check_raw(attempt, entry["raw_hashes"])
                if entry["status"] == "running" and _actor_ended(attempt):
                    entry.update(raw_hashes=_raw_hashes(attempt), failure_class=_failure_class(_read(attempt / "result.json"), attempt))
                    entry["status"] = ("infrastructure_failed" if entry["failure_class"] in {"infrastructure", "interrupted"}
                                       else "postprocess_pending")
                elif entry["status"] == "running":
                    entry.update(status="infrastructure_failed", failure_class="interrupted", raw_hashes=_raw_hashes(attempt))
                if entry["status"] == "postprocess_pending":
                    try:
                        _finalize(root, entry)
                    except Exception as exc:
                        write_json(root / "attempt_registry.json", registry)
                        _summary(root, registry, split)
                        raise RuntimeError("Actor retained; --resume retries postprocessing only") from exc
                    write_json(root / "attempt_registry.json", registry)
                    continue
                write_json(root / "attempt_registry.json", registry)
                if not retry_infrastructure:
                    _summary(root, registry, split)
                    raise ValueError("Retained infrastructure/interrupted attempt; use --resume --retry-infrastructure")
            attempt = root / "tasks" / _task_key(row["query_id"]) / f"attempt-{len(entries) + 1:03d}"
            if attempt.exists():
                raise ValueError("Unregistered attempt exists; refusing overwrite")
            entry = {"path": str(attempt.relative_to(root)), "status": "running",
                     "started_at": datetime.now(UTC).isoformat()}
            entries.append(entry)
            write_json(root / "attempt_registry.json", registry)
            task = BrowseCompAdapter.from_record(row, dataset_id=DATASET_ID, revision=data["revision"], split=split)
            task = replace(task, public_metadata={"task_group_id": row["task_group_id"], "research_split": split,
                                                 "upstream_split": "test"})
            actor_config = replace(config, dataset=replace(config.dataset, split=split))
            try:
                _run_actor(actor_config, task, attempt, manifest["run_id"])
            except (Exception, KeyboardInterrupt) as exc:
                entry.update(status="infrastructure_failed", failure_class="interrupted" if isinstance(exc, KeyboardInterrupt)
                             else "infrastructure", raw_hashes=_raw_hashes(attempt), error_type=type(exc).__name__)
                write_json(root / "attempt_registry.json", registry)
                _summary(root, registry, split)
                if isinstance(exc, KeyboardInterrupt):
                    raise
                raise RuntimeError("Actor infrastructure failure retained; explicit retry required") from exc
            entry["raw_hashes"] = _raw_hashes(attempt)
            if not _actor_ended(attempt):
                entry.update(status="infrastructure_failed", failure_class="interrupted")
            else:
                entry["failure_class"] = _failure_class(_read(attempt / "result.json"), attempt)
                entry["status"] = ("infrastructure_failed" if entry["failure_class"] in {"infrastructure", "interrupted"}
                                   else "postprocess_pending")
            write_json(root / "attempt_registry.json", registry)
            if entry["status"] == "infrastructure_failed":
                _summary(root, registry, split)
                raise RuntimeError("Actor infrastructure/interruption retained; use --resume --retry-infrastructure")
            try:
                _finalize(root, entry)
            except Exception as exc:
                _summary(root, registry, split)
                raise RuntimeError("Actor retained; --resume retries postprocessing only") from exc
            write_json(root / "attempt_registry.json", registry)
        return _summary(root, registry, split)


def export_run(run_root, output):
    root, output = Path(run_root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Export output already exists")
    if output.is_relative_to(root / "tasks"):
        raise ValueError("Export cannot write inside immutable raw attempt directories")
    with collector_lock(root):
        manifest, registry = _read(root / "manifest.json"), _read(root / "attempt_registry.json")
        selection = list(_jsonl(root / "selection.jsonl"))
        if sha(root / "selection.jsonl") != manifest["snapshot_hashes"]["selection.jsonl"]:
            raise ValueError("Frozen selection modified")
        if source_fingerprint() != manifest["frozen"]["source"]:
            raise ValueError("Frozen exporter source drift")
        _validate_registry(root, registry, selection)
        attempts = [_check_accepted(root, entries[-1]) for entries in registry["tasks"].values()
                    if entries and entries[-1]["status"] == "accepted"]
        if not attempts:
            raise ValueError("No accepted primary attempts to export")
        write_prediction_dataset(attempts, output)
        return _read(output / "manifest.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze public questions, grouped splits and full corpus index")
    for name in ("questions", "corpus", "output", "revision", "history-ids"):
        prepare.add_argument("--" + name, required=True)
    create = commands.add_parser("create-config", help="Create the fixed 4090 research profile")
    for name in ("prepared-data", "sdk-path", "run-root", "output", "base-url"):
        create.add_argument("--" + name, required=True)
    create.add_argument("--service-manifest")
    collector = commands.add_parser("collect", help="Collect immutable first infrastructure-valid attempts")
    collector.add_argument("--config", required=True)
    collector.add_argument("--split", choices=SPLITS, required=True)
    collector.add_argument("--resume", action="store_true")
    collector.add_argument("--retry-infrastructure", action="store_true")
    collector.add_argument("--limit", type=int, default=0)
    collector.add_argument("--release-test", action="store_true")
    exporter = commands.add_parser("export", help="Materialize v2 data from audited accepted primary attempts")
    exporter.add_argument("--run-root", required=True)
    exporter.add_argument("--output", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    def stop(signum, frame):
        raise KeyboardInterrupt("Collector received SIGTERM")
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        if command == "prepare":
            result = prepare_data(**args)
        elif command == "create-config":
            result = create_config(**args)
        elif command == "collect":
            args["config_path"] = args.pop("config")
            result = collect(**args)
        else:
            result = export_run(**args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print("Collection interrupted; raw attempts retained. Resume explicitly.", file=sys.stderr)
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
