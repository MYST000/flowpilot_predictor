"""Preparation and protocol tests only: no model or benchmark execution."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp import Client, FastMCP
from mcp import types

from benchmark_adapters import native_browsecomp
from benchmark_adapters import parallel_collection as collection
from benchmark_adapters.config import Config, DatasetConfig, RetrievalConfig
from benchmark_adapters.retrieval_tools import NativeSearchAction, RetrievalExecutor
from benchmark_adapters.sdk_bridge import Binding
from benchmark_adapters.tracing import Budget, TraceRecorder

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_timing_hook_preserves_content_and_error_metadata():
    timing = module("browsecomp_timing")
    for failed in (False, True):
        result = types.ServerResult(
            types.CallToolResult(
                content=[types.TextContent(type="text", text="original payload")],
                isError=failed,
                _meta={"original": True},
            )
        )

        async def original(request):
            return result

        handlers = {types.CallToolRequest: original}
        timing.install_timing(
            SimpleNamespace(_mcp_server=SimpleNamespace(request_handlers=handlers))
        )
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="search", arguments={"query": "fixture"}
            ),
        )
        actual = asyncio.run(handlers[types.CallToolRequest](request)).root
        assert actual.content[0].text == "original payload"
        assert actual.isError is failed
        assert actual.meta["original"] is True
        assert actual.meta["flowpilot_timing"]["executor_duration_ms"] >= 0
        assert actual.meta["flowpilot_timing"]["queue_wait_ms"] is None


def test_real_mcp_bridge_keeps_timing_out_of_observation(tmp_path, monkeypatch):
    server = FastMCP("fixture")

    @server.tool
    def search(query: str) -> list[dict]:
        return [{"docid": "1", "snippet": query}]

    @server.tool
    def get_document(docid: str) -> dict:
        return {"docid": docid, "text": "fixture"}

    module("browsecomp_timing").install_timing(server)
    monkeypatch.setattr(
        native_browsecomp, "Client", lambda url, **kw: Client(server, **kw)
    )
    config = Config(
        dataset=DatasetConfig(kind="browsecomp", id="Tevatron/browsecomp-plus"),
        retrieval=RetrievalConfig(backend="browsecomp_mcp"),
    )
    env = native_browsecomp.create_retrieval_environment(config)
    recorder = TraceRecorder(tmp_path, {"clock_domain": "fixture"})
    binding = Binding(env, recorder, Budget(2, 1, 60), 30, 16000)
    try:
        env.prepare()
        RetrievalExecutor(binding, "search")(NativeSearchAction(query="fixture"))
    finally:
        env.close()
        recorder.close()
    end = next(
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "tool_end"
    )
    assert 0 <= end["executor_duration_ms"] <= end["round_trip_ms"]
    assert "flowpilot_timing" not in end["model_observation"]
    assert json.loads(end["model_observation"])[0]["snippet"] == "fixture"


def test_round_robin_prepare_interleaves_adapters_without_execution(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(collection, "sdk_source_state", lambda *args: {})
    monkeypatch.setattr(collection, "adapter_source_digest", lambda: "fixture")
    for kind, identity, setting, split in (
        ("hotpot", "hotpotqa", "fullwiki-fixed-corpus-v1", "dev"),
        ("browsecomp", "Tevatron/browsecomp-plus", "openhands-search-read-v1", "test"),
    ):
        records = [
            {
                "_id": str(i),
                "question": f"Question {i}",
                "query_id": str(i),
                "query": f"Query {i}",
            }
            for i in range(3)
        ]
        (tmp_path / f"{kind}.json").write_text(json.dumps(records))
        (tmp_path / f"{kind}.toml").write_text(
            f'[dataset]\nkind="{kind}"\nid="{identity}"\nsetting="{setting}"\nrevision="fixture"\nsplit="{split}"\npath="{kind}.json"\n'
        )
    path = tmp_path / "campaign.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "run_id": "fixture",
                "runs_dir": "runs",
                "queue_policy": "round_robin",
                "concurrency": 4,
                "entries": [
                    {"config": f"{kind}.toml", "task_ids": ["0", "1", "2"]}
                    for kind in ("hotpot", "browsecomp")
                ],
            }
        )
    )
    root = collection.prepare_campaign(path)
    jobs = json.loads((root / "campaign.json").read_text())["jobs"]
    assert [j["adapter"] for j in jobs] == ["hotpot", "browsecomp"] * 3
    assert not (root / "collection_started.json").exists()


def test_selection_reproducible_and_preserves_frozen_split():
    prep = module("prepare_mixed_c4")
    rows = [
        {
            "question_id": str(i),
            "research_split": "fit" if i < 8 else "test",
            "difficulty": "easy" if i % 2 else "medium",
        }
        for i in range(12)
    ]
    selected = prep.choose(rows, "fit", 6, 7, code=True)
    assert selected == prep.choose(rows, "fit", 6, 7, code=True)
    assert len(set(selected)) == 6
    assert all(int(value) < 8 for value in selected)
    with pytest.raises(ValueError):
        prep.choose(rows, "test", 5, 7, code=True)


def test_lcb_adapter_carries_statement_hash_for_formal_split_validation():
    import hashlib
    from benchmark_adapters.code_tasks import LiveCodeBenchAdapter

    row = {
        "question_id": "fixture_a",
        "question_title": "Fixture",
        "question_content": "Return the input integer.",
        "platform": "atcoder",
        "difficulty": "easy",
        "contest_date": "2025-01-01T00:00:00",
        "metadata": "{}",
        "public_test_cases": json.dumps(
            [{"input": "1", "output": "1", "testtype": "stdin"}]
        ),
        "private_test_cases": json.dumps(
            [{"input": "2", "output": "2", "testtype": "stdin"}]
        ),
    }
    bundle = LiveCodeBenchAdapter.from_record(
        row, revision="fixture", checker_source=""
    )
    assert (
        bundle.task.public_metadata["statement_sha256"]
        == hashlib.sha256(row["question_content"].encode()).hexdigest()
    )
