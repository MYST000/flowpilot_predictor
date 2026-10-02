import json
from dataclasses import replace

import httpx
import pytest

from benchmark_adapters.config import Config, DatasetConfig, RetrievalConfig
from benchmark_adapters.contracts import Task
from benchmark_adapters.live_web import SEARCH_URL, LiveWebEnvironment, live_task
from benchmark_adapters.retrieval_tools import (
    NativeGetDocumentAction,
    NativeSearchAction,
    RetrievalExecutor,
)
from benchmark_adapters.sdk_bridge import Binding
from benchmark_adapters.tracing import Budget, TraceRecorder


@pytest.fixture
def live_config(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "fixture-key")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setattr(
        "benchmark_adapters.live_web.socket.getaddrinfo",
        lambda *a, **kw: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )
    return Config(
        dataset=DatasetConfig(
            kind="browsecomp", id="Tevatron/browsecomp-plus", setting="openhands-live-web-v1"
        ),
        retrieval=RetrievalConfig(
            backend="live_web", corpus_revision="live-web:brave:browsecomp:v1"
        ),
    )


def test_brave_to_page_native_schema_and_linked_timing(tmp_path, live_config):
    requests = []

    def handler(request):
        requests.append(request)
        if str(request.url).startswith(SEARCH_URL):
            assert request.headers["X-Subscription-Token"] == "fixture-key"
            return httpx.Response(
                200,
                json={
                    "web": {
                        "results": [
                            {
                                "url": "https://example.com/doc",
                                "title": "Evidence",
                                "description": "A <b>snippet</b>",
                            }
                        ]
                    }
                },
            )
        assert "X-Subscription-Token" not in request.headers
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            text="<title>Evidence</title><main>Fact one. Fact two.</main><script>ignore me</script>",
        )

    env = LiveWebEnvironment(live_config, transport=httpx.MockTransport(handler))
    recorder = TraceRecorder(tmp_path / "trace", {"task_id": "fixture"})
    binding = Binding(env, recorder, Budget(4, 1, 30), 10, 16000)
    try:
        metadata = env.prepare()
        assert metadata["trust_env"] is False
        RetrievalExecutor(binding, "search")(NativeSearchAction(query="fixture"))
        for _ in range(2):
            result = RetrievalExecutor(binding, "get_document")(
                NativeGetDocumentAction(docid="https://example.com/doc")
            )
            assert not result.is_error
    finally:
        env.close()
        recorder.close()
    assert len(requests) == 3  # Repeated reads must still fetch for the stock baseline.
    events = [
        json.loads(line) for line in (tmp_path / "trace/events.jsonl").read_text().splitlines()
    ]
    completed = [r for r in events if r["event"] == "tool_end"]
    attempts = [
        json.loads(line)
        for line in (tmp_path / "trace/network_attempts.jsonl").read_text().splitlines()
    ]
    assert len(completed) == len(attempts) == 3
    assert all(r["round_trip_ms"] >= r["executor_duration_ms"] >= 0 for r in completed)
    assert all(r["content_sha256"] and r["proxy_used"] is False for r in attempts)
    assert all("fixture-key" not in json.dumps(r) for r in events)
    assert all(r["tool_call_id"] == a["tool_call_id"] for r, a in zip(completed, attempts))


def test_hotpot_search_scope_and_live_sentence_identity(live_config):
    config = replace(
        live_config, dataset=replace(live_config.dataset, kind="hotpot", id="hotpotqa")
    )

    def handler(request):
        if str(request.url).startswith(SEARCH_URL):
            assert "site:en.wikipedia.org" in request.url.params["q"]
            return httpx.Response(
                200,
                json={
                    "web": {
                        "results": [
                            {"url": "https://en.wikipedia.org/wiki/X", "title": "X"},
                            {"url": "https://example.com/wrong"},
                        ]
                    }
                },
            )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            text="<title>X</title><main>First. Second.</main>",
        )

    env = LiveWebEnvironment(config, transport=httpx.MockTransport(handler))
    try:
        env.prepare()
        with env.operation_timeout(10):
            results = env.search("X", 5)
        assert len(results) == 1
        with env.operation_timeout(10):
            result = env.read(results[0]["docid"])
        assert result["sentence_identity"] == "live-extraction-not-official-hotpot"
        assert result["sentences"] == [[0, "First."], [1, "Second."]]
    finally:
        env.close()
    task = Task("hotpotqa", "rev", "dev", "x", "old instructions\n\nQuestion: Public question")
    assert "fixed corpus" not in live_task(task, "hotpot").instruction


@pytest.mark.parametrize("mode", ["limit", "redirect", "rate_limit"])
def test_failures_logged_without_retry_or_credential_redirect(live_config, mode):
    calls = []

    def handler(request):
        calls.append(request)
        if mode == "redirect":
            return httpx.Response(302, headers={"location": "https://example.com"})
        if mode == "rate_limit":
            return httpx.Response(429, headers={"Retry-After": "10"})
        return httpx.Response(200, content=b"x" * 100)

    config = replace(
        live_config, retrieval=replace(live_config.retrieval, web_max_response_bytes=10)
    )
    env = LiveWebEnvironment(config, transport=httpx.MockTransport(handler))
    try:
        env.prepare()
        with pytest.raises((ValueError, httpx.HTTPStatusError)), env.operation_timeout(10):
            env.search("fixture", 5)
        assert len(calls) == len(env.last_http_attempts) == 1
        assert env.last_http_attempts[0]["error_type"]
    finally:
        env.close()


def test_redirect_to_private_address_is_rejected(live_config, monkeypatch):
    env = LiveWebEnvironment(
        live_config,
        transport=httpx.MockTransport(
            lambda r: httpx.Response(302, headers={"location": "http://127.0.0.1/"})
        ),
    )
    monkeypatch.setattr(
        "benchmark_adapters.live_web.socket.getaddrinfo",
        lambda host, *a, **kw: [
            (2, 1, 6, "", ("127.0.0.1" if host == "127.0.0.1" else "93.184.216.34", 80))
        ],
    )
    try:
        env.prepare()
        env.known_urls.add("https://example.com/")
        with pytest.raises(ValueError, match="non-public"), env.operation_timeout(10):
            env.read("https://example.com/")
        assert len(env.last_http_attempts) == 2
    finally:
        env.close()
