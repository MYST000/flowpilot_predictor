import hashlib
import json
import threading
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from benchmark_adapters.config import LLMConfig, load_config
from benchmark_adapters.contracts import ConfigurationError


def profile(tmp_path, settings=""):
    path = tmp_path / "profile.toml"
    path.write_text(
        '[dataset]\npath = "fixture.jsonl"\nrevision = "fixture"\n'
        f"[llm]\nnum_retries = 0\n{settings}\n"
    )
    return load_config(path)


def test_unset_sampling_preserves_legacy_profile_identity(tmp_path):
    config = profile(tmp_path)
    legacy = asdict(config)
    legacy["llm"] = {
        "model": "openai/qwen3.5-9b",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_env": "LLM_API_KEY",
        "temperature": 0.0,
        "max_output_tokens": 4096,
        "timeout": 180,
        "num_retries": 0,
        "native_tool_calling": True,
    }
    assert config.to_dict() == legacy
    assert (
        config.fingerprint
        == hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
    )


@pytest.mark.parametrize(
    ("field", "literal", "value"),
    [
        ("seed", "20260916", 20260916),
        ("top_p", "0.95", 0.95),
        ("top_k", "20", 20),
        ("top_k", "-1", -1),
        ("presence_penalty", "0", 0),
        ("min_p", "0.05", 0.05),
        ("repetition_penalty", "1.1", 1.1),
        ("enable_thinking", "false", False),
    ],
)
def test_sampling_fields_load_and_change_profile_fingerprint(tmp_path, field, literal, value):
    baseline = profile(tmp_path)
    configured = profile(tmp_path, f"{field} = {literal}")
    assert getattr(configured.llm, field) == value
    assert configured.to_dict()["llm"][field] == value
    assert configured.fingerprint != baseline.fingerprint


@pytest.mark.parametrize(
    ("field", "literal"),
    [
        ("seed", "true"),
        ("seed", "1.5"),
        ("seed", "-1"),
        ("seed", "9223372036854775808"),
        ("top_p", "true"),
        ("top_p", "0"),
        ("top_p", "1.1"),
        ("top_p", "nan"),
        ("top_k", "true"),
        ("top_k", "20.0"),
        ("top_k", "0"),
        ("top_k", "-2"),
        ("presence_penalty", "false"),
        ("presence_penalty", "-2.1"),
        ("presence_penalty", "2.1"),
        ("presence_penalty", "inf"),
        ("min_p", '"0.1"'),
        ("min_p", "-0.01"),
        ("min_p", "1.01"),
        ("min_p", "nan"),
        ("repetition_penalty", "true"),
        ("repetition_penalty", "0"),
        ("repetition_penalty", "-1"),
        ("repetition_penalty", "inf"),
        ("enable_thinking", "0"),
        ("enable_thinking", '"false"'),
        ("temperature", "-0.1"),
        ("temperature", "nan"),
        ("temperature", "inf"),
    ],
)
def test_invalid_sampling_values_fail_before_collection(tmp_path, field, literal):
    with pytest.raises(ConfigurationError, match=rf"llm\.{field}"):
        profile(tmp_path, f"{field} = {literal}")


def test_unknown_llm_fields_remain_rejected(tmp_path):
    with pytest.raises(ConfigurationError, match="Unknown llm keys"):
        profile(tmp_path, "unknown_sampling_option = 42")


@pytest.mark.parametrize(
    "settings",
    [
        {"seed": True},
        {"top_p": float("nan")},
        {"top_k": 0},
        {"presence_penalty": 3},
        {"min_p": -1},
        {"repetition_penalty": 0},
        {"enable_thinking": "false"},
    ],
)
def test_direct_llm_config_construction_validates_sampling(settings):
    with pytest.raises(ConfigurationError, match="llm\\."):
        LLMConfig(**settings)


@pytest.fixture
def captured_model():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = json.dumps(
                {
                    "id": "chatcmpl-sampling",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "qwen3.5-9b",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "done"},
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def send_request(tmp_path, llm_config):
    from openhands.sdk import Message, TextContent

    from benchmark_adapters.runner import build_recorded_llm
    from benchmark_adapters.tracing import Budget, TraceRecorder

    recorder = TraceRecorder(tmp_path, {"task_id": "sampling-fixture"})
    try:
        llm = build_recorded_llm(llm_config, recorder, Budget(5, 5, 20))
        llm.completion(messages=[Message(role="user", content=[TextContent(text="Say done")])])
    finally:
        recorder.close()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    prepared = [event for event in events if event["event"] == "llm_request_prepared"]
    assert len(prepared) == 1
    snapshot = json.loads((tmp_path / prepared[0]["request"]["path"]).read_text())
    assert snapshot["num_retries"] == snapshot["max_retries"] == 0
    return snapshot


@pytest.mark.parametrize(
    ("temperature", "top_p", "top_k", "thinking"),
    [(0, 1, -1, True), (0.6, 0.95, 20, True), (0.6, 0.95, 20, False)],
)
def test_sampling_profiles_reach_http_and_t0_snapshot(
    tmp_path, captured_model, temperature, top_p, top_k, thinking
):
    base_url, requests = captured_model
    config = profile(
        tmp_path,
        f"temperature = {temperature}\nseed = 20260916\ntop_p = {top_p}\n"
        f"top_k = {top_k}\npresence_penalty = 0\nmin_p = 0\n"
        f"repetition_penalty = 1\nenable_thinking = {str(thinking).lower()}",
    )
    snapshot = send_request(tmp_path / "attempt", replace(config.llm, base_url=base_url))
    assert len(requests) == 1
    actual = requests[0]
    assert actual["temperature"] == snapshot["temperature"] == temperature
    assert actual["seed"] == snapshot["seed"] == 20260916
    assert actual["top_p"] == snapshot["top_p"] == top_p
    expected_extra = {
        "top_k": top_k,
        "presence_penalty": 0,
        "min_p": 0,
        "repetition_penalty": 1,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    assert snapshot["extra_body"] == expected_extra
    for key, value in expected_extra.items():
        assert actual[key] == value
    assert type(actual["top_k"]) is int
    assert actual["messages"] == snapshot["messages"]


def test_unset_sampling_preserves_legacy_wire_defaults(tmp_path, captured_model):
    base_url, requests = captured_model
    config = profile(tmp_path)
    snapshot = send_request(tmp_path / "attempt", replace(config.llm, base_url=base_url))
    assert len(requests) == 1
    assert requests[0]["temperature"] == snapshot["temperature"] == 0
    for name in (
        "seed",
        "top_p",
        "top_k",
        "presence_penalty",
        "min_p",
        "repetition_penalty",
        "chat_template_kwargs",
    ):
        assert name not in requests[0]
        assert snapshot.get(name) is None
    assert not snapshot.get("extra_body")
