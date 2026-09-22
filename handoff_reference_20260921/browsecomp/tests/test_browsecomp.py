import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def scripted_model():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply({"data": [{"id": "qwen3.5-9b"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            n = (len(received) - 1) % 3
            name, args = [
                ("search", {"query": "alpha", "top_k": 1, "summary": "Search"}),
                ("get_document", {"docid": "d1", "offset": 0, "summary": "Read"}),
                ("finish", {"message": "Alpha. Evidence d1.", "summary": "Done"}),
            ][n]
            self.reply({
                "id": f"reply-{len(received)}", "object": "chat.completion",
                "created": 1, "model": "qwen3.5-9b",
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [{
                        "id": f"call-{len(received)}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }],
                }}], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            })

        def reply(self, payload):
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", received
    server.shutdown()
    server.server_close()
    worker.join()


def test_real_sdk_retrieval_roundtrip_and_causal_export(tmp_path, scripted_model):
    from benchmark_adapters.config import Config, DatasetConfig, LLMConfig, RetrievalConfig, RuntimeConfig
    from benchmark_adapters.prediction_export import build_prediction_rows, load_t0_input, write_prediction_dataset
    from benchmark_adapters.retrieval import BrowseCompAdapter, build_index
    from benchmark_adapters.runner import run_task

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"docid": "d1", "text": "Alpha document."}) + "\n")
    index = tmp_path / "index.sqlite3"
    build_index(corpus, index, "fixture-corpus")
    url, received = scripted_model
    config = Config(
        dataset=DatasetConfig(kind="browsecomp", id="Tevatron/browsecomp-plus", split="fit", revision="fixture"),
        llm=LLMConfig(base_url=url, num_retries=0, max_output_tokens=128),
        retrieval=RetrievalConfig(index_path=str(index), corpus_revision="fixture-corpus"),
        runtime=replace(RuntimeConfig(), max_iterations=5, task_timeout=30),
    )
    task = BrowseCompAdapter.from_record(
        {"query_id": "q1", "query": "Find alpha", "answer": "PRIVATE_ANSWER"},
        dataset_id=config.dataset.id, revision="fixture", split="fit",
    )
    task = replace(task, public_metadata={"research_split": "fit", "task_group_id": "fixture/group1"})
    attempt = tmp_path / "attempt-001"
    result = run_task(config, task, attempt, run_id="fixture")
    assert result["execution_status"] == "completed", result
    rows = build_prediction_rows(attempt)
    assert [r["labels"]["next_tool_name"] for r in rows] == ["search", "get_document", "finish"]
    for row in rows[:2]:
        action = row["labels"]["actions"][0]
        assert action["round_trip_ms"] >= action["executor_duration_ms"] >= 0
        assert action["clock_domain"] == "host-retrieval-monotonic"
        assert action["execution_outcome"]["timed_out"] is False
        assert row["masks"]["first_executor_duration"]
        assert row["environment_id"]
        assert row["features"]["environment_known_at_t0"]["collector_host"]["cpu_logical_count"] > 0
    assert rows[2]["labels"]["actions"][0]["executor_duration_ms"] is None
    write_prediction_dataset([attempt], tmp_path / "export")
    first = json.loads((tmp_path / "export/inputs.jsonl").read_text().splitlines()[0])
    materialized = load_t0_input(attempt, first)
    assert "PRIVATE_ANSWER" not in json.dumps(received)
    assert "Alpha document." not in json.dumps(materialized)
    assert rows[0]["research_split"] == "fit"


@pytest.mark.parametrize('tool_limit,docid', [(10, 'd2'), (1, 'd1')])
def test_failed_read_and_budget_denial_remain_auditable(tmp_path, scripted_model, tool_limit, docid):
    from benchmark_adapters.config import Config, DatasetConfig, LLMConfig, RetrievalConfig, RuntimeConfig
    from benchmark_adapters.prediction_export import build_prediction_rows
    from benchmark_adapters.retrieval import BrowseCompAdapter, build_index
    from benchmark_adapters.runner import run_task

    corpus = tmp_path / 'corpus.jsonl'
    corpus.write_text(json.dumps({'docid': docid, 'text': 'Alpha document.'}) + '\n')
    index = tmp_path / 'index.sqlite3'
    build_index(corpus, index, 'fixture')
    config = Config(
        dataset=DatasetConfig(kind='browsecomp', id='Tevatron/browsecomp-plus', split='dev', revision='fixture'),
        llm=LLMConfig(base_url=scripted_model[0], num_retries=0, max_output_tokens=128),
        retrieval=RetrievalConfig(index_path=str(index), corpus_revision='fixture'),
        runtime=replace(RuntimeConfig(), max_iterations=5, max_tool_calls=tool_limit, task_timeout=30),
    )
    task = BrowseCompAdapter.from_record({'query_id': 'q', 'query': 'Find alpha'},
                                        dataset_id=config.dataset.id, revision='fixture', split='dev')
    attempt = tmp_path / 'attempt'
    result = run_task(config, task, attempt, run_id='failure-fixture')
    rows = build_prediction_rows(attempt)
    read = rows[1]['labels']['actions'][0]
    if tool_limit == 1:
        assert result['execution_status'] == 'budget_exhausted'
        assert read['execution_status'] == 'not_executed'
        assert read['not_executed_reason'] == 'max_tool_calls'
        assert read['executor_duration_ms'] is None
    else:
        assert read['execution_outcome']['exit_code'] == 1
        assert read['execution_status'] == 'completed'
        assert read['round_trip_ms'] >= read['executor_duration_ms'] >= 0
        assert read['normal_completion_right_censored'] is False


def test_full_collector_real_sdk_resume_and_frozen_profile(tmp_path, scripted_model):
    from benchmark_adapters.browsecomp_collection import prepare_data, create_config, collect, digest_json, export_run
    from benchmark_adapters.tracing import write_json

    questions, corpus, history = (tmp_path / name for name in ['questions.jsonl', 'corpus.jsonl', 'history.ids'])
    questions.write_text(''.join(json.dumps({'query_id': str(i), 'query': f'Find alpha item {i}', 'answer': 'PRIVATE_ANSWER'}) + '\n' for i in range(20)))
    corpus.write_text(json.dumps({'docid': 'd1', 'text': 'Alpha document.'}) + '\n')
    history.write_text('')
    prepared = tmp_path / 'prepared'
    prepare_data(questions, corpus, prepared, 'fixture', history)
    config_path = tmp_path / 'profile.json'
    # Locate the actually imported fixed SDK so this test also works after relocation.
    import openhands.sdk
    from pathlib import Path
    sdk = str(Path(openhands.sdk.__file__).resolve().parents[3])
    profile = create_config(prepared, sdk, tmp_path / 'run', config_path, scripted_model[0])
    profile['llm']['max_output_tokens'] = 128
    profile['runtime']['max_iterations'] = 5
    profile['runtime']['task_timeout'] = 30
    write_json(config_path, profile)
    payload = {'service_model_name': 'qwen3.5-9b', 'max_model_len': 32768, 'base_url': scripted_model[0]}
    write_json(profile['service_manifest'], {'fingerprint_payload': payload, 'fingerprint': digest_json(payload)})
    first = collect(config_path, 'dev', limit=1)
    assert first['accepted_tasks'] == 1
    assert len(scripted_model[1]) == 3
    assert collect(config_path, 'dev', resume=True, limit=1)['accepted_tasks'] == 1
    assert len(scripted_model[1]) == 3
    exported = export_run(profile['run_root'], tmp_path / 'export')
    assert exported['requests'] == 3
    inputs = (tmp_path / 'export/inputs.jsonl').read_text()
    assert 'PRIVATE_ANSWER' not in inputs
    profile['llm']['temperature'] = 0.5
    write_json(config_path, profile)
    with pytest.raises(ValueError, match='drift'):
        collect(config_path, 'dev', resume=True, limit=1)
    assert len(scripted_model[1]) == 3
