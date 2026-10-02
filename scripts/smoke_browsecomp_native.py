"""Four independent native clients; generic queries only, no benchmark tasks or LLM."""

import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
from pathlib import Path

from benchmark_adapters.config import load_config
from benchmark_adapters.native_browsecomp import create_retrieval_environment
from benchmark_adapters.retrieval_tools import (
    NativeGetDocumentAction,
    NativeSearchAction,
    RetrievalExecutor,
)
from benchmark_adapters.sdk_bridge import Binding
from benchmark_adapters.tracing import Budget, TraceRecorder

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence/retrieval/native_smoke_20260922"
BARRIER = threading.Barrier(4, timeout=60)
QUERIES = ("London museum", "mathematics history", "ocean research", "railway station")


def run_client(slot):
    config = load_config(ROOT / "configs/c4/browsecomp.toml")
    environment = create_retrieval_environment(config)
    recorder = TraceRecorder(
        OUTPUT / f"client_{slot}",
        {"clock_domain": "native-cpu-smoke", "worker_slot": slot},
    )
    binding = Binding(environment, recorder, Budget(2, 1, 120), 60, 3)
    try:
        identity = environment.prepare()
        BARRIER.wait()
        submitted_ns = time.monotonic_ns()
        RetrievalExecutor(binding, "search")(NativeSearchAction(query=QUERIES[slot]))
        returned_ns = time.monotonic_ns()
        assert len(recorder.retrieved_docids) == 5, recorder.retrieved_docids
        if slot == 0:
            docid = sorted(recorder.retrieved_docids)[0]
            RetrievalExecutor(binding, "get_document")(
                NativeGetDocumentAction(docid=docid)
            )
    finally:
        environment.close()
        recorder.close()
    events = [
        json.loads(line)
        for line in (OUTPUT / f"client_{slot}/events.jsonl").read_text().splitlines()
    ]
    assert not any(event["event"] == "tool_error" for event in events)
    ends = [event for event in events if event["event"] == "tool_end"]
    assert len(ends) == (2 if slot == 0 else 1)
    assert all(
        event["executor_duration_ms"] is None and event["queue_wait_ms"] is None
        for event in ends
    )
    assert all(event["round_trip_ms"] > 0 for event in ends)
    searches = json.loads(ends[0]["model_observation"])
    assert len(searches) == 5
    result = {
        "slot": slot,
        "query": QUERIES[slot],
        "submitted_ns": submitted_ns,
        "returned_ns": returned_ns,
        "search_round_trip_ms": ends[0]["round_trip_ms"],
        "results": len(searches),
        "identity": identity,
        "executor_duration_ms": None,
        "queue_wait_ms": None,
        "errors": [],
    }
    if slot == 0:
        doc = json.loads(ends[1]["model_observation"])
        assert len(doc["text"]) > 3
        result["document"] = {
            "docid": doc["docid"],
            "characters": len(doc["text"]),
            "sha256": hashlib.sha256(doc["text"].encode()).hexdigest(),
            "round_trip_ms": ends[1]["round_trip_ms"],
            "client_output_limit_chars": 3,
            "no_client_truncation": True,
        }
    return result


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    OUTPUT = parser.parse_args().output
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_client, range(4)))
    intervals = sorted(
        [(r["submitted_ns"], 1) for r in results]
        + [(r["returned_ns"], -1) for r in results]
    )
    active = peak = 0
    for _, delta in intervals:
        active += delta
        peak = max(peak, active)
    report = {
        "passed": True,
        "scope": "4 generic search calls and 1 get_document; no benchmark questions, no LLM or GPU",
        "independent_clients": 4,
        "client_search_intervals_peak_overlap": peak,
        "server_executor_parallelism_measured": False,
        "clients": results,
    }
    assert peak == 4, peak
    (OUTPUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "passed": True,
                "search_calls": 4,
                "read_calls": 1,
                "client_peak_overlap": peak,
                "report": str(OUTPUT / "report.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
