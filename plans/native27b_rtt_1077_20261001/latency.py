"""Future isolated predictor-cost benchmark; no labels or model updates are used."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PLAN = Path(__file__).resolve().parent
sys.path.insert(0, str(PLAN / "code"))
import numpy as np
from predictor import NAMES
from predictor.cli import load_artifact
from predictor.data import load_split, write_json
from predictor.models import monotone
from threadpoolctl import threadpool_limits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Refuse latency overwrite")
    artifact = load_artifact(args.model, args.data)
    model = artifact["model"]
    groups = defaultdict(list)
    for sample in sorted(load_split(args.data, "tune"), key=lambda s: s["sample_id"]):
        key = sample["adapter"] + "/" + sample["context"]["tool_name"]
        if len(groups[key]) < 64:
            groups[key].append(sample["context"])
    contexts = [(key, c) for key in sorted(groups) for c in groups[key]]

    def predict(context):
        prediction = model.predict(context)
        if prediction["duration_ms"] is not None and artifact["offsets"] is not None:
            prediction["duration_ms"] = dict(
                zip(
                    NAMES,
                    monotone(
                        np.array([prediction["duration_ms"][n] for n in NAMES])
                        + artifact["offsets"]
                    ).tolist(),
                )
            )
        return prediction

    times = defaultdict(list)
    supported = 0
    with threadpool_limits(limits=model.config["threads"]):
        for key in groups:
            for context in groups[key][:3]:
                predict(context)
        for _ in range(3):
            for key, context in contexts:
                start = time.perf_counter_ns()
                prediction = predict(context)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                times[key].append(elapsed)
                supported += prediction["duration_ms"] is not None
    all_times = [x for v in times.values() for x in v]

    def stats(v):
        return {
            "n": len(v),
            "p50_ms": float(np.quantile(v, 0.5)),
            "p95_ms": float(np.quantile(v, 0.95)),
            "p99_ms": float(np.quantile(v, 0.99)),
        }

    report = {
        "algorithm": model.name,
        "model_version": model.version,
        "calibration_version": artifact["calibration_version"],
        "source": "tune contexts only; labels not accessed by prediction",
        "threads": model.config["threads"],
        "benchmark_mode": "one process at a time after all training/evaluation children exit",
        "scope": "context feature extraction + prediction + calibration correction; excludes JSON/RPC/scheduler",
        "repeats": 3,
        "warmup_contexts_per_tool": 3,
        "supported_calls": supported,
        "overall": stats(all_times),
        "by_tool": {k: stats(v) for k, v in times.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
