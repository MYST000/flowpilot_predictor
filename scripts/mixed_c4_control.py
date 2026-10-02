"""Explicit preparation, collection and post-collection actions for mixed C4."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import httpx

from benchmark_adapters.config import load_config
from benchmark_adapters.native_browsecomp import create_retrieval_environment
from benchmark_adapters.parallel_collection import (
    _checked_manifest,
    evaluate_campaign,
    prepare_campaign,
    run_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def check_services(plan):
    with httpx.Client(trust_env=False, timeout=15) as client:
        response = client.get("http://127.0.0.1:8100/v1/models")
        response.raise_for_status()
        models = response.json()["data"]
        model = next(m for m in models if m["id"] == "qwen3.5-9b")
        if model.get("max_model_len", 0) < 262144:
            raise ValueError("The collection profile requires the 262144-token server")
    checks = {}
    for kind in ("hotpot", "browsecomp"):
        env = create_retrieval_environment(load_config(plan / f"{kind}.toml"))
        try:
            identity = env.prepare()
            if kind == "hotpot":
                env.search("FlowPilot timing preflight", 5)
            else:
                env.call_tool("search", {"query": "FlowPilot timing preflight"})
            timing = env.last_timing
            if timing.get("executor_duration_ms") is None:
                raise ValueError(
                    f"{kind}: executor timing missing; start the prepared timed service"
                )
            checks[kind] = {"identity": identity, "timing": timing}
        finally:
            env.close()
    return {
        "model": model,
        "retrieval_checks": checks,
        "preflight_queries": "One fixed search per service; can warm index pages; not training data",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "check", "run", "export", "evaluate")
    )
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/mixed_c4_v1")
    parser.add_argument(
        "--split", choices=("fit", "tune", "calibration", "test"), default="fit"
    )
    args = parser.parse_args()
    plan = args.plan.resolve()
    source = plan / f"{args.split}.json"
    config = json.loads(source.read_text())
    root = Path(config["runs_dir"]) / config["run_id"]
    if args.action == "prepare":
        print(
            json.dumps({"prepared": str(prepare_campaign(source)), "tasks_executed": 0})
        )
        return
    if args.action == "check":
        print(json.dumps(check_services(plan), ensure_ascii=False, indent=2))
        return
    if args.action == "run":
        if root.exists():
            _checked_manifest(root)
        else:
            prepare_campaign(source)
        if (root / "collection_started.json").exists():
            raise ValueError(
                "This run already started. Do not overwrite or silently retry tasks."
            )
        checks = check_services(plan)
        (root / "service_preflight.json").write_text(
            json.dumps(checks, ensure_ascii=False, indent=2)
        )
        if not (root / "validation.json").exists():
            validate_campaign(root)
        report = run_campaign(root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.action == "evaluate":
        print(json.dumps(evaluate_campaign(root), ensure_ascii=False, indent=2))
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_tool_timings.py"), str(root)],
        check=True,
    )


if __name__ == "__main__":
    main()
