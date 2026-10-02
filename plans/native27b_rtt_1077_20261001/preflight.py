"""Read-only preparation validation; never instantiates or fits a predictor."""

import argparse
import importlib.metadata
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

PLAN = Path(__file__).resolve().parent
DATA = Path("/data1/ql_flowpilot_predictor/predictor_prepared/native27b_1077_v1")
sys.path.insert(0, str(PLAN / "code"))
from predictor.data import digest, load_split, require
from predictor.features import group_key


def check(deep=False):
    protocol = json.loads((PLAN / "protocol.json").read_text())
    config = json.loads((PLAN / "model_config.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    require(manifest["protocol_sha256"] == digest(PLAN / "protocol.json"), "data/protocol changed")
    require(
        manifest["unique_tasks"] == 1077 and manifest["unique_valid_rtt_rows"] == 15239,
        "unexpected prepared counts",
    )
    require(config["target"] == "round_trip_ms", "RTT required")
    require(protocol["quixbugs_role"] == "auxiliary_fit", "auxiliary training protocol changed")
    require(
        digest(DATA / "task_audit.jsonl") == manifest["task_audit_sha256"], "task audit changed"
    )
    sealed = PLAN / "deployment.sha256.json"
    if sealed.exists():
        for name, want in json.loads(sealed.read_text()).items():
            require(digest(PLAN / name) == want, "deployment changed: " + name)
    dependencies = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "scipy", "scikit-learn", "lightgbm", "joblib", "threadpoolctl")
    }
    require(
        dependencies == json.loads((PLAN / "dependencies.json").read_text()),
        "dependency versions changed",
    )
    samples = {s: load_split(DATA, s) for s in manifest["splits"]}
    groups = {
        s: {r["task_group_id"] for r in samples[s]} for s in ("fit", "tune", "calibration", "test")
    }
    for i, a in enumerate(groups):
        for b in list(groups)[i + 1 :]:
            require(not groups[a] & groups[b], f"group leakage: {a}/{b}")
    for split in ("tune", "calibration", "test"):
        require(
            all(s["adapter"] != "quixbugs" for s in samples[split]),
            "auxiliary leaked into evaluation",
        )
    require(
        sum(s["adapter"] == "quixbugs" for s in samples["fit"]) == 394,
        "missing auxiliary training rows",
    )
    require(
        len(
            {
                s["sample_id"]
                for split in ("fit", "tune", "calibration", "test")
                for s in samples[split]
            }
        )
        == 15239,
        "duplicate primary sample",
    )
    fit_primary = samples["fit_primary"]
    forward = samples["tune_forward"]
    require(
        all(s["adapter"] != "quixbugs" for s in fit_primary), "late auxiliary in forward training"
    )
    require(
        max(s["observed_ns"] for s in fit_primary) < min(s["as_of_ns"] for s in forward),
        "online chronology violation",
    )
    require(
        {s["sample_id"] for s in forward} <= {s["sample_id"] for s in samples["tune"]},
        "forward tune changed",
    )
    require(
        len({s["clock_domain"] for split in samples for s in samples[split]}) == 1,
        "clock domain mismatch",
    )
    fit_support = defaultdict(list)
    for r in samples["fit"]:
        fit_support[group_key(r["context"])].append(r)
    support = {}
    for split in ("tune", "calibration", "test"):
        unknown = Counter(
            r["adapter"] + "/" + r["context"]["tool_name"]
            for r in samples[split]
            if group_key(r["context"]) not in fit_support
        )
        support[split] = {"unknown_identity_rows": sum(unknown.values()), "by_tool": dict(unknown)}
        require(not unknown, "new evaluation backend/schema missing from training: " + split)
    checked_sources = 0
    if deep:
        for path, want in manifest["source_sha256"].items():
            require(digest(path) == want, "source changed: " + path)
            checked_sources += 1
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    require(
        cpus >= config["threads"] * 5, "not enough visible CPUs for fixed five-way configuration"
    )
    return {
        "status": "ready",
        "training_started": False,
        "calibration_started": False,
        "model_test_started": False,
        "unique_tasks": 1077,
        "unique_valid_rtt_rows": 15239,
        "split_rows": {s: len(v) for s, v in samples.items()},
        "task_groups": {s: len(v) for s, v in groups.items()},
        "group_overlap": 0,
        "evaluation_support_identity_check": support,
        "five_parallel_processes": True,
        "threads_per_model": config["threads"],
        "visible_cpus": cpus,
        "dependencies": dependencies,
        "sources_hash_checked": checked_sources,
        "forward_online_guard": "fit_primary ends before increment tune; auxiliary excluded",
        "quixbugs_role": "394 auxiliary RTT rows added to fit only",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deep", action="store_true")
    args = p.parse_args()
    print(json.dumps(check(args.deep), indent=2))
