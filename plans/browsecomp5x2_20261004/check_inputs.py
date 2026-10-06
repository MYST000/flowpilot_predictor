"""CPU-only checks against the actual remote MCP and frozen campaign inputs."""

import hashlib
import json
import sys
from pathlib import Path

from benchmark_adapters.config import load_config
from benchmark_adapters.native_browsecomp import NativeBrowseCompEnvironment
from benchmark_adapters.parallel_collection import _checked_manifest, validate_campaign
from benchmark_adapters.sdk_provenance import check_sdk


def main():
    root = Path(sys.argv[1])
    config = load_config(root / "browsecomp.toml")
    provenance = check_sdk(config.runtime)
    profile = json.loads((root / "profile.json").read_text())
    baseline_bytes = (root / profile["workload"]["baseline_latency_path"]).read_bytes()
    assert (
        hashlib.sha256(baseline_bytes).hexdigest()
        == profile["workload"]["baseline_latency_sha256"]
    )
    baselines = {row["task_id"]: row for row in json.loads(baseline_bytes)["tasks"]}
    for n in (1, 2):
        spec = json.loads((root / f"campaign.round{n}.json").read_text())
        campaign_root, manifest = _checked_manifest(root / "rounds" / spec["run_id"])
        assert len(manifest["jobs"]) == 5
        for job in manifest["jobs"]:
            task = json.loads(Path(job["input_path"]).read_text())["task"]
            baseline = baselines[task["task_id"]]
            assert task["dataset_id"] == baseline["dataset_id"]
            assert task["revision"] == baseline["dataset_revision"]
            assert (
                hashlib.sha256(task["instruction"].encode()).hexdigest()
                == baseline["instruction_sha256"]
            )
        if not (campaign_root / "validation.json").exists():
            assert validate_campaign(campaign_root)["all_checks_passed"]
    env = NativeBrowseCompEnvironment(config)
    try:
        metadata = env.prepare()
        search = env.call_tool("search", {"query": "Mount Everest"})
        search_timing = dict(env.last_timing)
        assert isinstance(search.data, list) and search.data, (
            "MCP search returned no document list"
        )
        docid = str(search.data[0]["docid"])
        document = env.call_tool("get_document", {"docid": docid})
        assert document.text.strip(), "MCP get_document returned no content"
        report = {
            "status": "passed",
            "sdk": provenance,
            "mcp_tools_sha256": metadata["observed_tools_sha256"],
            "search_result_count": len(search.data),
            "read_docid": docid,
            "search_timing": search_timing,
            "read_timing": env.last_timing,
            "benchmark_attempts_started": 0,
            "slo_multiplier": profile["workload"]["slo_multiplier"],
        }
        (root / "mcp-check.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
