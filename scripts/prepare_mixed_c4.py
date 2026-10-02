"""Write frozen selections and launch configs. Never start a service or run a task."""

import argparse
import hashlib
import json
import random
import tarfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from benchmark_adapters.config import load_config
from benchmark_adapters.retrieval import BrowseCompAdapter, HotpotAdapter
from benchmark_adapters.retrieval_corpus import hotpot_public_records

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("fit", "tune", "calibration", "test")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_toml(path, config):
    lines = []
    for section, fields in config.to_dict().items():
        lines.append(f"[{section}]")
        for key, value in fields.items():
            if value is not None:
                lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines))


def known_exposure():
    known = {"hotpot": {"5a8b57f25542995d1e6f1371"}, "browsecomp": set()}
    known["browsecomp"].update(
        (ROOT / "data/browsecomp_plus/browsecomp_known_history.ids").read_text().split()
    )
    for path in (ROOT / "runs/campaigns").glob("*/campaign.json"):
        for job in json.loads(path.read_text()).get("jobs", []):
            kind = job["adapter"]
            if kind in known:
                payload = Path(job["input_path"])
                if payload.is_file():
                    known[kind].add(json.loads(payload.read_text())["task"]["task_id"])
    return known


def retrieval_manifest(config, known, seed):
    if config.dataset.kind == "hotpot":
        rows = hotpot_public_records(config.dataset.path)
        adapter = HotpotAdapter
    else:
        rows = (
            json.loads(line)
            for line in Path(config.dataset.path).read_text().splitlines()
            if line.strip()
        )
        adapter = BrowseCompAdapter
    tasks = [
        adapter.from_record(
            row,
            dataset_id=config.dataset.id,
            revision=config.dataset.revision,
            split=config.dataset.split,
        )
        for row in rows
    ]
    groups = defaultdict(list)
    for task in tasks:
        groups[hashlib.sha256(task.instruction.encode()).hexdigest()].append(task)
    records = []
    for fingerprint, members in sorted(groups.items()):
        exposed = any(t.task_id in known for t in members)
        score = (
            int(hashlib.sha256(f"{seed}:{fingerprint}".encode()).hexdigest(), 16)
            / 2**256
        )
        split = (
            "historical_dev"
            if exposed
            else (
                "fit"
                if score < 0.60
                else "tune"
                if score < 0.75
                else "calibration"
                if score < 0.85
                else "test"
            )
        )
        for task in members:
            records.append(
                {
                    "dataset_id": task.dataset_id,
                    "dataset_revision": task.revision,
                    "task_id": task.task_id,
                    "task_group_id": f"{task.dataset_id}:text:{fingerprint}",
                    "instruction_sha256": fingerprint,
                    "historically_exposed": exposed,
                    "research_split": split,
                }
            )
    return records


def choose(records, split, count, seed, code=False):
    pool = [r for r in records if r["research_split"] == split]
    rng = random.Random(seed)
    pool.sort(key=lambda r: r["question_id"] if code else r["task_id"])
    rng.shuffle(pool)
    if code:
        # Spread the first training tranche across supported difficulty strata.
        strata = defaultdict(list)
        for row in pool:
            strata[row["difficulty"]].append(row)
        pool = [
            part[i]
            for i in range(max(map(len, strata.values())))
            for _, part in sorted(strata.items())
            if i < len(part)
        ]
    if len(pool) < count:
        raise ValueError(
            f"Only {len(pool)} tasks available in {split}; requested {count}"
        )
    return [row["question_id" if code else "task_id"] for row in pool[:count]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/mixed_c4_v1")
    parser.add_argument("--seed", type=int, default=20260925)
    for name, count in zip(SPLITS, (100, 20, 10, 20), strict=True):
        parser.add_argument(f"--{name}", type=int, default=count)
    args = parser.parse_args()
    counts = {split: getattr(args, split) for split in SPLITS}
    if any(n <= 0 for n in counts.values()):
        parser.error("All split counts must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    known = known_exposure()
    configs = {}
    manifests = {}
    selections = {}
    for kind in ("hotpot", "browsecomp", "livecodebench"):
        config = load_config(ROOT / f"configs/c4/{kind}.toml")
        retrieval = config.retrieval
        if kind == "hotpot":
            retrieval = replace(
                retrieval, backend="hotpot_rpc", mcp_url="http://127.0.0.1:18124"
            )
        elif kind == "browsecomp":
            retrieval = replace(
                retrieval,
                backend="browsecomp_mcp",
                mcp_url="http://127.0.0.1:18123/mcp",
            )
        config = replace(
            config,
            retrieval=retrieval,
            llm=replace(
                config.llm,
                max_input_tokens=262144,
                max_output_tokens=32768,
                timeout=3600,
                num_retries=0,
            ),
            runtime=replace(
                config.runtime,
                max_iterations=80,
                max_tool_calls=160,
                max_llm_requests=100,
                task_timeout=14400,
                tool_timeout=120 if kind == "livecodebench" else 30,
            ),
        )
        write_toml(output / f"{kind}.toml", config)
        configs[kind] = config
        if kind == "livecodebench":
            path = ROOT / "data/livecodebench/protocol/livecodebench_v1.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
        else:
            path = output / f"{kind}_splits.jsonl"
            records = retrieval_manifest(config, known[kind], args.seed)
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
            )
        manifests[kind] = (path, digest(path))
        selections[kind] = {
            split: choose(
                records, split, count, args.seed, code=kind == "livecodebench"
            )
            for split, count in counts.items()
        }
    run_dirs = {}
    for split in SPLITS:
        entries = []
        for kind in configs:
            path, checksum = manifests[kind]
            entry = {
                "config": f"{kind}.toml",
                "task_ids": selections[kind][split],
                "research_split": split,
                "split_manifest": str(path),
                "split_manifest_sha256": checksum,
            }
            if kind == "livecodebench":
                entry.update(
                    checker=str(ROOT / "data/livecodebench/testing_util.reference.py"),
                    python_bin=str(ROOT / ".venv-task/bin"),
                )
            entries.append(entry)
        run_id = f"{output.name}_{split}"
        run_dirs[split] = str(ROOT / "runs/campaigns" / run_id)
        write_json(
            output / f"{split}.json",
            {
                "version": 1,
                "run_id": run_id,
                "runs_dir": str(ROOT / "runs/campaigns"),
                "concurrency": 4,
                "adapter_concurrency": {
                    k: 4 for k in ("hotpot", "browsecomp", "livecodebench", "quixbugs")
                },
                "seed": args.seed,
                "queue_policy": "round_robin",
                "partition": split,
                "episode_id": output.name,
                "replica_id": "qwen35-9b-tp4-c4-262k",
                "uid_base": 63200,
                "entries": entries,
            },
        )
    remote = output / "remote"
    remote.mkdir()
    files = {
        name: ROOT / "scripts" / name
        for name in (
            "browsecomp_timing.py",
            "serve_browsecomp_timed.py",
            "serve_corpus_c4.sh",
        )
    }
    files["hotpot_rpc.py"] = (
        ROOT
        / "repos/Openhands-software-agent-sdk/benchmarks/flowpilot/src/benchmark_adapters/hotpot_rpc.py"
    )
    for name, source in files.items():
        (remote / name).write_bytes(source.read_bytes())
    write_json(remote / "sha256.json", {name: digest(remote / name) for name in files})
    with tarfile.open(output / "remote.tar.gz", "w:gz") as archive:
        archive.add(remote, arcname="remote")
    summary = {
        "services_started": False,
        "benchmark_tasks_executed": 0,
        "seed": args.seed,
        "counts_per_benchmark": counts,
        "total_tasks": 3 * sum(counts.values()),
        "task_concurrency": 4,
        "ratio": "1:1:1 admitted task totals; instantaneous mix can vary",
        "run_dirs": run_dirs,
        "selections": selections,
        "known_retrieval_exposure": {k: sorted(v) for k, v in known.items()},
        "exposure_scope": "existing local campaign manifests plus preserved known-history IDs; not pretraining decontamination",
        "llm": {
            "context": 262144,
            "max_output_tokens": 32768,
            "tensor_parallel_size": 4,
            "max_num_seqs": 4,
        },
        "hardware_capacity_status": "launch configuration only; four full-length contexts not empirically validated",
    }
    write_json(output / "selection_summary.json", summary)
    print(
        json.dumps(
            {
                "prepared_configs": str(output),
                "tasks": summary["total_tasks"],
                "counts_per_benchmark": counts,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
