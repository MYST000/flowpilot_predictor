"""Collect explicitly selected QuixBugs/LiveCodeBench tasks using OpenHands."""

import argparse
import json
import re
import subprocess
from pathlib import Path

from .code_campaign import freeze_selection, sha, validate
from .code_tasks import LiveCodeBenchAdapter, QuixBugsAdapter
from .collection_runner import accepted_attempts, collector_lock, run_collection
from .config import load_config
from .controller_paths import private_controller_directory
from .prediction_export import write_prediction_dataset
from .research_splits import RESEARCH_SPLITS, apply_research_split
from .sdk_provenance import check_sdk


def checked_ids(task_ids):
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Provide nonempty, unique task IDs")
    for task_id in task_ids:
        if (
            not isinstance(task_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", task_id)
            or ".." in task_id
        ):
            raise ValueError("Invalid task ID; use a plain benchmark identifier")
    return list(task_ids)


def read_ids_file(path):
    """Read one explicit task ID per nonempty line, preserving file order."""
    return checked_ids(
        [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    )


def _split_policy(config, split_manifest, split_manifest_sha256, research_split):
    supplied = [
        value is not None for value in (split_manifest, split_manifest_sha256, research_split)
    ]
    if any(supplied) and not all(supplied):
        raise ValueError("Provide split manifest, SHA256 and research split together")
    if all(supplied):
        if research_split not in RESEARCH_SPLITS:
            raise ValueError("Unknown research split")
        if config.dataset.split != research_split:
            raise ValueError("dataset.split must match the requested research split")
    elif config.dataset.split != "dev":
        raise ValueError(
            "Formal collection requires a frozen split manifest and SHA256; default is dev"
        )


def select_lcb_records(paths, task_ids, *, revision, checker_source):
    task_ids = checked_ids(task_ids)
    requested = set(task_ids)
    found = {}
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                task_id = str(row["question_id"])
                if task_id not in requested:
                    continue
                if task_id in found:
                    raise ValueError(f"Duplicate dataset task ID: {task_id}")
                if row["difficulty"] not in {"easy", "medium"}:
                    raise ValueError("This profile supports easy/medium tasks only")
                found[task_id] = LiveCodeBenchAdapter.from_record(
                    row, revision=revision, checker_source=checker_source
                )
    if missing := requested - found.keys():
        raise ValueError(f"Unknown task IDs: {sorted(missing)}")
    return [found[task_id] for task_id in task_ids]


def load_selected(
    config,
    task_ids,
    checker_path=None,
    *,
    split_manifest=None,
    split_manifest_sha256=None,
    research_split=None,
):
    _split_policy(config, split_manifest, split_manifest_sha256, research_split)
    task_ids = checked_ids(task_ids)
    source = private_controller_directory(config.dataset.path)
    if config.dataset.kind == "quixbugs":
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ).strip()
        if actual != config.dataset.revision or dirty:
            raise ValueError("QuixBugs must be clean at the declared dataset revision")
        available = {
            p.stem.removeprefix("test_") for p in (source / "python_testcases").glob("test_*.py")
        }
        if missing := set(task_ids) - available:
            raise ValueError(f"Unknown task IDs: {sorted(missing)}")
        bundles = [QuixBugsAdapter.load_one(source, name, revision=actual) for name in task_ids]
        provenance = {"revision": actual, "source": str(source.resolve())}
    elif config.dataset.kind == "livecodebench":
        lock = json.loads((Path(__file__).parent / "data/lcb_release_v6.lock.json").read_text())
        if config.dataset.revision != lock["revision"]:
            raise ValueError("This profile requires the pinned LiveCodeBench release_v6")
        if checker_path is None:
            raise ValueError(
                "LiveCodeBench requires --checker pointing to the pinned official checker"
            )
        checker_path = Path(checker_path)
        if sha(checker_path) != lock["checker_sha256"]:
            raise ValueError("LiveCodeBench checker SHA256 mismatch")
        paths = [source / name for name in lock["files"]]
        for path in paths:
            if sha(path) != lock["files"][path.name]:
                raise ValueError(f"LiveCodeBench source SHA256 mismatch: {path.name}")
        bundles = select_lcb_records(
            paths,
            task_ids,
            revision=config.dataset.revision,
            checker_source=checker_path.read_text(),
        )
        provenance = {
            **lock,
            "source": str(source.resolve()),
            "checker_path": str(checker_path.resolve()),
        }
    else:
        raise ValueError("code_collection currently supports quixbugs and livecodebench")
    if split_manifest is not None:
        bundles, split_provenance = apply_research_split(
            bundles, split_manifest, split_manifest_sha256, research_split
        )
        provenance["research_split"] = split_provenance
    return bundles, {
        **provenance,
        "selection_rule": "explicit ordered task IDs; no score-based selection",
    }


def prepare_collection(
    config,
    task_ids,
    python_bin,
    run_id,
    *,
    checker_path=None,
    split_manifest=None,
    split_manifest_sha256=None,
    research_split=None,
):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_id) or ".." in run_id:
        raise ValueError("Invalid run_id; use one plain directory name")
    bundles, provenance = load_selected(
        config,
        task_ids,
        checker_path,
        split_manifest=split_manifest,
        split_manifest_sha256=split_manifest_sha256,
        research_split=research_split,
    )
    provenance["python_bin"] = str(Path(python_bin).absolute())
    output_root = private_controller_directory(config.runtime.runs_dir, create=True)
    ops = output_root / run_id
    private_controller_directory(ops, create=True)
    root = freeze_selection(ops, config.dataset.kind, bundles, provenance, config)
    return bundles, root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    ids = parser.add_mutually_exclusive_group(required=True)
    ids.add_argument("--ids", nargs="+", help="Explicit ordered benchmark task IDs")
    ids.add_argument("--ids-file", type=Path, help="One task ID per nonempty line")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--python-bin",
        type=Path,
        required=True,
        help="Task venv bin directory; preserve venv symlinks",
    )
    parser.add_argument("--checker", type=Path, help="Pinned LiveCodeBench testing_util.py")
    parser.add_argument("--split-manifest", type=Path, help="Frozen research split JSONL")
    parser.add_argument("--split-manifest-sha256", help="Expected SHA256 of the split manifest")
    parser.add_argument("--research-split", choices=sorted(RESEARCH_SPLITS))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the frozen run without repeating accepted attempts",
    )
    parser.add_argument(
        "--retry-infrastructure",
        action="store_true",
        help="With --resume, create a new attempt after an infrastructure failure",
    )
    parser.add_argument("--export-dir", type=Path, help="New output directory for --export")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="Freeze task selection without task execution or model requests",
    )
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Run independent reference/checker conformance checks",
    )
    mode.add_argument(
        "--run", action="store_true", help="Run selected tasks after successful validation"
    )
    mode.add_argument(
        "--collect",
        action="store_true",
        help="Validate if missing, then collect the selected tasks",
    )
    mode.add_argument(
        "--export", action="store_true", help="Export accepted attempts to a prediction dataset"
    )
    args = parser.parse_args()
    split_options = (args.split_manifest, args.split_manifest_sha256, args.research_split)
    if any(value is not None for value in split_options) and not all(
        value is not None for value in split_options
    ):
        parser.error(
            "--split-manifest, --split-manifest-sha256 and --research-split must be provided together"
        )
    if args.resume and not (args.run or args.collect):
        parser.error("--resume requires --run or --collect")
    if args.retry_infrastructure and not args.resume:
        parser.error("--retry-infrastructure requires --resume")
    if args.export != (args.export_dir is not None):
        parser.error("--export and --export-dir must be provided together")
    task_ids = read_ids_file(args.ids_file) if args.ids_file is not None else checked_ids(args.ids)
    config = load_config(args.config)
    sdk = check_sdk(config.runtime)
    validation_failed = False
    with collector_lock():
        bundles, root = prepare_collection(
            config,
            task_ids,
            args.python_bin,
            args.run_id,
            checker_path=args.checker,
            split_manifest=args.split_manifest,
            split_manifest_sha256=args.split_manifest_sha256,
            research_split=args.research_split,
        )
        if args.prepare:
            result = {
                "prepared": True,
                "root": str(root),
                "task_ids": [b.task.task_id for b in bundles],
                "sdk": sdk,
            }
        elif args.validate:
            result = validate(bundles, config, root, args.python_bin)
            validation_failed = not result["all_checks_passed"]
        elif args.export:
            attempts = accepted_attempts(root)
            if not attempts:
                raise ValueError("No accepted attempts to export")
            private_controller_directory(args.export_dir.parent, create=True)
            result = write_prediction_dataset(attempts, args.export_dir)
        else:
            if args.collect:
                validation_path = root / "validation.json"
                result = (
                    json.loads(validation_path.read_text())
                    if validation_path.exists()
                    else validate(bundles, config, root, args.python_bin)
                )
                validation_failed = not result["all_checks_passed"]
            if not validation_failed:
                result = run_collection(
                    bundles,
                    config,
                    root,
                    args.python_bin,
                    run_id=args.run_id,
                    resume=args.resume,
                    retry_infrastructure=args.retry_infrastructure,
                )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if validation_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
