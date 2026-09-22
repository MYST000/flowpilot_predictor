import hashlib
import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from benchmark_adapters import code_collection as cli
from benchmark_adapters.code_tasks import CodeBundle
from benchmark_adapters.config import Config, DatasetConfig, RuntimeConfig
from benchmark_adapters.contracts import Task


def test_ids_file_keeps_order_and_rejects_duplicate_or_unsafe_ids(tmp_path):
    path = tmp_path / "tasks.ids"
    path.write_text("abc302_a\n\nabc301_b\n")
    assert cli.read_ids_file(path) == ["abc302_a", "abc301_b"]
    for content in ("", "same\nsame\n", "../escape\n", "abc_a abc_b\n"):
        path.write_text(content)
        with pytest.raises(ValueError, match="task ID"):
            cli.read_ids_file(path)


@pytest.mark.parametrize("ids", [["../escape"], ["a/b"], ["a\nb"], [1]])
def test_explicit_ids_are_safe_plain_identifiers(ids):
    with pytest.raises(ValueError, match="task ID"):
        cli.checked_ids(ids)


@pytest.fixture
def lcb_fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    lock = json.loads((Path(cli.__file__).parent / "data/lcb_release_v6.lock.json").read_text())
    row = {
        "question_id": "abc400_a",
        "question_title": "Sum",
        "question_content": "Read and sum.",
        "difficulty": "easy",
        "platform": "atcoder",
        "contest_date": "2025-04-05T00:00:00",
        "metadata": {},
        "public_test_cases": [{"input": "1 2", "output": "3", "testtype": "stdin"}],
        "private_test_cases": [{"input": "PRIVATE", "output": "HIDDEN", "testtype": "stdin"}],
    }
    for name in lock["files"]:
        (source / name).write_text(json.dumps(row) + "\n" if name == "test.jsonl" else "")
    checker = source / "checker.py"
    checker.write_text("# fixture checker")
    manifest_row = {
        **{key: row[key] for key in ("question_id", "difficulty", "platform", "contest_date")},
        "statement_sha256": hashlib.sha256(row["question_content"].encode()).hexdigest(),
        "task_group_id": "livecodebench/atcoder:abc400",
        "research_split": "fit",
        "historically_selected": False,
    }
    manifest = tmp_path / "split.jsonl"
    manifest.write_text(json.dumps(manifest_row) + "\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    def fixture_controller_directory(path, *, create=False):
        path = Path(path)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(cli, "private_controller_directory", fixture_controller_directory)
    monkeypatch.setattr(
        cli,
        "sha",
        lambda path: (
            lock["checker_sha256"] if Path(path) == checker else lock["files"][Path(path).name]
        ),
    )
    config = Config(
        dataset=DatasetConfig(
            kind="livecodebench",
            id="livecodebench/code_generation_lite",
            path=str(source),
            revision=lock["revision"],
            split="fit",
            setting="openhands-python-code-v1",
        ),
        runtime=RuntimeConfig(runs_dir=str(tmp_path / "runs")),
    )
    return config, checker, manifest, digest


def test_prepare_projects_research_split_before_freezing_selection(lcb_fixture):
    config, checker, manifest, digest = lcb_fixture
    bundles, root = cli.prepare_collection(
        config,
        ["abc400_a"],
        "/usr/bin",
        "fit-run",
        checker_path=checker,
        split_manifest=manifest,
        split_manifest_sha256=digest,
        research_split="fit",
    )
    selection = json.loads((root / "selection.json").read_text())
    assert bundles[0].task.split == selection["tasks"][0]["split"] == "fit"
    assert selection["tasks"][0]["public_metadata"]["upstream_split"] == "test"
    assert (
        selection["tasks"][0]["public_metadata"]["task_group_id"] == "livecodebench/atcoder:abc400"
    )
    assert selection["provenance"]["research_split"]["manifest_sha256"] == digest
    assert "PRIVATE" not in json.dumps(selection)
    cli.prepare_collection(
        config,
        ["abc400_a"],
        "/usr/bin",
        "fit-run",
        checker_path=checker,
        split_manifest=manifest,
        split_manifest_sha256=digest,
        research_split="fit",
    )
    manifest.write_text(json.dumps(json.loads(manifest.read_text()), separators=(",", ":")) + "\n")
    new_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Frozen selection"):
        cli.prepare_collection(
            config,
            ["abc400_a"],
            "/usr/bin",
            "fit-run",
            checker_path=checker,
            split_manifest=manifest,
            split_manifest_sha256=new_digest,
            research_split="fit",
        )


@pytest.mark.parametrize(
    "provided",
    [
        {"research_split": "fit"},
        {"split_manifest": "split.jsonl"},
        {"split_manifest_sha256": "a" * 64},
    ],
)
def test_partial_research_policy_rejected_before_dataset_access(tmp_path, provided, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Dataset must not be accessed before split policy validation")

    monkeypatch.setattr(cli, "private_controller_directory", unexpected)
    with pytest.raises(ValueError, match="together"):
        cli.load_selected(Config(), ["abc400_a"], **provided)


def test_formal_split_requires_manifest_and_matching_config(lcb_fixture):
    config, checker, manifest, digest = lcb_fixture
    with pytest.raises(ValueError, match="manifest"):
        cli.load_selected(config, ["abc400_a"], checker)
    config = replace(config, dataset=replace(config.dataset, split="dev"))
    with pytest.raises(ValueError, match="dataset.split"):
        cli.load_selected(
            config,
            ["abc400_a"],
            checker,
            split_manifest=manifest,
            split_manifest_sha256=digest,
            research_split="fit",
        )
    bundles, provenance = cli.load_selected(config, ["abc400_a"], checker)
    assert bundles[0].task.split == "dev"
    assert "research_split" not in provenance


@pytest.fixture
def main_fixture(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    bundle = CodeBundle(
        "livecodebench", Task("lcb", "v", "dev", "abc400_a", "public"), {}, "solution.py", {}
    )
    state = {"locked": False, "calls": [], "root": root, "bundle": bundle}

    @contextmanager
    def locked():
        assert not state["locked"]
        state["locked"] = True
        try:
            yield
        finally:
            state["locked"] = False

    def prepare(*args, **kwargs):
        assert state["locked"]
        state["calls"].append(("prepare", args, kwargs))
        return [bundle], root

    monkeypatch.setattr(cli, "collector_lock", locked, raising=False)
    monkeypatch.setattr(cli, "load_config", lambda path: Config())
    monkeypatch.setattr(cli, "check_sdk", lambda config: {"checked": True})
    monkeypatch.setattr(cli, "prepare_collection", prepare)

    def invoke(*extra):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "collect",
                "--config",
                "fixture.toml",
                "--ids",
                "abc400_a",
                "--run-id",
                "run",
                "--python-bin",
                "/usr/bin",
                *extra,
            ],
        )
        cli.main()

    state["invoke"] = invoke
    return state


def test_run_forwards_explicit_resume_policy_under_collector_lock(main_fixture, monkeypatch):
    state = main_fixture

    def run(*args, **kwargs):
        assert state["locked"]
        state["calls"].append(("run", args, kwargs))
        return {"stage_complete": True}

    monkeypatch.setattr(cli, "run_collection", run, raising=False)
    state["invoke"]("--run", "--resume", "--retry-infrastructure")
    assert state["calls"][-1][2] == {"run_id": "run", "resume": True, "retry_infrastructure": True}
    assert not state["locked"]


@pytest.mark.parametrize("existing", [False, True])
def test_collect_validates_only_when_missing_then_runs(main_fixture, monkeypatch, existing):
    state = main_fixture
    if existing:
        (state["root"] / "validation.json").write_text('{"all_checks_passed": true}')

    def validate(*args):
        assert state["locked"]
        state["calls"].append(("validate", args, {}))
        return {"all_checks_passed": True}

    def run(*args, **kwargs):
        assert state["locked"]
        state["calls"].append(("run", args, kwargs))
        return {"stage_complete": True}

    monkeypatch.setattr(cli, "validate", validate)
    monkeypatch.setattr(cli, "run_collection", run, raising=False)
    state["invoke"]("--collect")
    assert [item[0] for item in state["calls"]] == (
        ["prepare", "run"] if existing else ["prepare", "validate", "run"]
    )


def test_collect_stops_when_validation_fails(main_fixture, monkeypatch):
    state = main_fixture
    monkeypatch.setattr(cli, "validate", lambda *args: {"all_checks_passed": False})
    monkeypatch.setattr(
        cli,
        "run_collection",
        lambda *a, **k: pytest.fail("Failed validation must prevent actor execution"),
        raising=False,
    )
    with pytest.raises(SystemExit) as exc:
        state["invoke"]("--collect")
    assert exc.value.code == 2
    assert not state["locked"]


def test_export_uses_only_registered_accepted_attempts(main_fixture, monkeypatch, tmp_path):
    state = main_fixture
    attempts = [state["root"] / "tasks/abc400_a/attempt-002"]
    monkeypatch.setattr(cli, "accepted_attempts", lambda root: attempts, raising=False)

    def export(actual_attempts, output):
        assert state["locked"]
        assert actual_attempts == attempts
        assert output == tmp_path / "export"
        return {"requests": 3}

    checked_paths = []

    def private_parent(path, *, create=False):
        assert state["locked"]
        checked_paths.append((path, create))
        return path

    monkeypatch.setattr(cli, "private_controller_directory", private_parent)
    monkeypatch.setattr(cli, "write_prediction_dataset", export, raising=False)
    state["invoke"]("--export", "--export-dir", str(tmp_path / "export"))
    assert checked_paths == [(tmp_path, True)]
    assert not state["locked"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--run", "--retry-infrastructure"],
        ["--prepare", "--resume"],
        ["--export"],
        ["--prepare", "--export-dir", "unused"],
        ["--prepare", "--research-split", "fit"],
        ["--prepare", "--ids-file", "also.ids"],
    ],
)
def test_invalid_cli_flag_combinations_fail_before_prepare(main_fixture, extra):
    with pytest.raises(SystemExit) as exc:
        main_fixture["invoke"](*extra)
    assert exc.value.code == 2
    assert main_fixture["calls"] == []


def test_cli_id_file_and_research_policy_reach_preparation(main_fixture, tmp_path, monkeypatch):
    path = tmp_path / "selection.ids"
    path.write_text("abc401_a\nabc400_a\n")
    manifest = tmp_path / "split.jsonl"
    digest = "a" * 64
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect",
            "--config",
            "profile.toml",
            "--ids-file",
            str(path),
            "--run-id",
            "run",
            "--python-bin",
            "/usr/bin",
            "--prepare",
            "--split-manifest",
            str(manifest),
            "--split-manifest-sha256",
            digest,
            "--research-split",
            "fit",
        ],
    )
    cli.main()
    _, args, kwargs = main_fixture["calls"][0]
    assert args[1] == ["abc401_a", "abc400_a"]
    assert kwargs["split_manifest"] == manifest
    assert kwargs["split_manifest_sha256"] == digest
    assert kwargs["research_split"] == "fit"
