import copy
import json

import pytest

from predictor.data import backend_signature, digest, load_split, signature, write_json
from predictor.migrate_backend_identity import migrate


@pytest.fixture
def legacy_data(tmp_path):
    raw = tmp_path / "raw"
    path = raw / "batch/prediction_dataset/inputs.jsonl"
    path.parent.mkdir(parents=True)
    inputs = [
        {
            "dataset_revision": "corpus-v1",
            "replica_id": actor,
            "features": {
                "environment_known_at_t0": {"backend": "browsecomp_mcp"},
                "tool_execution_profile": {"tool_timeout_seconds": 30},
            },
        }
        for actor in ("stock-model", "flowpilot-model")
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in inputs))
    samples = [
        {
            "sample_id": str(index),
            "task_group_id": "group-" + str(index),
            "context": {
                "backend_id": "browsecomp_mcp",
                "backend_version": signature(
                    {
                        "environment": row["features"]["environment_known_at_t0"],
                        "dataset_revision": row["dataset_revision"],
                        "actor": row["replica_id"],
                        "profile": row["features"]["tool_execution_profile"],
                    }
                ),
                "tool_schema_version": "schema-v1",
            },
            "labels": {"round_trip_ms": 12.0},
        }
        for index, row in enumerate(inputs)
    ]
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    split = prepared / "fit.jsonl"
    split.write_text("".join(json.dumps(row) + "\n" for row in samples))
    write_json(
        prepared / "manifest.json",
        {
            "source_sha256": {
                "/old/batch/prediction_dataset/inputs.jsonl": digest(path)
            },
            "splits": {"fit": {"sha256": digest(split), "rows": 2, "task_groups": 2}},
        },
    )
    return prepared, raw, inputs, samples


def test_migration_removes_experiment_identity_and_preserves_frozen_labels(
    tmp_path, legacy_data
):
    prepared, raw, inputs, original = legacy_data
    before = digest(prepared / "fit.jsonl")
    output = tmp_path / "new"
    manifest = migrate(prepared, output, "/old", raw)
    current = load_split(output, "fit")
    expected = backend_signature(
        inputs[0]["features"]["environment_known_at_t0"],
        "corpus-v1",
        inputs[0]["features"]["tool_execution_profile"],
    )
    for old, new in zip(original, current, strict=True):
        assert new["context"]["backend_version"] == expected
        restored = copy.deepcopy(new)
        restored["context"]["backend_version"] = old["context"]["backend_version"]
        assert restored == old
    assert digest(prepared / "fit.jsonl") == before
    assert manifest["backend_identity_version"] == 2
    assert manifest["identity_migration"]["labels_and_splits_unchanged"]


def test_migration_rejects_changed_original_inputs(tmp_path, legacy_data):
    prepared, raw, _, _ = legacy_data
    path = raw / "batch/prediction_dataset/inputs.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="inputs hash mismatch"):
        migrate(prepared, tmp_path / "new", "/old", raw)
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("changed", ["backend", "corpus", "profile"])
def test_backend_identity_keeps_execution_compatibility_boundaries(changed):
    original = backend_signature({"backend": "mcp"}, "v1", {"timeout": 30})
    current = backend_signature(
        {"backend": "rpc" if changed == "backend" else "mcp"},
        "v2" if changed == "corpus" else "v1",
        {"timeout": 60 if changed == "profile" else 30},
    )
    assert current != original
