"""Rebuild prepared backend identities from hash-verified original T0 inputs.

Writes a new dataset for retraining; never edits an existing model or dataset.
"""

import argparse
import json
import shutil
from pathlib import Path

from predictor.data import (
    backend_signature,
    digest,
    load_split,
    require,
    rows,
    signature,
    write_json,
)


def migrate(prepared_data, output, original_source_root, source_root):
    prepared_data, output = Path(prepared_data), Path(output)
    original_source_root, source_root = Path(original_source_root), Path(source_root)
    manifest_path = prepared_data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(
        manifest.get("backend_identity_version", 1) == 1, "expected legacy identities"
    )
    versions = {}
    sources = {}
    for original, expected in manifest["source_sha256"].items():
        if not original.endswith("/prediction_dataset/inputs.jsonl"):
            continue
        path = source_root / Path(original).relative_to(original_source_root)
        require(digest(path) == expected, "original prediction inputs hash mismatch")
        sources[str(path)] = expected
        for item in rows(path):
            features = item["features"]
            environment = features["environment_known_at_t0"]
            profile = features["tool_execution_profile"]
            revision = item["dataset_revision"]
            legacy = signature(
                {
                    "environment": environment,
                    "dataset_revision": revision,
                    "actor": item["replica_id"],
                    "profile": profile,
                }
            )
            current = backend_signature(environment, revision, profile)
            require(
                legacy not in versions or versions[legacy] == current,
                "ambiguous legacy backend identity",
            )
            versions[legacy] = current
    require(bool(sources), "no original prediction inputs recorded")
    # Validate every frozen split before creating the new dataset.
    splits = {name: load_split(prepared_data, name) for name in manifest["splits"]}
    require(
        all(
            row["context"]["backend_version"] in versions
            for split in splits.values()
            for row in split
        ),
        "original inputs do not cover every prepared backend",
    )
    output.mkdir(parents=True, exist_ok=False)
    for path in prepared_data.iterdir():
        if path.is_file() and path.name != "manifest.json" and path.stem not in splits:
            shutil.copyfile(path, output / path.name)
    for name, samples in splits.items():
        path = output / (name + ".jsonl")
        with path.open("w") as stream:
            for sample in samples:
                context = sample["context"]
                context["backend_version"] = versions[context["backend_version"]]
                stream.write(
                    json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n"
                )
        manifest["splits"][name]["sha256"] = digest(path)
    manifest["backend_identity_version"] = 2
    manifest["identity_migration"] = {
        "source_manifest_sha256": digest(manifest_path),
        "verified_input_sha256": sources,
        "backend_versions": versions,
        "changed_fields": ["context.backend_version"],
        "labels_and_splits_unchanged": True,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-data", "output", "original-source-root", "source-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = migrate(
        args.prepared_data, args.output, args.original_source_root, args.source_root
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "backend_identity_version": 2,
                "rows": {k: v["rows"] for k, v in result["splits"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
