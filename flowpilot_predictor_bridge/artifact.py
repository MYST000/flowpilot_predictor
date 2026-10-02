"""Reuse the user's artifact/code/data checks; never rewrite frozen weights."""

import ast
import importlib
import inspect
import json
from pathlib import Path

import joblib

from predictor import ARTIFACT_VERSION
from predictor.cli import load_artifact, versions
from predictor.data import digest, require


def _load_saved_code_artifact(model_path, prepared_data, artifact_code_root):
    """Validate an explicitly selected snapshot and its runtime compatibility.

    The 27B preparation snapshot differs only in CLI/data preparation logic.
    All estimator, feature and event implementations must match the imported
    runtime byte for byte; the runtime data `require` dependency must match AST.
    """
    path, code_root = Path(model_path), Path(artifact_code_root)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    expected = manifest.get("code_sha256", {})
    actual = {str(p.relative_to(code_root)): digest(p)
              for p in sorted((code_root / "predictor").glob("*.py"))}
    require(bool(expected) and actual == expected, "saved model/code snapshot mismatch")
    require(manifest.get("artifact_version") == ARTIFACT_VERSION, "unsupported artifact version")
    require(manifest.get("dependencies") == versions(), "model/runtime dependency mismatch")
    for name in ("__init__", "models", "features", "events", "evaluation"):
        module = importlib.import_module("predictor" if name == "__init__" else "predictor." + name)
        require(digest(module.__file__) == expected.get("predictor/" + name + ".py"),
                "runtime model implementation mismatch: " + name)
    saved_data = ast.parse((code_root / "predictor" / "data.py").read_text())
    saved_require = next(n for n in saved_data.body if isinstance(n, ast.FunctionDef) and n.name == "require")
    imported_require = ast.parse(inspect.getsource(require)).body[0]
    require(ast.dump(saved_require) == ast.dump(imported_require), "runtime data dependency mismatch")
    require(digest(path) == manifest.get("model_sha256"), "model hash mismatch")
    artifact = joblib.load(path)
    require(artifact["artifact_version"] == ARTIFACT_VERSION, "artifact payload version mismatch")
    require(artifact["data_manifest_sha256"] == digest(Path(prepared_data) / "manifest.json"),
            "data manifest mismatch")
    return artifact


def load_runtime_model(model_path, prepared_data, artifact_code_root=None):
    artifact = (_load_saved_code_artifact(model_path, prepared_data, artifact_code_root)
                if artifact_code_root else load_artifact(model_path, prepared_data))
    model = artifact["model"]
    if artifact["smoke"] or artifact["calibrated"]:
        raise ValueError("online bridge requires the full raw artifact")
    if model.name != "lightgbm" or model.config["target"] != "round_trip_ms":
        raise ValueError("online bridge requires LightGBM client RTT")
    return model, digest(Path(model_path))
