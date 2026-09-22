"""CPU-only safety checks; no tmux, GPU, package installation, or model launch."""
import importlib.util
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name("launcher_state.py")
        if not path.exists():
            raise AssertionError("launcher_state.py implementation is missing")
        spec = importlib.util.spec_from_file_location("launcher_state", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_occupied_port_is_refused(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            with self.assertRaisesRegex(ValueError, "occupied"):
                self.module.check_port(server.getsockname()[1])

    def test_remote_or_non_v1_url_is_refused(self):
        for url in ["http://example.com:8000/v1", "http://127.0.0.1:8000", "https://127.0.0.1:8000/v1"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.module.parse_local_url(url)
        self.assertEqual(self.module.parse_local_url("http://127.0.0.1:8000/v1"), 8000)

    def test_manifest_cannot_silently_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "service.json"
            self.module.write_manifest(path, {"fingerprint": "a", "profile": {"dtype": "bfloat16"}})
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                self.module.write_manifest(path, {"fingerprint": "b", "profile": {"dtype": "float16"}})
            self.assertEqual(original, path.read_bytes())

    def test_owner_state_rejects_untrusted_environment_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            owner = Path(temporary) / "run"
            owner.mkdir()
            (owner / "launch.json").write_text(json.dumps({"owner_dir": str(owner), "env": {"BASH_ENV": "/tmp/execute"}}))
            with self.assertRaisesRegex(ValueError, "environment"):
                self.module.load_state(owner)

    def test_stop_script_never_signals_foreign_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            owner = directory / "owner"
            owner.mkdir()
            source_root = Path(__file__).resolve().parents[1]
            state = {
                "owner_dir": str(owner), "token": "abc123", "service_session": "bc-service-abc123", "collector_session": "bc-collect-abc123",
                "env": {"HANDOFF_ROOT": str(source_root), "CONTROLLER_PYTHON": sys.executable, "VLLM_PYTHON": sys.executable},
            }
            (owner / "launch.json").write_text(json.dumps(state))
            stub = directory / "tmux"
            stub.write_text("#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$MOCK_TMUX_LOG\"\ncase \"$1\" in\n has-session) exit 0;;\n show-options) echo unrelated-token;;\n *) exit 88;;\nesac\n")
            stub.chmod(0o755)
            log = directory / "tmux.log"
            env = dict(os.environ, PATH=str(directory)+os.pathsep+os.environ["PATH"], MOCK_TMUX_LOG=str(log))
            result = subprocess.run(["bash", str(source_root/"scripts"/"stop_owned_service.sh"), str(owner)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn("owner token/path mismatch", result.stderr)
            self.assertNotIn("send-keys", log.read_text())

    def test_current_unrelated_process_is_not_a_service(self):
        state = {"env": {"HANDOFF_ROOT": str(Path(__file__).resolve().parents[1])}, "owner_dir": "/tmp/unrelated", "token": "abc"}
        with self.assertRaisesRegex(ValueError, "command"):
            self.module.verify_pid(state, os.getpid())

    def test_busy_gpu_is_refused_without_any_signal(self):
        with patch.object(self.module.subprocess, "check_output", side_effect=[
            "GPU-abcd, NVIDIA RTX 4090, 590.0, 24564, 512, 10\n", "4321\n"
        ]) as query:
            with self.assertRaisesRegex(ValueError, "busy"):
                self.module.gpu_info("0")
            self.assertTrue(all(call.args[0][0] == "nvidia-smi" for call in query.call_args_list))

    def test_gpu_selection_must_be_explicit_single_card(self):
        for device in ["", "0,1", "all", "MIG-abcd", "-1"]:
            with self.subTest(device=device), self.assertRaises(ValueError):
                self.module.gpu_info(device)

    def test_manifest_hashes_model_bytes_and_uses_collector_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            owner = directory / "launch" / "token"
            owner.mkdir(parents=True)
            model = directory / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"fake weights; no model is loaded")
            (model / "tokenizer.json").write_text('{"unicode":"检索"}')
            manifest = directory / "service_manifest.json"
            versions = {"python":"3.12.14", "vllm":"0.29.0+cu129", "torch":"2.13.0+cu129", "transformers":"5.17.0", "flashinfer-python":"0.6.18"}
            state = {"owner_dir":str(owner), "env":{"CUDA_VISIBLE_DEVICES":"0", "MODEL_PATH":str(model), "VLLM_PYTHON":sys.executable}, "port":8000, "token":"test", "model_name":"qwen3.5-9b", "base_url":"http://127.0.0.1:8000/v1", "manifest":str(manifest)}
            with patch.object(self.module, "gpu_info", return_value={"name":"fixture GPU"}), patch.object(self.module, "check_port"), patch.object(self.module.subprocess, "check_output", return_value=json.dumps(versions)):
                self.module.service_manifest(state)
                payload = json.loads(manifest.read_text())
                stable = payload["fingerprint_payload"]
                expected = hashlib.sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
                self.assertEqual(payload["fingerprint"], expected)
                self.assertEqual(stable["max_model_len"], 32768)
                weight = next(row for row in stable["model_files"] if row["path"] == "model.safetensors")
                self.assertEqual(weight["sha256"], hashlib.sha256((model/"model.safetensors").read_bytes()).hexdigest())
                self.module.service_manifest(state)
                (model/"model.safetensors").write_bytes(b"different weights")
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    self.module.service_manifest(state)


if __name__ == "__main__":
    unittest.main()
