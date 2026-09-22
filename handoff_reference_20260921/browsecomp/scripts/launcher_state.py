"""Portable launcher metadata and safety checks; uses only the standard library."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ENV_KEYS = {"HANDOFF_ROOT", "CONTROLLER_PYTHON", "VLLM_PYTHON", "MODEL_PATH", "CUDA_VISIBLE_DEVICES", "HEALTH_TIMEOUT_S", "STOP_SERVICE_ON_EXIT"}
FLAGS = ["--dtype", "bfloat16", "--tensor-parallel-size", "1", "--max-model-len", "32768", "--gpu-memory-utilization", "0.90", "--max-num-seqs", "1", "--max-num-batched-tokens", "2048", "--language-model-only", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3", "--enforce-eager"]


def parse_local_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path != "/v1" or parsed.query or parsed.fragment or parsed.username or parsed.password or not parsed.port:
        raise ValueError("llm.base_url must be http://127.0.0.1:PORT/v1")
    return parsed.port


def check_port(port):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise ValueError(f"port {port} is occupied; refusing to start or stop its owner") from exc


def gpu_info(device, require_idle=True):
    if not re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", device):
        raise ValueError("CUDA_VISIBLE_DEVICES must explicitly select one GPU index or UUID")
    result = subprocess.check_output(["nvidia-smi", "-i", device, "--query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if len(result) != 1:
        raise ValueError("exactly one physical GPU is required")
    fields = [value.strip() for value in result[0].split(",")]
    if len(fields) != 6:
        raise ValueError("unexpected nvidia-smi GPU output")
    if require_idle:
        processes = subprocess.check_output(["nvidia-smi", "-i", device, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        if processes or int(fields[4]) > 256 or int(fields[5]) > 0:
            raise ValueError("selected GPU is busy; choose an idle GPU without killing other workloads")
    return dict(zip(["uuid", "name", "driver_version", "memory_total_mib"], fields[:4]))


def load_state(owner):
    owner = Path(owner).resolve()
    state = json.loads((owner / "launch.json").read_text())
    if set(state.get("env", {})) - ENV_KEYS:
        raise ValueError("untrusted environment key in launcher state")
    if Path(state["owner_dir"]).resolve() != owner:
        raise ValueError("launcher owner path mismatch")
    return state


def write_manifest(path, payload):
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("fingerprint") != payload["fingerprint"]:
            raise ValueError("service fingerprint changed; create a new profile/run rather than overwriting")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def initialize(config_path, split, options):
    config_path = Path(config_path).resolve(strict=True)
    config = json.loads(config_path.read_text())
    for key in ["run_root", "service_manifest"]:
        if not Path(config[key]).is_absolute():
            raise ValueError(f"config {key} must be absolute after relocation")
    run_root = Path(config["run_root"]).resolve()
    package_root = Path(os.environ["HANDOFF_ROOT"]).resolve()
    if not run_root.is_relative_to(package_root.parent) or run_root == package_root.parent:
        raise ValueError("run_root must be inside the relocated flowpilot_predictor directory")
    base_url = config["llm"]["base_url"]
    port = parse_local_url(base_url)
    model_name = config.get("service_model_name", config["llm"]["model"].removeprefix("openai/"))
    if not re.fullmatch(r"[a-zA-Z0-9._/-]+", model_name):
        raise ValueError("invalid service model name")
    for key in ["CONTROLLER_PYTHON", "VLLM_PYTHON"]:
        if not Path(os.environ[key]).is_file() or not os.access(os.environ[key], os.X_OK):
            raise ValueError(f"missing executable {key}: restore and install environments first")
    if not Path(os.environ["MODEL_PATH"]).is_dir() or not os.environ["MODEL_PATH"]:
        raise ValueError("MODEL_PATH must point to local model weights")
    if os.environ["STOP_SERVICE_ON_EXIT"] not in {"0", "1"}:
        raise ValueError("STOP_SERVICE_ON_EXIT must be 0 or 1")
    if not 1 <= int(os.environ["HEALTH_TIMEOUT_S"]) <= 86400:
        raise ValueError("HEALTH_TIMEOUT_S must be within 1..86400")
    option_parser = argparse.ArgumentParser(prog="collector-options")
    option_parser.add_argument("--resume", action="store_true")
    option_parser.add_argument("--retry-infrastructure", action="store_true")
    option_parser.add_argument("--release-test", action="store_true")
    option_parser.add_argument("--limit", type=int)
    parsed_options = option_parser.parse_args(options)
    if parsed_options.limit is not None and parsed_options.limit < 1:
        raise ValueError("--limit must be positive")
    if split == "test" and not parsed_options.release_test:
        raise ValueError("test requires explicit --release-test after all protocol choices are frozen")
    check_port(port)
    gpu = gpu_info(os.environ["CUDA_VISIBLE_DEVICES"])
    token = uuid.uuid4().hex
    owner = run_root / ".launcher" / token
    owner.mkdir(parents=True, mode=0o700)
    state = {"owner_dir": str(owner), "token": token, "service_session": f"bc-service-{token[:12]}", "collector_session": f"bc-collect-{token[:12]}", "config": str(config_path), "split": split, "options": options, "base_url": base_url, "port": port, "model_name": model_name, "manifest": config["service_manifest"], "gpu": gpu, "env": {key: os.environ[key] for key in ENV_KEYS}}
    with (owner / "launch.json").open("x") as stream:
        json.dump(state, stream, indent=2)
    print(owner)


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def service_manifest(state):
    check_port(state["port"])
    gpu = gpu_info(state["env"]["CUDA_VISIBLE_DEVICES"])
    model = Path(state["env"]["MODEL_PATH"]).resolve()
    cache_path = Path(state["owner_dir"]).parent / "model_hash_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    new_cache, files = {}, []
    for path in sorted(model.rglob("*")):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(model).parts):
            continue
        if path.suffix not in {".safetensors", ".bin", ".json", ".model", ".txt", ".tiktoken", ".jinja"}:
            continue
        stat = path.stat()
        key = str(path.resolve())
        signature = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        entry = cache.get(key, {})
        digest = entry["sha256"] if entry.get("signature") == signature else hash_file(path)
        after = path.stat()
        if [after.st_size, after.st_mtime_ns, after.st_ctime_ns] != signature:
            raise ValueError(f"model file changed during hashing: {path}")
        new_cache[key] = {"signature": signature, "sha256": digest}
        files.append({"path": str(path.relative_to(model)), "size": stat.st_size, "sha256": digest})
    if not any(row["path"].endswith((".safetensors", ".bin")) for row in files):
        raise ValueError("MODEL_PATH has no model weight files")
    temporary = cache_path.with_name(f"model_hash_cache.{state['token']}.tmp")
    temporary.write_text(json.dumps(new_cache, indent=2))
    temporary.replace(cache_path)
    code = "import importlib.metadata as m,json,platform; print(json.dumps({'python':platform.python_version(),**{p:m.version(p) for p in ['vllm','torch','transformers','flashinfer-python']}}))"
    versions = json.loads(subprocess.check_output([state["env"]["VLLM_PYTHON"], "-c", code], text=True))
    expected = {"python": "3.12.14", "vllm": "0.29.0+cu129", "torch": "2.13.0+cu129", "transformers": "5.17.0", "flashinfer-python": "0.6.18"}
    if versions != expected:
        raise ValueError(f"runtime versions differ from frozen profile: {versions}; use a separately reviewed profile")
    flags = ["--served-model-name", state["model_name"], "--host", "127.0.0.1", "--port", str(state["port"]), *FLAGS]
    profile = {"model_path": str(model), "model_files": files, "gpu": gpu, "versions": versions, "base_url": state["base_url"], "service_model_name": state["model_name"], "max_model_len": 32768, "launch_flags": flags, "environment": {"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "8"}}
    fingerprint = hashlib.sha256(json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    payload = {"schema": "browsecomp_service_v1", "fingerprint": fingerprint, "fingerprint_payload": profile, "lifecycle": {"launch_token": state["token"], "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}}
    write_manifest(state["manifest"], payload)
    (Path(state["owner_dir"]) / "service_manifest.json").write_text(json.dumps(payload, indent=2))


def verify_pid(state, pid):
    process = Path("/proc") / str(int(pid))
    command = (process / "cmdline").read_bytes().split(b"\0")
    expected_script = str(Path(state["env"]["HANDOFF_ROOT"]) / "scripts" / "service.sh").encode()
    if expected_script not in command or state["owner_dir"].encode() not in command:
        raise ValueError("service pane PID command is no longer the owned wrapper")
    environment = (process / "environ").read_bytes().split(b"\0")
    if f"BROWSECOMP_OWNER_TOKEN={state['token']}".encode() not in environment:
        raise ValueError("service pane PID owner token mismatch")


def listener_owned(state):
    """Avoid accepting another service that raced our preflight port check."""
    root_pid = int((Path(state["owner_dir"]) / "service.wrapper.pid").read_text())
    verify_pid(state, root_pid)
    pids, pending = set(), [root_pid]
    while pending:
        pid = pending.pop()
        if pid in pids:
            continue
        pids.add(pid)
        try:
            pending.extend(int(value) for value in Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
        except OSError:
            pass
    inodes = set()
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        if int(fields[1].split(":")[1], 16) == state["port"] and fields[3] == "0A":
            inodes.add(f"socket:[{fields[9]}]")
    for pid in pids:
        try:
            for fd in Path(f"/proc/{pid}/fd").iterdir():
                try:
                    if os.readlink(fd) in inodes:
                        return True
                except OSError:
                    pass
        except OSError:
            pass
    return False


def health(state):
    deadline = time.monotonic() + int(state["env"]["HEALTH_TIMEOUT_S"])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    owner = Path(state["owner_dir"])
    while time.monotonic() < deadline:
        if (owner / "service.exit").exists():
            raise ValueError("owned service exited; inspect service.log")
        try:
            with opener.open(state["base_url"].removesuffix("/v1") + "/health", timeout=3) as response:
                if response.status != 200:
                    raise ValueError("service is unhealthy")
            with opener.open(state["base_url"] + "/models", timeout=3) as response:
                models = json.load(response)["data"]
            if any(item["id"] == state["model_name"] for item in models) and listener_owned(state):
                print("owned service health/model endpoint ready")
                return
        except (OSError, ValueError, urllib.error.URLError, KeyError):
            pass
        time.sleep(2)
    raise ValueError("health timeout; inspect service.log; do not silently shrink context or quantize")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "env", "field", "manifest", "health", "flags", "verify-pid", "preflight"])
    parser.add_argument("arguments", nargs="*")
    args, unknown = parser.parse_known_args()
    if args.action == "init":
        initialize(args.arguments[0], args.arguments[1], args.arguments[2:] + unknown)
        return
    if unknown:
        parser.error(f"unexpected arguments: {unknown}")
    state = load_state(args.arguments[0])
    if args.action == "env":
        for key, value in state["env"].items():
            sys.stdout.buffer.write(f"{key}={value}".encode() + b"\0")
    elif args.action == "field":
        value = state[args.arguments[1]]
        if isinstance(value, list):
            for item in value:
                sys.stdout.buffer.write(str(item).encode() + b"\0")
        else:
            print(value)
    elif args.action == "manifest":
        service_manifest(state)
    elif args.action == "health":
        health(state)
    elif args.action == "preflight":
        check_port(state["port"])
        if gpu_info(state["env"]["CUDA_VISIBLE_DEVICES"]) != state["gpu"]:
            raise ValueError("selected GPU identity changed after launch initialization")
    elif args.action == "verify-pid":
        verify_pid(state, args.arguments[1])
    elif args.action == "flags":
        for value in FLAGS:
            sys.stdout.buffer.write(value.encode() + b"\0")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
