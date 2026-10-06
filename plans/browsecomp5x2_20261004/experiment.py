"""Local preparation and explicit real-GPU execution of the frozen 5 x 2 replay."""

import argparse
import base64
import csv
import json
import os
import secrets
import shlex
import signal
import socket
import subprocess
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path(
    "/home/liyachen/workspace/experiments/flowpilot/browsecomp5x2-20261003T161126Z"
)
PREDICTOR = Path(__file__).resolve().parents[2]
FLOWPILOT = Path("/home/liyachen/workspace/flowpilot")
SDK = Path("/home/liyachen/openhands/software-agent-sdk")
VLLM = Path("/home/liyachen/vllm")
ML_SITE = (
    "/home/liyachen/.conda/envs/flowpilot-predictor-27b/lib/python3.12/site-packages"
)
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
DEPLOY = "/home/qiulin/flowpilot_predictor/deploy/native27b_fixed_c4_20260929"
REMOTE_ROOT = "/home/qiulin/flowpilot_predictor"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def environment(root, owner):
    credential_path = root / "services/credentials.json"
    if not credential_path.exists():
        with credential_path.open("x") as stream:
            json.dump(
                {
                    "api_key": secrets.token_urlsafe(32),
                    "dcs_key": base64.urlsafe_b64encode(
                        secrets.token_bytes(32)
                    ).decode(),
                },
                stream,
            )
        credential_path.chmod(0o600)
    credentials = json.loads(credential_path.read_text())
    env = dict(os.environ)
    for key in list(env):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
            del env[key]
    env.update(
        {
            "NO_PROXY": "*",
            "no_proxy": "*",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "OPENHANDS_SUPPRESS_BANNER": "1",
            "LLM_API_KEY": credentials["api_key"],
            "FLOWPILOT_INGRESS_API_KEY": credentials["api_key"],
            "FLOWPILOT_DCS_ENCRYPTION_KEY": credentials["dcs_key"],
            "FLOWPILOT_PREDICTOR_GATEWAY": "http://127.0.0.1:18852",
            "FLOWPILOT_EXPERIMENT_PROFILE": str(root / "profile.json"),
            "FLOWPILOT_PREDICTOR_CONFIG": str(
                PREDICTOR / "configs/predictor/runtime.json"
            ),
            "FLOWPILOT_REUSE_ENABLED": "1",
        }
    )
    env["PYTHONPATH"] = {
        "sdk": str(SDK / "benchmarks/flowpilot/src"),
        "gateway": ":".join([ML_SITE, str(PREDICTOR), str(FLOWPILOT)]),
        "vllm": str(FLOWPILOT),
    }[owner]
    return env


def command(root, owner, args, *, log=None):
    repo = {"sdk": SDK, "gateway": FLOWPILOT, "vllm": VLLM}[owner]
    cwd = FLOWPILOT if owner == "vllm" else repo
    argv = [str(repo / ".venv/bin/python"), "-B", *args]
    if log:
        with (root / "services" / log).open("a") as stream:
            subprocess.run(
                argv,
                cwd=cwd,
                env=environment(root, owner),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
    else:
        subprocess.run(argv, cwd=cwd, env=environment(root, owner), check=True)


def ledger(root):
    path = root / "services/pids.json"
    return json.loads(path.read_text()) if path.exists() else {}


def process_stamp(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def alive(row):
    try:
        fields = Path(f"/proc/{row['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and fields[19] == row["start_ticks"]
    except FileNotFoundError:
        return False


def launch(root, name, argv, *, cwd=None, env=None):
    records = ledger(root)
    if name in records and alive(records[name]):
        return records[name]
    with (root / "services" / (name + ".log")).open("a") as log:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    row = {"pid": process.pid, "start_ticks": process_stamp(process.pid), "argv": argv}
    records[name] = row
    write(root / "services/pids.json", records)
    return row


def stop(root, names=("gateway", "vllm", "tunnel")):
    records = ledger(root)
    for name in names:
        row = records.get(name)
        if row and alive(row):
            os.killpg(row["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 30
    while any(name in records and alive(records[name]) for name in names):
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Owned service has not exited; inspect services/pids.json and logs"
            )
        time.sleep(0.2)


def remote(root):
    expected = json.loads((root / "mcp-provenance.json").read_text())
    code = (
        "from pathlib import Path; import hashlib,json,subprocess; "
        f"p=Path({DEPLOY!r}); "
        f"expected={expected['deployment_sha256']!r}; "
        "actual={k:hashlib.sha256((p/k).read_bytes()).hexdigest() for k in expected}; "
        "assert actual==expected, 'Remote deployment changed'; "
        f"assert hashlib.sha256(Path({REMOTE_ROOT + '/runtime/browsecomp-native/serve.py'!r}).read_bytes()).hexdigest()=={expected['runtime_serve_sha256']!r}; "
        f"assert subprocess.check_output(['git','-C',{REMOTE_ROOT + '/repos/BrowseComp-Plus'!r},'rev-parse','HEAD'],text=True).strip()=={expected['browsecomp_commit']!r}; "
        "print('Remote MCP source hashes verified')"
    )
    subprocess.run(
        [*SSH, "qiulin_docker", "python3 -c " + shlex.quote(code)], check=True
    )
    invocation = f"bash {DEPLOY}/serve.sh browsecomp >> {REMOTE_ROOT}/logs/native27b_fixed_c4/browsecomp.log 2>&1"
    shell = (
        "if tmux -L flowpilot-corpus-native27b has-session -t browsecomp 2>/dev/null; then "
        "tmux -L flowpilot-corpus-native27b list-panes -t browsecomp -F '#{pane_pid} #{pane_dead}'; "
        "else tmux -L flowpilot-corpus-native27b new-session -d -s browsecomp "
        + shlex.quote(invocation)
        + "; fi"
    )
    subprocess.run([*SSH, "qiulin_docker", shell], check=True)


def tunnel(root):
    records = ledger(root)
    if "tunnel" in records and alive(records["tunnel"]):
        return
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 18123))
    launch(
        root,
        "tunnel",
        [
            *SSH,
            "-o",
            "ExitOnForwardFailure=yes",
            "-N",
            "-L",
            "127.0.0.1:18123:127.0.0.1:8123",
            "qiulin_docker",
        ],
    )
    deadline = time.monotonic() + 15
    while True:
        try:
            with socket.create_connection(("127.0.0.1", 18123), timeout=1):
                return
        except OSError:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "SSH tunnel did not start; inspect services/tunnel.log"
                ) from None
            time.sleep(0.2)


def check(root):
    remote(root)
    existing = ledger(root).get("tunnel")
    had_tunnel = existing is not None and alive(existing)
    try:
        tunnel(root)
        command(
            root,
            "sdk",
            [str(Path(__file__).with_name("check_inputs.py")), str(root)],
            log="check-inputs.log",
        )
        command(
            root,
            "gateway",
            [
                "-m",
                "flowpilot_predictor_bridge.launch_gateway",
                "--config",
                str(root / "profile.json"),
                "--run-dir",
                str(root / "gateway"),
                "--registry",
                str(root / "registry.json"),
                "--check",
            ],
            log="check-predictor.log",
        )
        command(
            root,
            "gateway",
            [str(Path(__file__).with_name("check_embedding.py")), str(root)],
            log="check-embedding.log",
        )
        write(
            root / "preflight.json",
            {
                "status": "passed",
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "gpu_inference_started": False,
                "benchmark_attempts_started": 0,
                "checks": [
                    "remote source hashes",
                    "SDK provenance and frozen tasks",
                    "MCP search/get_document",
                    "SLO inputs",
                    "predictor model load",
                    "gateway settings",
                    "CPU embedding",
                ],
            },
        )
    finally:
        if not had_tunnel:
            stop(root, ("tunnel",))
    print(
        "CPU/MCP preparation checks passed; no vLLM inference or benchmark task was started."
    )


def get(root, port, path):
    headers = {}
    if port == 18852:
        headers["x-flowpilot-api-key"] = json.loads(
            (root / "services/credentials.json").read_text()
        )["api_key"]
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers)
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        request, timeout=10
    ) as response:
        data = response.read()
        return json.loads(data) if data else {}


def wait_ready(root, service, port, path):
    deadline = time.monotonic() + 900
    while True:
        if not alive(ledger(root)[service]):
            raise RuntimeError(f"{service} exited; inspect its service log")
        try:
            return get(root, port, path)
        except (OSError, ValueError):
            if time.monotonic() > deadline:
                raise RuntimeError(f"{service} not ready after 900 seconds") from None
            time.sleep(2)


def start(root):
    check(root)
    records = ledger(root)
    if any(name in records and alive(records[name]) for name in ("vllm", "gateway")):
        raise RuntimeError(
            "Local services already running; use status, or stop-local before a fresh start"
        )
    gpu_rows = csv.reader(
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()
    )
    profile = json.loads((root / "profile.json").read_text())
    gpu_ids = {
        int(value)
        for value in profile["vllm"]["environment"]["CUDA_VISIBLE_DEVICES"].split(",")
    }
    selected = {row[1].strip(): row for row in gpu_rows if int(row[0]) in gpu_ids}
    applications = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    )
    if len(selected) != 4 or any(uuid in applications for uuid in selected):
        raise RuntimeError(
            f"GPUs {sorted(gpu_ids)} are missing or in use; no GPU process was started"
        )
    if any(float(row[3]) < float(row[2]) * 0.9 for row in selected.values()):
        raise RuntimeError(
            f"GPUs {sorted(gpu_ids)} lack memory for the 0.9-utilization profile"
        )
    for port in (18851, 18852):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    tunnel(root)
    launch(
        root,
        "vllm",
        [
            str(VLLM / ".venv/bin/python"),
            "-B",
            "-m",
            "examples.experiments.qwen35_27b_tp4.launch",
            "vllm",
            "--config",
            str(root / "profile.json"),
        ],
        cwd=FLOWPILOT,
        env=environment(root, "vllm"),
    )
    wait_ready(root, "vllm", 18851, "/health")
    write(
        root / "services/vllm-capabilities.json",
        get(root, 18851, "/v1/kv/capabilities"),
    )
    launch(
        root,
        "gateway",
        [
            str(FLOWPILOT / ".venv/bin/python"),
            "-B",
            "-m",
            "flowpilot_predictor_bridge.launch_gateway",
            "--config",
            str(root / "profile.json"),
            "--run-dir",
            str(root / "gateway"),
            "--registry",
            str(root / "registry.json"),
        ],
        cwd=FLOWPILOT,
        env=environment(root, "gateway"),
    )
    wait_ready(root, "gateway", 18852, "/flowpilot/health")
    snapshot(root, "services-ready")
    print("vLLM, predictor gateway and MCP tunnel ready.")


def snapshot(root, label):
    value = {
        name: get(root, 18852, path)
        for name, path in {
            "health": "/flowpilot/health",
            "predictor": "/flowpilot/v1/predictor",
            "scheduling": "/flowpilot/v1/scheduling/state",
            "reuse": "/flowpilot/v1/reuse",
            "dcs": "/flowpilot/v1/dcs",
            "tool_resolutions": "/flowpilot/v1/tool-resolutions",
            "metrics": "/flowpilot/metrics",
        }.items()
    }
    write(root / "services" / (label + ".json"), value)


def report(root):
    selection = json.loads((root / "selection.json").read_text())
    rows = []
    for round_number in (1, 2):
        campaign = json.loads((root / f"campaign.round{round_number}.json").read_text())
        for selected in selection["selected"]:
            path = (
                root
                / "rounds"
                / campaign["run_id"]
                / "tasks"
                / ("browsecomp--" + selected["task_id"])
                / "attempt-001/result.json"
            )
            result = json.loads(path.read_text()) if path.exists() else {}
            rows.append(
                {
                    "round": round_number,
                    "task_id": selected["task_id"],
                    "execution_status": result.get("execution_status", "not_run"),
                    "duration_s": result.get("duration_s"),
                    "slo_seconds": selected["slo_seconds"],
                    "slo_met": result.get("slo_met"),
                    "conversation_id": result.get("conversation_id"),
                    "job_id": result.get("slo", {}).get("job_id"),
                    "llm_requests": result.get("llm_requests"),
                    "tool_calls": result.get("tool_calls"),
                    "errors": result.get("errors"),
                    "result_path": str(path),
                }
            )
    value = {
        "attempts": rows,
        "status_counts": dict(Counter(row["execution_status"] for row in rows)),
        "slo_met_count": sum(row["slo_met"] is True for row in rows),
        "all_ten_jobs_distinct": (
            len({row["job_id"] for row in rows if row["job_id"]}) == 10
            and all(row["job_id"] == f"job-{row['conversation_id']}" for row in rows)
        ),
        "answer_accuracy_evaluated": False,
    }
    write(root / "benchmark-summary.json", value)
    print(json.dumps(value["status_counts"], ensure_ascii=False))
    return value


def run(root):
    for n in (1, 2):
        campaign = json.loads((root / f"campaign.round{n}.json").read_text())
        if (root / "rounds" / campaign["run_id"] / "collection_started.json").exists():
            raise RuntimeError(
                "An attempt already started; preserve it and prepare a new experiment directory"
            )
    try:
        start(root)
        for n in (1, 2):
            campaign = json.loads((root / f"campaign.round{n}.json").read_text())
            snapshot(root, f"before-round{n}")
            command(
                root,
                "sdk",
                [
                    "-m",
                    "benchmark_adapters.parallel_collection",
                    "run",
                    "--run-dir",
                    str(root / "rounds" / campaign["run_id"]),
                ],
                log=f"round{n}.log",
            )
            snapshot(root, f"after-round{n}")
    finally:
        result = report(root)
        stop(root)
    if (
        result["status_counts"] != {"completed": 10}
        or not result["all_ten_jobs_distinct"]
    ):
        raise RuntimeError(
            "The 10 attempts did not all complete with distinct job identities; see benchmark-summary.json"
        )


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("check", "start", "run", "report", "status", "stop-local")
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "status":
        print(
            json.dumps(
                {
                    name: {"pid": row["pid"], "alive": alive(row)}
                    for name, row in ledger(root).items()
                },
                indent=2,
            )
        )
    else:
        {
            "check": check,
            "start": start,
            "run": run,
            "report": report,
            "stop-local": stop,
        }[args.action](root)


if __name__ == "__main__":
    main()
