"""Launch only the owned BF16 TP2/C1 Hotpot micro-trial service.

The service is temporary. The parent records its PID, session, and GPU UUIDs
so shutdown can target only this launch, never unrelated GPU users.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


ROOT = Path("/home/qiulin/flowpilot_predictor")
OWNER = ROOT / "evidence" / "qwen_hotpot_trial_20260922"
MODEL = Path("/home/qiulin/HF_Model/Qwen3.5-9B")
PYTHON = ROOT / ".venv-vllm" / "bin" / "python"


def main() -> None:
    OWNER.mkdir(parents=True, exist_ok=True)
    state_file = OWNER / "service_state.json"
    if state_file.exists():
        raise SystemExit(
            "Existing owned launch state; inspect and stop it before relaunch."
        )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 8100))
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    devices = {}
    for line in query.splitlines():
        values = [part.strip() for part in line.split(",")]
        devices[int(values[0])] = dict(
            zip(
                (
                    "index",
                    "uuid",
                    "name",
                    "memory_total_mib",
                    "memory_used_mib",
                    "driver",
                ),
                values,
            )
        )
    selected = [devices[3], devices[4]]
    for gpu in selected:
        if int(gpu["memory_total_mib"]) - int(gpu["memory_used_mib"]) < 15000:
            raise SystemExit(
                f"GPU {gpu['index']} has less than the planned 15,000 MiB free"
            )
    versions = json.loads(
        subprocess.check_output(
            [
                str(PYTHON),
                "-c",
                "import importlib.metadata as m,json,platform; "
                "print(json.dumps({'python':platform.python_version(),"
                "**{p:m.version(p) for p in ['vllm','torch','transformers','tokenizers','flashinfer-python']}}))",
            ],
            text=True,
        )
    )
    env_settings = {
        "CUDA_VISIBLE_DEVICES": ",".join(gpu["uuid"] for gpu in selected),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_HOME": "/usr/local/cuda-12.5",
        "OMP_NUM_THREADS": "8",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "VLLM_CACHE_ROOT": str(OWNER / "cache" / "vllm"),
        "TORCHINDUCTOR_CACHE_DIR": str(OWNER / "cache" / "torchinductor"),
    }
    env = os.environ.copy()
    env.update(env_settings)
    env["PATH"] = "/usr/local/cuda-12.5/bin:" + env["PATH"]
    args = [
        str(PYTHON),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(MODEL),
        "--served-model-name",
        "qwen3.5-9b",
        "--host",
        "127.0.0.1",
        "--port",
        "8100",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.58",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "2048",
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3",
        "--enforce-eager",
    ]
    config_sha = hashlib.sha256((MODEL / "config.json").read_bytes()).hexdigest()
    template = MODEL / "chat_template.jinja"
    profile = {
        "profile_id": "qwen35_9b_bf16_tp2_c1_32k_hotpot_microtrial_20260922",
        "not_single_gpu_c4_acceptance": True,
        "intended_request_output_limit": 8192,
        "intended_enable_thinking": True,
        "model_path": str(MODEL),
        "model_config_sha256": config_sha,
        "chat_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest()
        if template.exists()
        else None,
        "model_full_weight_hash_reverified": False,
        "versions": versions,
        "selected_gpus_before_start": selected,
        "all_gpus_before_start_csv": query,
        "command": args,
        "environment": env_settings,
    }
    (OWNER / "service_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    with (OWNER / "service.log").open("ab", buffering=0) as output:
        child = subprocess.Popen(
            args,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    state = {
        "pid": child.pid,
        "process_group": child.pid,
        "started_at_unix": time.time(),
        "proc_start_ticks": Path(f"/proc/{child.pid}/stat").read_text().split()[21],
        "base_url": "http://127.0.0.1:8100/v1",
        "owner_dir": str(OWNER),
        "profile": str(OWNER / "service_profile.json"),
    }
    state_file.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state))


if __name__ == "__main__":
    main()
