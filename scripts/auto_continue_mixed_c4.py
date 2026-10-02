"""Arm a tmux watcher that resumes mixed C4 collection after fit completes."""

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/mixed_c4_v1"
LOG_DIR = ROOT / "logs/mixed_c4_v1"
STATE = LOG_DIR / "auto_continue.state.json"
SPLITS = ("tune", "calibration", "test")


def write_state(status, **details):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_name(f"{STATE.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps({"status": status, "updated_at": time.time(), **details}, indent=2)
        + "\n"
    )
    temporary.replace(STATE)
    print(f"auto_continue: {status} {details}", flush=True)


def tmux(*args):
    return subprocess.run(
        ["tmux", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def split_root(split):
    config = json.loads((PLAN / f"{split}.json").read_text())
    return Path(config["runs_dir"]) / config["run_id"]


def active_collectors():
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if (
            any(arg.endswith(b"/mixed_c4_control.py") for arg in arguments)
            and b"run" in arguments
        ):
            active.append(int(entry.name))
    return active


def verify_split(split):
    root = split_root(split)
    manifest = json.loads((root / "campaign.json").read_text())
    summary = json.loads((root / "collection_summary.json").read_text())
    if summary.get("status") != "collected":
        raise ValueError(f"{split}: collection status is {summary.get('status')!r}")
    receipts = list((root / "receipts").glob("*.json"))
    if len(receipts) != len(manifest["jobs"]):
        raise ValueError(f"{split}: {len(receipts)}/{len(manifest['jobs'])} receipts")
    for path in receipts:
        record = json.loads(path.read_text())
        if record.get("status") != "returned":
            raise ValueError(
                f"{split}: abnormal receipt {path.name}: {record.get('status')}"
            )
    for filename in ("inputs.jsonl", "targets.jsonl"):
        if not (root / "prediction_dataset" / filename).is_file():
            raise ValueError(f"{split}: missing prediction_dataset/{filename}")
    audit = json.loads((root / "tool_timing_dataset/audit.json").read_text())
    if audit.get("trace_issues"):
        raise ValueError(
            f"{split}: {len(audit['trace_issues'])} trace issues in audit.json"
        )
    measured = {job["adapter"]: 0 for job in manifest["jobs"]}
    with (root / "tool_timing_dataset/tools.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            label = row["labels"]
            if label.get("executed") and all(
                isinstance(label.get(key), (int, float))
                and not isinstance(label.get(key), bool)
                and label[key] >= 0
                for key in ("executor_duration_ms", "round_trip_ms")
            ):
                measured[row["adapter"]] += 1
    if any(count == 0 for count in measured.values()):
        raise ValueError(f"{split}: no measured tool time for an adapter: {measured}")
    return {"receipts": len(receipts), "measured_calls": measured}


def arm():
    pane = tmux("display-message", "-p", "-t", "fp-collect:tasks", "#{pane_id}")
    windows = tmux(
        "list-windows", "-t", "fp-collect", "-F", "#{window_name}"
    ).splitlines()
    if "auto" in windows:
        raise ValueError(
            "fp-collect:auto already exists; no second watcher was started"
        )
    if STATE.exists():
        state = json.loads(STATE.read_text())
        if state.get("status") in {"dispatched", "running"}:
            raise ValueError(
                f"automation already {state['status']}; no duplicate was started"
            )
    command = shlex.join(
        [sys.executable, "-u", str(Path(__file__).resolve()), "watch", "--pane", pane]
    )
    write_state("armed", pane=pane, window="fp-collect:auto")
    tmux(
        "new-window", "-d", "-t", "fp-collect:", "-n", "auto", "-c", str(ROOT), command
    )


def watch(pane):
    write_state("waiting_for_fit", pane=pane)
    root = split_root("fit")
    while True:
        current_command = tmux(
            "display-message", "-p", "-t", pane, "#{pane_current_command}"
        )
        summary_path = root / "collection_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if summary.get("status") != "collected":
                raise ValueError(f"fit ended with status {summary.get('status')!r}")
            if (
                root / "tool_timing_dataset/audit.json"
            ).exists() and current_command == "bash":
                evidence = verify_split("fit")
                if active_collectors():
                    raise ValueError("another collection process is running")
                invocation = shlex.join(
                    [sys.executable, "-u", str(Path(__file__).resolve()), "continue"]
                )
                write_state("dispatched", pane=pane, fit=evidence)
                tmux("send-keys", "-l", "-t", pane, invocation)
                tmux("send-keys", "-t", pane, "Enter")
                return
            if (
                current_command == "bash"
                and not (root / "tool_timing_dataset/audit.json").exists()
            ):
                raise ValueError("fit ended without tool_timing_dataset/audit.json")
        elif (root / "collection_started.json").exists() and current_command == "bash":
            raise ValueError("fit process exited without collection_summary.json")
        time.sleep(30)


def continue_collection():
    lock_path = LOG_DIR / "auto_continue.lock"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another auto continuation is running") from exc
        fit = verify_split("fit")
        write_state("running", split="tune", fit=fit)
        for split in SPLITS:
            root = split_root(split)
            if (root / "collection_summary.json").exists():
                evidence = verify_split(split)
                print(
                    f"auto_continue: {split} already completed: {evidence}", flush=True
                )
                continue
            if (root / "collection_started.json").exists():
                raise ValueError(
                    f"{split} already started but did not complete; refusing to rerun"
                )
            running = active_collectors()
            if running:
                raise ValueError(f"another collection process is running: {running}")
            write_state("running", split=split)
            log_path = LOG_DIR / f"{split}.log"
            with log_path.open("a", buffering=1) as log:
                command = [
                    "bash",
                    str(ROOT / "scripts/collect_mixed_c4.sh"),
                    "run",
                    "--split",
                    split,
                ]
                with subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                ) as process:
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    if process.wait() != 0:
                        raise RuntimeError(f"{split} collection failed; see {log_path}")
            evidence = verify_split(split)
            print(f"auto_continue: {split} verified: {evidence}", flush=True)
        write_state("completed", splits=list(SPLITS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("arm", "watch", "continue", "status"))
    parser.add_argument("--pane")
    args = parser.parse_args()
    if args.action == "status":
        print(STATE.read_text() if STATE.exists() else '{"status": "not_armed"}')
        return
    try:
        if args.action == "arm":
            arm()
        elif args.action == "watch":
            if not args.pane:
                parser.error("watch requires --pane")
            watch(args.pane)
        else:
            continue_collection()
    except Exception as exc:
        if args.action in {"watch", "continue"}:
            write_state(
                "failed", action=args.action, error=f"{type(exc).__name__}: {exc}"
            )
        raise


if __name__ == "__main__":
    main()
