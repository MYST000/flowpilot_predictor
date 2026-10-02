"""Explicit future execution: five algorithms, calibration, frozen tests and reports."""

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from preflight import DATA, PLAN, check

sys.path.insert(0, str(PLAN / "code"))
from predictor.data import digest

PYTHON = Path("/root/flowpilot_predictor/.venv-predictor/bin/python")
ALGORITHMS = ("empirical", "ewma", "cluster", "qrf", "lightgbm")
BASE = Path("/data1/ql_flowpilot_predictor/predictor_experiments")


def commands(output):
    cli = [str(PYTHON), "-B", "-u", "-m", "predictor.cli"]
    common = ["--data", str(DATA)]
    config = ["--config", str(PLAN / "model_config.json")]
    training = []
    calibration = []
    testing = []
    for algorithm in ALGORITHMS:
        model = output / "train" / algorithm / "model.joblib"
        corrected = output / "final" / (algorithm + "_calibrated")
        training.append(
            (
                algorithm,
                cli
                + [
                    "train",
                    "--algorithm",
                    algorithm,
                    *common,
                    *config,
                    "--output",
                    str(model.parent),
                ],
            )
        )
        calibration.append(
            (
                algorithm,
                cli + ["calibrate", *common, "--model", str(model), "--output", str(corrected)],
            )
        )
        testing.extend(
            [
                (
                    algorithm + "_raw",
                    cli
                    + [
                        "evaluate",
                        *common,
                        "--model",
                        str(model),
                        "--split",
                        "test",
                        "--allow-test",
                        "--mode",
                        "frozen",
                        "--output",
                        str(output / "final" / (algorithm + "_test_raw")),
                    ],
                ),
                (
                    algorithm + "_calibrated",
                    cli
                    + [
                        "evaluate",
                        *common,
                        "--model",
                        str(corrected / "model.joblib"),
                        "--split",
                        "test",
                        "--allow-test",
                        "--mode",
                        "frozen",
                        "--output",
                        str(output / "final" / (algorithm + "_test_calibrated")),
                    ],
                ),
            ]
        )
    forward_model = output / "online" / "ewma_forward_fit" / "model.joblib"
    forward = [
        (
            "ewma_forward_fit",
            cli
            + [
                "train",
                "--algorithm",
                "ewma",
                *common,
                *config,
                "--fit-split",
                "fit_primary",
                "--tune-split",
                "tune_forward",
                "--output",
                str(forward_model.parent),
            ],
        ),
        (
            "ewma_forward_online",
            cli
            + [
                "evaluate",
                *common,
                "--model",
                str(forward_model),
                "--split",
                "tune_forward",
                "--mode",
                "online",
                "--output",
                str(output / "online" / "ewma_forward_online"),
            ],
        ),
    ]
    latency = [
        (
            algorithm,
            [
                str(PYTHON),
                "-B",
                "-u",
                str(PLAN / "latency.py"),
                "--data",
                str(DATA),
                "--model",
                str(output / "final" / (algorithm + "_calibrated") / "model.joblib"),
                "--output",
                str(output / "latency" / (algorithm + ".json")),
            ],
        )
        for algorithm in ALGORITHMS
    ]
    report = [
        ("report", [str(PYTHON), "-B", "-u", str(PLAN / "summarize.py"), "--root", str(output)])
    ]
    # Five processes at a time; raw and calibrated use distinct artifact directories.
    return [
        ("train", training, 5),
        ("calibration", calibration, 5),
        ("test", testing, 5),
        ("forward_online", forward, 1),
        ("isolated_latency", latency, 1),
        ("summary", report, 1),
    ]


class Runner:
    def __init__(self, output):
        self.output = output
        self.children = []
        self.state = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "stage": "initializing",
            "finished": [],
            "active": [],
        }

    def save(self):
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        path = self.output / "status.json"
        tmp = self.output / "status.tmp"
        tmp.write_text(json.dumps(self.state, indent=2) + "\n")
        tmp.replace(path)

    def cancel_children(self):
        alive = [p for p in self.children if p.poll() is None]
        for p in alive:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        for p in alive:
            try:
                p.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()

    def stage(self, name, jobs, parallelism):
        self.state["stage"] = name
        self.save()
        pending = list(jobs)
        active = []
        while pending or active:
            while pending and len(active) < parallelism:
                label, command = pending.pop(0)
                log = (self.output / "logs" / (name + "_" + label + ".log")).open("x")
                try:
                    child = subprocess.Popen(
                        command,
                        cwd=PLAN / "code",
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                finally:
                    log.close()
                self.children.append(child)
                active.append((label, child))
                print(name, label, "started", child.pid, flush=True)
            self.state["active"] = [{"label": label, "pid": p.pid} for label, p in active]
            self.save()
            time.sleep(0.5)
            for label, p in list(active):
                code = p.poll()
                if code is None:
                    continue
                if code != 0:
                    raise RuntimeError(
                        f"{name}/{label} exited {code}; inspect logs/{name}_{label}.log"
                    )
                active.remove((label, p))
                self.state["finished"].append(name + "/" + label)
                print(name, label, "complete", flush=True)
        self.state["active"] = []
        self.save()

    def freeze(self):
        records = []
        for algorithm in ALGORITHMS:
            model = self.output / "train" / algorithm
            m = json.loads((model / "manifest.json").read_text())
            metrics = json.loads((model / "metrics.json").read_text())
            if m["smoke_only"] or m["training_split"] != "fit":
                raise ValueError("not a full main fit model")
            records.append(
                {
                    "algorithm": algorithm,
                    "model_sha256": digest(model / "model.joblib"),
                    "manifest_sha256": digest(model / "manifest.json"),
                    "tune_metrics_sha256": digest(model / "metrics.json"),
                    "tune_tool_macro_mean_pinball_ms": metrics["tool_macro_mean_pinball_ms"],
                    "tune_supported_rows": metrics["micro"]["supported_rows"],
                    "tune_rows": metrics["micro"]["rows"],
                }
            )
        eligible = [
            r
            for r in records
            if r["tune_supported_rows"] == r["tune_rows"]
            and r["tune_tool_macro_mean_pinball_ms"] is not None
        ]
        frozen = {
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "selection_rule": "lowest tune tool-macro mean four-quantile pinball among full-support candidates; no automatic deployment",
            "tune_candidate": min(
                eligible, key=lambda r: (r["tune_tool_macro_mean_pinball_ms"], r["algorithm"])
            )["algorithm"]
            if eligible
            else None,
            "all_five_fixed_models_will_be_tested": True,
            "protocol_sha256": digest(PLAN / "protocol.json"),
            "config_sha256": digest(PLAN / "model_config.json"),
            "data_manifest_sha256": digest(DATA / "manifest.json"),
            "models": records,
        }
        (self.output / "frozen_before_test.json").write_text(json.dumps(frozen, indent=2) + "\n")

    def execute(self):
        try:
            for name, jobs, parallelism in commands(self.output):
                if name == "calibration":
                    self.freeze()
                self.stage(name, jobs, parallelism)
            self.state["status"] = "complete"
            self.state["stage"] = "complete"
            self.state["exit_code"] = 0
        except BaseException as exc:
            self.cancel_children()
            self.state.update(
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                exit_code=1,
                active=[],
            )
            raise
        finally:
            self.state["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.save()
            (self.output / "exit_code").write_text(str(self.state.get("exit_code", 1)) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually fit/calibrate/evaluate; omitted means print commands only",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != BASE or not output.name:
        parser.error("output must be a direct child of " + str(BASE))
    if not args.execute:
        import shlex

        for stage, jobs, parallelism in commands(output):
            print(f"# {stage}: max processes={parallelism}")
            for _, command in jobs:
                print(shlex.join(command))
        return
    os.environ.update(
        PYTHONPATH=str(PLAN / "code"),
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="8",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
    )
    lock = Path("/data1/ql_flowpilot_predictor/predictor_27b_training.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another 27B predictor experiment is active")
    checked = check(deep=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    snapshot = output / "deployment_snapshot"
    shutil.copytree(PLAN, snapshot, ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache"))
    shutil.copy2(DATA / "manifest.json", output / "data_manifest.json")
    (output / "preflight.json").write_text(json.dumps(checked, indent=2) + "\n")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    Runner(output).execute()


if __name__ == "__main__":
    main()
