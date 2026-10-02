"""Score frozen BrowseComp answers using the existing local 9B API.

Uses the pinned upstream judging criteria with constrained JSON output,
not the official Qwen3-32B leaderboard protocol. No Agent is run here.
"""

import argparse
import fcntl
import hashlib
import json
import os
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("fit", "tune", "calibration", "test")
REPO = ROOT / "repos/BrowseComp-Plus"
COMMIT = "046949032b0328319cc9a02663a759ec601d9402"
SOURCE = "scripts_evaluation/evaluate_run.py"
OUTPUT = ROOT / "runs/evaluations/mixed_c4_v1_qwen35_9b_json_v2"
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "extracted_final_answer": {"type": "string"},
        "correct_answer": {"type": "string"},
        "reasoning": {"type": "string"},
        "correct": {"type": "string", "enum": ["yes", "no"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 100},
    },
    "required": ["extracted_final_answer", "correct_answer", "reasoning", "correct", "confidence"],
    "additionalProperties": False,
}
JSON_INSTRUCTION = (
    "\n\nOutput format override only: preserve all judging criteria above, but return "
    "one JSON object with exactly these required keys: extracted_final_answer (string), "
    "correct_answer (string), reasoning (string), correct (the string yes or no), "
    "confidence (number from 0 to 100). Include correct in every response. "
    "Do not use Markdown or text outside the JSON object."
)


def parse_json_judgement(text):
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != set(JUDGE_SCHEMA["required"]):
        raise ValueError("Judge JSON has missing or unexpected fields")
    if any(not isinstance(value[k], str) for k in ("extracted_final_answer", "correct_answer", "reasoning")):
        raise ValueError("Judge answer and reasoning must be strings")
    if value["correct"] not in ("yes", "no"):
        raise ValueError("Judge correct must be yes or no")
    if type(value["confidence"]) not in (int, float) or not 0 <= value["confidence"] <= 100:
        raise ValueError("Invalid judge confidence")
    return {**value, "correct": value["correct"] == "yes", "parse_error": False}



def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def official_helpers():
    raw = (REPO / SOURCE).read_bytes()
    original = subprocess.check_output(["git", "-C", str(REPO), "show", f"{COMMIT}:{SOURCE}"])
    if raw != original:
        raise ValueError("BrowseComp official scorer differs from the pinned commit")
    # run_name is not __main__: importing definitions never constructs a vLLM engine.
    helpers = runpy.run_path(str(REPO / SOURCE), run_name="flowpilot_judge_helpers")
    return helpers, digest(raw)


def load_jobs(split):
    campaign = ROOT / "runs/campaigns" / ("mixed_c4_v1_" + split)
    manifest = read(campaign / "campaign.json")
    if read(campaign / "collection_summary.json")["status"] != "collected":
        raise ValueError(f"{split}: collection is not finished")
    marker = read(campaign / "collection_started.json")
    if marker["campaign_sha256"] != digest((campaign / "campaign.json").read_bytes()):
        raise ValueError(f"{split}: campaign changed since collection")
    datasets, jobs = {}, []
    for job in manifest["jobs"]:
        if job["adapter"] != "browsecomp":
            continue
        raw = Path(job["input_path"]).read_bytes()
        if digest(raw) != job["input_sha256"]:
            raise ValueError(f"Frozen input changed: {job['job_id']}")
        payload = json.loads(raw)
        dataset = payload["config"]["dataset"]
        identity = (dataset["path"], dataset["sha256"])
        if identity not in datasets:
            raw_gold = Path(dataset["path"]).read_bytes()
            if digest(raw_gold) != dataset["sha256"]:
                raise ValueError("BrowseComp ground truth checksum mismatch")
            rows = [json.loads(line) for line in raw_gold.splitlines() if line.strip()]
            datasets[identity] = {str(row["query_id"]): row for row in rows}
        task_id = str(payload["task"]["task_id"])
        gold = datasets[identity][task_id]
        submission_path = Path(job["attempt_dir"]) / "submission.json"
        submission = read(submission_path) if submission_path.exists() else {
            "query_id": task_id, "status": "failed", "result": []
        }
        if str(submission.get("query_id")) != task_id:
            raise ValueError(f"Submission ID mismatch: {job['job_id']}")
        parts = submission.get("result", [])
        answer = parts[-1].get("output", "") if parts and parts[-1].get("type") == "output_text" else ""
        jobs.append({"job_id": job["job_id"], "query_id": task_id,
                     "question": gold["query"], "gold": gold["answer"],
                     "response": answer, "submission": submission,
                     "input_sha256": job["input_sha256"]})
    if not jobs:
        raise ValueError(f"{split}: no BrowseComp jobs")
    return jobs


def judge_one(job, protocol, helpers, client, previous=None):
    identity = fingerprint({"job": job, "protocol": protocol})
    if previous is not None:
        if previous["identity"] != identity:
            raise ValueError("Judge/input changed; use a new --output directory")
        if previous["status"] in ("judged", "no_answer"):
            return previous
    record = {"job_id": job["job_id"], "query_id": job["query_id"],
              "identity": identity, "protocol": protocol,
              "response": job["response"], "correct_answer": job["gold"],
              "correct": None, "attempts": list((previous or {}).get("attempts", []))}
    if job["submission"].get("status") != "completed" or not job["response"].strip():
        record.update(status="no_answer", correct=False)
        return record
    prompt = helpers["create_judge_prompt"](job["question"], job["response"], job["gold"]) + JSON_INSTRUCTION
    attempt = {"at": datetime.now(timezone.utc).isoformat(), "prompt": prompt}
    try:
        result = client.chat.completions.create(
            model=protocol["model"], messages=[{"role": "user", "content": prompt}],
            temperature=0.7, top_p=0.8, max_tokens=4096, seed=20260926,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
            response_format={"type": "json_schema", "json_schema": {
                "name": "browsecomp_judgement", "strict": True, "schema": JUDGE_SCHEMA}},
        )
        choice = result.choices[0]
        text = choice.message.content or ""
        attempt.update(raw_response=text, finish_reason=choice.finish_reason,
                       usage=result.usage.model_dump() if result.usage else None)
        parsed = parse_json_judgement(text)
        attempt["parsed"] = parsed
        if choice.finish_reason != "stop" or parsed.get("parse_error") or type(parsed.get("correct")) is not bool:
            record.update(status="judge_error", error="Truncated or unparseable judge response")
        else:
            record.update(status="judged", correct=parsed["correct"])
    except Exception as exc:
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        record.update(status="judge_error", error=attempt["error"])
    record["attempts"].append(attempt)
    return record


def summarize(records, protocol):
    total = len(records)
    errors = sum(row["status"] == "judge_error" for row in records)
    correct = sum(row.get("correct") is True for row in records)
    return {"status": "needs_review" if errors else "scored",
            "protocol": protocol, "total": total, "correct": correct,
            "judged": sum(row["status"] == "judged" for row in records),
            "no_answer": sum(row["status"] == "no_answer" for row in records),
            "judge_errors": errors,
            "accuracy": correct / total if total and not errors else None,
            "accuracy_lower_bound": correct / total if total else None,
            "note": "Local Qwen3.5-9B judge; not official Qwen3-32B results. Judge errors are unresolved, not wrong answers."}


def combined_summary(split, browsecomp, destination):
    source = ROOT / "runs/campaigns" / ("mixed_c4_v1_" + split) / "evaluation_summary.json"
    combined = {"split": split, "browsecomp": browsecomp}
    if source.exists():
        rows = read(source)
        hp = [r["evaluation"] for r in rows if r["job_id"].startswith("hotpot--")]
        lcb = [r["evaluation"] for r in rows if r["job_id"].startswith("livecodebench--")]
        hp_ok = bool(hp) and all(r.get("evaluation_status") == "scored" for r in hp)
        lcb_ok = bool(lcb) and all(r.get("status") in ("evaluated", "no_submission") for r in lcb)
        combined["hotpot"] = {"status": "scored" if hp_ok else "needs_review", "total": len(hp),
            "metrics": {k: sum(r["metrics"][k] for r in hp) / len(hp)
                        for k in hp[0]["metrics"]} if hp_ok else None}
        combined["livecodebench"] = {"status": "scored" if lcb_ok else "needs_review", "total": len(lcb),
            "passed": sum(r.get("passed") is True for r in lcb),
            "pass_rate": sum(r.get("passed") is True for r in lcb) / len(lcb) if lcb_ok else None}
    combined["status"] = "scored" if all(combined.get(k, {}).get("status") == "scored"
        for k in ("hotpot", "livecodebench", "browsecomp")) else "incomplete"
    save(destination / "task_evaluation_summary.json", combined)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", *SPLITS), default="all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="Validate inputs and model endpoint; no scoring")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    selected = SPLITS if args.split == "all" else (args.split,)
    helpers, source_hash = official_helpers()
    jobs = {s: load_jobs(s) for s in selected}
    import httpx
    from openai import OpenAI

    with httpx.Client(trust_env=False, timeout=180) as http:
        with OpenAI(api_key="EMPTY", base_url="http://127.0.0.1:8100/v1",
                    http_client=http, max_retries=0) as client:
            models = client.models.list().data
            if "qwen3.5-9b" not in {m.id for m in models}:
                raise ValueError("Start the existing qwen3.5-9b service on localhost:8100 first")
            config = ROOT / "models/Qwen3.5-9B/config.json"
            protocol = {"name": "browsecomp_official_criteria_local_9b_json_v2", "model": "qwen3.5-9b",
                        "base_url": "http://127.0.0.1:8100/v1", "official_commit": COMMIT,
                        "official_source_sha256": source_hash,
                        "model_config_sha256": digest(config.read_bytes()),
                        "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                        "max_tokens": 4096, "enable_thinking": False, "seed": 20260926,
                        "response_schema": JUDGE_SCHEMA, "format_instruction": JSON_INSTRUCTION}
            print(json.dumps({"mode": "check" if args.check else "run", "tasks": {s: len(v) for s, v in jobs.items()},
                              "judge": protocol["model"], "output": str(args.output)}, ensure_ascii=False), flush=True)
            if args.check:
                return
            os.umask(0o077)
            args.output.mkdir(parents=True, exist_ok=True)
            with (args.output / ".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                errors = 0
                for split, items in jobs.items():
                    records = []
                    destination = args.output / split
                    def score_job(job):
                        path = destination / "results" / (job["job_id"] + ".json")
                        previous = read(path) if path.exists() else None
                        result = judge_one(job, protocol, helpers, client, previous)
                        if result is not previous:
                            save(path, result)
                        return result
                    with ThreadPoolExecutor(max_workers=args.workers) as pool:
                        for index, result in enumerate(pool.map(score_job, items), 1):
                            records.append(result)
                            print(f"{split} {index}/{len(items)} {result['job_id']} {result['status']}", flush=True)
                    summary = summarize(records, protocol)
                    save(destination / "summary.json", summary)
                    combined_summary(split, summary, destination)
                    errors += summary["judge_errors"]
                    print(json.dumps({"split": split, "status": summary["status"], "accuracy": summary["accuracy"],
                                      "judge_errors": summary["judge_errors"]}), flush=True)
                if errors:
                    raise SystemExit("Judge errors remain. Rerun the same command to retry only failed requests; saved successful judgements are reused.")


if __name__ == "__main__":
    main()
