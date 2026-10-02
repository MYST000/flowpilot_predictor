#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs/mixed_c4_v1/evaluation"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/evaluate_mixed_c4.sh [all|fit|tune|calibration|test]

The script evaluates completed trajectories sequentially. It never reruns Agent
tasks and never overwrites an existing evaluation. HotpotQA and LiveCodeBench
are scored locally. BrowseComp-Plus evaluation plans are prepared, but its
answer correctness remains pending until the separate Qwen3-32B judge is run.
EOF
}

selection="${1:-all}"
case "$selection" in
  all) splits=(fit tune calibration test) ;;
  fit|tune|calibration|test) splits=("$selection") ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

mkdir -p "$LOG_DIR"

verify_summary() {
  local split="$1" summary="$2" expected="$3" actual bad
  actual="$(jq 'length' "$summary")"
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: $split evaluation_summary has $actual/$expected records" >&2
    exit 1
  fi
  bad="$(jq '[.[] | select(
    ((.job_id | startswith("hotpot--")) and .evaluation.evaluation_status != "scored") or
    ((.job_id | startswith("browsecomp--")) and .evaluation.evaluation_status != "pending") or
    ((.job_id | startswith("livecodebench--")) and
      (.evaluation.status != "evaluated" and .evaluation.status != "no_submission"))
  )] | length' "$summary")"
  if [[ "$bad" != "0" ]]; then
    echo "ERROR: $split has $bad evaluator errors or unexpected statuses in $summary" >&2
    exit 1
  fi
  echo "$split: $actual/$expected evaluation records; HotpotQA and LiveCodeBench scored, BrowseComp-Plus judge pending"
}

for split in "${splits[@]}"; do
  run_root="$ROOT/runs/campaigns/mixed_c4_v1_${split}"
  collection_summary="$run_root/collection_summary.json"
  evaluation_summary="$run_root/evaluation_summary.json"
  log="$LOG_DIR/${split}.log"

  if [[ ! -f "$collection_summary" ]]; then
    echo "ERROR: $split has no collection_summary.json" >&2
    exit 1
  fi
  collection_status="$(jq -r '.status' "$collection_summary")"
  if [[ "$collection_status" != "collected" ]]; then
    echo "ERROR: $split collection status is $collection_status, not collected" >&2
    exit 1
  fi
  expected="$(jq '.records | length' "$collection_summary")"

  if [[ -f "$evaluation_summary" ]]; then
    verify_summary "$split" "$evaluation_summary" "$expected"
    echo "$split: already processed; skipping"
    continue
  fi

  partial="$(find "$run_root/tasks" -type d -path '*/attempt-001/evaluation' | wc -l)"
  if [[ "$partial" != "0" ]]; then
    echo "ERROR: $split has $partial partial evaluation directories but no evaluation_summary.json" >&2
    echo "Inspect them before recovery; the evaluator intentionally refuses to overwrite results." >&2
    exit 1
  fi

  echo "$split: evaluating $expected trajectories; log=$log"
  if ! bash "$ROOT/scripts/collect_mixed_c4.sh" evaluate --split "$split" >"$log" 2>&1; then
    echo "ERROR: $split evaluation failed; last log lines:" >&2
    tail -40 "$log" >&2
    exit 1
  fi

  if [[ ! -f "$evaluation_summary" ]]; then
    echo "ERROR: $split finished without evaluation_summary.json" >&2
    exit 1
  fi
  verify_summary "$split" "$evaluation_summary" "$expected"
done

echo "Local scoring and BrowseComp-Plus judge preparation complete for: ${splits[*]}"
echo "Logs: $LOG_DIR"
