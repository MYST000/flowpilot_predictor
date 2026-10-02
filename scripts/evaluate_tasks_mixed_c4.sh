#!/usr/bin/env bash
# Full local task evaluation using the already served Qwen3.5-9B as BC judge.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
selection="${1:-all}"
case "$selection" in
  all|fit|tune|calibration|test) ;;
  --help|-h)
    echo "Usage: bash scripts/evaluate_tasks_mixed_c4.sh [all|fit|tune|calibration|test]"
    echo "Hotpot official scorer + LCB tests + local Qwen3.5-9B BrowseComp judge."
    echo "Uses the existing localhost:8100 service; downloads no model."
    exit 0 ;;
  *) echo "Invalid split: $selection" >&2; exit 2 ;;
esac
cd "$ROOT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
PYTHON="$ROOT/.venv-vllm/bin/python"
"$PYTHON" scripts/evaluate_browsecomp_local.py --split "$selection" --check
bash scripts/evaluate_mixed_c4.sh "$selection"
"$PYTHON" -u scripts/evaluate_browsecomp_local.py --split "$selection"
echo "Results: $ROOT/runs/evaluations/mixed_c4_v1_qwen35_9b_json_v2/<split>/task_evaluation_summary.json"
