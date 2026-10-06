#!/usr/bin/env bash
set -euo pipefail
umask 077
PLAN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/root/flowpilot_predictor/.venv-predictor/bin/python
OUT=/data/ql_flowpilot_predictor/predictor_experiments/native27b_calibration_online_v1
SOCKET=flowpilot-predictor-calibration-online
SESSION=native27b_calibration_online_v1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PLAN/../native27b_rtt_1077_20261001/code"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
case "${1:-status}" in
    check)
        TEST_TMP="$(mktemp -d /tmp/native27b-predictor-calibration-test.XXXXXX)"
        trap 'rm -rf -- "$TEST_TMP"' EXIT
        "$PYTHON" -B -m pytest -q -p no:cacheprovider --basetemp "$TEST_TMP" "$PLAN/test_experiment.py"
        ;;
    start)
        [[ ! -e "$OUT" ]] || { echo "Existing experiment: $OUT; refusing overwrite." >&2; exit 1; }
        [[ ! -e "$OUT.workflow.log" ]] || { echo 'Existing workflow log; inspect before retry.' >&2; exit 1; }
        "$PYTHON" -B "$PLAN/experiment.py" >/dev/null
        mkdir -p "$(dirname "$OUT")"
        printf -v command '%q ' "$PYTHON" -B -u "$PLAN/experiment.py" --execute
        printf -v command 'exec %s >%q 2>&1' "$command" "$OUT.workflow.log"
        tmux -L "$SOCKET" new-session -d -s "$SESSION" -c "$PLAN" "$command"
        tmux -L "$SOCKET" set-option -t "$SESSION" remain-on-exit off
        echo "Started CPU experiment: $SOCKET / $SESSION"
        echo "Output: $OUT"
        ;;
    status)
        tmux -L "$SOCKET" list-panes -t "$SESSION" -F '#{session_name} dead=#{pane_dead} pid=#{pane_pid}' 2>/dev/null || true
        [[ ! -f "$OUT/status.json" ]] || cat "$OUT/status.json"
        ;;
    *) echo 'Usage: run.sh check|start|status' >&2; exit 2 ;;
esac
