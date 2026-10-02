#!/usr/bin/env bash
# Preparation checks/dry-run are read-only. Only 'start' launches model fitting.
set -euo pipefail
umask 077
PLAN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/root/flowpilot_predictor/.venv-predictor/bin/python
SOCKET=flowpilot-predictor-27b
MODE=${1:-dry-run}
RUN_ID=${2:-native27b_1077_v1}
[[ "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
OUT="/data1/ql_flowpilot_predictor/predictor_experiments/$RUN_ID"
LOGS=/data1/ql_flowpilot_predictor/logs/predictor_native27b_1077_v1
export PYTHONPATH="$PLAN/code" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
case "$MODE" in
    check) exec "$PYTHON" -B "$PLAN/preflight.py" --deep ;;
    dry-run)
        "$PYTHON" -B "$PLAN/preflight.py"
        exec "$PYTHON" -B "$PLAN/workflow.py" --output "$OUT"
        ;;
    start)
        [[ ! -e "$OUT" ]] || { echo "Existing experiment: $OUT; refusing overwrite or duplicate training." >&2; exit 1; }
        "$PYTHON" -B "$PLAN/preflight.py" >/dev/null
        if tmux -L "$SOCKET" has-session -t "$RUN_ID" 2>/dev/null; then
            echo "Session already exists: $RUN_ID" >&2; exit 1
        fi
        mkdir -p "$LOGS"
        printf -v command '%q ' "$PYTHON" -B -u "$PLAN/workflow.py" --execute --output "$OUT"
        printf -v command 'exec %s >>%q 2>&1' "$command" "$LOGS/$RUN_ID.workflow.log"
        tmux -L "$SOCKET" new-session -d -s "$RUN_ID" -c "$PLAN/code"
        tmux -L "$SOCKET" set-option -t "$RUN_ID" remain-on-exit on
        tmux -L "$SOCKET" respawn-pane -k -t "$RUN_ID":0.0 "$command"
        echo "Started server tmux: $SOCKET / $RUN_ID"
        echo "Output: $OUT"
        echo "Log: $LOGS/$RUN_ID.workflow.log"
        ;;
    status)
        tmux -L "$SOCKET" list-panes -t "$RUN_ID" -F '#{session_name} dead=#{pane_dead} exit=#{pane_dead_status} pid=#{pane_pid}' 2>/dev/null || true
        if [[ -f "$OUT/status.json" ]]; then cat "$OUT/status.json"; else echo 'No training status file; inspect workflow log if a session exists.'; fi
        ;;
    stop)
        if tmux -L "$SOCKET" has-session -t "$RUN_ID" 2>/dev/null; then
            if [[ "$(tmux -L "$SOCKET" display-message -p -t "$RUN_ID":0.0 '#{pane_dead}')" == 0 ]]; then
                tmux -L "$SOCKET" send-keys -t "$RUN_ID":0.0 C-c
                echo 'Interrupt sent to this workflow; it will terminate only its own child process groups.'
            else
                echo 'Workflow already exited.'
            fi
        else
            echo 'No session for this run.'
        fi
        ;;
    *) echo 'Usage: run.sh check|dry-run|start|status|stop [run_id]' >&2; exit 2 ;;
esac
