#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-dry-run}
SESSION=${2:-predictor-five}
DATA=${3:-runs/predictor_prepared/v1}
OUT=${4:-runs/predictor_experiments/$(date -u +%Y%m%dT%H%M%SZ)}
CONFIG=${5:-configs/predictor/default.json}
[[ "$SESSION" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid tmux session name' >&2; exit 2; }
if [[ "$MODE" == dry-run ]]; then
    exec bash scripts/run_predictor_suite.sh dry-run "$DATA" "$OUT" "$CONFIG"
fi
[[ "$MODE" == smoke || "$MODE" == full ]] || { echo 'mode must be dry-run, smoke or full' >&2; exit 2; }
command -v tmux >/dev/null
# Validate before creating a detached session.
bash scripts/run_predictor_suite.sh dry-run "$DATA" "$OUT" "$CONFIG" >/dev/null
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session already exists: $SESSION" >&2; exit 2
fi
printf -v run_command '%q ' env "PREDICTOR_PYTHON=${PREDICTOR_PYTHON:-$PWD/.venv-predictor/bin/python}" bash "$PWD/scripts/run_predictor_suite.sh" "$MODE" "$DATA" "$OUT" "$CONFIG"
# Keep the pane after completion so failures and final status remain inspectable.
tmux new-session -d -s "$SESSION" -c "$PWD"
tmux set-option -t "$SESSION" remain-on-exit on
tmux send-keys -t "$SESSION" -l "exec $run_command"
tmux send-keys -t "$SESSION" Enter
printf 'Started %s (%s). Attach: tmux attach -t %s\nOutput: %s\n' "$SESSION" "$MODE" "$SESSION" "$OUT"
