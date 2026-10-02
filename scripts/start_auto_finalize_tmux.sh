#!/usr/bin/env bash
# Usage: start_auto_finalize_tmux.sh SESSION --run-dir RUN [--output OUT] --parameters-frozen --allow-test
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION=${1:?First argument must be a tmux session name}
shift
[[ "$SESSION" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid tmux session name' >&2; exit 2; }
PREDICTOR_PYTHON=${PREDICTOR_PYTHON:-$PWD/.venv-predictor/bin/python}
command -v tmux >/dev/null
[[ -x "$PREDICTOR_PYTHON" ]] || { echo 'Missing predictor Python environment' >&2; exit 2; }
"$PREDICTOR_PYTHON" scripts/auto_finalize_predictors.py --validate-only "$@" >/dev/null
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session already exists: $SESSION" >&2; exit 2
fi
printf -v run_command '%q ' "$PREDICTOR_PYTHON" -u "$PWD/scripts/auto_finalize_predictors.py" "$@"
tmux new-session -d -s "$SESSION" -c "$PWD"
tmux set-option -t "$SESSION" remain-on-exit on
tmux send-keys -t "$SESSION" -l "exec $run_command"
tmux send-keys -t "$SESSION" Enter
printf 'Started automatic finalization watcher: %s\nAttach: tmux attach -t %s\n' "$SESSION" "$SESSION"
