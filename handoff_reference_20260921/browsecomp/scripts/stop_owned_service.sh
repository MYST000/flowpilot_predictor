#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ ${1:-} == --help || $# != 1 ]]; then
    printf '%s\n' 'Usage: stop_owned_service.sh OWNER_DIRECTORY (printed by launch_two_tmux.sh)'
    exit 0
fi
source "$SCRIPT_DIR/common.sh"
load_owner "$1"
if ! tmux has-session -t "=$SERVICE_SESSION" 2>/dev/null; then
    echo 'Owned tmux session no longer exists; no process was signaled.'
    exit 0
fi
owns_service || { echo 'Refusing shutdown: service owner token/path mismatch.' >&2; exit 1; }
if [[ $(tmux display-message -p -t "=$SERVICE_SESSION:0.0" '#{pane_dead}') != 1 ]]; then
    pane_pid=$(tmux display-message -p -t "=$SERVICE_SESSION:0.0" '#{pane_pid}')
    "$CONTROLLER_PYTHON" "$SCRIPT_DIR/launcher_state.py" verify-pid "$OWNER_DIR" "$pane_pid"
    tmux send-keys -t "=$SERVICE_SESSION:0.0" C-c
fi
for ((attempt=0; attempt<60; attempt++)); do
    owns_service || { echo 'Owner session changed/disappeared; no further action.'; exit 0; }
    if [[ $(tmux display-message -p -t "=$SERVICE_SESSION:0.0" '#{pane_dead}') == 1 ]]; then
        printf '%s\n' 'Owned service pane exited. Logs, attempts, model hashes and caches are retained.'
        exit 0
    fi
    sleep 1
done
echo 'Owned service did not exit within 60 seconds. Inspect its tmux/log; no escalation to global or PID kill.' >&2
exit 1
