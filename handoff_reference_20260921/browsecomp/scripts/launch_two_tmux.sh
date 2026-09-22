#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ ${1:-} == --help || $# -lt 2 ]]; then
    printf '%s\n' 'Usage: launch_two_tmux.sh CONFIG_JSON SPLIT [--resume] [--retry-infrastructure] [--limit N] [--release-test]'
    exit 0
fi
source "$SCRIPT_DIR/env.sh"
source "$SCRIPT_DIR/common.sh"
command -v tmux >/dev/null
command -v nvidia-smi >/dev/null
case "$2" in dev|fit|tune|calibration|test|historical_dev) ;; *) echo 'Invalid split' >&2; exit 2;; esac
OWNER_DIR=$("$CONTROLLER_PYTHON" "$SCRIPT_DIR/launcher_state.py" init "$@")
load_owner "$OWNER_DIR"
failed=1
cleanup_launch() {
    if [[ $failed == 1 ]]; then bash "$SCRIPT_DIR/stop_owned_service.sh" "$OWNER_DIR" || true; fi
}
trap cleanup_launch EXIT
tmux new-session -d -s "$SERVICE_SESSION" -e "BROWSECOMP_OWNER_TOKEN=$TOKEN" bash "$SCRIPT_DIR/service.sh" "$OWNER_DIR"
tmux set-option -t "=$SERVICE_SESSION" remain-on-exit on
tmux set-option -t "=$SERVICE_SESSION" @browsecomp_token "$TOKEN"
tmux set-option -t "=$SERVICE_SESSION" @browsecomp_owner "$OWNER_DIR"
touch "$OWNER_DIR/owner.ready"
tmux new-session -d -s "$COLLECTOR_SESSION" bash "$SCRIPT_DIR/collector.sh" "$OWNER_DIR"
tmux set-option -t "=$COLLECTOR_SESSION" remain-on-exit on
tmux set-option -t "=$COLLECTOR_SESSION" @browsecomp_token "$TOKEN"
tmux set-option -t "=$COLLECTOR_SESSION" @browsecomp_owner "$OWNER_DIR"
touch "$OWNER_DIR/collector.ready"
failed=0
printf 'Owner directory: %s\nService: tmux attach -t %s\nCollector: tmux attach -t %s\n' "$OWNER_DIR" "$SERVICE_SESSION" "$COLLECTOR_SESSION"
printf 'Stop safely: bash %q %q\n' "$SCRIPT_DIR/stop_owned_service.sh" "$OWNER_DIR"

