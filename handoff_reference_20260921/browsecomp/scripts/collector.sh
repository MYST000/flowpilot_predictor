#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_owner "${1:?internal script: owner directory required}"
exec > >(tee -a "$OWNER_DIR/collector.log") 2>&1
printf '%s\n' "$$" > "$OWNER_DIR/collector.wrapper.pid"
finish_collector() {
    local rc=$?
    trap - EXIT INT TERM
    printf '%s\n' "$rc" > "$OWNER_DIR/collector.exit"
    if [[ $STOP_SERVICE_ON_EXIT == 1 ]]; then
        bash "$SCRIPT_DIR/stop_owned_service.sh" "$OWNER_DIR" || true
    fi
    exit "$rc"
}
trap finish_collector EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# The adapter also takes its project-wide collector lock when collecting.
exec 9>"${TMPDIR:-/tmp}/flowpilot-browsecomp-launcher-${UID}.lock"
flock -n 9 || { echo 'Another BrowseComp launcher is collecting'; exit 1; }
for ((attempt=0; attempt<100; attempt++)); do
    [[ -f "$OWNER_DIR/collector.ready" ]] && break
    sleep 0.1
 done
owns_service || { echo 'Service owner identity mismatch'; exit 1; }
"$CONTROLLER_PYTHON" "$SCRIPT_DIR/launcher_state.py" health "$OWNER_DIR"
owns_service || { echo 'Service ownership changed while waiting'; exit 1; }
mapfile -d '' -t options < <(state_field options)
"$CONTROLLER_PYTHON" -m benchmark_adapters.browsecomp_collection collect \
    --config "$(state_field config)" --split "$(state_field split)" "${options[@]}"

