#!/usr/bin/env bash
# Internal functions. State is JSON, parsed without eval or shell sourcing.
set -euo pipefail
unset PYTHONPATH PYTHONHOME
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

load_owner() {
    OWNER_DIR=${1:?owner directory required}
    local rows row
    rows=$(mktemp)
    if ! python3 "$SCRIPT_DIR/launcher_state.py" env "$OWNER_DIR" > "$rows"; then
        rm -f -- "$rows"
        return 1
    fi
    while IFS= read -r -d '' row; do export "$row"; done < "$rows"
    rm -f -- "$rows"
    source "$SCRIPT_DIR/env.sh"
    OWNER_DIR=$("$CONTROLLER_PYTHON" "$SCRIPT_DIR/launcher_state.py" field "$OWNER_DIR" owner_dir)
    TOKEN=$(state_field token)
    SERVICE_SESSION=$(state_field service_session)
    COLLECTOR_SESSION=$(state_field collector_session)
}

state_field() { "$CONTROLLER_PYTHON" "$SCRIPT_DIR/launcher_state.py" field "$OWNER_DIR" "$1"; }

owns_service() {
    tmux has-session -t "=$SERVICE_SESSION" 2>/dev/null &&
    [[ $(tmux show-options -qv -t "=$SERVICE_SESSION" @browsecomp_token) == "$TOKEN" ]] &&
    [[ $(tmux show-options -qv -t "=$SERVICE_SESSION" @browsecomp_owner) == "$OWNER_DIR" ]]
}

