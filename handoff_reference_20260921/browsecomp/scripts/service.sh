#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
load_owner "${1:?internal script: owner directory required}"
exec > >(tee -a "$OWNER_DIR/service.log") 2>&1
printf '%s\n' "$$" > "$OWNER_DIR/service.wrapper.pid"
finish_service() { local rc=$?; printf '%s\n' "$rc" > "$OWNER_DIR/service.exit"; }
trap finish_service EXIT
for ((attempt=0; attempt<100; attempt++)); do
    [[ -f "$OWNER_DIR/owner.ready" ]] && break
    sleep 0.1
done
owns_service || { echo 'Service owner identity mismatch; refusing to start'; exit 1; }
export BROWSECOMP_OWNER_TOKEN="$TOKEN"
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
# Isolate only compilation caches; never redirect or remove model/corpus caches.
export VLLM_CACHE_ROOT="$OWNER_DIR/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$OWNER_DIR/cache/torchinductor"
mkdir -p -- "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
"$CONTROLLER_PYTHON" "$SCRIPT_DIR/launcher_state.py" manifest "$OWNER_DIR"
mapfile -d '' -t flags < <("$CONTROLLER_PYTHON" "$SCRIPT_DIR/launcher_state.py" flags "$OWNER_DIR")
"$CONTROLLER_PYTHON" "$SCRIPT_DIR/launcher_state.py" preflight "$OWNER_DIR"
printf 'Starting owned BF16 32K service, token=%s\n' "$TOKEN"
# Foreground child receives tmux C-c along with this shell; no PID-wide killing.
set +e
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server --model "$MODEL_PATH" \
    --served-model-name "$(state_field model_name)" --host 127.0.0.1 --port "$(state_field port)" "${flags[@]}"
status=$?
set -e
exit "$status"

