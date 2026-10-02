#!/usr/bin/env bash
set -euo pipefail
FLOWPILOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$FLOWPILOT_ROOT/scripts/openhands_env.sh"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export LITELLM_LOCAL_MODEL_COST_MAP=True HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$OPENHANDS_REPO/.venv/bin/python" "$FLOWPILOT_ROOT/scripts/mixed_c4_control.py" "$@"
