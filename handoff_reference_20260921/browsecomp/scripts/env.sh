#!/usr/bin/env bash
# Source after your trusted config/env.local.sh; no package is installed here.
HANDOFF_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
export HANDOFF_ROOT
export CONTROLLER_PYTHON="${CONTROLLER_PYTHON:-$HANDOFF_ROOT/.venv-controller/bin/python}"
export VLLM_PYTHON="${VLLM_PYTHON:-$HANDOFF_ROOT/.envs/vllm/bin/python}"
export MODEL_PATH="${MODEL_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"
export STOP_SERVICE_ON_EXIT="${STOP_SERVICE_ON_EXIT:-1}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OPENHANDS_SUPPRESS_BANNER=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export TOKENIZERS_PARALLELISM=false
# Remove inherited source injection; install_controller.sh creates local editables.
unset PYTHONPATH PYTHONHOME

