# Copy to env.local.sh, edit, then source it explicitly before scripts/env.sh.
# The handoff root is inferred from scripts/env.sh and can relocate.
export MODEL_PATH="/data/HF_MODELS/Qwen3.5-9B"
# Select ONE idle card after inspecting nvidia-smi. No implicit GPU 0 fallback.
export CUDA_VISIBLE_DEVICES="0"
# Optional overrides; defaults use .venv-controller and .envs/vllm in this package.
# export CONTROLLER_PYTHON="$HOME/flowpilot_predictor/handoff_browsecomp_4090_20260920/.venv-controller/bin/python"
# export VLLM_PYTHON="$HOME/flowpilot_predictor/handoff_browsecomp_4090_20260920/.envs/vllm/bin/python"
export STOP_SERVICE_ON_EXIT=1
export HEALTH_TIMEOUT_S=1800

