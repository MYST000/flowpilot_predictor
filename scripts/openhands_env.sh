# Source this file before OpenHands development and local smoke commands.
export FLOWPILOT_ROOT=/home/qiulin/flowpilot_predictor
export OPENHANDS_REPO="$FLOWPILOT_ROOT/repos/Openhands-software-agent-sdk"
export PATH="$OPENHANDS_REPO/.venv/bin:$FLOWPILOT_ROOT/.venv-data/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$FLOWPILOT_ROOT/.python"
export UV_CACHE_DIR="$FLOWPILOT_ROOT/cache/uv"
export UV_FROZEN=1
# The adapter is an independent editable package, outside the uv workspace.
export UV_NO_SYNC=1
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
