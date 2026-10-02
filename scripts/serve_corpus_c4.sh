#!/usr/bin/env bash
set -euo pipefail
FLOWPILOT_ROOT="${FLOWPILOT_ROOT:-/home/qiulin/flowpilot_predictor}"
export FLOWPILOT_ROOT
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LITELLM_LOCAL_MODEL_COST_MAP=True CUDA_VISIBLE_DEVICES=""
export JAVA_HOME="$FLOWPILOT_ROOT/runtime/browsecomp-native/java21/jdk-21.0.12.1+1-jre"
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xms512m -Xmx4g -XX:ActiveProcessorCount=4}"
export PATH="$JAVA_HOME/bin:$PATH"
export HF_HUB_CACHE="$FLOWPILOT_ROOT/runtime/browsecomp-native/hf-cache"
case "${1:-}" in
  browsecomp)
    exec "$FLOWPILOT_ROOT/.venv-retrieval/bin/python" "$script_dir/serve_browsecomp_timed.py" --port 8123
    ;;
  hotpot)
    exec "$FLOWPILOT_ROOT/repos/Openhands-software-agent-sdk/.venv/bin/python" "$script_dir/hotpot_rpc.py" --config "$FLOWPILOT_ROOT/configs/c4/hotpot.toml" --port 8124
    ;;
  *) echo "Usage: $0 {hotpot|browsecomp}" >&2; exit 2 ;;
esac
