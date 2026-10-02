#!/usr/bin/env bash
set -euo pipefail
FLOWPILOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FLOWPILOT_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/usr/local/cuda OMP_NUM_THREADS=8
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_CACHE_ROOT="$FLOWPILOT_ROOT/cache/vllm-c4"
export TORCHINDUCTOR_CACHE_DIR="$FLOWPILOT_ROOT/cache/torchinductor-c4"
export PATH="/usr/local/cuda/bin:$PATH"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
command=("$FLOWPILOT_ROOT/.venv-vllm/bin/python" -m vllm.entrypoints.openai.api_server
  --model "$FLOWPILOT_ROOT/models/Qwen3.5-9B" --served-model-name qwen3.5-9b
  --host 127.0.0.1 --port 8100 --dtype bfloat16 --tensor-parallel-size 4
  --max-model-len 262144 --gpu-memory-utilization 0.90
  --max-num-seqs 4 --max-num-batched-tokens 2048
  --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3 --enforce-eager)
if [[ "${1:-}" == "--print" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
if [[ $# -gt 0 ]]; then echo "Usage: $0 [--print]" >&2; exit 2; fi
IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#devices[@]} -ne 4 ]]; then echo "TP4 requires exactly four visible GPUs" >&2; exit 2; fi
exec "${command[@]}"
