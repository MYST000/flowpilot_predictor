#!/usr/bin/env bash
# Run five independent CPU processes; default is dry-run, full training is explicit.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-dry-run}
DATA=${2:-runs/predictor_prepared/v1}
OUT=${3:-runs/predictor_experiments/$(date -u +%Y%m%dT%H%M%SZ)}
CONFIG=${4:-configs/predictor/default.json}
PREDICTOR_PYTHON=${PREDICTOR_PYTHON:-$PWD/.venv-predictor/bin/python}
case "$MODE" in dry-run|smoke|full) ;; *) echo 'Usage: run_predictor_suite.sh {dry-run|smoke|full} [data] [output] [config]' >&2; exit 2;; esac
[[ -x "$PREDICTOR_PYTHON" ]] || { echo "Missing interpreter: $PREDICTOR_PYTHON" >&2; exit 2; }
[[ -f "$DATA/manifest.json" && -f "$CONFIG" ]] || { echo 'Prepare data and config first' >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=""
# Cap native BLAS pools; model-specific forest/LightGBM parallelism is in config.
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
extra=()
[[ "$MODE" != smoke ]] || extra=(--smoke)
if [[ "$MODE" == dry-run ]]; then
    for algorithm in empirical ewma cluster qrf lightgbm; do
        printf '%q ' "$PREDICTOR_PYTHON" -m predictor.cli train --algorithm "$algorithm" --data "$DATA" --output "$OUT/$algorithm" --config "$CONFIG"
        printf '\n'
    done
    exit 0
fi
mkdir -p "$OUT/logs"
pids=(); names=()
cleanup() {
    for pid in "${pids[@]}"; do if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi; done
}
trap 'cleanup; exit 130' INT TERM
for algorithm in empirical ewma cluster qrf lightgbm; do
    "$PREDICTOR_PYTHON" -u -m predictor.cli train --algorithm "$algorithm" --data "$DATA" --output "$OUT/$algorithm" --config "$CONFIG" "${extra[@]}" >"$OUT/logs/$algorithm.log" 2>&1 &
    pids+=("$!"); names+=("$algorithm")
done
status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        printf '%s: OK\n' "${names[$i]}"
    else
        rc=$?; printf '%s: FAILED (%s); see %s/logs/%s.log\n' "${names[$i]}" "$rc" "$OUT" "${names[$i]}" >&2; status=1
    fi
    pids[$i]=""
done
# Extra frozen-vs-online comparison for EWMA, reset from the same fit artifact.
if [[ "$status" == 0 ]]; then
    "$PREDICTOR_PYTHON" -u -m predictor.cli evaluate --data "$DATA" --model "$OUT/ewma/model.joblib" --output "$OUT/ewma_online" --mode online --split tune "${extra[@]}" >"$OUT/logs/ewma_online.log" 2>&1 || status=1
fi
if [[ "$status" == 0 ]]; then
    "$PREDICTOR_PYTHON" scripts/summarize_time_predictors.py --root "$OUT" || status=1
fi
printf '%s\n' "$status" > "$OUT/exit_code"
exit "$status"
