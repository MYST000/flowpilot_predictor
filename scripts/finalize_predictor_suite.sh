#!/usr/bin/env bash
# After tuning/freeze: calibrate and evaluate the five selected FIT models.
# No training and no hyperparameter selection occur here. Default: print only.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-dry-run}
MODEL_ROOT=${2:?Usage: finalize_predictor_suite.sh dry-run|full MODEL_ROOT OUTPUT_ROOT [DATA] [--allow-test]}
OUT=${3:?OUTPUT_ROOT is required}
DATA=${4:-runs/predictor_prepared/v1}
ALLOW_TEST=${5:-}
PREDICTOR_PYTHON=${PREDICTOR_PYTHON:-$PWD/.venv-predictor/bin/python}
case "$MODE" in dry-run|full) ;; *) echo 'Mode must be dry-run or full' >&2; exit 2;; esac
[[ "$MODE" != full || "$ALLOW_TEST" == --allow-test ]] || { echo 'Full final evaluation requires --allow-test after freezing the protocol' >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
[[ -x "$PREDICTOR_PYTHON" && -f "$DATA/manifest.json" ]] || { echo 'Missing Python environment or prepared data' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Validate all selections before creating output or touching evaluation labels.
for algorithm in empirical ewma cluster qrf lightgbm; do
    [[ -f "$MODEL_ROOT/$algorithm/model.joblib" && -f "$MODEL_ROOT/$algorithm/manifest.json" ]] || { echo "Missing selection: $algorithm" >&2; exit 2; }
done
if [[ "$MODE" == full ]]; then
    "$PREDICTOR_PYTHON" - "$MODEL_ROOT" "$DATA" <<'PY'
import sys
from pathlib import Path
from predictor.cli import load_artifact
for algorithm in ('empirical','ewma','cluster','qrf','lightgbm'):
    a = load_artifact(Path(sys.argv[1])/algorithm/'model.joblib', sys.argv[2])
    if a['smoke'] or a['calibrated'] or a['model'].name != algorithm:
        raise SystemExit(f'{algorithm}: final pipeline requires a full uncalibrated fit model')
PY
    mkdir -p "$OUT/logs"
    trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit_code"' EXIT
fi
run_step() {
    local label=$1
    shift
    if [[ "$MODE" == dry-run ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        printf 'Running %s\n' "$label"
        "$@" >"$OUT/logs/$label.log" 2>&1
        printf '%s: OK\n' "$label"
    fi
}
for algorithm in empirical ewma cluster qrf lightgbm; do
    model="$MODEL_ROOT/$algorithm/model.joblib"
    calibrated="$OUT/${algorithm}_calibrated"
    run_step "${algorithm}_calibrate" "$PREDICTOR_PYTHON" -u -m predictor.cli calibrate \
        --data "$DATA" --model "$model" --output "$calibrated"
    run_step "${algorithm}_test_raw" "$PREDICTOR_PYTHON" -u -m predictor.cli evaluate \
        --data "$DATA" --model "$model" --split test --allow-test --mode frozen --output "$OUT/${algorithm}_test_raw"
    run_step "${algorithm}_test_calibrated" "$PREDICTOR_PYTHON" -u -m predictor.cli evaluate \
        --data "$DATA" --model "$calibrated/model.joblib" --split test --allow-test --mode frozen --output "$OUT/${algorithm}_test_calibrated"
done
run_step summary "$PREDICTOR_PYTHON" scripts/summarize_time_predictors.py --root "$OUT"
