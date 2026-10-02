#!/usr/bin/env bash
set -euo pipefail
FLOWPILOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec ssh -F "$FLOWPILOT_ROOT/configs/ssh/whu4090.conf" -NT \
  -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18123:127.0.0.1:8123 \
  -L 127.0.0.1:18124:127.0.0.1:8124 WHU_4090_6
