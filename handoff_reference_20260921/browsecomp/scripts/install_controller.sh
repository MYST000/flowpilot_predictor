#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source "$SCRIPT_DIR/env.sh"
if [[ ${1:-} == --help ]]; then
    printf '%s\n' 'Usage: bash scripts/install_controller.sh' 'Requires restored .envs/controller-base; clones fixed SDK into sdk and installs editables only into .venv-controller.'
    exit 0
fi
[[ $# == 0 ]] || { echo 'Unexpected arguments' >&2; exit 2; }
SDK_COMMIT=a6db5dcba26a3acfaeac58c8ba5195433a0e223d
SDK_PATH="$HANDOFF_ROOT/sdk"
BASE_PYTHON="$HANDOFF_ROOT/.envs/controller-base/bin/python"
VENV_PATH="$HANDOFF_ROOT/.venv-controller"
[[ -x "$BASE_PYTHON" ]] || { echo 'Restore controller-base conda-pack archive first; see requirements/README.md.' >&2; exit 1; }
[[ ! -L "$VENV_PATH" ]] || { echo 'Refusing symlink controller environment' >&2; exit 1; }
"$BASE_PYTHON" -c 'import sys; assert sys.version_info[:3] == (3,12,14), sys.version'
if [[ ! -d "$SDK_PATH" ]]; then
    git clone --no-checkout https://github.com/MYST000/Openhands-software-agent-sdk.git "$SDK_PATH"
    git -C "$SDK_PATH" checkout --detach "$SDK_COMMIT"
fi
[[ $(git -C "$SDK_PATH" rev-parse HEAD) == "$SDK_COMMIT" ]] || { echo 'SDK commit mismatch; refusing to modify existing SDK checkout.' >&2; exit 1; }
[[ -z $(git -C "$SDK_PATH" status --porcelain --untracked-files=no) ]] || { echo 'SDK tracked source has modifications.' >&2; exit 1; }
"$BASE_PYTHON" -m venv --system-site-packages "$VENV_PATH"
"$VENV_PATH/bin/python" - "$VENV_PATH" <<'PY'
from pathlib import Path
import sys
assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve()
assert sys.prefix != sys.base_prefix
PY
"$VENV_PATH/bin/python" -m pip install --no-deps --no-build-isolation \
    -e "$SDK_PATH/openhands-sdk" -e "$SDK_PATH/openhands-tools" \
    -e "$SDK_PATH/openhands-workspace" -e "$HANDOFF_ROOT/adapter"
"$VENV_PATH/bin/python" - "$HANDOFF_ROOT" <<'PY'
from importlib import import_module, metadata
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
for package, module, source in [
    ('openhands-sdk', 'openhands.sdk', root/'sdk'/'openhands-sdk'),
    ('openhands-tools', 'openhands.tools', root/'sdk'/'openhands-tools'),
    ('openhands-workspace', 'openhands.workspace', root/'sdk'/'openhands-workspace'),
    ('flowpilot-benchmark-adapters', 'benchmark_adapters', root/'adapter'),
]:
    imported = import_module(module)
    assert Path(imported.__file__).resolve().is_relative_to(source), imported.__file__
    if package.startswith('openhands-'):
        assert metadata.version(package) == '1.31.1', (package, metadata.version(package))
    print(package, metadata.version(package), imported.__file__)
PY
printf 'Controller ready: %s/bin/python\n' "$VENV_PATH"
