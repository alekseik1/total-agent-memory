#!/usr/bin/env bash
set -euo pipefail

[ "$(uname)" = Darwin ] || { echo 'This check requires macOS' >&2; exit 1; }
TAM_CHECK_REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TAM_CHECK_ROOT="$(mktemp -d /tmp/tam-native-macos-XXXXXX)"
export PIP_CACHE_DIR="$TAM_CHECK_ROOT/pip-cache"
TAM_CHECK_PYTHON="${TAM_CHECK_PYTHON:-/opt/homebrew/bin/python3.12}"
TAM_CHECK_LABEL="com.total-agent-memory.release-check.$$"
TAM_CHECK_DOMAIN="gui/$(id -u)"
TAM_CHECK_STARTED=0
cleanup() {
  result=$?
  if [ "$TAM_CHECK_STARTED" = 1 ]; then
    launchctl bootout "$TAM_CHECK_DOMAIN/$TAM_CHECK_LABEL" || result=1
  fi
  echo "Test directory: $TAM_CHECK_ROOT"
  exit "$result"
}
trap cleanup EXIT

"$TAM_CHECK_PYTHON" -m zipfile -e "$TAM_CHECK_REPO/docs/benchmarks/platform-v14-20260915/windows-candidate.zip" "$TAM_CHECK_ROOT/source"
"$TAM_CHECK_PYTHON" -m venv "$TAM_CHECK_ROOT/wheel-env"
"$TAM_CHECK_ROOT/wheel-env/bin/python" -m pip install "$TAM_CHECK_ROOT"/source/dist/*.whl pytest
export PATH="$TAM_CHECK_ROOT/wheel-env/bin:$PATH"
export FASTEMBED_CACHE_PATH="${FASTEMBED_CACHE_PATH:-/tmp/tam-v14-model-cache}"
export TAM_MEMORY_DIR="$TAM_CHECK_ROOT/memory"
export MEMORY_LLM_ENABLED=false MEMORY_MODE=fast MEMORY_EMBED_THREADS=1 PYTHONIOENCODING=utf-8
cd "$TAM_CHECK_ROOT/source"
python -m pip check
python tests/smoke/installed_runtime.py
python -m pytest tests/test_team_memory.py tests/test_team_lifecycle.py tests/test_store_windows_encoding.py -q --junitxml="$TAM_CHECK_ROOT/native-results.xml"

mkdir -p "$TAM_CHECK_ROOT/home"
env HOME="$TAM_CHECK_ROOT/home" bash install.sh --ide cursor
export PATH="$TAM_CHECK_ROOT/source/.venv/bin:$PATH"
.venv/bin/python tests/smoke/installed_runtime.py

export TAM_CHECK_ROOT TAM_CHECK_LABEL
.venv/bin/python - <<'PY'
import os
import plistlib
import socket
from pathlib import Path

root = Path(os.environ['TAM_CHECK_ROOT'])
source = root / 'source'
with (source / 'launchagents/com.total-agent-memory.dashboard.plist').open('rb') as f:
    job = plistlib.load(f)
replacements = {'__INSTALL_DIR__': str(source), '__MEMORY_DIR__': str(root / 'memory'), '__HOME__': str(root / 'home')}
def render(value):
    if isinstance(value, str):
        for marker, replacement in replacements.items():
            value = value.replace(marker, replacement)
        return value
    if isinstance(value, list):
        return [render(item) for item in value]
    if isinstance(value, dict):
        return {key: render(item) for key, item in value.items()}
    return value
job = render(job)
job['Label'] = os.environ['TAM_CHECK_LABEL']
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
job['EnvironmentVariables']['DASHBOARD_PORT'] = str(port)
job['EnvironmentVariables']['MEMORY_LLM_ENABLED'] = 'false'
(root / 'memory/logs').mkdir(parents=True, exist_ok=True)
(root / 'dashboard-port').write_text(str(port))
with (root / 'dashboard.plist').open('wb') as f:
    plistlib.dump(job, f)
PY
plutil -lint "$TAM_CHECK_ROOT/dashboard.plist"
launchctl bootstrap "$TAM_CHECK_DOMAIN" "$TAM_CHECK_ROOT/dashboard.plist"
TAM_CHECK_STARTED=1
DASHBOARD_PORT="$(cat "$TAM_CHECK_ROOT/dashboard-port")" .venv/bin/python "$TAM_CHECK_REPO/tests/smoke/check_dashboard.py"
echo 'Native macOS wheel, source installation and LaunchAgent checks passed'
