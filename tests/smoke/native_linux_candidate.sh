#!/usr/bin/env bash
set -euo pipefail

TAM_CHECK_ROOT="$(mktemp -d /tmp/tam-native-linux-XXXXXX)"
export PIP_CACHE_DIR="$TAM_CHECK_ROOT/pip-cache"
unzip -q /candidate/windows-candidate.zip -d "$TAM_CHECK_ROOT/source"
python3 -m venv "$TAM_CHECK_ROOT/wheel-env"
"$TAM_CHECK_ROOT/wheel-env/bin/python" -m pip install "$TAM_CHECK_ROOT"/source/dist/*.whl pytest
export PATH="$TAM_CHECK_ROOT/wheel-env/bin:$PATH"
export FASTEMBED_CACHE_PATH="$TAM_CHECK_ROOT/models"
export TAM_MEMORY_DIR="$TAM_CHECK_ROOT/memory"
export MEMORY_LLM_ENABLED=false MEMORY_MODE=fast MEMORY_EMBED_THREADS=1 PYTHONIOENCODING=utf-8
cd "$TAM_CHECK_ROOT/source"
python -m pip check
python tests/smoke/installed_runtime.py
python -m pytest tests/test_team_memory.py tests/test_team_lifecycle.py tests/test_store_windows_encoding.py -q --junitxml="$TAM_CHECK_ROOT/native-results.xml"
bash install.sh --ide claude-code
export PATH="$TAM_CHECK_ROOT/source/.venv/bin:$PATH"
.venv/bin/python tests/smoke/installed_runtime.py
.venv/bin/python tests/smoke/check_dashboard.py
systemctl --user is-active claude-memory-dashboard.service
echo "Native Linux wheel, source installation and systemd checks passed: $TAM_CHECK_ROOT"
