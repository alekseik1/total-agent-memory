#!/usr/bin/env bash
# Local CI parity: the same lint, test selection and recall-quality gate as
# .github/workflows/ci.yml, in one command.
#
#   scripts/check.sh               ruff + test suite + platform-sensitive + eval gate
#   scripts/check.sh --coverage    also enforce the coverage floor from ci.yml
#   scripts/check.sh --quick       ruff + eval gate only (no full suite)
#
# Environment:
#   PYTHON                 interpreter (default: .venv/bin/python, else python3)
#   FASTEMBED_CACHE_PATH   model cache; the eval gate needs the default model
#                          cached (python tests/smoke/prewarm_models.py)
#   PYTEST_ARGS            extra arguments appended to every pytest call
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CI_FILE="$ROOT/.github/workflows/ci.yml"
cd "$ROOT"

run_coverage=0
run_suite=1
for arg in "$@"; do
  case "$arg" in
    --coverage) run_coverage=1 ;;
    --quick) run_suite=0 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON="$ROOT/.venv/bin/python"; else PYTHON="python3"; fi
fi

ci_env() {
  # Read a top-level env value from ci.yml so the two never drift apart.
  sed -n "s/^  $1: *'\{0,1\}\([^']*\)'\{0,1\} *$/\1/p" "$CI_FILE" | head -n 1
}

BLOCKING_MARKERS="$(ci_env PYTEST_BLOCKING_MARKERS)"
COVERAGE_FLOOR="$(ci_env COVERAGE_FLOOR)"
RUFF_VERSION="$(ci_env RUFF_VERSION)"
if [[ -z "$BLOCKING_MARKERS" || -z "$COVERAGE_FLOOR" || -z "$RUFF_VERSION" ]]; then
  echo "could not read PYTEST_BLOCKING_MARKERS / COVERAGE_FLOOR / RUFF_VERSION from $CI_FILE" >&2
  exit 2
fi

export MCP_TRANSPORT=stdio
export MEMORY_LLM_ENABLED=false
export MEMORY_MODE=fast
export MEMORY_CROSS_RERANK=off
export MEMORY_EMBED_THREADS=1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

TAM_MEMORY_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tam-check.XXXXXX")"
export TAM_MEMORY_DIR
cleanup() { rm -rf -- "$TAM_MEMORY_DIR"; }
trap cleanup EXIT

read -r -a extra_args <<< "${PYTEST_ARGS:-}"
pytest_run() {
  "$PYTHON" -m pytest tests -q --strict-markers -p no:cacheprovider "$@" ${extra_args[@]+"${extra_args[@]}"}
}

step() { printf '\n==> %s\n' "$*"; }

step "ruff (expected version $RUFF_VERSION)"
local_ruff="$("$PYTHON" -m ruff --version 2>/dev/null | awk '{print $2}')" || {
  echo "ruff is not installed for $PYTHON; pip install -r requirements-dev.txt" >&2
  exit 1
}
if [[ "$local_ruff" != "$RUFF_VERSION" ]]; then
  echo "warning: local ruff $local_ruff differs from CI ruff $RUFF_VERSION; results may differ" >&2
fi
"$PYTHON" -m ruff check .

if [[ "$run_suite" -eq 1 ]]; then
  if [[ "$run_coverage" -eq 1 ]]; then
    step "test suite with coverage floor ${COVERAGE_FLOOR}% (-m 'not quarantined')"
    pytest_run -m "not quarantined" --cov=src --cov-report=term:skip-covered --cov-fail-under="$COVERAGE_FLOOR"
  else
    step "test suite (-m '$BLOCKING_MARKERS')"
    pytest_run -m "$BLOCKING_MARKERS"
  fi

  step "platform-sensitive tests"
  rc=0
  pytest_run -m "platform_sensitive and not quarantined" || rc=$?
  # Exit code 5: no test carries the marker.
  if [[ "$rc" -ne 0 && "$rc" -ne 5 ]]; then exit "$rc"; fi
fi

step "recall-quality gate (-m eval)"
TAM_EVAL_REQUIRE_EMBEDDER=1 pytest_run -s -m eval

step "all checks passed"
