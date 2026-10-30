#!/usr/bin/env bash
# total-agent-memory — cross-platform health check
#
# Usage:
#   bash scripts/diagnose.sh
#
# Env:
#   DIAG_TEST_MODE=1     Skip real checks, emit mock OK report (for tests).
#   TAM_MEMORY_DIR       Override memory/state dir. Resolution order: TAM_MEMORY_DIR,
#                        legacy CLAUDE_MEMORY_DIR, ~/.tam, ~/.claude-memory (if it
#                        exists), else ~/.tam.
#   DASHBOARD_PORT       Override dashboard port (default 37737).
#
# Exit code:
#   0   all checks passed
#   1   one or more checks failed

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Same order as src/paths.py memory_dir().
if [ -n "${TAM_MEMORY_DIR:-}" ]; then
    MEMORY_DIR="$TAM_MEMORY_DIR"
elif [ -n "${CLAUDE_MEMORY_DIR:-}" ]; then
    MEMORY_DIR="$CLAUDE_MEMORY_DIR"
elif [ -d "$HOME/.tam" ] || [ ! -d "$HOME/.claude-memory" ]; then
    MEMORY_DIR="$HOME/.tam"
else
    MEMORY_DIR="$HOME/.claude-memory"
fi
DASHBOARD_PORT="${DASHBOARD_PORT:-37737}"

FAILED=0
REPORT=()

print_ok()   { REPORT+=("  OK   $1"); }
print_fail() { REPORT+=("  FAIL $1"); FAILED=1; }
print_warn() { REPORT+=("  WARN $1"); }
print_info() { REPORT+=("  INFO $1"); }

detect_os() {
    local uname_s
    uname_s="$(uname -s 2>/dev/null || echo unknown)"
    if [ "$uname_s" = "Darwin" ]; then
        echo "macOS"
    elif [ "$uname_s" = "Linux" ]; then
        if grep -qi microsoft /proc/version 2>/dev/null || [ -n "${WSL_DISTRO_NAME:-}" ]; then
            echo "WSL2"
        else
            echo "Linux"
        fi
    else
        echo "$uname_s"
    fi
}

check_os() {
    # Sets global DETECTED_OS so callers don't need a subshell (which would
    # discard appends to the REPORT array).
    DETECTED_OS="$(detect_os)"
    print_ok "OS: $DETECTED_OS ($(uname -srm 2>/dev/null || echo unknown))"
}

check_python() {
    local venv_py="$INSTALL_DIR/.venv/bin/python"
    if [ -x "$venv_py" ]; then
        local ver
        ver="$("$venv_py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "?")"
        print_ok "Python $ver venv: $venv_py"
    else
        print_fail "Python venv missing at $venv_py (run install.sh)"
    fi
}

check_mcp_importable() {
    local venv_py="$INSTALL_DIR/.venv/bin/python"
    if [ ! -x "$venv_py" ]; then
        print_fail "MCP server not checkable: venv python missing"
        return
    fi
    if "$venv_py" -c "import sys; sys.path.insert(0, '$INSTALL_DIR/src'); import server" >/dev/null 2>&1; then
        print_ok "MCP server module importable"
    else
        print_fail "MCP server import failed (run: $venv_py -c 'import server')"
    fi
}

check_services_macos() {
    local loaded
    loaded="$(launchctl list 2>/dev/null | grep -c 'com\.claude\.memory\|com\.claude-total-memory' || true)"
    if [ "${loaded:-0}" -ge 1 ]; then
        print_ok "LaunchAgents loaded: $loaded"
    else
        print_fail "No claude.memory LaunchAgents found (expected 3-4)"
    fi
}

check_services_linux() {
    if ! command -v systemctl >/dev/null 2>&1; then
        print_warn "systemctl not available — services may use shell-loop fallback"
        return
    fi
    if ! systemctl --user show-environment >/dev/null 2>&1; then
        print_warn "systemd --user not running (WSL1 or container?) — fallback mode"
        return
    fi
    local units="claude-memory-reflection.path claude-total-memory-dashboard.service"
    local active=0 total=0
    for u in $units; do
        total=$((total + 1))
        if systemctl --user is-active --quiet "$u" 2>/dev/null || \
           systemctl --user is-enabled --quiet "$u" 2>/dev/null; then
            active=$((active + 1))
        fi
    done
    if [ "$active" -eq "$total" ]; then
        print_ok "systemd --user units active: $active/$total"
    else
        print_fail "systemd --user units: $active/$total active"
    fi
}

check_dashboard() {
    local url="http://127.0.0.1:$DASHBOARD_PORT"
    local code=""
    if command -v curl >/dev/null 2>&1; then
        code="$(curl -sS -o /dev/null -m 3 -w '%{http_code}' "$url" 2>/dev/null || echo "")"
    elif command -v wget >/dev/null 2>&1; then
        if wget -q -T 3 -O /dev/null "$url" 2>/dev/null; then
            code="200"
        fi
    fi
    if [ "$code" = "200" ]; then
        print_ok "Dashboard: HTTP 200 at $url"
    else
        print_fail "Dashboard unreachable at $url (got: ${code:-no-response})"
    fi
}

check_ollama() {
    local url="${OLLAMA_HOST:-http://127.0.0.1:11434}/api/tags"
    if command -v curl >/dev/null 2>&1; then
        if curl -sS -o /dev/null -m 2 "$url" 2>/dev/null; then
            print_ok "Ollama detected at ${OLLAMA_HOST:-http://127.0.0.1:11434}"
        else
            print_warn "Ollama not reachable (optional) at ${OLLAMA_HOST:-http://127.0.0.1:11434}"
        fi
    else
        print_warn "curl not installed — skipping Ollama check"
    fi
}

check_db() {
    local db="$MEMORY_DIR/memory.db"
    if [ ! -f "$db" ]; then
        # install.sh creates the directory; the MCP server creates the database on its first start.
        if [ -d "$MEMORY_DIR" ]; then
            print_warn "DB not created yet: $db (the MCP server creates it on its first start)"
        else
            print_fail "DB missing: $db (no memory dir; run install.sh)"
        fi
        return
    fi
    local venv_py="$INSTALL_DIR/.venv/bin/python"
    if [ ! -x "$venv_py" ]; then
        print_warn "DB present ($db) but venv missing — can't check migrations"
        return
    fi
    local mig
    mig="$("$venv_py" - <<PY 2>/dev/null
import sqlite3, sys
try:
    con = sqlite3.connect(r"$db")
    cur = con.cursor()
    row = cur.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    print(row[0] if row and row[0] is not None else "none")
except Exception as exc:
    print(f"err:{exc}")
PY
)"
    case "$mig" in
        err:*|"")
            print_warn "DB migrations unknown: ${mig:-no-output}"
            ;;
        none)
            print_fail "DB has no schema_migrations rows"
            ;;
        *)
            print_ok "DB migrations at version $mig ($db)"
            ;;
    esac
}

# --- Test mode: emit mock report, skip real checks ---
if [ "${DIAG_TEST_MODE:-0}" = "1" ]; then
    cat <<'EOF'
total-agent-memory diagnostic (TEST MODE)
==========================================
  OK   OS: TestOS (mock)
  OK   Python 3.13 venv: mock
  OK   MCP server module importable
  OK   Services: mock
  OK   Dashboard: HTTP 200 (mock)
  OK   Ollama detected (mock)
  OK   DB migrations at version 42
==========================================
Result: 7/7 passed
EOF
    exit 0
fi

# --- Real run ---
echo "total-agent-memory diagnostic"
echo "============================="
DETECTED_OS=""
check_os
check_python
check_mcp_importable
case "$DETECTED_OS" in
    macOS)  check_services_macos ;;
    Linux|WSL2) check_services_linux ;;
    *)      print_warn "OS '$DETECTED_OS' has no service check" ;;
esac
check_dashboard
check_ollama
check_db

echo ""
for line in "${REPORT[@]}"; do
    echo "$line"
done
echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "Result: all checks passed"
else
    echo "Result: one or more checks FAILED (see above)"
fi
exit "$FAILED"
