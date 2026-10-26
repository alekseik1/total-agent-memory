#!/usr/bin/env bash
# ===========================================
# Common hook utilities — Portable version (bash 3.2+)
#
# Source this file: source "$(dirname "$0")/lib/common.sh"
#
# Environment variables:
#   CLAUDE_MEMORY_INSTALL_DIR — TAM checkout holding src/ (install-env.sh, written by
#                               the installer next to this file, provides the default)
#   TAM_MEMORY_DIR            — memory storage (default ~/.tam)
#   CLAUDE_MEMORY_DIR         — legacy name; wins when set. Both are re-exported equal.
#   HOOK_PARSE_FIELDS=0       — set before sourcing to skip the payload field parse
#   HOOK_SKIP_STDIN=1         — set before sourcing when the payload does not come on
#                               stdin (Codex notify passes it as an argument)
#
# Reading payload fields (one python spawn per hook, done here on source):
#   hook_get '.tool_input.file_path'     -> value or empty
#   $HOOK_EVENT $HOOK_EXIT_CODE $HOOK_ERROR_TEXT $HOOK_INTERRUPTED (normalized)
#
# Sending text to the model:
#   hook_set_event PostToolUse            # default when the payload lacks hook_event_name
#   hook_add_context "message"            # repeatable; merged into one JSON object
#   The buffer is flushed on exit as
#   {"hookSpecificOutput":{"hookEventName":...,"additionalContext":...}}.
#   Never echo model-directed text from PreToolUse/PostToolUse hooks directly.
#
# IMPORTANT: Reads stdin immediately on source because subshells lose it.
# ===========================================
# Variables defined here are consumed by the sourcing hooks.
# shellcheck disable=SC2034

# Read JSON from stdin IMMEDIATELY (before any subshell calls)
if [ "${HOOK_SKIP_STDIN:-0}" = "1" ]; then
    HOOK_INPUT=""
else
    HOOK_INPUT=$(cat)
fi

HOOK_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
HOOK_HELPER="${HOOK_LIB_DIR}/tam_hook.py"

# Installer-generated defaults (install dir + memory dir) for copies living outside the checkout.
if [ -f "${HOOK_LIB_DIR}/install-env.sh" ]; then
    # shellcheck source=/dev/null
    . "${HOOK_LIB_DIR}/install-env.sh"
fi

# Resolve install dir (where src/ lives)
CLAUDE_MEMORY_INSTALL_DIR="${CLAUDE_MEMORY_INSTALL_DIR:-$(cd "${HOOK_LIB_DIR}/../.." 2>/dev/null && pwd)}"

# Resolve memory storage dir. Order (unchanged since v10): CLAUDE_MEMORY_DIR (explicit
# per-hook override), TAM_MEMORY_DIR, ~/.tam, then an unmigrated ~/.claude-memory.
# Both names are exported with the resolved value so src/ scripts started from a hook
# (paths.memory_dir() reads TAM_MEMORY_DIR first) use the very same directory.
if [ -n "${CLAUDE_MEMORY_DIR:-}" ]; then
    TAM_MEMORY_DIR="$CLAUDE_MEMORY_DIR"
elif [ -z "${TAM_MEMORY_DIR:-}" ]; then
    if [ -d "$HOME/.tam" ] || [ ! -d "$HOME/.claude-memory" ]; then
        TAM_MEMORY_DIR="$HOME/.tam"
    else
        TAM_MEMORY_DIR="$HOME/.claude-memory"
    fi
fi
CLAUDE_MEMORY_DIR="$TAM_MEMORY_DIR"
export TAM_MEMORY_DIR CLAUDE_MEMORY_DIR CLAUDE_MEMORY_INSTALL_DIR
[ -z "${TAM_HOOKS_VERSION:-}" ] || export TAM_HOOKS_VERSION

# Python: the venv of the checkout, then the server venv the plugin bootstrap creates in the memory
# dir (or ~/.tam), POSIX and Windows layouts both; finally a system python, newest first (ambient
# memory needs 3.11+, so plain python3 comes last).
hook_find_python() {
    local candidate dir
    for dir in "${CLAUDE_MEMORY_INSTALL_DIR}" "${TAM_MEMORY_DIR}" "${HOME}/.tam"; do
        for candidate in "$dir/.venv/bin/python" "$dir/.venv/Scripts/python.exe"; do
            if [ -x "$candidate" ]; then
                printf '%s' "$candidate"
                return 0
            fi
        done
    done
    for candidate in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    printf '%s' "python3"
}
if [ -z "${HOOK_PYTHON:-}" ]; then
    HOOK_PYTHON="$(hook_find_python)"
fi

# Recovery and state directories
HOOK_SRC_DIR="${CLAUDE_MEMORY_INSTALL_DIR}/src"
HOOK_RECOVERY_DIR="${CLAUDE_MEMORY_DIR}/recovery"
HOOK_STATE_DIR="${CLAUDE_MEMORY_DIR}/state"
HOOK_LOG_FILE="${CLAUDE_MEMORY_DIR}/hooks.log"
TAM_HOOK_LOG_FILE="$HOOK_LOG_FILE"
export TAM_HOOK_LOG_FILE

# Run the shared helper: hook_helper <command> [args...]
hook_helper() {
    "$HOOK_PYTHON" "$HOOK_HELPER" "$@"
}

# ===========================================
# Payload fields — parsed once
# ===========================================

HOOK_EVENT=""
HOOK_EXIT_CODE=""
HOOK_ERROR_TEXT=""
HOOK_INTERRUPTED="0"
HOOK_FIELDS_PARSED=""
if [ "${HOOK_PARSE_FIELDS:-1}" != "0" ] && [ -n "$HOOK_INPUT" ]; then
    HOOK_FIELD_ASSIGNMENTS=$(printf '%s' "$HOOK_INPUT" | hook_helper fields)
    # Output is NAME='shlex-quoted value' lines produced by tam_hook.py only.
    eval "$HOOK_FIELD_ASSIGNMENTS"
    unset HOOK_FIELD_ASSIGNMENTS
fi

# Extract a field from the parsed payload
# Usage: hook_get '.tool_input.command'
#        hook_get 'cwd'
hook_get() {
    local key="${1#.}"
    key="${key//./__}"
    key="${key//[!A-Za-z0-9_]/_}"
    local var="HOOK_F_${key}"
    if [ -n "${!var:-}" ]; then
        printf '%s\n' "${!var}"
    fi
}

# ===========================================
# Model context output (PreToolUse / PostToolUse / PostToolUseFailure / ...)
# ===========================================

HOOK_DEFAULT_EVENT=""
HOOK_CONTEXT_BUF=""

hook_set_event() {
    HOOK_DEFAULT_EVENT="$1"
}

hook_add_context() {
    [ -z "$1" ] && return 0
    if [ -n "$HOOK_CONTEXT_BUF" ]; then
        HOOK_CONTEXT_BUF="${HOOK_CONTEXT_BUF}

$1"
    else
        HOOK_CONTEXT_BUF="$1"
    fi
}

hook_flush_context() {
    [ -z "$HOOK_CONTEXT_BUF" ] && return 0
    local event="${HOOK_EVENT:-$HOOK_DEFAULT_EVENT}"
    if [ -z "$event" ]; then
        hook_log "CONTEXT_DROPPED: no hook event name for: ${HOOK_CONTEXT_BUF:0:120}"
        HOOK_CONTEXT_BUF=""
        return 0
    fi
    printf '%s' "$HOOK_CONTEXT_BUF" | hook_helper emit "$event"
    HOOK_CONTEXT_BUF=""
}
trap hook_flush_context EXIT

# Get project name from cwd
hook_project_name() {
    if [ -n "${MEMORY_PROJECT:-}" ]; then
        echo "$MEMORY_PROJECT"
        return 0
    fi
    local cwd
    cwd=$(hook_get 'cwd')
    [ -z "$cwd" ] && cwd="$PWD"
    basename "$cwd"
}

# Get git branch from cwd (read-only, safe)
hook_git_branch() {
    local cwd
    cwd=$(hook_get 'cwd')
    [ -z "$cwd" ] && cwd="$PWD"
    git -C "$cwd" rev-parse --abbrev-ref HEAD 2>/dev/null || echo ""
}

# Get short model name
hook_model_short() {
    local model
    model=$(hook_get 'model__display_name')
    [ -z "$model" ] && model=$(hook_get 'model__id')
    [ -z "$model" ] && model=$(hook_get 'model')
    case "$model" in
        *[Oo]pus*)   echo "Opus" ;;
        *[Ss]onnet*) echo "Sonnet" ;;
        *[Hh]aiku*)  echo "Haiku" ;;
        "")          echo "" ;;
        *)           echo "$model" ;;
    esac
}

# Build context string: "project @ branch"
hook_context() {
    local project branch ctx
    project=$(hook_project_name)
    branch=$(hook_git_branch)
    ctx="$project"
    [ -n "$branch" ] && ctx="$ctx @ $branch"
    echo "$ctx"
}

# Send macOS notification (no-op elsewhere). Text goes through argv, never into the script.
# Usage: hook_notify "message" "title" "sound"
hook_notify() {
    local message="${1:-}"
    local title="${2:-Claude Memory}"
    local sound="${3:-default}"

    if [ "$(uname)" = "Darwin" ] && [ "${TAM_HOOK_NOTIFY:-1}" != "0" ]; then
        osascript -e 'on run argv' \
            -e 'display notification (item 1 of argv) with title (item 2 of argv) sound name (item 3 of argv)' \
            -e 'end run' "$message" "$title" "$sound" >/dev/null 2>&1 \
            || hook_log "NOTIFY_FAILED: $title"
    fi
}

# Log to file
# Usage: hook_log "message"
hook_log() {
    local message="$1"
    mkdir -p "$(dirname "$HOOK_LOG_FILE")"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" >> "$HOOK_LOG_FILE"
}

# Run a Python script from src/ directory (background, with logging)
#
# Two modes:
#   - Native:  execs via local venv Python.
#   - Docker:  proxies to `docker exec` in the running MCP container when
#              CLAUDE_MEMORY_DOCKER=1 (or TAM_DOCKER=1). Container name via
#              CLAUDE_MEMORY_CONTAINER / TAM_CONTAINER (default: claude-memory-mcp).
#
# Usage: hook_run_script "auto_self_improve.py" "error" "--description" "msg"
hook_run_script() {
    local script_name="$1"
    shift

    if [ "${CLAUDE_MEMORY_DOCKER:-${TAM_DOCKER:-0}}" = "1" ]; then
        local container="${CLAUDE_MEMORY_CONTAINER:-${TAM_CONTAINER:-claude-memory-mcp}}"
        if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
            docker exec "$container" python "src/${script_name}" "$@" \
                >> "$HOOK_LOG_FILE" 2>&1 &
            return 0
        fi
        hook_log "RUN_SKIPPED: container ${container} not running for ${script_name}"
        return 1
    fi

    local script_path="${HOOK_SRC_DIR}/${script_name}"
    if [ -f "$script_path" ]; then
        "$HOOK_PYTHON" "$script_path" "$@" >> "$HOOK_LOG_FILE" 2>&1 &
        return 0
    fi
    hook_log "RUN_SKIPPED: ${script_path} not found (set CLAUDE_MEMORY_INSTALL_DIR)"
    return 1
}
