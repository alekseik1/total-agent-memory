#!/usr/bin/env bash
# ===========================================
# Memory nudge library — read access to the per-session
# writes-vs-memory_save counters.
#
# The counters live in $TAM_MEMORY_DIR/state/nudge-<session>.json and are
# maintained by post-tool-use.{sh,ps1} through `tam_hook.py nudge`, which
# also owns the thresholds (MEMORY_NUDGE_SOFT/HARD/STEP). This file only
# exposes shell helpers on top of the same Python implementation, so the
# bash and PowerShell hooks can never drift apart.
#
# Source from any hook AFTER common.sh (or with HOOK_PYTHON and
# CLAUDE_MEMORY_DIR/TAM_MEMORY_DIR exported):
#     source "$(dirname "$0")/lib/memory-nudge.sh"
# ===========================================

NUDGE_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
NUDGE_MEMORY_DIR="${CLAUDE_MEMORY_DIR:-${TAM_MEMORY_DIR:-$HOME/.tam}}"
NUDGE_STATE_DIR="${NUDGE_MEMORY_DIR}/state"

# nudge_path SESSION_ID -> echoes the counter file path.
nudge_path() {
    local sid="${1:-unknown}"
    # Strip anything that looks unsafe in a filename
    sid=$(printf '%s' "$sid" | tr -c 'A-Za-z0-9._-' '_')
    echo "${NUDGE_STATE_DIR}/nudge-${sid}.json"
}

# nudge_summary SESSION_ID PROJECT — one final-status line
# (MEMORY_FINAL_WARNING / MEMORY_FINAL_NOTE / MEMORY_FINAL_OK); silent when
# nothing relevant happened in the session.
nudge_summary() {
    "${HOOK_PYTHON:-python3}" "${NUDGE_LIB_DIR}/tam_hook.py" nudge-summary \
        --memory-dir "$NUDGE_MEMORY_DIR" --session "${1:-unknown}" --project "${2:-unknown}"
}
