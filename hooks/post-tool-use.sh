#!/usr/bin/env bash
# ===========================================
# PostToolUse Hook — observation capture + memory_save nudges
#
# Runs AFTER a tool call (PostToolUse) or a failed one (PostToolUseFailure).
# Three responsibilities:
#
#  0. (default on, TAM_ACTION_LOG) log the call in agent_actions
#     (tam_hook.py record-action, src/ambient/actions.py): tool, short
#     redacted summary, exit status. Synchronous so the Stop hook sees it.
#  1. (always-on) bump per-session counters (writes / edits / saves)
#     and, when the writes-without-saves ratio crosses a threshold,
#     send a nudge to the model as hookSpecificOutput.additionalContext.
#     This addresses the "model never calls memory_save on its own"
#     pattern (reported 2026-05-14 by client running Sonnet).
#  2. (opt-in) when MEMORY_POST_TOOL_CAPTURE=1, enqueue a deferred
#     observation for the extractor (detached worker).
#
# Env:
#   MEMORY_POST_TOOL_CAPTURE  — "1" to enable observation capture
#   MEMORY_NUDGE_DISABLE      — "1" to disable nudges entirely
#   TAM_HOOKS_IN_SUBAGENTS    — "1" to count and nudge subagent calls too
#   MEMORY_NUDGE_SOFT/HARD/STEP — tune thresholds (see lib/tam_hook.py)
#   TAM_MEMORY_DIR / CLAUDE_MEMORY_INSTALL_DIR — resolved by lib/common.sh
#
# Hook: PostToolUse (matcher: "*"), PostToolUseFailure (matcher: "*")
# ===========================================

HOOK_PARSE_FIELDS=0
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"

printf '%s' "$HOOK_INPUT" | hook_helper record-action --src "$HOOK_SRC_DIR" --memory-dir "$CLAUDE_MEMORY_DIR"

# Nudge counter + emission (synchronous; stdlib only, no DB or network I/O).
if [ "${MEMORY_NUDGE_DISABLE:-0}" != "1" ]; then
    printf '%s' "$HOOK_INPUT" | hook_helper nudge --memory-dir "$CLAUDE_MEMORY_DIR" --emit PostToolUse
fi

# Opt-in guard — observation capture only runs when explicitly enabled.
if [ "${MEMORY_POST_TOOL_CAPTURE:-0}" != "1" ]; then
    exit 0
fi

# Cache stdin so the detached worker can read it after this shell exits.
TMP_INPUT="$(mktemp -t cmm-pthook.XXXXXX)" || { hook_log "POST_TOOL_CAPTURE: mktemp failed"; exit 0; }
printf '%s' "$HOOK_INPUT" > "$TMP_INPUT"

hook_helper capture-observation --src "$HOOK_SRC_DIR" --memory-dir "$CLAUDE_MEMORY_DIR" \
    --input "$TMP_INPUT" </dev/null >>"$HOOK_LOG_FILE" 2>&1 &
disown 2>/dev/null || true
exit 0
