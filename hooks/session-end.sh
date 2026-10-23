#!/usr/bin/env bash
# ===========================================
# Session End Hook — Portable version
#
# 1. Parses the transcript (nested Claude Code JSONL and the legacy flat
#    format, via src/auto_episode_capture.extract_messages_from_transcript)
# 2. Writes this session's recovery file (one per session)
# 3. Starts auto_session_save.py, auto_episode_capture.py and auto_session_end.py
#    in background
# 4. Cleans error-fix state
#
# SessionEnd output is not shown to the model; progress goes to hooks.log.
#
# Hook: SessionEnd (matcher: "")
# ===========================================

# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"

REASON=$(hook_get 'reason')
CTX=$(hook_context)

# Human-readable reason
case "$REASON" in
    clear)                       REASON_TEXT="User ran /clear" ;;
    compact)                     REASON_TEXT="User ran /compact" ;;
    logout)                      REASON_TEXT="User logged out" ;;
    prompt_input_exit)           REASON_TEXT="User exited" ;;
    bypass_permissions_disabled) REASON_TEXT="Permissions changed" ;;
    *)                           REASON_TEXT="Session ended (${REASON:-unknown})" ;;
esac

printf '%s' "$HOOK_INPUT" | hook_helper session-end \
    --src "$HOOK_SRC_DIR" \
    --recovery-dir "$HOOK_RECOVERY_DIR" \
    --reason-text "$REASON_TEXT"

hook_notify "$CTX | $REASON_TEXT" "Claude Memory | Done" "Submarine"
hook_log "SESSION_END: $CTX | $REASON_TEXT"

# ======= CLEAN UP ERROR-FIX STATE =======
if [ -d "$HOOK_STATE_DIR" ]; then
    rm -f "$HOOK_STATE_DIR"/last-error-* 2>/dev/null
fi
exit 0
