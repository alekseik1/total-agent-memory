#!/usr/bin/env bash
# ===========================================
# On Stop Hook — Portable version
#
# Runs at the end of EVERY assistant turn, so it stays cheap and quiet:
#
# 1. Refreshes ONE recovery snapshot per session
#    ($TAM_MEMORY_DIR/recovery/pending-<session>.md, at most once every
#    STOP_SNAPSHOT_MIN_INTERVAL_SEC) in case the session dies without
#    SessionEnd. SessionEnd overwrites it with the transcript excerpt.
# 2. Once per session, when there were >=3 edits and no memory_save, shows
#    the USER a systemMessage reminder. Stop hooks cannot add model context
#    and must never block, so nothing model-directed is printed.
# 3. Ambient turn end (src/ambient/turn_end.py): checks the turn's logged
#    actions against the rules surfaced to the session (a broken
#    prohibition is reported in the same systemMessage) and, when the turn
#    changed something, auto-saves a short turn summary
#    (TAM_RULE_COMPLIANCE / TAM_TURN_AUTOSAVE, both default true).
# 4. Prunes nudge and ambient state older than 7 days.
#
# No desktop notification per turn.
#
# Hook: Stop (matcher: "")
# ===========================================

HOOK_PARSE_FIELDS=0
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"

printf '%s' "$HOOK_INPUT" | hook_helper stop \
    --memory-dir "$CLAUDE_MEMORY_DIR" \
    --recovery-dir "$HOOK_RECOVERY_DIR" \
    --src "$HOOK_SRC_DIR"
exit 0
