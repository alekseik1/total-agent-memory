#!/usr/bin/env bash
# ===========================================
# Session Start Hook — Portable version
#
# Plain stdout of SessionStart is added to the model context.
#
#   startup / clear : memory hint, SOUL rules, error patterns, project
#                     knowledge count
#   compact / resume: memory hint + the latest session_continuity summary,
#                     next steps, pitfalls and top active rules injected
#                     directly (bounded, see RESTORE_MAX_CHARS in tam_hook.py)
#   startup / resume: the newest recovery file of a previous session in the
#                     same directory (consumed once; stale ones are removed)
#
# Also cleans old error-fix state.
#
# Hook: SessionStart (matcher: "" or "startup|resume|clear|compact")
# ===========================================

# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"

CWD=$(hook_get 'cwd')
[ -z "$CWD" ] && CWD="$PWD"
PROJECT=$(hook_project_name)
BRANCH=$(hook_git_branch)
MODEL=$(hook_model_short)
SOURCE=$(hook_get 'source')
SESSION_ID=$(hook_get 'session_id')
[ -z "$SESSION_ID" ] && SESSION_ID="${CLAUDE_SESSION_ID:-unknown}"

CTX="$PROJECT"
[ -n "$BRANCH" ] && CTX="$CTX @ $BRANCH"

# Human-readable source
case "$SOURCE" in
    startup) SOURCE_TEXT="New session" ;;
    resume)  SOURCE_TEXT="Resumed" ;;
    clear)   SOURCE_TEXT="After /clear" ;;
    compact) SOURCE_TEXT="After /compact" ;;
    *)       SOURCE_TEXT="Started" ;;
esac

NOTIFY_MSG="$CTX | $SOURCE_TEXT"
[ -n "$MODEL" ] && NOTIFY_MSG="$NOTIFY_MSG | $MODEL"
hook_notify "$NOTIFY_MSG" "Claude Memory | Session" "Glass"
hook_log "SESSION_START: $NOTIFY_MSG"

echo "Session: $NOTIFY_MSG"
echo ""

# ======= MEMORY HINT =======
HINT="MEMORY_HINT: Project: ${PROJECT}"
[ -n "$BRANCH" ] && HINT="${HINT}, Branch: ${BRANCH}"
HINT="${HINT}. Use memory_recall(query=\"your task\", project=\"${PROJECT}\") to search past knowledge."
HINT="${HINT} Also run self_rules_context(project=\"${PROJECT}\") to load behavioral rules."
echo "$HINT"

# ======= RESTORE / RULES / RECOVERY =======
CONTEXT=$(hook_helper session-start-context \
    --db "$CLAUDE_MEMORY_DIR/memory.db" \
    --project "$PROJECT" \
    --cwd "$CWD" \
    --session "$SESSION_ID" \
    --source "$SOURCE" \
    --recovery-dir "$HOOK_RECOVERY_DIR")
if [ -n "$CONTEXT" ]; then
    echo ""
    echo "$CONTEXT"
fi

# ======= CLEAN UP OLD STATE =======
if [ -d "$HOOK_STATE_DIR" ]; then
    # Remove error states older than 1 hour
    find "$HOOK_STATE_DIR" -maxdepth 1 -name "last-error-*" -mmin +60 -delete 2>/dev/null
fi
exit 0
