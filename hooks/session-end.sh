#!/usr/bin/env bash
# ===========================================
# Session End Hook — Portable version
#
# 1. Saves recovery context from transcript
# 2. Calls auto_session_save.py for context preservation
# 3. Calls auto_episode_capture.py for episode capture
# 4. Cleans old recovery files
#
# Hook: SessionEnd (matcher: "")
# ===========================================

source "$(dirname "$0")/lib/common.sh"

REASON=$(hook_get 'reason')
CTX=$(hook_context)
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
DATE_SHORT=$(date '+%Y%m%d-%H%M%S')
CWD=$(hook_get 'cwd')
[ -z "$CWD" ] && CWD="$PWD"
# hook_project_name resolves a worktree's cwd to the main repo's own name -
# a bare `basename "$CWD"` names the worktree directory instead, so a
# session_end call from the tool side (project = git-root basename) and this
# hook's own session_end (see auto_session_end.py, producer="hook") mint two
# different dedup keys for the same close and the dedup policy never engages.
PROJECT=$(hook_project_name)

# Human-readable reason
case "$REASON" in
    clear)                       REASON_TEXT="User ran /clear" ;;
    compact)                     REASON_TEXT="User ran /compact" ;;
    logout)                      REASON_TEXT="User logged out" ;;
    prompt_input_exit)           REASON_TEXT="User exited" ;;
    bypass_permissions_disabled) REASON_TEXT="Permissions changed" ;;
    *)                           REASON_TEXT="Session ended ($REASON)" ;;
esac

# Recovery directory
mkdir -p "$HOOK_RECOVERY_DIR"

# Try to extract info from transcript
TRANSCRIPT=$(hook_get 'transcript_path')
TRANSCRIPT_SUMMARY=""
LAST_USER_MSGS=""
LAST_ASSISTANT=""

if [ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ]; then
    LINES=$(wc -l < "$TRANSCRIPT" 2>/dev/null | tr -d ' ')

    # Extract last user messages and assistant context via Python
    EXTRACTED=$("$HOOK_PYTHON" -c "
import json, sys

transcript = '$TRANSCRIPT'
user_msgs = []
assistant_msgs = []

try:
    with open(transcript) as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                role = d.get('role', '')
                content = d.get('content', '')

                if isinstance(content, list):
                    text = ' '.join(c.get('text', '') for c in content if c.get('type') == 'text')
                elif isinstance(content, str):
                    text = content
                else:
                    continue

                if not text.strip():
                    continue

                if role == 'user' and d.get('type') == 'human':
                    user_msgs.append(text.strip()[:500])
                elif role == 'assistant':
                    assistant_msgs.append(text.strip()[:500])
            except:
                continue

    # Output last 5 user messages
    print('USER_MSGS_START')
    for msg in user_msgs[-5:]:
        print(msg)
    print('USER_MSGS_END')

    print('ASSISTANT_START')
    for msg in assistant_msgs[-3:]:
        print(msg[:2000])
    print('ASSISTANT_END')
except:
    pass
" 2>/dev/null)

    LAST_USER_MSGS=$(echo "$EXTRACTED" | sed -n '/^USER_MSGS_START$/,/^USER_MSGS_END$/p' | sed '1d;$d')
    LAST_ASSISTANT=$(echo "$EXTRACTED" | sed -n '/^ASSISTANT_START$/,/^ASSISTANT_END$/p' | sed '1d;$d' | head -c 2000)
    TRANSCRIPT_SUMMARY="~${LINES} transcript events"

    # Save transcript recovery with context
    if [ -n "$LAST_USER_MSGS" ] || [ -n "$LAST_ASSISTANT" ]; then
        RECOVERY_FILE="$HOOK_RECOVERY_DIR/pending-${DATE_SHORT}.md"
        cat > "$RECOVERY_FILE" <<EOF
# Session Recovery - ${TIMESTAMP}

## Context
- **Project**: ${PROJECT}
- **Path**: ${CWD}
- **Reason**: ${REASON_TEXT}
- **Transcript**: ${TRANSCRIPT_SUMMARY}

## Last User Requests
${LAST_USER_MSGS:-No user messages extracted}

## Last Assistant Context
${LAST_ASSISTANT:-No assistant content extracted}

## Recovery Action
1. Review what was being discussed
2. Use memory_recall() to find related knowledge
3. Continue where left off or save summary to memory
EOF
        hook_log "SESSION_END: Recovery saved to $RECOVERY_FILE"
    fi
fi

# Keep only last 5 recovery files
ls -t "$HOOK_RECOVERY_DIR"/pending-*.md 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null

# ======= AUTO-SAVE SESSION CONTEXT (via auto_session_save.py) =======
if [ -n "$LAST_USER_MSGS" ]; then
    hook_run_script "auto_session_save.py" \
        --project "$PROJECT" \
        --cwd "$CWD" \
        --reason "$REASON_TEXT" \
        --user-context "$LAST_USER_MSGS" \
        --assistant-context "$(echo "$LAST_ASSISTANT" | head -c 1000)"
    hook_log "AUTO_SESSION_SAVE: Started for project $PROJECT"
fi

# ======= AUTO EPISODE CAPTURE (via auto_episode_capture.py) =======
EXTRACT_SESSION_ID=""
if [ -n "$TRANSCRIPT" ] && [ -f "$TRANSCRIPT" ]; then
    EXTRACT_SESSION_ID=$(basename "$TRANSCRIPT" .jsonl)
else
    # Fallback: find latest transcript by CWD
    PROJECT_HASH=$(echo "$CWD" | sed 's|^/||;s|/|-|g')
    TRANSCRIPT_DIR="$HOME/.claude/projects/-${PROJECT_HASH}"
    FALLBACK_TRANSCRIPT=$(ls -t "$TRANSCRIPT_DIR"/*.jsonl 2>/dev/null | head -1)
    if [ -n "$FALLBACK_TRANSCRIPT" ] && [ -f "$FALLBACK_TRANSCRIPT" ]; then
        EXTRACT_SESSION_ID=$(basename "$FALLBACK_TRANSCRIPT" .jsonl)
    fi
fi

if [ -n "$EXTRACT_SESSION_ID" ]; then
    hook_run_script "auto_episode_capture.py" \
        --session-id "$EXTRACT_SESSION_ID" \
        --project "$PROJECT"
    hook_log "AUTO_EPISODE: Started capture for session $EXTRACT_SESSION_ID (project: $PROJECT)"
fi

# Build notification
STATS=""
[ -n "$TRANSCRIPT_SUMMARY" ] && STATS=" | $TRANSCRIPT_SUMMARY"
NOTIFY_MSG="$CTX | $REASON_TEXT$STATS"

hook_notify "$NOTIFY_MSG" "Claude Memory | Done" "Submarine"
hook_log "SESSION_END: $NOTIFY_MSG"

echo "Session ended: $TIMESTAMP ($REASON_TEXT)"
echo ""
echo "MEMORY_AUTO_SAVE: Session context auto-saved to recovery + memory."

# ======= CLEAN UP ERROR-FIX STATE =======
if [ -d "$HOOK_STATE_DIR" ]; then
    rm -f "$HOOK_STATE_DIR"/last-error-* 2>/dev/null
fi
