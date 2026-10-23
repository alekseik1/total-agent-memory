#!/usr/bin/env bash
# ===========================================
# UserPromptSubmit Hook — prompt capture + ambient recall
#
# 1. Caches the hook JSON and hands it to a detached worker
#    (tam_hook.py save-intent) that stores the prompt in `intents`.
# 2. Ambient recall (tam_hook.py prompt-context, src/ambient/prompt_context.py):
#    rules whose text matches the prompt and a short background block
#    (warm through the shared daemon, cold through FTS) are sent to the
#    model as hookSpecificOutput.additionalContext. Bounded by
#    TAM_AMBIENT_LATENCY_MS and TAM_AMBIENT_TOKEN_BUDGET, deduped per session;
#    any failure is logged and the prompt goes through unchanged.
#
# Env: TAM_AMBIENT_RECALL=false disables step 2 (also settable in the dashboard)
#      TAM_MEMORY_DIR / CLAUDE_MEMORY_INSTALL_DIR — resolved by lib/common.sh
#
# Hook: UserPromptSubmit (matcher: "")
# ===========================================

HOOK_PARSE_FIELDS=0
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"

# Ambient recall first: it reports "no store" itself, and with TAM_DOCKER=1 the
# store lives in the container, not under CLAUDE_MEMORY_DIR.
printf '%s' "$HOOK_INPUT" | hook_helper prompt-context --src "$HOOK_SRC_DIR" \
    --memory-dir "$CLAUDE_MEMORY_DIR" --emit UserPromptSubmit

DB_PATH="${CLAUDE_MEMORY_DIR}/memory.db"
[ -f "$DB_PATH" ] || exit 0

TMP_INPUT="$(mktemp -t cmm-uprompt.XXXXXX)" || { hook_log "USER_PROMPT: mktemp failed"; exit 0; }
printf '%s' "$HOOK_INPUT" > "$TMP_INPUT"

hook_helper save-intent --src "$HOOK_SRC_DIR" --db "$DB_PATH" --input "$TMP_INPUT" \
    </dev/null >>"$HOOK_LOG_FILE" 2>&1 &
disown 2>/dev/null || true
exit 0
