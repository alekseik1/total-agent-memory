#!/usr/bin/env bash
# ===========================================
# Auto Capture Hook — Portable version
#
# PostToolUse:Write|Edit — suggests memory_observe after file changes
# (delivered as hookSpecificOutput.additionalContext).
#
# Env: TAM_AUTO_CAPTURE_HINT=0|false disables it; silent in subagents unless
#      TAM_HOOKS_IN_SUBAGENTS=1
#
# Hook: PostToolUse (matcher: "Write|Edit")
# ===========================================

# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
hook_set_event "PostToolUse"

[ "$HOOK_QUIET_SUBAGENT" = "1" ] && exit 0
case "${TAM_AUTO_CAPTURE_HINT:-}" in 0|false) exit 0 ;; esac

TOOL_NAME=$(hook_get 'tool_name')
FILE_PATH=$(hook_get 'tool_input.file_path')

if [ -z "$TOOL_NAME" ] || [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Skip config/doc files — usually not worth observing
case "$FILE_PATH" in
    *.md|*.txt|*.log|*.json|*.yaml|*.yml|*.toml|*.lock|*.sum)
        exit 0
        ;;
esac

FILENAME=$(basename "$FILE_PATH")
hook_add_context "MEMORY_HINT: File changed: ${FILENAME}. Consider: memory_observe(tool_name=\"${TOOL_NAME}\", summary=\"Modified ${FILENAME}\", files_affected=[\"${FILE_PATH}\"])"
exit 0
