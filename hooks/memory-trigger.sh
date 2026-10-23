#!/usr/bin/env bash
# ===========================================
# Memory Trigger Hook — Portable version
#
# PostToolUse / PostToolUseFailure:Bash — detects errors, fix pairs and
# significant commands. The decision table lives in lib/tam_hook.py
# (shared with memory-trigger.ps1):
#
# 1. Bash ERRORS -> self_error_log reminder + auto_self_improve.py error
# 2. error->fix pairs -> "save the fix" reminder + auto_self_improve.py fix
# 3. significant commands -> memory_save reminder
# 4. test/build/lint success -> rate SOUL rules
# 5. full-suite pass -> reusable save reminder
#
# Output: one hookSpecificOutput.additionalContext JSON object (or nothing).
#
# Hook: PostToolUse (matcher: "Bash"), PostToolUseFailure (matcher: "Bash")
# ===========================================

HOOK_PARSE_FIELDS=0
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"

# Project resolution (MEMORY_PROJECT, else basename of the payload cwd) happens in the helper.
printf '%s' "$HOOK_INPUT" | hook_helper memory-trigger \
    --state-dir "$HOOK_STATE_DIR" --src "$HOOK_SRC_DIR" --emit PostToolUse
exit 0
