#!/usr/bin/env bash
# ===========================================
# PreToolUse Hook — rule reminder before a Bash call or a file edit
#
# Matches the command line (or the edited path plus the new text) against the
# active SOUL rules (tam_hook.py rule-reminder, src/ambient/reminders.py).
# A matching rule — above all a prohibition the action seems to break — is
# sent to the model as hookSpecificOutput.additionalContext and recorded as a
# rule surface, so the Stop hook can check whether it was followed. The same
# rule is reminded at most once per 15 minutes per session. No permission
# decision is ever made.
#
# Env: TAM_RULE_REMINDERS=false disables it (also settable in the dashboard);
#      silent in subagents unless TAM_HOOKS_IN_SUBAGENTS=1
#
# Hook: PreToolUse (matcher: "Bash|Edit|Write|MultiEdit|NotebookEdit")
# ===========================================

HOOK_PARSE_FIELDS=0
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"

printf '%s' "$HOOK_INPUT" | hook_helper rule-reminder --src "$HOOK_SRC_DIR" \
    --memory-dir "$CLAUDE_MEMORY_DIR" --emit PreToolUse
exit 0
