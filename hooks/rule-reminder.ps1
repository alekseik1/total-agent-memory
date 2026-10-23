# ===========================================
# PreToolUse Hook (PowerShell) — rule reminder before a Bash call or an edit
#
# Twin of hooks/rule-reminder.sh. Matches the command line (or the edited
# path plus the new text) against the active SOUL rules
# (tam_hook.py rule-reminder); matching rules go to the model as
# hookSpecificOutput.additionalContext and are recorded as rule surfaces for
# the Stop-hook compliance check. No permission decision is ever made.
#
# Env: TAM_RULE_REMINDERS=false disables it
#
# Hook: PreToolUse (matcher: "Bash|Edit|Write|MultiEdit|NotebookEdit")
# ===========================================

. (Join-Path $PSScriptRoot "lib\common.ps1")

Invoke-HookHelper rule-reminder --src $HookSrcDir --memory-dir $MemoryDir --emit PreToolUse
exit 0
