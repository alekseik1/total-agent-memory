# ===========================================
# Memory Trigger Hook (PowerShell) - twin of hooks/memory-trigger.sh
#
# PostToolUse / PostToolUseFailure:Bash - error capture (self_error_log +
# auto_self_improve.py), error->fix pairs, significant commands and
# test/build/lint success reminders. The decision table is shared with the
# bash hook in lib/tam_hook.py; output is one
# hookSpecificOutput.additionalContext JSON object (or nothing).
#
# Hook: PostToolUse (matcher: "Bash"), PostToolUseFailure (matcher: "Bash")
# ===========================================

. (Join-Path $PSScriptRoot "lib\common.ps1")

Invoke-HookHelper memory-trigger --state-dir $HookStateDir --src $HookSrcDir --emit PostToolUse
exit 0
