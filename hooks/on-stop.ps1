# ===========================================
# Stop Hook (PowerShell) - twin of hooks/on-stop.sh
#
# Runs at the end of EVERY assistant turn, so it stays cheap and quiet:
# 1. Refreshes ONE recovery snapshot per session (rate-limited; SessionEnd
#    overwrites it with the transcript excerpt).
# 2. Once per session, when there were >=3 edits and no memory_save, shows
#    the USER a systemMessage reminder. Stop hooks cannot add model context
#    and must never block, so nothing model-directed is printed.
# 3. Ambient turn end: rule compliance check against the turn's logged
#    actions (a broken prohibition is reported in the same systemMessage)
#    and the turn summary auto-save (TAM_RULE_COMPLIANCE / TAM_TURN_AUTOSAVE).
# 4. Prunes nudge and ambient state older than 7 days.
#
# Hook: Stop (matcher: "")
# ===========================================

. (Join-Path $PSScriptRoot "lib\common.ps1")

Invoke-HookHelper stop --memory-dir $MemoryDir --recovery-dir $HookRecoveryDir --src $HookSrcDir
exit 0
