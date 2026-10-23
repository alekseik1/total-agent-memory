# ===========================================
# Session End Hook (PowerShell) — twin of hooks/session-end.sh
#
# 1. Parses the transcript (nested Claude Code JSONL and the legacy flat
#    format, via src/auto_episode_capture.extract_messages_from_transcript)
# 2. Writes this session's recovery file (one per session)
# 3. Starts auto_session_save.py and auto_episode_capture.py in background
# 4. Cleans error-fix state
#
# SessionEnd output is not shown to the model; progress goes to hooks.log.
#
# Hook: SessionEnd (matcher: "")
# ===========================================

. (Join-Path $PSScriptRoot "lib\common.ps1")

$Reason = Get-HookField 'reason'
$ReasonText = switch ($Reason) {
    "clear"                       { "User ran /clear" }
    "compact"                     { "User ran /compact" }
    "logout"                      { "User logged out" }
    "prompt_input_exit"           { "User exited" }
    "bypass_permissions_disabled" { "Permissions changed" }
    default                       { if ($Reason) { "Session ended ($Reason)" } else { "Session ended (unknown)" } }
}

Invoke-HookHelper session-end --src $HookSrcDir --recovery-dir $HookRecoveryDir --reason-text $ReasonText
Write-HookLog "SESSION_END: $(Get-HookProject) | $ReasonText"

if (Test-Path $HookStateDir) {
    Get-ChildItem -Path $HookStateDir -Filter "last-error-*" -File | Remove-Item -ErrorAction SilentlyContinue
}
exit 0
