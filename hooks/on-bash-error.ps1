# ===========================================
# Bash failure hook (PowerShell) — v7.0 learn_error trigger
#
# Twin of hooks/on-bash-error.sh. Fires when a Bash call fails
# (PostToolUseFailure, or PostToolUse with a non-zero exit code) and error
# text is present; asks the model to call learn_error(...) through
# hookSpecificOutput.additionalContext.
#
# Hook: PostToolUseFailure (matcher: "Bash"), PostToolUse (matcher: "Bash")
# ===========================================

. (Join-Path $PSScriptRoot "lib\common.ps1")
Set-HookEvent "PostToolUse"

function Invoke-Hook {
    if ((Get-HookField 'tool_name') -ne "Bash") { return }
    if ($HookExitCode -in @("", "0", "null")) { return }
    if ($HookInterrupted) { return }

    $stderr = $HookErrorText
    if ($stderr.Length -gt 500) { $stderr = $stderr.Substring(0, 500) }
    foreach ($pat in @("permission denied by user", "User denied", "SIGINT")) {
        if ($stderr -like "*$pat*") { return }
    }
    if (-not $stderr) { return }

    Add-HookContext @"
v7.0 learn_error trigger: bash exited $HookExitCode. If the root cause is
reproducible and fixable, call:
  learn_error(
      file="<path if relevant>",
      error="<short stderr>",
      root_cause="<what actually failed>",
      fix="<what resolves it>",
      pattern="<short slug, e.g. sqlite-locked-during-ddl>"
  )
Skip if this is user-aborted, interactive, or benign. After N>=3 same patterns
it auto-consolidates into a rule - do not re-log if you just fixed it and the
root cause is identical to an earlier call this turn.
"@
}

Invoke-Hook
Complete-Hook
exit 0
