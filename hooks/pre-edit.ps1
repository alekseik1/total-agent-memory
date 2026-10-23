# ===========================================
# PreToolUse Hook (PowerShell) — v7.0 file_context guard
#
# Twin of hooks/pre-edit.sh. Asks the model to call file_context(path)
# BEFORE editing a file, delivered as hookSpecificOutput.additionalContext.
#
# Hook: PreToolUse (matcher: "Write|Edit")
# ===========================================

. (Join-Path $PSScriptRoot "lib\common.ps1")
Set-HookEvent "PreToolUse"

function Invoke-Hook {
    $tool = Get-HookField 'tool_name'
    $filePath = Get-HookField 'tool_input.file_path'

    # Only guard Write/Edit, not NotebookEdit
    if ($tool -ne "Write" -and $tool -ne "Edit") { return }
    if (-not $filePath) { return }

    # Skip vendored / temp paths (Windows + POSIX separators).
    $normalized = $filePath.Replace("\", "/")
    foreach ($pat in @("/.git/", "/node_modules/", "/.venv/", "/Temp/")) {
        if ($normalized -like "*$pat*") { return }
    }
    if ($normalized.StartsWith("/tmp/")) { return }

    Add-HookContext @"
v7.0 pre-edit guard: before editing ``$filePath``, call
  file_context(path="$filePath")
If risk_score > 0.3, read the returned warnings (past errors / hot spots) and
incorporate them into the edit. Skip if file_context was already called for this
path in the current turn.
"@
}

Invoke-Hook
Complete-Hook
exit 0
