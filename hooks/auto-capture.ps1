# ===========================================
# Auto Capture Hook (PowerShell) - twin of hooks/auto-capture.sh
#
# PostToolUse:Write|Edit - suggests memory_observe after file changes
# (delivered as hookSpecificOutput.additionalContext).
#
# Hook: PostToolUse (matcher: "Write|Edit")
# ===========================================

. (Join-Path $PSScriptRoot "lib\common.ps1")
Set-HookEvent "PostToolUse"

function Invoke-Hook {
    $toolName = Get-HookField 'tool_name'
    $filePath = Get-HookField 'tool_input.file_path'
    if (-not $toolName -or -not $filePath) { return }

    # Skip config/doc files - usually not worth observing
    $ext = [System.IO.Path]::GetExtension($filePath)
    if ($ext -in ".md", ".txt", ".log", ".json", ".yaml", ".yml", ".toml", ".lock", ".sum") { return }

    $fileName = [System.IO.Path]::GetFileName($filePath)
    Add-HookContext "MEMORY_HINT: File changed: $fileName. Consider: memory_observe(tool_name=`"$toolName`", summary=`"Modified $fileName`", files_affected=[`"$filePath`"])"
}

Invoke-Hook
Complete-Hook
exit 0
