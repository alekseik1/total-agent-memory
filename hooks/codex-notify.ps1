#Requires -Version 5.1
<#
.SYNOPSIS
    Codex CLI notify hook — captures every finished Codex turn into TAM

.DESCRIPTION
    Twin of hooks/codex-notify.sh. Codex runs the notify program with the
    notification JSON as its last argument. Each "agent-turn-complete" becomes
    one codex_turn row in the agent action log and, unless
    TAM_TURN_AUTOSAVE=false, one auto-saved turn summary
    (tam_hook.py codex-notify). Other notification types are ignored. A JSON
    piped on stdin (older wrappers) still works.

    Add to ~/.codex/config.toml:
      notify = ["powershell", "-ExecutionPolicy", "Bypass", "-File", "C:/path/to/hooks/codex-notify.ps1"]

    Env: TAM_HOOK_NOTIFY=0 turns the desktop notification off.
#>

$HookSkipStdin = $args.Count -gt 0
. (Join-Path $PSScriptRoot "lib\common.ps1")

$Payload = if ($args.Count -gt 0) { [string]$args[-1] } else { $HookInput }
if (-not $Payload -or -not $Payload.Trim()) { exit 0 }

if (-not $HookPython) {
    Write-HookLog "CODEX_NOTIFY_SKIPPED: no python found for tam_hook.py"
    exit 0
}
& $HookPython $HookHelper codex-notify --src $HookSrcDir --memory-dir $MemoryDir --payload $Payload 2>&1 |
    ForEach-Object { Write-HookLog "CODEX_NOTIFY: $_" }

if ($env:TAM_HOOK_NOTIFY -ne "0" -and $Payload -match '"agent-turn-complete"') {
    $Project = Split-Path -Leaf (Get-Location).Path
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $notify = New-Object System.Windows.Forms.NotifyIcon
        try {
            $notify.Icon = [System.Drawing.SystemIcons]::Information
            $notify.Visible = $true
            $notify.ShowBalloonTip(3000, "Total Memory — $Project", "Turn captured in TAM memory",
                [System.Windows.Forms.ToolTipIcon]::Info)
            Start-Sleep -Seconds 3
        } finally {
            $notify.Dispose()
        }
    } catch {
        Write-HookLog "CODEX_NOTIFY: desktop notification failed: $($_.Exception.Message)"
    }
}
exit 0
