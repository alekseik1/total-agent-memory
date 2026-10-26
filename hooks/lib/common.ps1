# ===========================================
# Common hook utilities (PowerShell twin of lib/common.sh)
#
# Dot-source it first thing in a hook:
#     . (Join-Path $PSScriptRoot "lib\common.ps1")
#
# Provides:
#   $HookInput      raw stdin (read once; set $HookSkipStdin = $true before
#                   dot-sourcing when the payload is an argument, as for Codex notify)
#   $HookData       parsed payload (ConvertFrom-Json) or $null
#   Get-HookField 'tool_input.file_path'   -> string or ""
#   $HookEvent / $HookExitCode / $HookErrorText / $HookInterrupted (normalized
#   exactly like tam_hook.py derived fields)
#   Set-HookEvent 'PostToolUse'            default event name
#   Add-HookContext "message"              repeatable; merged
#   Complete-Hook                          writes ONE
#       {"hookSpecificOutput":{"hookEventName":...,"additionalContext":...}}
#   Invoke-HookHelper <args>               runs lib/tam_hook.py, payload on stdin
#   Write-HookLog "message"
#
# Path resolution matches common.sh: install-env.ps1 defaults, then
# CLAUDE_MEMORY_INSTALL_DIR; memory dir CLAUDE_MEMORY_DIR, TAM_MEMORY_DIR,
# ~/.tam, unmigrated ~/.claude-memory. Both memory variables are exported
# with the resolved value.
# ===========================================

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = [Console]::OutputEncoding

$HookLibDir = $PSScriptRoot
$HookHelper = Join-Path $HookLibDir "tam_hook.py"

$HookInput = if ($HookSkipStdin) { "" } else { [Console]::In.ReadToEnd() }
$HookData = $null
if ($HookInput -and $HookInput.Trim()) {
    try {
        $HookData = $HookInput | ConvertFrom-Json
    } catch {
        [Console]::Error.WriteLine("[tam-hook] hook payload is not valid JSON: $($_.Exception.Message)")
    }
}

$InstallEnv = Join-Path $HookLibDir "install-env.ps1"
if (Test-Path $InstallEnv) { . $InstallEnv }

if (-not $env:CLAUDE_MEMORY_INSTALL_DIR) {
    $env:CLAUDE_MEMORY_INSTALL_DIR = Split-Path -Parent (Split-Path -Parent $HookLibDir)
}

if ($env:CLAUDE_MEMORY_DIR) {
    $env:TAM_MEMORY_DIR = $env:CLAUDE_MEMORY_DIR
} elseif (-not $env:TAM_MEMORY_DIR) {
    $NewDir = Join-Path $HOME ".tam"
    $OldDir = Join-Path $HOME ".claude-memory"
    if ((Test-Path $NewDir) -or -not (Test-Path $OldDir)) {
        $env:TAM_MEMORY_DIR = $NewDir
    } else {
        $env:TAM_MEMORY_DIR = $OldDir
    }
}
$env:CLAUDE_MEMORY_DIR = $env:TAM_MEMORY_DIR

$InstallDir = $env:CLAUDE_MEMORY_INSTALL_DIR
$MemoryDir = $env:TAM_MEMORY_DIR
$HookSrcDir = Join-Path $InstallDir "src"
$HookRecoveryDir = Join-Path $MemoryDir "recovery"
$HookStateDir = Join-Path $MemoryDir "state"
$HookLogFile = Join-Path $MemoryDir "hooks.log"
$env:TAM_HOOK_LOG_FILE = $HookLogFile

$HookPython = $env:HOOK_PYTHON
if (-not $HookPython) {
    # The checkout's venv, then the server venv the plugin bootstrap creates in the memory dir or ~/.tam.
    foreach ($base in @($InstallDir, $MemoryDir, (Join-Path $HOME ".tam"))) {
        foreach ($candidate in @(
                [System.IO.Path]::Combine($base, ".venv", "Scripts", "python.exe"),
                [System.IO.Path]::Combine($base, ".venv", "bin", "python"))) {
            if (Test-Path $candidate) { $HookPython = $candidate; break }
        }
        if ($HookPython) { break }
    }
}
if (-not $HookPython) {
    foreach ($name in @("python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { $HookPython = $cmd.Source; break }
    }
}

function Write-HookLog([string]$Message) {
    try {
        $dir = Split-Path -Parent $HookLogFile
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        Add-Content -Path $HookLogFile -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" -Encoding UTF8
    } catch {
        [Console]::Error.WriteLine("[tam-hook] cannot write ${HookLogFile}: $($_.Exception.Message)")
    }
}

function Get-HookField([string]$Path) {
    $node = $HookData
    foreach ($key in $Path.TrimStart('.').Split('.')) {
        if ($null -eq $node) { return "" }
        $prop = $node.PSObject.Properties[$key]
        if ($null -eq $prop) { return "" }
        $node = $prop.Value
    }
    if ($null -eq $node) { return "" }
    if ($node -is [string]) { return $node }
    if ($node -is [bool]) { return $node.ToString().ToLowerInvariant() }
    if ($node -is [System.Management.Automation.PSCustomObject] -or $node -is [array]) {
        return ($node | ConvertTo-Json -Compress -Depth 10)
    }
    return [string]$node
}

function Get-HookProject {
    if ($env:MEMORY_PROJECT) { return $env:MEMORY_PROJECT }
    $cwd = Get-HookField 'cwd'
    if (-not $cwd) { $cwd = (Get-Location).Path }
    $leaf = Split-Path -Leaf $cwd.TrimEnd('/', '\')
    if ($leaf) { return $leaf }
    return "unknown"
}

function Get-HookExitCode {
    foreach ($container in @('tool_response', 'tool_output')) {
        foreach ($key in @('exit_code', 'exitCode', 'returncode')) {
            $value = Get-HookField "$container.$key"
            if ($value -ne "") { return $value }
        }
    }
    if ((Get-HookField 'hook_event_name') -eq 'PostToolUseFailure') {
        $m = [regex]::Match((Get-HookField 'error'), 'exit code:?\s*(-?\d+)', 'IgnoreCase')
        if ($m.Success) { return $m.Groups[1].Value }
        return "1"
    }
    return ""
}

function Get-HookErrorText {
    foreach ($container in @('tool_response', 'tool_output')) {
        foreach ($key in @('stderr', 'error')) {
            $value = Get-HookField "$container.$key"
            if ($value.Trim()) { return $value }
        }
    }
    return (Get-HookField 'error')
}

$HookEvent = Get-HookField 'hook_event_name'
$HookExitCode = Get-HookExitCode
$HookErrorText = Get-HookErrorText
$HookInterrupted = ((Get-HookField 'is_interrupt') -eq 'true') -or
    ((Get-HookField 'tool_response.interrupted') -eq 'true')

$script:HookDefaultEvent = ""
$script:HookContext = New-Object System.Collections.Generic.List[string]
$HookContextMaxChars = 9500

function Set-HookEvent([string]$Name) { $script:HookDefaultEvent = $Name }

function Add-HookContext([string]$Text) {
    if ($Text -and $Text.Trim()) { $script:HookContext.Add($Text.Trim()) }
}

function Complete-Hook {
    if ($script:HookContext.Count -eq 0) { return }
    $eventName = if ($HookEvent) { $HookEvent } else { $script:HookDefaultEvent }
    $text = ($script:HookContext -join "`n`n")
    $script:HookContext.Clear()
    if (-not $eventName) {
        Write-HookLog "CONTEXT_DROPPED: no hook event name"
        return
    }
    if ($text.Length -gt $HookContextMaxChars) {
        $text = $text.Substring(0, $HookContextMaxChars - 16) + "`n[... truncated]"
    }
    $payload = [ordered]@{
        hookSpecificOutput = [ordered]@{ hookEventName = $eventName; additionalContext = $text }
    }
    [Console]::Out.WriteLine(($payload | ConvertTo-Json -Compress -Depth 5))
}

# Starts tam_hook.py detached (fire-and-forget worker); output goes to hooks.log.
function Start-HookWorker([string[]]$HelperArgs) {
    if (-not $HookPython) {
        Write-HookLog "WORKER_SKIPPED: no python found for tam_hook.py $($HelperArgs -join ' ')"
        return
    }
    # Start-Process joins ArgumentList with spaces, so every element is quoted here.
    $quoted = @($HookHelper) + $HelperArgs | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }
    try {
        # Worker failures are written to hooks.log by tam_hook.py itself (TAM_HOOK_LOG_FILE).
        Start-Process -FilePath $HookPython -ArgumentList ($quoted -join ' ') -WindowStyle Hidden | Out-Null
    } catch {
        Write-HookLog "WORKER_FAILED: $($HelperArgs[0]): $($_.Exception.Message)"
    }
}

# Runs tam_hook.py with the payload on stdin; returns its stdout lines.
# A simple (non-advanced) function on purpose: "--flag" tokens must reach
# python verbatim through $args instead of being bound as PowerShell parameters.
function Invoke-HookHelper {
    if (-not $HookPython) {
        Write-HookLog "HELPER_SKIPPED: no python found for tam_hook.py $($args -join ' ')"
        return
    }
    $HookInput | & $HookPython $HookHelper @args
}
