#Requires -Version 5.1
<#
.SYNOPSIS
    total-agent-memory - cross-platform health check (Windows).

.DESCRIPTION
    Windows/PowerShell equivalent of scripts/diagnose.sh.
    Prints a human-readable report of Python venv, MCP importability,
    Scheduled Tasks, dashboard HTTP, Ollama, and DB migration version.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\diagnose.ps1

.NOTES
    Exit code 0 = all checks passed.
    Exit code 1 = one or more checks failed.
    Set $env:DIAG_TEST_MODE = 1 to emit a mock OK report (used by tests).
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallDir = Split-Path -Parent $ScriptDir
$HomeDir    = if ($env:USERPROFILE) { $env:USERPROFILE } else { $env:HOME }
# Same order as src/paths.py memory_dir().
$MemoryDir  = if ($env:TAM_MEMORY_DIR) {
    $env:TAM_MEMORY_DIR
} elseif ($env:CLAUDE_MEMORY_DIR) {
    $env:CLAUDE_MEMORY_DIR
} elseif (Test-Path (Join-Path $HomeDir ".tam")) {
    Join-Path $HomeDir ".tam"
} elseif (Test-Path (Join-Path $HomeDir ".claude-memory")) {
    Join-Path $HomeDir ".claude-memory"
} else {
    Join-Path $HomeDir ".tam"
}
$Port       = if ($env:DASHBOARD_PORT) { $env:DASHBOARD_PORT } else { "37737" }

$script:Failed = 0
$script:Report = @()

function PrintOk   ([string]$msg) { $script:Report += "  OK   $msg" }
function PrintFail ([string]$msg) { $script:Report += "  FAIL $msg"; $script:Failed = 1 }
function PrintWarn ([string]$msg) { $script:Report += "  WARN $msg" }

function Get-OsName {
    # $IsWindows/$IsMacOS/$IsLinux do not exist in Windows PowerShell 5.1, and StrictMode makes reading them an error.
    if ($env:OS -eq "Windows_NT") { return "Windows" }
    $kernel = (& uname -s 2>$null)
    if ($kernel -eq "Darwin") { return "macOS" }
    if ($kernel -eq "Linux") { return "Linux" }
    return "Unknown"
}

function Check-Python {
    $venvPy = Join-Path $InstallDir ".venv\Scripts\python.exe"
    if (Test-Path $venvPy) {
        $ver = & $venvPy -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        PrintOk "Python $ver venv: $venvPy"
        return $venvPy
    } else {
        PrintFail "Python venv missing at $venvPy (run install.ps1)"
        return $null
    }
}

function Check-McpImport ([string]$venvPy) {
    if (-not $venvPy) {
        PrintFail "MCP server not checkable: venv python missing"
        return
    }
    $src = Join-Path $InstallDir "src"
    & $venvPy -c "import sys; sys.path.insert(0, r'$src'); import server" 2>$null
    if ($LASTEXITCODE -eq 0) {
        PrintOk "MCP server module importable"
    } else {
        PrintFail "MCP server import failed"
    }
}

function Check-ScheduledTasks {
    try {
        $tasks = Get-ScheduledTask -TaskName "ClaudeTotal*" -ErrorAction Stop
        $count = ($tasks | Measure-Object).Count
        if ($count -ge 1) {
            PrintOk "Scheduled Tasks registered: $count"
        } else {
            PrintFail "No ClaudeTotal* Scheduled Tasks found"
        }
    } catch {
        PrintFail "Scheduled Task query failed: $($_.Exception.Message)"
    }
}

function Check-Dashboard {
    $url = "http://127.0.0.1:$Port"
    try {
        $resp = Invoke-WebRequest -Uri $url -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            PrintOk "Dashboard: HTTP 200 at $url"
        } else {
            PrintFail "Dashboard: HTTP $($resp.StatusCode) at $url"
        }
    } catch {
        PrintFail "Dashboard unreachable at $url"
    }
}

function Check-Ollama {
    # NOTE: $host is a PowerShell automatic variable - use $ollamaHost instead.
    $ollamaHost = if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST } else { "http://127.0.0.1:11434" }
    try {
        Invoke-WebRequest -Uri "$ollamaHost/api/tags" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop | Out-Null
        PrintOk "Ollama detected at $ollamaHost"
    } catch {
        PrintWarn "Ollama not reachable (optional) at $ollamaHost"
    }
}

function Check-Db ([string]$venvPy) {
    $db = Join-Path $MemoryDir "memory.db"
    if (-not (Test-Path $db)) {
        # install.ps1 creates the directory; the MCP server creates the database on its first start.
        if (Test-Path $MemoryDir) {
            PrintWarn "DB not created yet: $db (the MCP server creates it on its first start)"
        } else {
            PrintFail "DB missing: $db (no memory dir; run install.ps1)"
        }
        return
    }
    if (-not $venvPy) {
        PrintWarn "DB present but venv missing - can't check migrations"
        return
    }
    $code = @"
import sqlite3, sys
try:
    con = sqlite3.connect(r'$db')
    row = con.cursor().execute('SELECT MAX(version) FROM schema_migrations').fetchone()
    print(row[0] if row and row[0] is not None else 'none')
except Exception as exc:
    print(f'err:{exc}')
"@
    $mig = & $venvPy -c $code 2>$null
    if ($mig -match '^err:' -or [string]::IsNullOrEmpty($mig)) {
        PrintWarn "DB migrations unknown: $mig"
    } elseif ($mig -eq 'none') {
        PrintFail "DB has no schema_migrations rows"
    } else {
        PrintOk "DB migrations at version $mig ($db)"
    }
}

# --- Test mode ---
if ($env:DIAG_TEST_MODE -eq "1") {
    @"
total-agent-memory diagnostic (TEST MODE)
==========================================
  OK   OS: Windows (mock)
  OK   Python 3.13 venv: mock
  OK   MCP server module importable
  OK   Scheduled Tasks: 4 (mock)
  OK   Dashboard: HTTP 200 (mock)
  OK   Ollama detected (mock)
  OK   DB migrations at version 42
==========================================
Result: 7/7 passed
"@ | Write-Output
    exit 0
}

# --- Real run ---
Write-Host "total-agent-memory diagnostic"
Write-Host "============================="
$os = Get-OsName
PrintOk "OS: $os ($([Environment]::OSVersion.VersionString))"

$py = Check-Python
Check-McpImport $py
Check-ScheduledTasks
Check-Dashboard
Check-Ollama
Check-Db $py

Write-Host ""
$script:Report | ForEach-Object { Write-Host $_ }
Write-Host ""
if ($script:Failed -eq 0) {
    Write-Host "Result: all checks passed"
} else {
    Write-Host "Result: one or more checks FAILED (see above)"
}
exit $script:Failed
