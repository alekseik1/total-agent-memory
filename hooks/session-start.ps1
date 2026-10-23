# ===========================================
# Session Start Hook (PowerShell) — twin of hooks/session-start.sh
#
# Plain stdout of SessionStart is added to the model context.
#   startup / clear : memory hint, SOUL rules, error patterns, project
#                     knowledge count
#   compact / resume: memory hint + latest session_continuity summary,
#                     next steps, pitfalls and top active rules (bounded)
#   startup / resume: newest recovery file of a previous session in the
#                     same directory (consumed once; stale ones removed)
#
# Hook: SessionStart (matcher: "" or "startup|resume|clear|compact")
# ===========================================

. (Join-Path $PSScriptRoot "lib\common.ps1")

$Cwd = Get-HookField 'cwd'
if (-not $Cwd) { $Cwd = (Get-Location).Path }
$Project = Get-HookProject
$Source = Get-HookField 'source'
$SessionId = Get-HookField 'session_id'
if (-not $SessionId) { $SessionId = if ($env:CLAUDE_SESSION_ID) { $env:CLAUDE_SESSION_ID } else { "unknown" } }

$Branch = ""
try { $Branch = (git -C $Cwd rev-parse --abbrev-ref HEAD 2>$null) } catch { $Branch = "" }

$SourceText = switch ($Source) {
    "startup" { "New session" }
    "resume"  { "Resumed" }
    "clear"   { "After /clear" }
    "compact" { "After /compact" }
    default   { "Started" }
}
$Ctx = if ($Branch) { "$Project @ $Branch" } else { $Project }
Write-HookLog "SESSION_START: $Ctx | $SourceText"

Write-Output "Session: $Ctx | $SourceText"
Write-Output ""

$Hint = "MEMORY_HINT: Project: $Project"
if ($Branch) { $Hint += ", Branch: $Branch" }
$Hint += ". Use memory_recall(query=`"your task`", project=`"$Project`") to search past knowledge."
$Hint += " Also run self_rules_context(project=`"$Project`") to load behavioral rules."
Write-Output $Hint

# "--opt=value" keeps empty values: Windows PowerShell 5.1 drops empty native arguments.
$Context = Invoke-HookHelper session-start-context "--db=$(Join-Path $MemoryDir 'memory.db')" `
    "--project=$Project" "--cwd=$Cwd" "--session=$SessionId" "--source=$Source" "--recovery-dir=$HookRecoveryDir"
if ($Context) {
    Write-Output ""
    $Context | ForEach-Object { Write-Output $_ }
}

# Remove error-fix states older than 1 hour
if (Test-Path $HookStateDir) {
    Get-ChildItem -Path $HookStateDir -Filter "last-error-*" -File |
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddHours(-1) } |
        Remove-Item -ErrorAction SilentlyContinue
}
exit 0
