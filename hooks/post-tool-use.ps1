# ===========================================
# PostToolUse Hook (PowerShell) — observation capture + memory_save nudges
#
# Twin of hooks/post-tool-use.sh:
#  0. (default on, TAM_ACTION_LOG) log the call in agent_actions
#     (tam_hook.py record-action); synchronous so the Stop hook sees it.
#  1. (always-on) bump per-session counters and, when writes-without-saves
#     crosses a threshold, send a nudge to the model as
#     hookSpecificOutput.additionalContext (lib/tam_hook.py nudge).
#  2. (opt-in) MEMORY_POST_TOOL_CAPTURE=1 enqueues a deferred observation
#     through a detached worker (tam_hook.py capture-observation).
#
# Env:
#   MEMORY_POST_TOOL_CAPTURE  — "1" to enable observation capture
#   MEMORY_NUDGE_DISABLE      — "1" to disable nudges entirely
#   MEMORY_NUDGE_SOFT/HARD/STEP — tune thresholds (defaults: 3/7/3)
#   TAM_MEMORY_DIR / CLAUDE_MEMORY_INSTALL_DIR — resolved by lib/common.ps1
#
# Hook: PostToolUse (matcher: "*"), PostToolUseFailure (matcher: "*")
# ===========================================

. (Join-Path $PSScriptRoot "lib\common.ps1")

Invoke-HookHelper record-action --src $HookSrcDir --memory-dir $MemoryDir

if ($env:MEMORY_NUDGE_DISABLE -ne "1") {
    Invoke-HookHelper nudge --memory-dir $MemoryDir --emit PostToolUse
}

if ($env:MEMORY_POST_TOOL_CAPTURE -ne "1") { exit 0 }

$TmpInput = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "cmm-pthook-$(New-Guid).json")
[System.IO.File]::WriteAllText($TmpInput, $HookInput)
Start-HookWorker @("capture-observation", "--src", $HookSrcDir, "--memory-dir", $MemoryDir, "--input", $TmpInput)
exit 0
