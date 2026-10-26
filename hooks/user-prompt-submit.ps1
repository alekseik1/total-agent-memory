# ===========================================
# UserPromptSubmit Hook (PowerShell) - prompt capture + ambient recall
#
# Twin of hooks/user-prompt-submit.sh:
# 1. Caches the hook JSON and hands it to a detached worker
#    (tam_hook.py save-intent) that stores the prompt in `intents`.
# 2. Ambient recall (tam_hook.py prompt-context): matching rules and a short
#    background block as hookSpecificOutput.additionalContext, bounded by
#    TAM_AMBIENT_LATENCY_MS / TAM_AMBIENT_TOKEN_BUDGET, deduped per session.
#
# Env: TAM_AMBIENT_RECALL=false disables step 2
#      TAM_MEMORY_DIR / CLAUDE_MEMORY_INSTALL_DIR - resolved by lib/common.ps1
#
# Hook: UserPromptSubmit (matcher: "")
# ===========================================

. (Join-Path $PSScriptRoot "lib\common.ps1")

# Ambient recall first: it reports "no store" itself, and with TAM_DOCKER=1 the
# store lives in the container, not under the memory dir.
Invoke-HookHelper prompt-context --src $HookSrcDir --memory-dir $MemoryDir --emit UserPromptSubmit

$DbPath = Join-Path $MemoryDir "memory.db"
if (-not (Test-Path $DbPath)) { exit 0 }

$TmpInput = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "cmm-uprompt-$(New-Guid).json")
[System.IO.File]::WriteAllText($TmpInput, $HookInput)
Start-HookWorker @("save-intent", "--src", $HookSrcDir, "--db", $DbPath, "--input", $TmpInput)
exit 0
