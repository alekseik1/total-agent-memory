# ===========================================
# PreToolUse Hook (PowerShell) - provenance for memory writes
#
# Twin of hooks/inject-provenance.sh: adds `_ctx` {agent, session_id, workspace, repo, tool_use_id}
# to the input of mcp__memory__* tool calls (hookSpecificOutput.updatedInput); the TAM server strips
# and records it (src/provenance.py).
#
# Hook: PreToolUse (matcher: "mcp__memory__.*")
# ===========================================

. (Join-Path $PSScriptRoot "lib\common.ps1")

Invoke-HookHelper inject-provenance
exit 0
