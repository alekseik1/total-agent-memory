#!/usr/bin/env bash
# ===========================================
# PreToolUse Hook — provenance for memory writes
#
# Adds `_ctx` {agent, session_id, workspace (repo-relative), repo, tool_use_id} to the input of
# mcp__memory__* tool calls through hookSpecificOutput.updatedInput. The TAM server strips `_ctx`
# before validating the arguments and records it next to each memory the call wrote
# (src/provenance.py, table memory_provenance). Tool schemas are unchanged.
#
# Registered by the installers (install.sh / setup wizard), not by the plugin's hooks.json.
# Hook: PreToolUse (matcher: "mcp__memory__.*")
# ===========================================

HOOK_PARSE_FIELDS=0
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"

printf '%s' "$HOOK_INPUT" | hook_helper inject-provenance
exit 0
