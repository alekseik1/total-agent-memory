#!/usr/bin/env bash
# ===========================================
# PreToolUse hook for Write|Edit — v7.0 file_context guard
#
# Asks the model to call file_context(path) BEFORE editing a file.
# Delivered as hookSpecificOutput.additionalContext (plain stdout of
# PreToolUse never reaches the model). No permission decision is made.
#
# Hook: PreToolUse (matcher: "Write|Edit")
# ===========================================

# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
hook_set_event "PreToolUse"

TOOL=$(hook_get '.tool_name')
FILE_PATH=$(hook_get '.tool_input.file_path')

# Only guard Write/Edit, not NotebookEdit
case "$TOOL" in
    Write|Edit) ;;
    *) exit 0 ;;
esac

[ -z "$FILE_PATH" ] && exit 0

# Skip vendored / temp paths
case "$FILE_PATH" in
    */.git/*|*/node_modules/*|*/.venv/*|/tmp/*) exit 0 ;;
esac

hook_add_context "v7.0 pre-edit guard: before editing \`$FILE_PATH\`, call
  file_context(path=\"$FILE_PATH\")
If risk_score > 0.3, read the returned warnings (past errors / hot spots) and
incorporate them into the edit. Skip if file_context was already called for this
path in the current turn."

exit 0
