#!/usr/bin/env bash
# ===========================================
# Bash failure hook — v7.0 learn_error trigger
#
# Fires when a Bash call fails (PostToolUseFailure, or PostToolUse with a
# non-zero exit code in the response) and stderr/error text is present.
# Asks the model to call learn_error(...) via additionalContext;
# N>=3 similar patterns auto-consolidate into a rule on the MCP side.
#
# Hook: PostToolUseFailure (matcher: "Bash"), PostToolUse (matcher: "Bash")
# ===========================================

# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"
hook_set_event "PostToolUse"

TOOL=$(hook_get '.tool_name')
[ "$TOOL" != "Bash" ] && exit 0

# Empty or zero -> no error, skip
case "$HOOK_EXIT_CODE" in
    ""|0|null) exit 0 ;;
esac

# Skip user aborts
[ "$HOOK_INTERRUPTED" = "1" ] && exit 0

STDERR="${HOOK_ERROR_TEXT:0:500}"

# Skip noise: user aborts, interactive prompts, benign warnings
case "$STDERR" in
    *"permission denied by user"*|*"User denied"*|*"SIGINT"*) exit 0 ;;
esac

# Only react if there is an actionable signal
[ -z "$STDERR" ] && exit 0

hook_add_context "v7.0 learn_error trigger: bash exited $HOOK_EXIT_CODE. If the root cause is
reproducible and fixable, call:
  learn_error(
      file=\"<path if relevant>\",
      error=\"<short stderr>\",
      root_cause=\"<what actually failed>\",
      fix=\"<what resolves it>\",
      pattern=\"<short slug, e.g. sqlite-locked-during-ddl>\"
  )
Skip if this is user-aborted, interactive, or benign. After N>=3 same patterns
it auto-consolidates into a rule — do not re-log if you just fixed it and the
root cause is identical to an earlier call this turn."

exit 0
