#!/usr/bin/env bash
#
# Codex CLI notify hook — captures every finished Codex turn into TAM
#
# Codex runs the notify program with the notification JSON as its last
# argument ({"type":"agent-turn-complete","thread-id":...,"cwd":...,
# "input-messages":[...],"last-assistant-message":...}). Each finished turn
# becomes one `codex_turn` row in the agent action log and, unless
# TAM_TURN_AUTOSAVE=false, one auto-saved turn summary
# (tam_hook.py codex-notify, src/ambient/codex.py). Other notification types
# are ignored. Older wrappers that pipe the JSON on stdin keep working.
#
# Optional; register it in ~/.codex/config.toml:
#   notify = ["/path/to/total-agent-memory/hooks/codex-notify.sh"]
#
# Env: TAM_HOOK_NOTIFY=0 turns the macOS notification off.

if [ $# -gt 0 ]; then
    HOOK_SKIP_STDIN=1
fi
HOOK_PARSE_FIELDS=0
# shellcheck source=lib/common.sh
source "$(dirname "$0")/lib/common.sh"

if [ $# -gt 0 ]; then
    PAYLOAD="${!#}"
else
    PAYLOAD="$HOOK_INPUT"
fi
[ -z "$PAYLOAD" ] && exit 0

hook_helper codex-notify --src "$HOOK_SRC_DIR" --memory-dir "$CLAUDE_MEMORY_DIR" --payload "$PAYLOAD" \
    </dev/null >>"$HOOK_LOG_FILE" 2>&1

case "$PAYLOAD" in
    *'"agent-turn-complete"'*)
        PROJECT=$(basename "$PWD")
        hook_notify "Turn captured in TAM memory" "Total Memory — $PROJECT" "default"
        ;;
esac
exit 0
