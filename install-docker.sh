#!/usr/bin/env bash
#
# total-agent-memory — Docker Installer
#
# Registers the Dockerized MCP server in ~/.claude.json (where Claude Code reads
# MCP servers) and adds hooks in Docker-exec mode to ~/.claude/settings.json.
# Existing hooks are kept, both files are backed up, invalid JSON aborts the run.
#
# Prerequisites:
#   - Docker Engine / Docker Desktop running
#   - `docker compose up -d --build` already executed in this directory
#     (or will be executed by --with-compose flag below)
#
# Usage:
#   bash install-docker.sh                    # assumes stack is already up
#   bash install-docker.sh --with-compose     # also runs docker compose up -d --build
#
set -euo pipefail

echo ""
echo "======================================================="
echo "  total-agent-memory v12.0 — Docker Installer"
echo "======================================================="
echo ""

INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"

WITH_COMPOSE=0
for arg in "$@"; do
    case "$arg" in
        --with-compose) WITH_COMPOSE=1 ;;
    esac
done

# -- 1. Docker sanity --
echo "-> Step 1: Checking Docker..."
if ! command -v docker &>/dev/null; then
    echo "  ERROR: docker not found. Install Docker Desktop or Docker Engine first."
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "  ERROR: Docker daemon not reachable. Start Docker Desktop / systemctl start docker."
    exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
    echo "  ERROR: 'docker compose' plugin missing. Upgrade Docker Desktop or install compose-v2."
    exit 1
fi
echo "  OK: Docker ready"

# -- 2. Optional: build & start stack --
if [ "$WITH_COMPOSE" -eq 1 ]; then
    echo "-> Step 2: Building and starting the stack..."
    docker compose -f "$INSTALL_DIR/docker-compose.yml" up -d --build
    echo "  OK: Stack is up"
else
    echo "-> Step 2: Skipped (run 'docker compose up -d --build' yourself)."
fi

# -- 3. Wait for MCP healthz --
echo "-> Step 3: Waiting for MCP HTTP endpoint (http://127.0.0.1:3737)..."
ATTEMPTS=0
until curl -fsS http://127.0.0.1:3737/healthz >/dev/null 2>&1; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ "$ATTEMPTS" -gt 60 ]; then
        echo "  WARN: MCP did not become healthy within 60s."
        echo "        Check: docker compose logs mcp"
        break
    fi
    sleep 1
done
[ "$ATTEMPTS" -le 60 ] && echo "  OK: MCP responding"

# -- 4. Access token (the container's servers bind beyond loopback and require it) --
echo "-> Step 4: Reading the MCP access token from the container..."
CONTAINER="${TAM_CONTAINER:-tam-mcp}"
MCP_URL="${TAM_MCP_URL:-http://127.0.0.1:3737/mcp}"
MCP_TOKEN="${TAM_HTTP_TOKEN:-}"
if [ -z "$MCP_TOKEN" ]; then
    MCP_TOKEN="$(docker exec "$CONTAINER" cat /data/state/http-token 2>/dev/null || true)"
fi
if [ -z "$MCP_TOKEN" ]; then
    echo "  WARN: no access token found in container '$CONTAINER' (/data/state/http-token)."
    echo "        The MCP entry is written without one; set TAM_HTTP_TOKEN and re-run if /mcp shows 401."
fi

# -- 5. Register hooks in Docker-exec mode --
echo "-> Step 5: Registering hooks (Docker mode)..."

HOOK_SESSION="$INSTALL_DIR/hooks/session-start.sh"
HOOK_SESSION_END="$INSTALL_DIR/hooks/session-end.sh"
HOOK_STOP="$INSTALL_DIR/hooks/on-stop.sh"
HOOK_BASH="$INSTALL_DIR/hooks/memory-trigger.sh"
HOOK_WRITE="$INSTALL_DIR/hooks/auto-capture.sh"
# Ambient memory: prompt recall, rule reminders, the agent action log and Codex
# turns. They run `docker exec -i <container> python -m ambient ...` because the
# store lives in the container's volume.
HOOK_PROMPT="$INSTALL_DIR/hooks/user-prompt-submit.sh"
HOOK_RULES="$INSTALL_DIR/hooks/rule-reminder.sh"
HOOK_ACTIONS="$INSTALL_DIR/hooks/post-tool-use.sh"
HOOK_CODEX="$INSTALL_DIR/hooks/codex-notify.sh"
# Provenance: adds _ctx (agent, session, workspace) to mcp__memory__* tool input; payload-only, no docker exec.
HOOK_PROVENANCE="$INSTALL_DIR/hooks/inject-provenance.sh"

# Hooks read TAM_DOCKER=1 at runtime and switch to `docker exec`.
# We wrap each via a small launcher that exports the env var.
# Legacy CLAUDE_MEMORY_DOCKER / CLAUDE_MEMORY_CONTAINER also exported for
# hooks that haven't been updated yet (see hooks/lib/common.sh).
LAUNCHER_DIR="$INSTALL_DIR/hooks/_docker-launchers"
mkdir -p "$LAUNCHER_DIR"
for src in "$HOOK_SESSION" "$HOOK_SESSION_END" "$HOOK_STOP" "$HOOK_BASH" "$HOOK_WRITE" \
           "$HOOK_PROMPT" "$HOOK_RULES" "$HOOK_ACTIONS" "$HOOK_CODEX" "$HOOK_PROVENANCE"; do
    name=$(basename "$src")
    dest="$LAUNCHER_DIR/$name"
    cat > "$dest" <<LAUNCHER
#!/usr/bin/env bash
export TAM_DOCKER=1
export TAM_CONTAINER="\${TAM_CONTAINER:-tam-mcp}"
# Legacy aliases (kept until all hook scripts switch over)
export CLAUDE_MEMORY_DOCKER="\${CLAUDE_MEMORY_DOCKER:-\$TAM_DOCKER}"
export CLAUDE_MEMORY_CONTAINER="\${CLAUDE_MEMORY_CONTAINER:-\$TAM_CONTAINER}"
exec "$src" "\$@"
LAUNCHER
    chmod +x "$dest"
done

# One merge for hooks and MCP: refuses on invalid JSON, backs files up first, never removes user hooks.
echo "-> Step 5b: Registering MCP server and hooks (settings are backed up first)..."
if ! printf '%s' "$MCP_TOKEN" | PYTHONPATH="$INSTALL_DIR/src" python3 -m setup_wizard.docker_install \
        --home "$HOME" --launcher-dir "$LAUNCHER_DIR" --url "$MCP_URL" --token-stdin; then
    echo "  ERROR: Claude Code configuration was not changed (see the message above)."
    exit 1
fi

# Codex CLI has its own config; the installer only prepares the launcher.
echo "  Codex CLI (optional): add to ~/.codex/config.toml"
echo "    notify = [\"$LAUNCHER_DIR/codex-notify.sh\"]"

# -- 6. Summary --
echo ""
echo "======================================================="
echo ""
echo "  INSTALLED (Docker mode)"
echo ""
echo "  Services:"
echo "    MCP          http://127.0.0.1:3737/mcp"
echo "    Dashboard    http://127.0.0.1:37737/login?token=<contents of /data/state/http-token>"
echo "    Ollama       http://127.0.0.1:11434"
echo ""
echo "  Management:"
echo "    Logs:        docker compose logs -f"
echo "    Restart:     docker compose restart"
echo "    Stop:        docker compose down"
echo "    Wipe data:   docker compose down -v"
echo ""
echo "  Restart Claude Code, then /mcp → 'memory' should show Connected."
echo ""
echo "======================================================="
