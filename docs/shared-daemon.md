# Shared warm daemon (`tam daemon` / `tam proxy`)

Opt-in. By default every MCP client session (each Claude Code window, each
Codex run) spawns its own TAM process, and each one loads the embedding
model: a cold start per session and N copies in RAM. The shared daemon runs
**one** warm TAM server per memory directory; clients launch the light
`tam proxy`, which relays their stdio MCP traffic to it.

Measured on an M-series Mac, fastembed model already downloaded, median of 5:

| Launch command                  | `initialize` answered | first `memory_recall` done |
|---------------------------------|-----------------------|----------------------------|
| direct stdio server (default)   | 1.47 s                | 1.51 s                     |
| `tam proxy`, no daemon yet      | 0.81 s                | 1.67 s (spawns the daemon) |
| `tam proxy`, daemon running     | 0.15 s                | 0.18 s                     |

## Turning it on

```bash
tam setup register --client claude-code --client codex --shared-daemon
```

This writes `command = <tam console script>`, `args = ["proxy"]` (or
`python <install>/src/tam_daemon proxy` for a git checkout) into each
client's config, keeping `TAM_MEMORY_DIR` and the other env. Nothing else
changes; to go back, register again without `--shared-daemon`. Re-running the
interactive `tam setup` wizard writes the default (direct) entry again.

By hand, the entry is just the usual one with a `proxy` argument:

```jsonc
// ~/.claude.json  (Claude Code), mcpServers.memory
{ "command": "/path/to/venv/bin/total-agent-memory", "args": ["proxy"],
  "env": { "TAM_MEMORY_DIR": "/Users/me/.tam" } }
```

```toml
# ~/.codex/config.toml
[mcp_servers.memory]
command = "/path/to/venv/bin/total-agent-memory"
args = ["proxy"]
[mcp_servers.memory.env]
TAM_MEMORY_DIR = "/Users/me/.tam"
```

## What the proxy does

* Reuses the memory dir's running daemon, or starts one detached and waits
  for `/internal/health` (at most `TAM_DAEMON_START_TIMEOUT_SEC`).
* A daemon of a **different** TAM version (you upgraded) is asked to shut
  down and a fresh one is started. Only at proxy start-up: a proxy that
  loses its daemon mid-session reconnects to whichever version runs, so two
  installs sharing one memory dir never take turns restarting it.
* If no daemon can be started at all, the proxy runs the ordinary
  in-process stdio server instead — the client always works.
* If the daemon goes away mid-session (idle exit, `tam daemon stop`, an
  upgrade) the next request restarts it transparently; when that fails the
  proxy switches to a private stdio server and replays the handshake.
* Frames are relayed raw, so tool lists, `structuredContent`, `isError` and
  protocol errors pass through unchanged, for both the initialize-handshake
  protocol revisions and the 2026-07-28 per-request envelope.

## Managing the daemon

```bash
tam daemon start            # background; reports the existing one when running
tam daemon start --foreground   # same as `tam daemon serve`, for launchd/systemd
tam daemon status [--json]  # pid, port, version, uptime, RSS, memory dir, warm flags
tam daemon stop             # graceful: drains requests, flushes background work
tam daemon restart
```

`status` exits 0 when running and 3 when not (LSB convention). Without the
console script use `python <install>/src/tam_daemon daemon <command>`.

| Setting                        | Default | Meaning                                        |
|--------------------------------|---------|------------------------------------------------|
| `TAM_DAEMON_PORT`              | `0`     | loopback port; 0 = pick a free one             |
| `TAM_DAEMON_IDLE_EXIT_MIN`     | `60`    | exit after this many idle minutes; 0 = never   |
| `TAM_DAEMON_START_TIMEOUT_SEC` | `30`    | how long starters wait for `/internal/health`  |
| `TAM_DAEMON_STOP_TIMEOUT_SEC`  | `20`    | graceful stop before SIGTERM, then SIGKILL     |

The daemon inherits the environment of whoever started it (`MEMORY_MODE`,
provider keys, …). Clients sharing a memory dir share one configuration;
start it yourself (`tam daemon start` with the env you want) if your clients
disagree. All clients also share one TAM session id (`memory_extract_session`
and the raw call log see one session for the daemon's lifetime).

## Files (`<memory dir>/daemon/`, directory 0700)

| File           | Purpose                                                                  |
|----------------|--------------------------------------------------------------------------|
| `daemon.lock`  | OS lock held by the running daemon — the singleton and liveness check    |
| `start.lock`   | serialises starters so concurrent clients spawn one daemon               |
| `daemon.json`  | registry: pid, host, port, version, started_at, memory_dir, python       |
| `daemon.token` | bearer token, 0600, created on first start and reused                    |
| `daemon.log`   | stderr of detached daemons (truncated at start past 5 MiB)               |

A registry whose lock is free is stale (crashed daemon) and is removed by the
next `status`/`start`/proxy.

## HTTP surface

Bound to `127.0.0.1` only. Every route requires
`Authorization: Bearer <daemon.token>` (401 otherwise) and a loopback
`Host`/`Origin` (421/403 otherwise — the same DNS-rebinding rules as
`MCP_TRANSPORT=http`).

### `POST|GET|DELETE /mcp`

MCP streamable HTTP, stateless sessions. `initialize`, `ping` and
`tools/list` answer immediately; tool calls wait for the store, which is
built and pre-warmed (embedding model + first inference, cross-encoder load
when enabled) on a background thread right after the socket is bound.

### `GET /internal/health`

```json
{"status": "starting|ready|failed", "version": "14.8.0", "pid": 123, "port": 50123,
 "memory_dir": "/Users/me/.tam", "uptime_sec": 12.3, "started_at": 1791896412.7,
 "idle_exit_min": 60.0, "in_flight": 0, "error": null,
 "warm": {"store": true, "embedding": true, "reranker": "disabled|pending|loading|ready"}}
```

Health probes do not count as activity for the idle timer.

### `POST /internal/hook/recall`

Warm per-prompt recall for client hooks. The bundled UserPromptSubmit hook
(`hooks/user-prompt-submit.{sh,ps1}`, `src/ambient/prompt_context.py`) calls it
on every prompt when a daemon is registered, and falls back to FTS-only recall
when none answers within `TAM_AMBIENT_LATENCY_MS` (see the ambient section of
[configuration.md](configuration.md)). Request (unknown keys → 400):

```json
{"query": "how do we rotate credentials",   // required, non-empty; cut to 4000 chars
 "project": "api",                           // optional string|null
 "session_id": "claude-session-uuid",        // optional string|null
 "budget_chars": 2000,                       // optional, 200..20000
 "limit": 8,                                 // optional, 1..20
 "dedupe": true}                             // optional; skip records this session already got
```

Response 200:

```json
{"ok": true,
 "text": "[TAM memory: records relevant to this prompt (project api)]\n- (fact #12, api, 2026-10-13) Credentials rotate every 30 days",
 "count": 1, "ids": [12], "skipped_seen": 0, "truncated": false, "elapsed_ms": 7}
```

`text` is empty when nothing (new) is relevant. Errors are
`{"ok": false, "error": "..."}` with 400 / 401 / 413 / 503 (store still
warming after 10 s) / 500.

```bash
TOKEN=$(cat ~/.tam/daemon/daemon.token)
PORT=$(tam daemon status --json | python -c 'import json,sys; print(json.load(sys.stdin)["port"])')
curl -s -H "Authorization: Bearer $TOKEN" -d '{"query":"deploy steps","project":"api"}' \
  http://127.0.0.1:$PORT/internal/hook/recall
```

### `POST /internal/shutdown`

`202 {"ok": true, "pid": 123}`; the daemon drains in-flight requests, stops
the enrichment worker (its queue is persisted), closes the caches and the
database, removes the registry and exits.

## Metrics

The daemon bumps the in-process telemetry counters `memory_perf_report`
shows: `daemon_http_requests`, `daemon_auth_rejected`,
`daemon_requests_gated`, `daemon_hook_recall_requests`,
`daemon_hook_recall_errors`, `daemon_shutdown_requests`, and latency
histograms `daemon_http_ms`, `daemon_hook_recall_ms`, `daemon_bootstrap_ms`,
`daemon_prewarm_embed_ms` (`*_count`, `*_bucket_le_*`).
