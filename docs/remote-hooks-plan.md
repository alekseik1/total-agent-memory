# Remote hooks mode: plan (phase 1 of moving TAM memory off the MacBook)

Owner decisions (2026-11-08): host = vps-layer7 over Tailscale; hooks = HTTP mode in this fork; offsite backup = PC only; `MEMORY_CROSS_RERANK=off`, index recall `limit=30`; config in `alekseik1/agent-config` with symlinks and auto pull/push.

Goal: with `TAM_REMOTE_URL` set, hooks and MCP on the laptop never touch `memory.db`; all DB work goes through the HTTP server. Without it, local mode is unchanged.

## Verified facts
| Fact | Where |
|---|---|
| HTTP app routes: `/healthz` + `Mount("/mcp")`; bearer guard wraps only `/mcp` | `src/server.py:9407,9419-9426` |
| Serialisation seam exists: Docker mode sends `HookCall` JSON | `hooks/lib/tam_hook.py:1284`, `src/ambient/hooks.py:40,170` |
| Route to copy: daemon `POST /internal/hook/recall` | `src/tam_daemon/service.py:426-475` |
| Stop reads `agent_actions`; record-action kept synchronous for ordering | `hooks/post-tool-use.sh:10` |
| Tool calls block the server event loop | `src/server.py:9573` |
| Server session id comes from its own env `CLAUDE_CODE_SESSION_ID` | `src/server.py:9324` |
| Claude Code exports `CLAUDE_CODE_SESSION_ID` to child processes; `headersHelper` runs per MCP connection | `env`, code.claude.com/docs/en/mcp |
| Transcripts: median 346 KB, max 9.9 MB; home uplink ~40 KB/s -> extract on laptop | measured; `src/auto_episode_capture.py:82-141` |
| rules-harvest edits rules with raw `UPDATE rules`; no MCP action | `~/.claude/skills/rules-harvest/SKILL.md:109` |

## Slices (one branch + PR each)
| # | What | Files | ~lines |
|---|---|---|---|
| S1 server | `POST /internal/hook/{cmd}` behind bearer guard, run via `asyncio.to_thread`; cmds `session-start`, `prompt-context`, `save-intent`, `actions-batch`, `turn-end`, `session-end`, `error-capture`; accepts pre-extracted text, never reads files; header `X-TAM-Session` -> per-request session id (contextvar) instead of env; LLM work of session-end in a background thread | `src/server.py`, new `src/hook_endpoint.py`, tests | 300 |
| S2 client | `tam_hook.py` remote mode when `TAM_REMOTE_URL` set: no server imports; git branch, transcript text (<=64 KB), tool summary (<=400 chars) built on laptop; post-tool-use only appends to spool `~/.cache/tam/spool/<sid>.jsonl`; Stop/UserPromptSubmit/SessionEnd flush spool first (order kept); network down -> spool kept; sync hooks timeout 3 s -> empty context, exit 0; `hooks/tam-headers` for `headersHelper`: token from Keychain/`secret-tool` + `X-TAM-Session` | `hooks/lib/tam_hook.py`, `hooks/*.sh`, new `hooks/tam-headers`, tests | 400 |
| S3 rules | `self_rules(action="update", id, content)` for rules-harvest | `src/server.py`, test | 100 |
| S4 agent-config | `countActiveRules` -> HTTP; `delivery-smoke` -> HTTP (else every agent-config commit fails); rules-harvest -> `self_rules`; drop sqlite fallbacks in foreman-coder/test-fixer; move `review-roi`, `review-backtest`, `usage-guard` from `~/.tam` to `agent-config/data/` | 12 sites, see direct-db-users.md | 150 |

Out of scope (phase 2): VPS deploy, reflection/dashboard services, VPS->PC backup, `OLLAMA_URL`.

## Acceptance
1. E2E: a Claude session with `TAM_REMOTE_URL` -> local HTTP server on a DB copy; start -> prompts -> tools -> stop -> end yields the same rows as local mode: `session_summaries`, ordered `agent_actions`, `intents`, turn summary in `knowledge`, `episodes`.
2. Hook process imports no `server`/`memory_core` (test).
3. post-tool-use p50 < 30 ms; prompt-context <= 300 ms + RTT.
4. Server down -> every hook exits 0 within timeout; spool survives and the next flush delivers it.
5. MCP calls in remote mode record the same session id as hooks.

## Tests
`.venv/bin/pytest` on new/touched hook and server modules; E2E run above with sqlite row comparison; foreman-review T2 on S1/S2, T1 on S3/S4.

## Rollback
Unset `TAM_REMOTE_URL` -> old local mode. Slices merge independently; S1 and S3 are additive.

## Risks
- Does `headersHelper` see `CLAUDE_CODE_SESSION_ID`? Check first in S2; fallback: SessionStart hook binds session_id to `mcp-session-id`.
- Live checkout: every new Claude session runs `server.py` and hooks from `~/claude-memory-server` -> implementation must happen in a worktree, not in the live tree.
