# Who wrote a memory (provenance)

Every record a tool call writes gets a row in `memory_provenance` (migration 047): the client agent, its
session id, the workspace relative to the repository root, the repository name and the tool-use / turn ids.
Tool schemas are unchanged; nothing is asked of the model.

Two sources feed it:

| Source | What it carries | Set up by |
|---|---|---|
| `_ctx` in the tool input | agent, session id, repo-relative workspace, repo, tool-use id | the PreToolUse hook `hooks/inject-provenance.*` (matcher `mcp__memory__.*`), registered by `install.sh`, the setup wizard and `install-docker.sh`. The plugin's `hooks.json` does not carry it |
| MCP request `_meta` | `claudecode/toolUseId`, Codex `x-codex-turn-metadata` (session, turn, workspace name), conversation / thread ids | the client, automatically |

The hook sends `_ctx` only to the TAM server. The installers record how they registered the `memory`
server (command and arguments, or URL; never env or headers) in `~/.claude/tam-provenance.json`, and the hook
injects only when the effective `memory` registration, project-scoped and `.mcp.json` overrides included,
matches that marker, or when the tool belongs to the total-agent-memory plugin. A server of another vendor
that is also named `memory` gets nothing. An install made before 14.10.0 has no marker, so the hook stays
silent until `update.sh`, `tam setup --reconfigure` or `install-docker.sh` runs once.

The server strips `_ctx` before it validates, logs or passes on the arguments. The request `_meta` is
authoritative for the tool-use id; the hook is authoritative for the workspace. Both are labels the client
supplies, not authentication. Absolute paths are never stored: a workspace is a path below the repository
root, or empty. Deleting a record deletes its provenance rows (trigger in the migration).

Read it back with `provenance.provenance_for(db, knowledge_id)`.
