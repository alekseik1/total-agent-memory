-- Who wrote each memory (src/provenance.py). One row per record a tool call wrote:
-- the client agent, its session, the workspace relative to the repository root and the
-- tool-use / turn ids the client sent. The data comes from the PreToolUse hook (`_ctx`
-- injected into the tool input) and from the MCP request _meta; both are advisory labels
-- supplied by the client, not authentication.
-- Additive: no existing table changes; the trigger removes a record's rows when the record
-- itself is deleted (hard delete / erasure), so provenance never outlives the memory.
CREATE TABLE IF NOT EXISTS memory_provenance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    knowledge_id INTEGER NOT NULL,
    tool TEXT NOT NULL DEFAULT '',
    agent TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    workspace TEXT NOT NULL DEFAULT '',
    repo TEXT NOT NULL DEFAULT '',
    tool_use_id TEXT NOT NULL DEFAULT '',
    turn_id TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '' CHECK (source IN ('', 'hook', 'meta', 'hook+meta')),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_provenance_knowledge ON memory_provenance(knowledge_id);
CREATE INDEX IF NOT EXISTS idx_memory_provenance_session ON memory_provenance(session_id, created_at);

CREATE TRIGGER IF NOT EXISTS trg_memory_provenance_cascade
AFTER DELETE ON knowledge
BEGIN
    DELETE FROM memory_provenance WHERE knowledge_id = OLD.id;
END;
