-- Base schema — the core tables every other migration builds on.
--
-- This file is the single source of truth: `Store._schema()` executes it at
-- startup and the pytest `db` fixture loads the same text, so a test can never
-- pass against a schema production does not have. Before this file existed the
-- fixture hand-rolled its own `knowledge` table, which is how `fact_merger`
-- shipped writing a non-existent `updated_at` column with a green suite.
--
-- Everything here must stay idempotent (`IF NOT EXISTS`): it re-runs on every
-- startup. Lives under `src/sql/` (not `migrations/`) so it ships inside the
-- `src` package-data glob in a wheel install; `_apply_sql_migrations` never
-- sees this file and does not record a "000" row in the migrations table.
-- Columns added to existing databases belong in `Store._migrate()`, not here.

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
    project TEXT DEFAULT 'general', status TEXT DEFAULT 'open',
    summary TEXT, log_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, type TEXT NOT NULL,
    content TEXT NOT NULL, context TEXT DEFAULT '',
    project TEXT DEFAULT 'general', tags TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active', superseded_by INTEGER,
    confidence REAL DEFAULT 1.0, source TEXT DEFAULT 'explicit',
    created_at TEXT NOT NULL, last_confirmed TEXT,
    recall_count INTEGER DEFAULT 0, last_recalled TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS relations (
    from_id INTEGER, to_id INTEGER, type TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS timeline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, ts TEXT NOT NULL,
    event TEXT NOT NULL, summary TEXT NOT NULL,
    details TEXT DEFAULT '', project TEXT DEFAULT 'general', files TEXT DEFAULT '[]'
);
-- Audit trail for semantic fact merges (src/fact_merger.py). One row per
-- merged cluster: which record replaced which sources, and why.
CREATE TABLE IF NOT EXISTS knowledge_merges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    merged_knowledge_id INTEGER NOT NULL,
    source_ids TEXT NOT NULL,
    rationale TEXT,
    created_at TEXT NOT NULL
);
-- Origin session for records synthesized by src/fact_merger.py (never a real
-- user session). Seeded once here, not per-merge, so a merger whose contract
-- is "db: SQLite connection" doesn't also decide session composition, and so
-- the row exists before the first merge rather than being created on demand.
INSERT OR IGNORE INTO sessions (id, started_at, project, status, summary)
VALUES ('fact-merge', '1970-01-01T00:00:00Z', 'general', 'closed',
        'Synthesized fact-merge origin (see knowledge_merges)');

CREATE INDEX IF NOT EXISTS idx_k_status ON knowledge(status);
CREATE INDEX IF NOT EXISTS idx_k_type ON knowledge(type);
CREATE INDEX IF NOT EXISTS idx_k_project ON knowledge(project);
CREATE INDEX IF NOT EXISTS idx_k_session ON knowledge(session_id);
CREATE INDEX IF NOT EXISTS idx_k_last_confirmed ON knowledge(last_confirmed);
CREATE INDEX IF NOT EXISTS idx_rel_from ON relations(from_id);
CREATE INDEX IF NOT EXISTS idx_rel_to ON relations(to_id);
CREATE INDEX IF NOT EXISTS idx_t_session ON timeline(session_id);
CREATE INDEX IF NOT EXISTS idx_s_started ON sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_km_merged ON knowledge_merges(merged_knowledge_id);
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    content, context, tags, content='knowledge', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS k_fts_i AFTER INSERT ON knowledge BEGIN
    INSERT INTO knowledge_fts(rowid,content,context,tags)
    VALUES (new.id,new.content,new.context,new.tags);
END;
CREATE TRIGGER IF NOT EXISTS k_fts_u AFTER UPDATE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts,rowid,content,context,tags)
    VALUES ('delete',old.id,old.content,old.context,old.tags);
    INSERT INTO knowledge_fts(rowid,content,context,tags)
    VALUES (new.id,new.content,new.context,new.tags);
END;
CREATE TABLE IF NOT EXISTS embeddings (
    knowledge_id INTEGER PRIMARY KEY,
    binary_vector BLOB NOT NULL,
    float32_vector BLOB NOT NULL,
    embed_model TEXT NOT NULL,
    embed_dim INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
