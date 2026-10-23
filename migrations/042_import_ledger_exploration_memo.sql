-- Import ledger: one row per unit `tam import` stored (a chat turn, a note
-- section, a mem0 memory, a bundle record). The key is a SHA-256 over the
-- importer kind, the source text and its original timestamp, so running the
-- same import again skips what is already in the store instead of relying on
-- near-duplicate detection. knowledge_id points at the stored (or the
-- pre-existing duplicate) record.
CREATE TABLE IF NOT EXISTS import_ledger (
    kind TEXT NOT NULL,
    project TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    knowledge_id INTEGER,
    source_ref TEXT NOT NULL DEFAULT '',
    imported_at TEXT NOT NULL,
    PRIMARY KEY (kind, project, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_import_ledger_knowledge ON import_ledger(knowledge_id);

-- Exploration memos: a conclusion an agent reached about a codebase question,
-- keyed to the SHA-256 of every file it read. memory_exploration_recall
-- re-hashes those files and reports the memo as fresh (all byte-identical)
-- or stale (lists the changed and missing files). One memo per
-- (project, normalised question); recording the question again replaces it.
CREATE TABLE IF NOT EXISTS exploration_memos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project TEXT NOT NULL,
    question TEXT NOT NULL,
    question_hash TEXT NOT NULL,
    conclusion TEXT NOT NULL,
    root TEXT NOT NULL,
    question_vector BLOB,
    vector_model TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (project, question_hash)
);
CREATE INDEX IF NOT EXISTS idx_exploration_memos_project ON exploration_memos(project, updated_at DESC);

-- Files are stored relative to the project root (POSIX separators) with the
-- content hash and size taken at record time.
CREATE TABLE IF NOT EXISTS exploration_memo_files (
    memo_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size INTEGER NOT NULL,
    PRIMARY KEY (memo_id, path)
);
