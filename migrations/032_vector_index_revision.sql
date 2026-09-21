CREATE TABLE IF NOT EXISTS vector_index_revision (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    revision INTEGER NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO vector_index_revision(singleton, revision) VALUES (1, 0);

CREATE TRIGGER IF NOT EXISTS vector_revision_embedding_insert AFTER INSERT ON embeddings
BEGIN UPDATE vector_index_revision SET revision = revision + 1 WHERE singleton = 1; END;
CREATE TRIGGER IF NOT EXISTS vector_revision_embedding_update AFTER UPDATE ON embeddings
BEGIN UPDATE vector_index_revision SET revision = revision + 1 WHERE singleton = 1; END;
CREATE TRIGGER IF NOT EXISTS vector_revision_embedding_delete AFTER DELETE ON embeddings
BEGIN UPDATE vector_index_revision SET revision = revision + 1 WHERE singleton = 1; END;
CREATE TRIGGER IF NOT EXISTS vector_revision_knowledge_insert AFTER INSERT ON knowledge
BEGIN UPDATE vector_index_revision SET revision = revision + 1 WHERE singleton = 1; END;
CREATE TRIGGER IF NOT EXISTS vector_revision_knowledge_delete AFTER DELETE ON knowledge
BEGIN UPDATE vector_index_revision SET revision = revision + 1 WHERE singleton = 1; END;
CREATE TRIGGER IF NOT EXISTS vector_revision_knowledge_update
AFTER UPDATE OF id, status, project, type, branch ON knowledge
BEGIN UPDATE vector_index_revision SET revision = revision + 1 WHERE singleton = 1; END;
