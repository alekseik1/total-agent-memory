-- The vector cache reloaded every pool after any write: one saved record made the
-- next unscoped recall on a 1M-record store re-read a million binary vectors
-- (about 5 s instead of 0.35 s). Each revision bump now also logs which record
-- changed, so the cache can patch its pools with just those rows.
-- The log keeps the last 50,000 changes (the DELETE in each trigger below); a
-- cache older than that reloads as before.

CREATE TABLE IF NOT EXISTS vector_changes (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    revision INTEGER NOT NULL,
    knowledge_id INTEGER NOT NULL
);
-- The cache reads the log by revision range; without this it scanned all rows.
CREATE INDEX IF NOT EXISTS idx_vector_changes_revision ON vector_changes(revision);

DROP TRIGGER IF EXISTS vector_revision_embedding_insert;
DROP TRIGGER IF EXISTS vector_revision_embedding_update;
DROP TRIGGER IF EXISTS vector_revision_embedding_delete;
DROP TRIGGER IF EXISTS vector_revision_knowledge_insert;
DROP TRIGGER IF EXISTS vector_revision_knowledge_delete;
DROP TRIGGER IF EXISTS vector_revision_knowledge_update;

CREATE TRIGGER vector_revision_embedding_insert AFTER INSERT ON embeddings BEGIN
    UPDATE vector_index_revision SET revision = revision + 1 WHERE singleton = 1;
    INSERT INTO vector_changes(revision, knowledge_id)
        SELECT revision, NEW.knowledge_id FROM vector_index_revision WHERE singleton = 1;
    DELETE FROM vector_changes WHERE seq <= (SELECT MAX(seq) FROM vector_changes) - 50000;
END;
CREATE TRIGGER vector_revision_embedding_update AFTER UPDATE ON embeddings BEGIN
    UPDATE vector_index_revision SET revision = revision + 1 WHERE singleton = 1;
    INSERT INTO vector_changes(revision, knowledge_id)
        SELECT revision, OLD.knowledge_id FROM vector_index_revision WHERE singleton = 1;
    INSERT INTO vector_changes(revision, knowledge_id)
        SELECT revision, NEW.knowledge_id FROM vector_index_revision WHERE singleton = 1;
    DELETE FROM vector_changes WHERE seq <= (SELECT MAX(seq) FROM vector_changes) - 50000;
END;
CREATE TRIGGER vector_revision_embedding_delete AFTER DELETE ON embeddings BEGIN
    UPDATE vector_index_revision SET revision = revision + 1 WHERE singleton = 1;
    INSERT INTO vector_changes(revision, knowledge_id)
        SELECT revision, OLD.knowledge_id FROM vector_index_revision WHERE singleton = 1;
    DELETE FROM vector_changes WHERE seq <= (SELECT MAX(seq) FROM vector_changes) - 50000;
END;
CREATE TRIGGER vector_revision_knowledge_insert AFTER INSERT ON knowledge BEGIN
    UPDATE vector_index_revision SET revision = revision + 1 WHERE singleton = 1;
    INSERT INTO vector_changes(revision, knowledge_id)
        SELECT revision, NEW.id FROM vector_index_revision WHERE singleton = 1;
    DELETE FROM vector_changes WHERE seq <= (SELECT MAX(seq) FROM vector_changes) - 50000;
END;
CREATE TRIGGER vector_revision_knowledge_delete AFTER DELETE ON knowledge BEGIN
    UPDATE vector_index_revision SET revision = revision + 1 WHERE singleton = 1;
    INSERT INTO vector_changes(revision, knowledge_id)
        SELECT revision, OLD.id FROM vector_index_revision WHERE singleton = 1;
    DELETE FROM vector_changes WHERE seq <= (SELECT MAX(seq) FROM vector_changes) - 50000;
END;
CREATE TRIGGER vector_revision_knowledge_update
AFTER UPDATE OF id, status, project, type, branch ON knowledge BEGIN
    UPDATE vector_index_revision SET revision = revision + 1 WHERE singleton = 1;
    INSERT INTO vector_changes(revision, knowledge_id)
        SELECT revision, OLD.id FROM vector_index_revision WHERE singleton = 1;
    INSERT INTO vector_changes(revision, knowledge_id)
        SELECT revision, NEW.id FROM vector_index_revision WHERE singleton = 1;
    DELETE FROM vector_changes WHERE seq <= (SELECT MAX(seq) FROM vector_changes) - 50000;
END;
