ALTER TABLE knowledge ADD COLUMN source_format TEXT NOT NULL DEFAULT 'auto'
    CHECK(source_format IN ('auto','conversation'));

CREATE TABLE passage_sources (
    knowledge_id INTEGER PRIMARY KEY REFERENCES knowledge(id) ON DELETE CASCADE,
    fingerprint TEXT NOT NULL,
    model TEXT NOT NULL
);
CREATE TABLE evidence_passages (
    id INTEGER PRIMARY KEY,
    knowledge_id INTEGER NOT NULL REFERENCES passage_sources(knowledge_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    content TEXT NOT NULL,
    vector BLOB,
    UNIQUE(knowledge_id, ordinal)
);
CREATE VIRTUAL TABLE evidence_passages_fts USING fts5(content, content='evidence_passages', content_rowid='id');
CREATE TRIGGER evidence_passages_insert AFTER INSERT ON evidence_passages BEGIN
    INSERT INTO evidence_passages_fts(rowid,content) VALUES(new.id,new.content);
END;
CREATE TRIGGER evidence_passages_delete AFTER DELETE ON evidence_passages BEGIN
    INSERT INTO evidence_passages_fts(evidence_passages_fts,rowid,content) VALUES('delete',old.id,old.content);
END;
CREATE TRIGGER evidence_passages_update AFTER UPDATE ON evidence_passages BEGIN
    INSERT INTO evidence_passages_fts(evidence_passages_fts,rowid,content) VALUES('delete',old.id,old.content);
    INSERT INTO evidence_passages_fts(rowid,content) VALUES(new.id,new.content);
END;
CREATE TRIGGER evidence_source_update AFTER UPDATE OF content,status,project,branch,type,source_format ON knowledge BEGIN
    DELETE FROM passage_sources WHERE knowledge_id=old.id;
END;
CREATE TRIGGER evidence_source_delete AFTER DELETE ON knowledge BEGIN
    DELETE FROM passage_sources WHERE knowledge_id=old.id;
END;
CREATE TRIGGER passage_source_delete AFTER DELETE ON passage_sources BEGIN
    DELETE FROM evidence_passages WHERE knowledge_id=old.knowledge_id;
END;
