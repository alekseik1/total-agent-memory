CREATE TABLE atomic_facts (
    id INTEGER PRIMARY KEY,
    knowledge_id INTEGER NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    temporal_text TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    UNIQUE(knowledge_id, subject, predicate, object, temporal_text)
);
CREATE INDEX idx_atomic_parent ON atomic_facts(knowledge_id);
CREATE TABLE atomic_fact_sources (
    fact_id INTEGER NOT NULL,
    knowledge_id INTEGER NOT NULL,
    quote TEXT NOT NULL,
    PRIMARY KEY(fact_id, knowledge_id)
);
CREATE INDEX idx_atomic_source ON atomic_fact_sources(knowledge_id);
CREATE TABLE atomic_fact_runs (
    knowledge_id INTEGER PRIMARY KEY,
    source_content TEXT NOT NULL,
    fact_count INTEGER NOT NULL,
    model TEXT NOT NULL
);
CREATE VIRTUAL TABLE atomic_facts_fts USING fts5(content, tokenize='unicode61');
CREATE TRIGGER atomic_fact_insert AFTER INSERT ON atomic_facts BEGIN
    INSERT INTO atomic_facts_fts(rowid, content) VALUES(new.id, new.content);
END;
CREATE TRIGGER atomic_fact_delete AFTER DELETE ON atomic_facts BEGIN
    DELETE FROM atomic_facts_fts WHERE rowid=old.id;
    DELETE FROM atomic_fact_sources WHERE fact_id=old.id;
END;
CREATE TRIGGER atomic_fact_update AFTER UPDATE OF content ON atomic_facts BEGIN
    DELETE FROM atomic_facts_fts WHERE rowid=old.id;
    INSERT INTO atomic_facts_fts(rowid, content) VALUES(new.id, new.content);
END;
CREATE TRIGGER atomic_source_update
AFTER UPDATE OF content, status, project, session_id, branch ON knowledge BEGIN
    DELETE FROM atomic_fact_runs WHERE knowledge_id IN (
        SELECT knowledge_id FROM atomic_facts WHERE id IN (
            SELECT fact_id FROM atomic_fact_sources WHERE knowledge_id=old.id
        )
    ) OR knowledge_id=old.id;
    DELETE FROM atomic_facts WHERE id IN (
        SELECT fact_id FROM atomic_fact_sources WHERE knowledge_id=old.id
    ) OR knowledge_id=old.id;
END;
CREATE TRIGGER atomic_source_delete AFTER DELETE ON knowledge BEGIN
    DELETE FROM atomic_fact_runs WHERE knowledge_id IN (
        SELECT knowledge_id FROM atomic_facts WHERE id IN (
            SELECT fact_id FROM atomic_fact_sources WHERE knowledge_id=old.id
        )
    ) OR knowledge_id=old.id;
    DELETE FROM atomic_facts WHERE id IN (
        SELECT fact_id FROM atomic_fact_sources WHERE knowledge_id=old.id
    ) OR knowledge_id=old.id;
END;
