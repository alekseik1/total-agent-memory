ALTER TABLE enrichment_queue ADD COLUMN lease_token TEXT;
ALTER TABLE enrichment_queue ADD COLUMN heartbeat_at TEXT;
ALTER TABLE enrichment_queue ADD COLUMN atomic_only INTEGER NOT NULL DEFAULT 0;
CREATE INDEX idx_eq_lease ON enrichment_queue(status, heartbeat_at);
CREATE INDEX idx_knowledge_neighbors ON knowledge(project, session_id, status, created_at, id);

ALTER TABLE atomic_facts ADD COLUMN observed_at TEXT NOT NULL DEFAULT '';
ALTER TABLE atomic_facts ADD COLUMN event_start TEXT NOT NULL DEFAULT '';
ALTER TABLE atomic_facts ADD COLUMN event_end TEXT NOT NULL DEFAULT '';
ALTER TABLE atomic_facts ADD COLUMN event_precision TEXT NOT NULL DEFAULT 'unknown';
ALTER TABLE atomic_facts ADD COLUMN event_key TEXT NOT NULL DEFAULT '';
ALTER TABLE atomic_facts ADD COLUMN temporal_anchor_at TEXT NOT NULL DEFAULT '';
CREATE INDEX idx_atomic_event_time ON atomic_facts(event_start, event_end);

CREATE TABLE atomic_fact_dependencies (
    target_id INTEGER NOT NULL,
    source_id INTEGER NOT NULL,
    PRIMARY KEY(target_id, source_id)
);
CREATE INDEX idx_atomic_dependency_source ON atomic_fact_dependencies(source_id);
INSERT OR IGNORE INTO atomic_fact_dependencies
    SELECT f.knowledge_id, s.knowledge_id FROM atomic_facts f
    JOIN atomic_fact_sources s ON s.fact_id=f.id;
INSERT OR IGNORE INTO atomic_fact_dependencies SELECT knowledge_id, knowledge_id FROM atomic_fact_runs;
CREATE TABLE atomic_fact_rebuild (
    knowledge_id INTEGER PRIMARY KEY
);
INSERT INTO atomic_fact_rebuild SELECT knowledge_id FROM atomic_fact_runs;
DELETE FROM atomic_fact_runs;

DROP TRIGGER atomic_source_update;
DROP TRIGGER atomic_source_delete;
CREATE TRIGGER atomic_source_update
AFTER UPDATE OF content, status, project, session_id, branch, created_at ON knowledge
WHEN old.content IS NOT new.content OR old.status IS NOT new.status
  OR old.project IS NOT new.project OR old.session_id IS NOT new.session_id
  OR old.branch IS NOT new.branch OR old.created_at IS NOT new.created_at
BEGIN
    INSERT OR IGNORE INTO atomic_fact_rebuild
        SELECT target_id FROM atomic_fact_dependencies WHERE source_id=old.id;
    INSERT OR IGNORE INTO atomic_fact_rebuild
        SELECT knowledge_id FROM atomic_fact_runs WHERE knowledge_id=old.id;
    DELETE FROM atomic_fact_runs WHERE knowledge_id IN (
        SELECT target_id FROM atomic_fact_dependencies WHERE source_id=old.id
    ) OR knowledge_id=old.id;
    DELETE FROM atomic_facts WHERE knowledge_id IN (
        SELECT target_id FROM atomic_fact_dependencies WHERE source_id=old.id
    ) OR knowledge_id=old.id;
END;
CREATE TRIGGER atomic_source_delete AFTER DELETE ON knowledge BEGIN
    INSERT OR IGNORE INTO atomic_fact_rebuild
        SELECT target_id FROM atomic_fact_dependencies WHERE source_id=old.id AND target_id!=old.id;
    DELETE FROM atomic_fact_runs WHERE knowledge_id IN (
        SELECT target_id FROM atomic_fact_dependencies WHERE source_id=old.id
    ) OR knowledge_id=old.id;
    DELETE FROM atomic_facts WHERE knowledge_id IN (
        SELECT target_id FROM atomic_fact_dependencies WHERE source_id=old.id
    ) OR knowledge_id=old.id;
    DELETE FROM atomic_fact_dependencies WHERE target_id=old.id;
    DELETE FROM atomic_fact_rebuild WHERE knowledge_id=old.id;
END;
