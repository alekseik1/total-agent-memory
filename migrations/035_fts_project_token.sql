-- Project-scoped full-text search at 1M records spent its time ranking every
-- match in the store before the project filter: a word common across projects
-- ("хран*", 139k records) was scored 139k times for a project holding 5k.
-- knowledge_fts gains a fifth column, fts_project: one token per project
-- ("p" + hex of its UTF-8 bytes), read from a generated column of knowledge.
-- A scoped search ANDs that token into MATCH, so FTS5 intersects doclists.
-- The token cannot collide with a word; scoped bm25() gives the column weight 0.
--
-- The update trigger now fires only when an indexed column changes. It used to
-- re-index a row on every UPDATE, including the recall counter bumped for each
-- result of every recall. The missing delete trigger is added.
--
-- The index is rebuilt from knowledge: about 30 s per million records (M2 Max).
-- Keep this DDL identical to memory_core/fts_schema.py (tests/test_fts_schema.py).

ALTER TABLE knowledge ADD COLUMN fts_project TEXT
    GENERATED ALWAYS AS ('p' || lower(hex(project))) VIRTUAL;

DROP TRIGGER IF EXISTS k_fts_i;
DROP TRIGGER IF EXISTS k_fts_u;
DROP TRIGGER IF EXISTS k_fts_d;
DROP TABLE IF EXISTS knowledge_fts;

CREATE VIRTUAL TABLE knowledge_fts USING fts5(
    content, context, tags, fts_project, content='knowledge', content_rowid='id'
);
CREATE TRIGGER k_fts_i AFTER INSERT ON knowledge BEGIN
    INSERT INTO knowledge_fts(rowid, content, context, tags, fts_project)
    VALUES (new.id, new.content, new.context, new.tags, new.fts_project);
END;
CREATE TRIGGER k_fts_u AFTER UPDATE OF content, context, tags, project ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, content, context, tags, fts_project)
    VALUES ('delete', old.id, old.content, old.context, old.tags, old.fts_project);
    INSERT INTO knowledge_fts(rowid, content, context, tags, fts_project)
    VALUES (new.id, new.content, new.context, new.tags, new.fts_project);
END;
CREATE TRIGGER k_fts_d AFTER DELETE ON knowledge BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, content, context, tags, fts_project)
    VALUES ('delete', old.id, old.content, old.context, old.tags, old.fts_project);
END;

INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild');
