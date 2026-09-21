"""The full-text index over `knowledge`, shared by migrations, repair and rebuild.

`knowledge_fts` is an external-content FTS5 table over content, context,
tags and `fts_project`. `fts_project` is a generated column of `knowledge`
(migration 035) holding one token per project, "p" + the hex of its UTF-8
bytes, so a project-scoped search can AND that token into MATCH and let FTS5
intersect doclists instead of ranking every match in the store. The token
cannot collide with a word, and scoped ranking gives the column weight 0.

The update trigger fires only when an indexed column changes; recall
counters and status updates no longer re-index the row.
"""

from __future__ import annotations

import sqlite3

KNOWLEDGE_FTS_DDL = """
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
"""

# bm25() weights for (content, context, tags, fts_project).
SCOPED_BM25_WEIGHTS = "1.0, 1.0, 1.0, 0.0"


def project_token(project: str) -> str:
    """The `fts_project` token of `project`; matches SQL `'p' || lower(hex(project))`."""
    return "p" + project.encode("utf-8").hex()


def scoped_match(terms: str, project: str) -> str:
    """MATCH expression: `terms` in the text columns, restricted to `project`."""
    return f"{{content context tags}} : ({terms}) AND fts_project : {project_token(project)}"


def recreate_knowledge_fts(db: sqlite3.Connection) -> None:
    """Drop and rebuild `knowledge_fts` and its triggers from `knowledge`."""
    db.execute("DROP TRIGGER IF EXISTS k_fts_i")
    db.execute("DROP TRIGGER IF EXISTS k_fts_u")
    db.execute("DROP TRIGGER IF EXISTS k_fts_d")
    db.execute("DROP TABLE IF EXISTS knowledge_fts")
    db.executescript(KNOWLEDGE_FTS_DDL)
    db.execute("INSERT INTO knowledge_fts(knowledge_fts) VALUES('rebuild')")
    db.commit()
