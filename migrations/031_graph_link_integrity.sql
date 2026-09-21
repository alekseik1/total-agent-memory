CREATE TABLE knowledge_nodes_quarantine (
    knowledge_id INTEGER,
    node_id TEXT,
    role TEXT,
    strength REAL,
    reason TEXT NOT NULL,
    archived_at TEXT NOT NULL
);

INSERT INTO knowledge_nodes_quarantine
SELECT knowledge_id, node_id, role, strength, 'missing_parent',
       strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
FROM knowledge_nodes
WHERE NOT EXISTS (SELECT 1 FROM knowledge WHERE id=knowledge_id)
   OR NOT EXISTS (SELECT 1 FROM graph_nodes WHERE id=node_id);

DELETE FROM knowledge_nodes
WHERE NOT EXISTS (SELECT 1 FROM knowledge WHERE id=knowledge_id)
   OR NOT EXISTS (SELECT 1 FROM graph_nodes WHERE id=node_id);

CREATE TRIGGER knowledge_nodes_insert_guard BEFORE INSERT ON knowledge_nodes
WHEN NOT EXISTS (SELECT 1 FROM knowledge WHERE id=new.knowledge_id)
  OR NOT EXISTS (SELECT 1 FROM graph_nodes WHERE id=new.node_id)
BEGIN
    SELECT RAISE(ABORT, 'Graph link requires existing knowledge and node');
END;

CREATE TRIGGER knowledge_nodes_update_guard
BEFORE UPDATE OF knowledge_id, node_id ON knowledge_nodes
WHEN NOT EXISTS (SELECT 1 FROM knowledge WHERE id=new.knowledge_id)
  OR NOT EXISTS (SELECT 1 FROM graph_nodes WHERE id=new.node_id)
BEGIN
    SELECT RAISE(ABORT, 'Graph link requires existing knowledge and node');
END;

CREATE TRIGGER knowledge_links_delete AFTER DELETE ON knowledge
BEGIN
    DELETE FROM knowledge_nodes WHERE knowledge_id=old.id;
END;

CREATE TRIGGER node_links_delete AFTER DELETE ON graph_nodes
BEGIN
    DELETE FROM knowledge_nodes WHERE node_id=old.id;
END;

CREATE TRIGGER knowledge_links_id_guard BEFORE UPDATE OF id ON knowledge
WHEN old.id IS NOT new.id
 AND EXISTS (SELECT 1 FROM knowledge_nodes WHERE knowledge_id=old.id)
BEGIN
    SELECT RAISE(ABORT, 'Cannot change linked knowledge identity');
END;

CREATE TRIGGER node_links_id_guard BEFORE UPDATE OF id ON graph_nodes
WHEN old.id IS NOT new.id
 AND EXISTS (SELECT 1 FROM knowledge_nodes WHERE node_id=old.id)
BEGIN
    SELECT RAISE(ABORT, 'Cannot change linked node identity');
END;
