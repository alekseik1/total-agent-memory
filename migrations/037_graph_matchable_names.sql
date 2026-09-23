-- The concept extractor refreshes its name cache every 60 s from active graph
-- nodes whose name can match a text token (no colon). Every save adds a
-- "save:<id>:<time>" event node, so without an index the refresh scanned the
-- whole table: 320 ms per refresh at 1M records for about 200 usable names.
-- The partial covering index holds just the matchable names.
CREATE INDEX IF NOT EXISTS idx_graph_nodes_matchable
    ON graph_nodes(id, name, type) WHERE status = 'active' AND instr(name, ':') = 0;
