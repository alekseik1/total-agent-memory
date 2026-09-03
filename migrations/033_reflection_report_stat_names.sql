-- 033_reflection_report_stat_names.sql
-- Renames three `reflection_reports` columns to what they actually hold, and
-- drops a fourth that never held anything.
--
-- `_save_report` (src/reflection/agent.py) wrote synthesis['edges_strengthened']
-- into `new_nodes`, synthesis['clusters_found'] into `patterns_found`, and
-- synthesis['skills_proposed'] into `skills_refined` — none of those pairs
-- share a meaning. On a real database this reads `new_nodes=116176`, which is
-- strengthened graph edges, not new nodes. Renaming to match what is actually
-- stored; the existing rows carry over unchanged, `RENAME COLUMN` only
-- relabels them.
--
-- `rules_proposed` was hardcoded to 0 in every INSERT `_save_report` ever
-- issued — nothing in this codebase computes or writes a real value for it.
-- A column that always lies is worse than no column; dropped rather than
-- renamed.

ALTER TABLE reflection_reports RENAME COLUMN new_nodes TO edges_strengthened;
ALTER TABLE reflection_reports RENAME COLUMN patterns_found TO clusters_found;
ALTER TABLE reflection_reports RENAME COLUMN skills_refined TO skills_proposed;
ALTER TABLE reflection_reports DROP COLUMN rules_proposed;
