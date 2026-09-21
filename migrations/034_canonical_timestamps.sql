-- One stored timestamp format for knowledge: 2026-09-21T08:21:37.622445Z (UTC).
-- Older writers produced "...Z", "...+00:00" and zone-less "YYYY-MM-DD HH:MM:SS".
-- Mixed widths broke string ordering and showed readers inconsistent dates.
-- Zone-less values came from Python datetime.now() (auto_session_save,
-- analyze_project), i.e. the machine's local time, so they are converted with
-- SQLite's 'utc' modifier, which applies the local zone and its DST rules.
-- The instants do not change, so the atomic-fact trigger (which rebuilds facts
-- when created_at changes) is suspended for this rewrite and restored verbatim
-- from migration 030. Unparseable values are left untouched.

DROP TRIGGER atomic_source_update;

UPDATE knowledge SET
    created_at = CASE
        WHEN created_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN created_at
        WHEN created_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]+00:00' THEN substr(created_at, 1, 26) || 'Z'
        WHEN created_at NOT LIKE '%Z' AND created_at NOT GLOB '*[+-][0-9][0-9]:[0-9][0-9]'
             AND strftime('%Y-%m-%dT%H:%M:%f000Z', created_at, 'utc') IS NOT NULL
            THEN strftime('%Y-%m-%dT%H:%M:%f000Z', created_at, 'utc')
        WHEN strftime('%Y-%m-%dT%H:%M:%f000Z', created_at) IS NOT NULL THEN strftime('%Y-%m-%dT%H:%M:%f000Z', created_at)
        ELSE created_at
    END,
    last_confirmed = CASE
        WHEN last_confirmed IS NULL THEN NULL
        WHEN last_confirmed GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z' THEN last_confirmed
        WHEN last_confirmed GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]+00:00' THEN substr(last_confirmed, 1, 26) || 'Z'
        WHEN last_confirmed NOT LIKE '%Z' AND last_confirmed NOT GLOB '*[+-][0-9][0-9]:[0-9][0-9]'
             AND strftime('%Y-%m-%dT%H:%M:%f000Z', last_confirmed, 'utc') IS NOT NULL
            THEN strftime('%Y-%m-%dT%H:%M:%f000Z', last_confirmed, 'utc')
        WHEN strftime('%Y-%m-%dT%H:%M:%f000Z', last_confirmed) IS NOT NULL THEN strftime('%Y-%m-%dT%H:%M:%f000Z', last_confirmed)
        ELSE last_confirmed
    END
WHERE NOT created_at GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'
   OR (last_confirmed IS NOT NULL
       AND NOT last_confirmed GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z');

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
