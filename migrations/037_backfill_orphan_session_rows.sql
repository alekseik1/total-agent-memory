-- 037_backfill_orphan_session_rows.sql
-- Gives a session row to every summary that names a session `sessions` never
-- heard of.
--
-- 036 could only close rows that already existed, and on a real database that
-- was 25 of 256 summaries. The other 231 name sessions the MCP process never
-- opened: the Claude Code session id a hook reports (a bare uuid), an id a
-- caller made up (`mcp_20260830_ag2683_expired_draft_loop`), an API session
-- (`session_01TVefg...`). They were real sessions — they just ended under an
-- identity this server does not issue, and `session_end` had no reason to
-- create a row. It does now; this repairs the history.
--
-- started_at is the earliest ended_at among that session's summaries rather
-- than an invented start: the only thing known for certain is that the session
-- was alive then. ended_at is the latest.

INSERT INTO sessions (id, started_at, project, branch, ended_at)
SELECT ss.session_id,
       MIN(ss.ended_at),
       COALESCE(
           (SELECT s2.project FROM session_summaries s2
             WHERE s2.session_id = ss.session_id
               AND s2.project IS NOT NULL AND s2.project <> 'general'
          ORDER BY s2.ended_at DESC LIMIT 1),
           'general'
       ),
       COALESCE(
           (SELECT s3.branch FROM session_summaries s3
             WHERE s3.session_id = ss.session_id
               AND COALESCE(s3.branch, '') <> ''
          ORDER BY s3.ended_at DESC LIMIT 1),
           ''
       ),
       MAX(ss.ended_at)
  FROM session_summaries ss
 WHERE NOT EXISTS (SELECT 1 FROM sessions s WHERE s.id = ss.session_id)
 GROUP BY ss.session_id;
