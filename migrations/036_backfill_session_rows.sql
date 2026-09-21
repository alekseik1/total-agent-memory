-- 036_backfill_session_rows.sql
-- Closes session rows that were only ever recorded in `session_summaries`.
--
-- `SessionContinuity.session_end` wrote the summary and nothing else, and
-- `Store.session_start` stamps 'general' because the server had no project
-- resolution at bootstrap. The result on a real database: every row in
-- `sessions` open (`ended_at IS NULL`) and in one project bucket, while
-- `session_summaries` held the truth — 1,685 rows against 77 summaries in the
-- last 30 days alone. Both leaks are fixed in code; this repairs the history.
--
-- Only sessions that actually have a summary are touched. A session with no
-- summary was never ended (a crash, a killed client), and saying otherwise
-- would invent an end time — those rows stay open on purpose.
--
-- Where one session_id somehow carries several summaries, the latest one wins
-- (MAX(ended_at)); project/branch are taken from that same newest row.

UPDATE sessions
   SET ended_at = (
           SELECT MAX(s.ended_at) FROM session_summaries s
            WHERE s.session_id = sessions.id
       )
 WHERE ended_at IS NULL
   AND EXISTS (SELECT 1 FROM session_summaries s WHERE s.session_id = sessions.id);

-- Project: only where the row still holds the 'general' placeholder, so a
-- session that already knows its project is never overwritten.
UPDATE sessions
   SET project = (
           SELECT s.project FROM session_summaries s
            WHERE s.session_id = sessions.id
              AND s.project IS NOT NULL
              AND s.project <> 'general'
         ORDER BY s.ended_at DESC
            LIMIT 1
       )
 WHERE COALESCE(project, 'general') = 'general'
   AND EXISTS (
           SELECT 1 FROM session_summaries s
            WHERE s.session_id = sessions.id
              AND s.project IS NOT NULL
              AND s.project <> 'general'
       );

UPDATE sessions
   SET branch = (
           SELECT s.branch FROM session_summaries s
            WHERE s.session_id = sessions.id
              AND COALESCE(s.branch, '') <> ''
         ORDER BY s.ended_at DESC
            LIMIT 1
       )
 WHERE COALESCE(branch, '') = ''
   AND EXISTS (
           SELECT 1 FROM session_summaries s
            WHERE s.session_id = sessions.id
              AND COALESCE(s.branch, '') <> ''
       );
