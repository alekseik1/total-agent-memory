-- 038_resolve_learned_errors.sql
-- Closes `errors` rows that `ErrorCapture.learn_error` wrote as 'open' even
-- though they carried a fix.
--
-- `learn_error` requires `fix` by contract (it raises if the field is empty)
-- but hardcoded `status = 'open'` in its INSERT and never set `resolved_at`,
-- so every row it ever wrote was born already-fixed and stuck open forever.
-- On a real database: 150 open errors, 148 of them matching this signature
-- (a learn_error row, non-empty fix) - this repairs the history. Fixed in
-- code (src/error_capture.py) so no new rows land like this.
--
-- Only rows that are unmistakably learn_error's are touched: the context
-- column carries its 'root_cause: ... | pattern: ...' signature, which
-- server.py's log_error and auto_self_improve.py's log_error never write.
-- The 2 rows with no fix stay open on purpose - they are genuinely open.
--
-- resolved_at is stamped from the row's own created_at, not the migration's
-- run time: the fix was already known when the row was written, and inventing
-- a later timestamp would be a lie.

UPDATE errors
   SET status = 'resolved',
       resolved_at = created_at
 WHERE status = 'open'
   AND COALESCE(fix, '') <> ''
   AND context LIKE 'root_cause:%';
