-- 038_resolve_learned_errors.sql
-- Closes `errors` rows that `ErrorCapture.learn_error` wrote as 'open'
-- although it requires a fix. Only its rows are touched: their context
-- carries the 'root_cause: ... | pattern: ...' signature. resolved_at takes
-- the row's created_at, when the fix was already known. Rows without a fix
-- stay open.

UPDATE errors
   SET status = 'resolved',
       resolved_at = created_at
 WHERE status = 'open'
   AND COALESCE(fix, '') <> ''
   AND context LIKE 'root_cause:%';
