-- PostgreSQL twin of migrations/043_resolve_learned_errors.sql (data only): closes `errors` rows that
-- ErrorCapture.learn_error wrote as 'open' although it requires a fix. Only its rows are touched: their
-- context starts with the exact, case-sensitive 'root_cause:'. resolved_at takes the row's created_at.

UPDATE errors
   SET status = 'resolved',
       resolved_at = created_at
 WHERE status = 'open'
   AND COALESCE(fix, '') <> ''
   AND substr(context, 1, 11) = 'root_cause:';

INSERT INTO migrations (version, description, applied_at) VALUES ('043', 'resolve learned errors', tam_utc_now());
