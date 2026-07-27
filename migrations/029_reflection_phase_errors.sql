-- 029_reflection_phase_errors.sql
-- Make failing reflection phases visible.
--
-- run_full() runs six phases and each one catches its own exceptions, returning
-- {"error": "..."} instead of raising. _save_report then persisted only digest
-- and synthesis counters, so every one of those errors was dropped on the floor
-- and survived only in stderr. That is how `fact_merge` stayed broken for
-- months with "table knowledge has no column named updated_at": nothing the
-- report kept ever mentioned it.
--
-- One JSON blob of {phase: message} per report, so a failing phase is queryable
-- instead of archaeological.

ALTER TABLE reflection_reports ADD COLUMN phase_errors JSON;
