-- Agent action log (src/ambient/actions.py): one row per tool call a client
-- hook observed (Claude Code PostToolUse / PostToolUseFailure) or per turn a
-- Codex notify reported. The Stop hook reads a session's rows to check the
-- turn against the rules surfaced to it and to write the turn summary.
--
-- summary is a short, secret-redacted description (the command line, the file
-- path plus the start of the new text); it is never the full tool input or
-- output. status: ok | failed | interrupted; exit_code is the process exit
-- code when the tool reported one. Rows older than
-- TAM_ACTION_LOG_RETENTION_DAYS (and beyond the row cap) are pruned by the
-- writer; memory_forget and `tam forget-auto --include-actions` prune too.
-- ts is canonical UTC text (memory_core.timestamps), so range scans compare text.
CREATE TABLE IF NOT EXISTS agent_actions (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    client TEXT NOT NULL DEFAULT 'claude-code',
    session_id TEXT NOT NULL DEFAULT '',
    project TEXT NOT NULL DEFAULT 'general',
    tool TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ok' CHECK (status IN ('ok', 'failed', 'interrupted')),
    exit_code INTEGER
);
-- The turn read at Stop: one session's actions after a timestamp.
CREATE INDEX IF NOT EXISTS idx_agent_actions_session_ts ON agent_actions(session_id, ts);
-- Retention DELETE and project-scoped forget.
CREATE INDEX IF NOT EXISTS idx_agent_actions_ts ON agent_actions(ts);
CREATE INDEX IF NOT EXISTS idx_agent_actions_project_ts ON agent_actions(project, ts);

-- Rated outcomes. success_count / fail_count / success_rate stay the explicit
-- ratings (self_rules(action='rate')), the only input of auto-suspension.
-- success_lower_bound is the lower end of their 95% Wilson interval, 0.0 while
-- unrated, so a rule with one lucky rating does not rank above one with forty.
-- The Stop-hook compliance check is a heuristic and writes only its own
-- counters: auto_success_count (rule followed) and auto_fail_count (rule
-- violated); they never feed success_rate, the bound or suspension.
ALTER TABLE rules ADD COLUMN success_lower_bound REAL DEFAULT 0.0;
ALTER TABLE rules ADD COLUMN auto_success_count INTEGER DEFAULT 0;
ALTER TABLE rules ADD COLUMN auto_fail_count INTEGER DEFAULT 0;
