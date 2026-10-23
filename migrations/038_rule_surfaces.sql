-- Rule exposure vs. rule firing.
--
-- self_rules_context bumped rules.fire_count for every rule it returned, while
-- self_rules(action='rate') divided success_count by fire_count and suspended a
-- rule once fire_count >= 10 with a rate below 0.2. Ten session starts and one
-- helpful rating therefore suspended a good rule. Loads are now counted in
-- shown_count; fire_count only moves on an explicit fire or a confirmed follow.
--
-- Historic fire_count values cannot be split into loads and real fires, so all
-- of them become exposures, and fire_count keeps at most the number of ratings
-- (every rating is evidence of one real firing). success_rate becomes
-- success / (success + fail), the share of rated outcomes that were helpful.
ALTER TABLE rules ADD COLUMN shown_count INTEGER DEFAULT 0;

UPDATE rules SET
    shown_count = COALESCE(fire_count, 0),
    fire_count = MIN(COALESCE(fire_count, 0), COALESCE(success_count, 0) + COALESCE(fail_count, 0)),
    success_rate = CASE
        WHEN COALESCE(success_count, 0) + COALESCE(fail_count, 0) > 0
        THEN CAST(COALESCE(success_count, 0) AS REAL)
             / (COALESCE(success_count, 0) + COALESCE(fail_count, 0))
        ELSE 0.0
    END;

-- One row per rule shown to a session for a query (reused within the same
-- session and hour). followed: 1 followed, 0 ignored, -1 not applicable,
-- NULL not yet judged.
CREATE TABLE IF NOT EXISTS rule_surfaces (
    id TEXT PRIMARY KEY,
    rule_id INTEGER NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    project TEXT NOT NULL DEFAULT 'general',
    query TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    surfaced_at TEXT NOT NULL,
    followed INTEGER CHECK (followed IS NULL OR followed IN (-1, 0, 1)),
    follow_note TEXT NOT NULL DEFAULT '',
    followed_at TEXT
);
-- Reuse lookup (rule, session, hour) and the per-session follow check.
CREATE INDEX IF NOT EXISTS idx_rule_surfaces_session ON rule_surfaces(session_id, rule_id, surfaced_at);
-- Per-rule follow statistics (damping, effectiveness report).
CREATE INDEX IF NOT EXISTS idx_rule_surfaces_rule ON rule_surfaces(rule_id, followed);
-- Project-scoped effectiveness report.
CREATE INDEX IF NOT EXISTS idx_rule_surfaces_project ON rule_surfaces(project, surfaced_at);
