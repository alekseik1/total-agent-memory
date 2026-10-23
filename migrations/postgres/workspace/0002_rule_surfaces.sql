-- PostgreSQL twin of migrations/038_rule_surfaces.sql (same mapping rules as
-- 0001_baseline.sql): rules.shown_count separates rule loads from real fires,
-- historic fire_count values move into it, success_rate becomes
-- success / (success + fail), and rule_surfaces records each rule exposure
-- with its follow verdict.
ALTER TABLE rules ADD COLUMN shown_count bigint DEFAULT 0;

UPDATE rules SET
    shown_count = COALESCE(fire_count, 0),
    fire_count = LEAST(COALESCE(fire_count, 0), COALESCE(success_count, 0) + COALESCE(fail_count, 0)),
    success_rate = CASE
        WHEN COALESCE(success_count, 0) + COALESCE(fail_count, 0) > 0
        THEN CAST(COALESCE(success_count, 0) AS double precision)
             / (COALESCE(success_count, 0) + COALESCE(fail_count, 0))
        ELSE 0.0
    END;

CREATE TABLE rule_surfaces (
    id text PRIMARY KEY,
    rule_id bigint NOT NULL,
    session_id text NOT NULL DEFAULT '',
    project text NOT NULL DEFAULT 'general',
    query text NOT NULL DEFAULT '',
    source text NOT NULL DEFAULT '',
    surfaced_at text NOT NULL,
    followed bigint CHECK (followed IS NULL OR followed IN (-1, 0, 1)),
    follow_note text NOT NULL DEFAULT '',
    followed_at text
);
CREATE INDEX idx_rule_surfaces_session ON rule_surfaces(session_id, rule_id, surfaced_at);
CREATE INDEX idx_rule_surfaces_rule ON rule_surfaces(rule_id, followed NULLS FIRST);
CREATE INDEX idx_rule_surfaces_project ON rule_surfaces(project, surfaced_at);

INSERT INTO migrations (version, description, applied_at)
VALUES ('038', 'rule surfaces', tam_utc_now());
