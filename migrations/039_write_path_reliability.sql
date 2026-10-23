-- Write-path reliability: embedding backfill queue, outbox leases, error journal,
-- importance budget audit. PostgreSQL twin: migrations/postgres/workspace/0003_write_path_reliability.sql.

-- Outbox: a reconciler claims a pending intent by setting a lease. Another
-- reconciler skips intents whose lease has not expired; a crashed reconciler's
-- lease runs out and the intent becomes claimable again.
ALTER TABLE write_intents ADD COLUMN lease_until TEXT;
CREATE INDEX IF NOT EXISTS idx_write_intents_pending ON write_intents(created_at) WHERE status = 'pending';

-- Records saved while the embedding provider was unavailable. The drain claims
-- due rows by pushing next_attempt_at forward (a lease), embeds them, and
-- deletes the row on success. Failures back off exponentially; after the
-- attempt budget the row stays as status='failed' (dead letter) until retried.
CREATE TABLE IF NOT EXISTS embedding_pending (
    knowledge_id    INTEGER PRIMARY KEY,
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    next_attempt_at TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'failed')),
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embedding_pending_due ON embedding_pending(status, next_attempt_at);

-- Best-effort failures that used to reach stderr only. One row is one
-- occurrence, or a rollup of rate-limited occurrences (occurrences > 1).
CREATE TABLE IF NOT EXISTS error_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    component   TEXT NOT NULL,
    message     TEXT NOT NULL,
    exc_type    TEXT NOT NULL,
    trace_head  TEXT,
    occurrences INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_error_log_ts ON error_log(ts);
CREATE INDEX IF NOT EXISTS idx_error_log_component_ts ON error_log(component, ts);

-- Saves whose requested importance exceeded the per-project daily budget and
-- was lowered; the knowledge row keeps the applied level, this keeps the request.
CREATE TABLE IF NOT EXISTS importance_budget_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    knowledge_id INTEGER NOT NULL,
    project      TEXT NOT NULL,
    ktype        TEXT NOT NULL,
    requested    TEXT NOT NULL,
    applied      TEXT NOT NULL,
    budget       INTEGER NOT NULL,
    used         INTEGER NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_importance_budget_log_project ON importance_budget_log(project, created_at);
CREATE INDEX IF NOT EXISTS idx_importance_budget_log_knowledge ON importance_budget_log(knowledge_id);
