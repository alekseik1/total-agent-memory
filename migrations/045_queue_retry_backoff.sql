-- Retry scheduling for the durable background queues. A failed row is parked
-- until next_attempt_at instead of being re-claimed on the next tick, so a
-- short LLM/provider outage no longer burns through a row's attempts.
-- deferrals counts parks caused by an unreachable provider (not charged as
-- attempts); requeues counts automatic re-runs of rows that had failed.
ALTER TABLE enrichment_queue ADD COLUMN next_attempt_at TEXT;
ALTER TABLE enrichment_queue ADD COLUMN deferrals INTEGER NOT NULL DEFAULT 0;
ALTER TABLE enrichment_queue ADD COLUMN requeues INTEGER NOT NULL DEFAULT 0;
ALTER TABLE deep_enrichment_queue ADD COLUMN next_attempt_at TEXT;
ALTER TABLE deep_enrichment_queue ADD COLUMN deferrals INTEGER NOT NULL DEFAULT 0;
ALTER TABLE triple_extraction_queue ADD COLUMN next_attempt_at TEXT;
ALTER TABLE triple_extraction_queue ADD COLUMN deferrals INTEGER NOT NULL DEFAULT 0;
ALTER TABLE ingest_queue ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE ingest_queue ADD COLUMN next_attempt_at TEXT;

-- Stages of an enrichment task that already completed, so a retry does not
-- repeat non-idempotent work (episodic Event nodes, audit rows).
CREATE TABLE IF NOT EXISTS enrichment_stage_done (
    queue_id INTEGER NOT NULL,
    stage TEXT NOT NULL,
    done_at TEXT NOT NULL,
    PRIMARY KEY (queue_id, stage)
);
