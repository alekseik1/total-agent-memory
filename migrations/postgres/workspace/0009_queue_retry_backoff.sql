-- PostgreSQL twin of migrations/045_queue_retry_backoff.sql (same mapping rules as 0001_baseline.sql):
-- retry scheduling columns of the durable background queues and the stages of an enrichment task that
-- already completed. The team server runs without background queues, but the schema stays identical.

ALTER TABLE enrichment_queue ADD COLUMN next_attempt_at text;
ALTER TABLE enrichment_queue ADD COLUMN deferrals bigint NOT NULL DEFAULT 0;
ALTER TABLE enrichment_queue ADD COLUMN requeues bigint NOT NULL DEFAULT 0;
ALTER TABLE deep_enrichment_queue ADD COLUMN next_attempt_at text;
ALTER TABLE deep_enrichment_queue ADD COLUMN deferrals bigint NOT NULL DEFAULT 0;
ALTER TABLE triple_extraction_queue ADD COLUMN next_attempt_at text;
ALTER TABLE triple_extraction_queue ADD COLUMN deferrals bigint NOT NULL DEFAULT 0;
ALTER TABLE ingest_queue ADD COLUMN attempts bigint NOT NULL DEFAULT 0;
ALTER TABLE ingest_queue ADD COLUMN next_attempt_at text;

CREATE TABLE enrichment_stage_done (
    queue_id bigint NOT NULL,
    stage text NOT NULL,
    done_at text NOT NULL,
    PRIMARY KEY (queue_id, stage)
);

INSERT INTO migrations (version, description, applied_at) VALUES ('045', 'queue retry backoff', tam_utc_now());
