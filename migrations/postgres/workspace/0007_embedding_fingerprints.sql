-- PostgreSQL twin of migrations/044_embedding_fingerprints.sql: the recipe each embedding group was produced
-- with, so the startup fingerprint check also runs on PostgreSQL. Same mapping rules as 0001_baseline.sql
-- (INTEGER -> bigint, TEXT -> text, BLOB -> bytea).

CREATE TABLE embedding_fingerprints (
    space text NOT NULL,
    model text NOT NULL,
    provider text NOT NULL,
    dim bigint NOT NULL,
    pooling text NOT NULL DEFAULT '',
    library text NOT NULL DEFAULT '',
    probe_vector bytea NOT NULL,
    recorded_at text NOT NULL,
    PRIMARY KEY (space, model)
);

INSERT INTO migrations (version, description, applied_at) VALUES ('044', 'embedding fingerprints', tam_utc_now());
