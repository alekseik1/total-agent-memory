-- Recipe each embedding group was produced with. A vector's identity is more
-- than its model name: the same name can come out of a different pooling or
-- library version. The probe vector is the model's output for a fixed text at
-- the time the group was written; a startup check compares today's output with
-- it and retires the group's vectors when they no longer agree.
CREATE TABLE IF NOT EXISTS embedding_fingerprints (
    space        TEXT NOT NULL,
    model        TEXT NOT NULL,
    provider     TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    pooling      TEXT NOT NULL DEFAULT '',
    library      TEXT NOT NULL DEFAULT '',
    probe_vector BLOB NOT NULL,
    recorded_at  TEXT NOT NULL,
    PRIMARY KEY (space, model)
);
