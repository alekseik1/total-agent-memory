-- Browsers an account has signed in from (SQLite: Registry.SCHEMA known_devices). The digest is the SHA-256 of an
-- opaque cookie value; a known device is throttled by its own failure counter instead of the shared account lock.
-- Applied by team_memory.pg_provision into schema tam_control with search_path set to it.

CREATE TABLE known_devices (
    digest text PRIMARY KEY,
    user_id text NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    created_at double precision NOT NULL,
    last_seen_at double precision NOT NULL
);
CREATE INDEX known_devices_user ON known_devices (user_id);
