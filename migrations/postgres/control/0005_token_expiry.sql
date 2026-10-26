-- Token expiry and last use (SQLite: Registry.ADDED_COLUMNS). All columns are nullable, so existing
-- tokens stay valid, never expire and never record a last use. Epoch times are double precision like sessions.
-- Applied by team_memory.pg_provision into schema tam_control with search_path set to it.

ALTER TABLE tokens ADD COLUMN expires_at double precision;
ALTER TABLE tokens ADD COLUMN last_used_at double precision;
