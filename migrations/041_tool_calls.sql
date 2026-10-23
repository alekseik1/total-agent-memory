-- Per-tool MCP call telemetry (src/tool_telemetry.py): one row per tool call
-- with its latency, outcome and payload sizes. Argument and result contents are
-- never stored, only their UTF-8 byte counts, so a call carrying a secret leaves
-- nothing but a number behind. memory_perf_report reads it for per-tool count,
-- p50/p95 and error rate; rows older than 7 days are pruned by the writer.
-- ts is canonical UTC text (memory_core.timestamps), so range scans compare text.

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    tool TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    ok INTEGER NOT NULL CHECK (ok IN (0, 1)),
    error_class TEXT,
    args_bytes INTEGER NOT NULL DEFAULT 0,
    result_bytes INTEGER NOT NULL DEFAULT 0,
    profile TEXT NOT NULL DEFAULT 'full'
);
-- The report and the retention DELETE both filter on ts.
CREATE INDEX IF NOT EXISTS idx_tool_calls_ts ON tool_calls(ts);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_ts ON tool_calls(tool, ts);
