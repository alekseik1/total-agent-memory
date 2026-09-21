import json
import sqlite3
from contextlib import contextmanager

from team_memory.contracts import Actor, Conflict

SYSTEM = Actor(user_id="system", display_name="System", client="server")
LEGACY = Actor(user_id="unknown", display_name="Legacy / unknown", client="legacy")
SNAPSHOT_COLUMNS = ("id", "type", "content", "context", "project", "tags", "status", "superseded_by")


class AuditedConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = SYSTEM
        self.origin: Actor | None = None
        self.reason = ""
        self.atomic = False
        self.failed = False
        self.create_function("tam_actor", 0, lambda: self.actor.model_dump_json())
        self.create_function("tam_origin", 0, lambda: (self.origin or self.actor).model_dump_json())
        self.create_function("tam_reason", 0, lambda: self.reason)

    def commit(self):
        if not self.atomic:
            super().commit()

    def rollback(self):
        if self.atomic:
            self.failed = True
        super().rollback()

    def executescript(self, script):
        if not self.atomic:
            return super().executescript(script)
        statement = ""
        cursor = self.cursor()
        for char in script:
            statement += char
            if char == ";" and sqlite3.complete_statement(statement):
                cursor.execute(statement)
                statement = ""
        if statement.strip():
            cursor.execute(statement)
        return cursor

    @contextmanager
    def transaction(self, actor: Actor, reason: str = ""):
        self.commit()
        self.actor, self.reason = actor, reason
        self.atomic, self.failed = True, False
        self.execute("BEGIN IMMEDIATE")
        try:
            yield
            if self.failed:
                raise Conflict("Operation rolled back")
            super().commit()
        except BaseException:
            super().rollback()
            raise
        finally:
            self.atomic = False
            self.actor, self.origin, self.reason = SYSTEM, None, ""


def install(db: AuditedConnection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS tam_authorship (
            record_id INTEGER PRIMARY KEY, created_by TEXT NOT NULL,
            updated_by TEXT NOT NULL, revision INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS tam_history (
            sequence INTEGER PRIMARY KEY, record_id INTEGER NOT NULL,
            at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            operation TEXT NOT NULL, actor TEXT NOT NULL, reason TEXT NOT NULL,
            revision INTEGER NOT NULL, before_state TEXT, after_state TEXT);
        CREATE INDEX IF NOT EXISTS tam_history_record ON tam_history(record_id,sequence);
        CREATE TABLE IF NOT EXISTS tam_requests (
            user_id TEXT NOT NULL, request_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
            result TEXT NOT NULL, completed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            PRIMARY KEY(user_id,request_id));
    """)
    db.execute("INSERT OR IGNORE INTO tam_authorship SELECT id,?,?,1 FROM knowledge",
               (LEGACY.model_dump_json(), LEGACY.model_dump_json()))
    for operation, prefix in (("INSERT", "NEW"), ("UPDATE", "NEW"), ("DELETE", "OLD")):
        def snapshot(alias):
            return "json_object(" + ",".join(f"'{col}',{alias}.{col}" for col in SNAPSHOT_COLUMNS) + ")"
        before = "NULL" if operation == "INSERT" else snapshot("OLD")
        after = "NULL" if operation == "DELETE" else snapshot("NEW")
        when = ""
        if operation == "UPDATE":
            when = "WHEN " + " OR ".join(f"OLD.{c} IS NOT NEW.{c}" for c in SNAPSHOT_COLUMNS[1:])
        revision = (
            "INSERT INTO tam_authorship VALUES (NEW.id,tam_origin(),tam_actor(),1);"
            if operation == "INSERT" else
            f"UPDATE tam_authorship SET updated_by=tam_actor(),revision=revision+1 WHERE record_id={prefix}.id;"
        )
        db.executescript(f"""
            CREATE TRIGGER IF NOT EXISTS tam_audit_{operation.lower()}
            AFTER {operation} ON knowledge {when} BEGIN
                {revision}
                INSERT INTO tam_history(record_id,operation,actor,reason,revision,before_state,after_state)
                SELECT {prefix}.id,'{operation.lower()}',tam_actor(),tam_reason(),revision,{before},{after}
                FROM tam_authorship WHERE record_id={prefix}.id;
            END;
        """)
    db.commit()


def authorship(db: sqlite3.Connection, record_id: int) -> dict:
    row = db.execute("SELECT created_by,updated_by,revision FROM tam_authorship WHERE record_id=?",
                     (record_id,)).fetchone()
    if row is None:
        raise Conflict("Record unavailable")
    return {"created_by": json.loads(row[0]), "updated_by": json.loads(row[1]), "revision": row[2]}
