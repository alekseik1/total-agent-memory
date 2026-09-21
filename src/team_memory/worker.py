import hashlib
import json
import logging
import multiprocessing
import os
import threading
from collections import OrderedDict
from pathlib import Path

from team_memory.audit import SNAPSHOT_COLUMNS, AuditedConnection, authorship, install
from team_memory.contracts import (
    Actor,
    Conflict,
    DomainError,
    Reply,
    Unavailable,
    Work,
    Workspace,
)
from team_memory.lifecycle import ServerLease

DEFAULT_WORKERS = 3
DEFAULT_OPERATION_TIMEOUT = 120
STOP_TIMEOUT_SECONDS = 5
LOGGER = logging.getLogger(__name__)


class Runtime:
    def __init__(self, root: str):
        os.environ.update(TAM_MEMORY_DIR=root, CLAUDE_MEMORY_DIR=root,
                          USE_BINARY_SEARCH="true", MEMORY_ASYNC_ENRICHMENT="false")
        import server
        self.store = server.Store(connection_factory=AuditedConnection)
        self.recall = server.Recall(self.store)
        install(self.store.db)
        self.session = "workspace"
        self.store.session_start(self.session)

    def record(self, record_id: int) -> dict:
        row = self.store.q1("SELECT * FROM knowledge WHERE id=?", (record_id,))
        if row is None:
            raise Conflict("Record unavailable")
        return {**row, **authorship(self.store.db, record_id)}

    def execute(self, work: Work):
        args = work.arguments
        operation = work.operation
        if operation == "memory_recall":
            result = self.recall.search(args["query"], project=args.get("project"), limit=args["limit"])
            records = [item for group in result.get("results", {}).values() for item in group]
            records.sort(key=lambda item: float(item.get("score", 0)), reverse=True)
            return [{**item, **authorship(self.store.db, item["id"])} for item in records[:args["limit"]]]
        if operation == "memory_get":
            return self.record(args["id"])
        if operation == "memory_history":
            self.record(args["id"])
            rows = self.store.db.execute("""
                WITH RECURSIVE predecessors(id) AS (
                    SELECT ? UNION SELECT k.id FROM knowledge k JOIN predecessors p ON k.superseded_by=p.id
                ) SELECT h.* FROM tam_history h JOIN predecessors p ON p.id=h.record_id
                  WHERE h.sequence>? ORDER BY h.sequence LIMIT ?
                """, (args["id"], args["after"], args["limit"])).fetchall()
            return [{**dict(row), **{k: json.loads(row[k]) if row[k] else None
                                    for k in ("actor", "before_state", "after_state")}} for row in rows]
        if operation == "memory_export":
            ids = self.store.db.execute("SELECT id FROM knowledge WHERE id>? ORDER BY id LIMIT ?",
                                        (args["after"], args["limit"])).fetchall()
            return [self.record(row[0]) for row in ids]
        with self.store.db.transaction(work.actor, args.get("reason", "")):
            request_id = args.get("request_id")
            fingerprint = hashlib.sha256(json.dumps({"operation": operation, "arguments": args},
                                                     sort_keys=True).encode()).hexdigest()
            if request_id:
                previous = self.store.db.execute("SELECT fingerprint,result FROM tam_requests WHERE user_id=? AND request_id=?",
                                                 (work.actor.user_id, request_id)).fetchone()
                if previous:
                    if previous[0] != fingerprint:
                        raise Conflict("request_id was already used with different arguments")
                    return json.loads(previous[1])
            if operation == "memory_save":
                result = self.save(args)
            elif operation in ("memory_update", "memory_delete"):
                old = self.record(args["id"])
                if old["revision"] != args["expected_revision"] or old["status"] != "active":
                    raise Conflict("Revision changed; read the record before retrying")
                if operation == "memory_delete":
                    self.store.delete_knowledge(old["id"])
                    result = self.record(old["id"])
                else:
                    self.store.db.origin = Actor.model_validate(old["created_by"])
                    result = self.save({**args, "type": old["type"], "project": old["project"],
                                        "tags": json.loads(old["tags"]), "context": old["context"],
                                        "branch": old.get("branch", ""), "source_format": old.get("source_format", "auto"),
                                        "importance": old.get("importance", "medium")}, replacing=True)
                    if result.get("saved") is False:
                        raise Conflict("Replacement rejected by quality gate")
                    self.store.db.execute("UPDATE knowledge SET status='superseded',superseded_by=? WHERE id=?",
                                          (result["id"], old["id"]))
                    self.store._delete_embedding(old["id"])
                    result["previous_id"] = old["id"]
            else:
                raise DomainError("Unknown workspace operation")
            if request_id:
                self.store.db.execute("INSERT INTO tam_requests(user_id,request_id,fingerprint,result) VALUES (?,?,?,?)",
                                      (work.actor.user_id, request_id, fingerprint, json.dumps(result, ensure_ascii=False)))
        self.invalidate()
        return result

    def save(self, args, replacing=False):
        record_id, dedup, redacted, sections, quality = self.store.save_knowledge(
            self.session, args["content"], args["type"], project=args["project"],
            tags=args["tags"], context=args["context"], importance=args["importance"],
            branch=args["branch"], source_format=args["source_format"],
            skip_dedup=replacing)
        if record_id is None:
            return {"saved": False, "quality": quality}
        record = self.record(record_id)
        if dedup:
            state = json.dumps({key: record.get(key) for key in SNAPSHOT_COLUMNS}, ensure_ascii=False)
            self.store.db.execute("""
                INSERT INTO tam_history(record_id,operation,actor,reason,revision,before_state,after_state)
                VALUES (?,'confirm',tam_actor(),tam_reason(),?,?,?)
                """, (record_id, record["revision"], state, state))
        return {**record, "saved": True, "deduplicated": dedup,
                "privacy_redacted": redacted, "privacy_redacted_sections": sections}

    def invalidate(self):
        if self.store.cache is not None:
            self.store.cache.invalidate()
        if getattr(self.store, "v9_cache", None) is not None:
            self.store.v9_cache.invalidate_all()


def serve(connection, root: str):
    with ServerLease(Path(root)):
        serve_locked(connection, root)


def serve_locked(connection, root: str):
    runtime = None
    try:
        runtime = Runtime(root)
        while True:
            payload = connection.recv()
            if payload is None:
                break
            try:
                result = runtime.execute(Work.model_validate_json(payload))
                response = Reply(data=result)
            except DomainError as exc:
                runtime.store.db.rollback()
                runtime.invalidate()
                response = Reply(error=str(exc), code=exc.code)
            except Exception:
                LOGGER.exception('workspace_operation_failed')
                runtime.store.db.rollback()
                runtime.invalidate()
                response = Reply(error="Workspace operation failed", code="unavailable")
            connection.send(response.model_dump_json())
    except EOFError:
        LOGGER.info('workspace_connection_closed')
    finally:
        if runtime is not None:
            runtime.store.db.close()
        connection.close()


class WorkerPool:
    def __init__(self, root: Path, maximum: int = DEFAULT_WORKERS, timeout: float = DEFAULT_OPERATION_TIMEOUT):
        if maximum < 1 or timeout <= 0:
            raise ValueError("Worker count and timeout must be positive")
        self.root, self.maximum, self.timeout = root, maximum, timeout
        self.workers = OrderedDict()
        self.lock = threading.Lock()
        self.context = multiprocessing.get_context("spawn")
        from team_memory.registry import Registry
        self.registry = Registry(root)

    def search_order(self, workspaces: list[Workspace]) -> list[Workspace]:
        with self.lock:
            return sorted(workspaces, key=lambda workspace: workspace.key not in self.workers)

    def invoke(self, work: Work, token: str):
        with self.lock:
            actor = self.registry.authenticate(token)
            self.registry.authorize(actor, work.workspace.scope, work.operation in ("memory_save", "memory_update", "memory_delete"))
            if actor != work.actor:
                raise Conflict("Identity changed; retry the request")
            key = work.workspace.key
            if key not in self.workers:
                if len(self.workers) >= self.maximum:
                    _, worker = self.workers.popitem(last=False)
                    self.stop(worker)
                parent, child = self.context.Pipe()
                process = self.context.Process(target=serve, args=(child, str(self.root / "workspaces" / key)), daemon=True)
                process.start()
                child.close()
                self.workers[key] = (process, parent)
            self.workers.move_to_end(key)
            process, connection = self.workers[key]
            try:
                connection.send(work.model_dump_json())
                if not connection.poll(self.timeout):
                    raise Unavailable("Workspace timeout; check history before retrying a write")
                reply = Reply.model_validate_json(connection.recv())
            except (EOFError, OSError, Unavailable) as exc:
                self.stop(self.workers.pop(key))
                raise Unavailable("Workspace unavailable; check history before retrying a write") from exc
            if reply.error:
                if reply.code == "conflict":
                    raise Conflict(reply.error)
                raise Unavailable(reply.error)
            return reply.data

    @staticmethod
    def stop(worker):
        process, connection = worker
        if process.is_alive():
            process.terminate()
        process.join(STOP_TIMEOUT_SECONDS)
        if process.is_alive():
            process.kill()
            process.join()
        connection.close()
        process.close()

    def close(self):
        with self.lock:
            for worker in self.workers.values():
                self.stop(worker)
            self.workers.clear()
