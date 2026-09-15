import asyncio
import json
import logging
import time
from collections import Counter

from pydantic import JsonValue, ValidationError

from team_memory.contracts import (
    Browse,
    Delete,
    DomainError,
    Empty,
    History,
    RecordRequest,
    Save,
    Search,
    Update,
    Work,
)
from team_memory.registry import Registry
from team_memory.worker import WorkerPool

LOGGER = logging.getLogger(__name__)
LATENCY_BUCKETS = (0.1, 0.5, 1, 5, 30, 120)
TOOLS = {
    "memory_scopes": (Empty, "List your personal, team and shared workspaces."),
    "memory_save": (Save, "Save memory. Default scope is personal; author comes from your token."),
    "memory_recall": (Search, "Search all accessible workspaces, or one selected scope."),
    "memory_get": (RecordRequest, "Read an exact record with author and revision."),
    "memory_update": (Update, "Replace an exact record with revision checking; returns its new ID."),
    "memory_delete": (Delete, "Soft-delete an exact record with revision checking."),
    "memory_history": (History, "Page through this record and its predecessors; after is the last sequence."),
    "memory_export": (Browse, "Page through workspace records with authorship; pass last ID as after."),
}
WRITES = frozenset(("memory_save", "memory_update", "memory_delete"))


class MemoryService:
    def __init__(self, registry: Registry, pool: WorkerPool):
        self.registry, self.pool = registry, pool
        self.counts = Counter()
        self.histogram = Counter()

    async def call(self, token: str, name: str, arguments: dict[str, JsonValue]) -> JsonValue:
        started = time.monotonic()
        status = "ok"
        try:
            return await self._execute(token, name, arguments)
        except (DomainError, ValidationError):
            status = "rejected"
            raise
        except Exception:
            status = "error"
            LOGGER.exception('gateway_call_failed')
            raise
        finally:
            elapsed = time.monotonic() - started
            operation = name if name in TOOLS else "unknown"
            self.counts[(operation, status)] += 1
            for bound in LATENCY_BUCKETS:
                if elapsed <= bound:
                    self.histogram[(operation, str(bound))] += 1
            self.histogram[(operation, "+Inf")] += 1
            LOGGER.info(json.dumps({"event": "memory_call", "tool": operation,
                                    "status": status, "duration_seconds": elapsed}))

    async def _execute(self, token: str, name: str, arguments: dict[str, JsonValue]) -> JsonValue:
        actor = self.registry.authenticate(token)
        if name not in TOOLS:
            raise DomainError("Unknown tool")
        request = TOOLS[name][0].model_validate(arguments)
        if name == "memory_scopes":
            return {"actor": actor.model_dump(), "workspaces": [w.model_dump(mode="json")
                    for w in self.registry.workspaces(actor)]}
        scopes = (self.registry.workspaces(actor) if isinstance(request, Search) and request.scope is None
                  else [self.registry.authorize(actor, request.scope, name in WRITES)])
        execution_order = (await asyncio.to_thread(self.pool.search_order, scopes)
                           if isinstance(request, Search) and len(scopes) > 1 else scopes)
        output = []
        for workspace in execution_order:
            work = Work(actor=actor, workspace=workspace, operation=name,
                        arguments=request.model_dump(mode="json", exclude={"scope"}))
            data = await asyncio.to_thread(self.pool.invoke, work, token)
            current_actor = self.registry.authenticate(token)
            self.registry.authorize(current_actor, workspace.scope, False)
            output.append({"scope": workspace.scope.model_dump(mode="json"),
                           "scope_tags": workspace.scope.system_tags(), "data": data})
        current_actor = self.registry.authenticate(token)
        for workspace in scopes:
            self.registry.authorize(current_actor, workspace.scope, False)
        if isinstance(request, Search):
            ranked = []
            for group in output:
                ranked.extend({"scope": group["scope"], "scope_tags": group["scope_tags"], "record": record, "scope_rank": rank}
                              for rank, record in enumerate(group["data"], 1))
            scope_order = {(workspace.scope.kind.value, workspace.scope.team_id): index
                           for index, workspace in enumerate(scopes)}
            ranked.sort(key=lambda item: (item["scope_rank"], -float(item["record"].get("score", 0)),
                                          scope_order[(item['scope']['kind'], item['scope']['team_id'])]))
            return {"results": ranked[:request.limit], "ordering": "scope_rank_then_score"}
        return output[0]
