from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

MAX_CONTENT_CHARS = 200_000
MAX_QUERY_CHARS = 8_000
MAX_RESULTS = 50


class DomainError(Exception):
    code = "invalid_request"


class Unauthorized(DomainError):
    code = "unauthorized"


class Forbidden(DomainError):
    code = "forbidden"


class Conflict(DomainError):
    code = "conflict"


class Unavailable(DomainError):
    code = "unavailable"


class ScopeKind(str, Enum):
    personal = "personal"
    team = "team"
    shared = "shared"


class DTO(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Actor(DTO):
    user_id: str
    display_name: str
    client: str


class Scope(DTO):
    kind: ScopeKind = ScopeKind.personal
    team_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,64}$")

    @model_validator(mode="after")
    def validate_team(self):
        if (self.kind == ScopeKind.team) != (self.team_id is not None):
            raise ValueError("team_id is required only for team scope")
        return self

    def system_tags(self) -> list[str]:
        tags = ["scope:" + self.kind.value]
        if self.team_id is not None:
            tags.append("team:" + self.team_id)
        return tags


class Workspace(DTO):
    key: str
    scope: Scope
    owner_id: str | None = None
    writable: bool


class Save(DTO):
    request_id: UUID | None = None
    scope: Scope = Field(default_factory=Scope)
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    type: Literal["fact", "decision", "solution", "lesson", "convention"] = "fact"
    project: str = Field(default="general", min_length=1, max_length=128)
    tags: list[str] = Field(default_factory=list, max_length=64)
    context: str = Field(default="", max_length=MAX_CONTENT_CHARS)
    importance: Literal["low", "medium", "high", "critical"] = "medium"
    source_format: Literal["auto", "conversation"] = "auto"
    branch: str = Field(default="", max_length=128)

    @model_validator(mode="after")
    def validate_tags(self):
        if any(len(t) > 128 or t.startswith(("scope:", "team:", "user:")) for t in self.tags):
            raise ValueError("Use scope fields for access; reserved or oversized tag")
        return self


class Search(DTO):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    scope: Scope | None = None
    project: str | None = Field(default=None, max_length=128)
    limit: int = Field(default=10, ge=1, le=MAX_RESULTS)


class RecordRequest(DTO):
    scope: Scope = Field(default_factory=Scope)
    id: int = Field(gt=0)


class History(RecordRequest):
    after: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=MAX_RESULTS)


class Update(RecordRequest):
    request_id: UUID | None = None
    expected_revision: int = Field(gt=0)
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    reason: str = Field(min_length=1, max_length=2000)


class Delete(RecordRequest):
    request_id: UUID | None = None
    expected_revision: int = Field(gt=0)
    reason: str = Field(min_length=1, max_length=2000)


class Browse(DTO):
    scope: Scope = Field(default_factory=Scope)
    after: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=MAX_RESULTS)


class Empty(DTO):
    pass


class Work(DTO):
    actor: Actor
    workspace: Workspace
    operation: str
    arguments: dict[str, JsonValue]


class Reply(DTO):
    data: JsonValue = None
    error: str | None = None
    code: str | None = None
