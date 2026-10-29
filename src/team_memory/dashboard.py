import asyncio
import hmac
import inspect
import ipaddress
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from functools import lru_cache, wraps
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ValidationError, field_validator
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from team_memory.accounts import DEVICE_TTL_SECONDS, Session
from team_memory.contracts import (
    DTO,
    Actor,
    AuditPage,
    DomainError,
    Forbidden,
    InviteRedeem,
    MembershipChange,
    MemoryCall,
    OrganizationUpdate,
    OrgRoleChange,
    PasswordChange,
    PasswordLogin,
    ProviderTest,
    SettingsChange,
    TeamCreate,
    TeamRef,
    TeamRename,
    TokenCreate,
    TokenFilter,
    TokenLogin,
    TokenRef,
    Unauthorized,
    UserActive,
    UserCreate,
    UserRef,
)
from team_memory.dashboard_service import DashboardService
from team_memory.database_contracts import (
    DATABASE_API,
    DATABASE_MIGRATE_API,
    DATABASE_MIGRATION_API,
    DATABASE_MIGRATION_CANCEL_API,
    DATABASE_PLAN_API,
    DATABASE_REPOINT_API,
    DATABASE_ROLLBACK_API,
    DATABASE_TEST_API,
    DatabasePlanRequest,
    DatabaseRepointRequest,
    DatabaseRollbackRequest,
    DatabaseTestRequest,
    MigrationStartRequest,
)
from team_memory.metrics import route_class
from team_memory.sections import (
    SECTIONS,
    STATIC_PREFIX,
    Capability,
    Section,
    capabilities,
    register,
)

LOGGER = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
API_PREFIX = "/dashboard/api"
STATIC_NAME = re.compile(r"(?:fonts/)?[a-z0-9_-]{1,64}\.(js|css|svg|woff2|txt)")
CONTENT_TYPES = {"js": "text/javascript; charset=utf-8", "css": "text/css; charset=utf-8",
                 "svg": "image/svg+xml", "woff2": "font/woff2", "txt": "text/plain; charset=utf-8"}
IMMUTABLE_TYPES = frozenset(("woff2",))
ASSET_MAX_AGE_SECONDS = 7 * 24 * 3600
TOKEN_METHOD = "token"
COOKIE = "tam_session"
SECURE_COOKIE = "__Host-tam_session"
DEVICE_COOKIE = "tam_device"
SECURE_DEVICE_COOKIE = "__Host-tam_device"
CSRF_HEADER = "x-csrf-token"
SAFE_METHODS = frozenset(("GET", "HEAD"))
STATUS = {"unauthorized": 401, "forbidden": 403, "conflict": 409, "unavailable": 503, "rate_limited": 429,
          "not_found": 404}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "font-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
SECURITY_HEADERS = {"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
                    "Referrer-Policy": "no-referrer", "Cross-Origin-Opener-Policy": "same-origin"}
HSTS_MAX_AGE_SECONDS = 365 * 24 * 3600
HSTS_VALUE = f"max-age={HSTS_MAX_AGE_SECONDS}"
TRUSTED_PROXIES_ENV = "TAM_TEAM_TRUSTED_PROXIES"
HEALTH_DETAILS_ENV = "TAM_TEAM_HEALTH_VERSION"
# A reverse proxy normally reaches the app over loopback or a private network. A direct client on a
# public address can therefore never smuggle X-Forwarded-* headers in.
PRIVATE_PROXY_RANGES = ("127.0.0.0/8", "::1/128", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16",
                        "fc00::/7", "fe80::/10")
TRUTHY = ("1", "true", "yes")
UNKNOWN_CLIENT = "unknown"
TOKEN_SESSION_DENIED = ("This action needs a password sign-in: a session opened with an access token cannot "
                        "create credentials or change settings")
Model = TypeVar("Model", bound=BaseModel)
# Next to Providers (order 84) under Administration; ties sort by id, so Database comes first.
DATABASE_SECTION = Section(id="database", title="Database", capability=Capability.superadmin,
                           script=STATIC_PREFIX + "database.js", mount="TamSections.database", order=84, icon="database")


class DashboardConfig(DTO):
    cookie_secure: Literal["auto", "always", "never"] = "auto"
    trust_proxy: bool = False
    trusted_proxies: tuple[str, ...] = PRIVATE_PROXY_RANGES
    health_version: bool = True

    @field_validator("trusted_proxies")
    @classmethod
    def _networks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            ipaddress.ip_network(item, strict=False)
        return value

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "DashboardConfig":
        env = os.environ if environ is None else environ
        listed = tuple(part.strip() for part in env.get(TRUSTED_PROXIES_ENV, "").split(",") if part.strip())
        return cls(cookie_secure=env.get("TAM_TEAM_COOKIE_SECURE", "auto").strip().lower(),
                   trust_proxy=env.get("TAM_TEAM_TRUST_PROXY", "").strip().lower() in TRUTHY,
                   trusted_proxies=listed or PRIVATE_PROXY_RANGES,
                   health_version=env.get(HEALTH_DETAILS_ENV, "true").strip().lower() in TRUTHY)


@lru_cache(maxsize=16)
def _proxy_networks(cidrs: tuple[str, ...]) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    return tuple(ipaddress.ip_network(item, strict=False) for item in cidrs)


def _address(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(value.strip())
    except ValueError:
        return None


def _is_trusted_proxy(config: "DashboardConfig", value: str) -> bool:
    address = _address(value)
    return address is not None and any(address in network for network in _proxy_networks(config.trusted_proxies))


def secured(response: Response) -> Response:
    response.headers.update(SECURITY_HEADERS)
    response.headers.setdefault("Cache-Control", "no-store")
    return response


def service_of(request: Request) -> DashboardService:
    return request.app.state.dashboard


def config_of(request: Request) -> DashboardConfig:
    return request.app.state.dashboard_config


def peer_address(scope_client: tuple[str, int] | None) -> str:
    return scope_client[0] if scope_client else UNKNOWN_CLIENT


def request_is_https(config: DashboardConfig, scheme: str, peer: str, forwarded_proto: str) -> bool:
    if config.cookie_secure != "auto":
        return config.cookie_secure == "always"
    proxied = config.trust_proxy and _is_trusted_proxy(config, peer)
    return scheme == "https" or (proxied and forwarded_proto.split(",")[-1].strip().lower() == "https")


def is_https(request: Request) -> bool:
    return request_is_https(config_of(request), request.url.scheme, peer_address(request.scope.get("client")),
                            request.headers.get("x-forwarded-proto", ""))


def forwarded_client(config: "DashboardConfig", peer: str, forwarded_for: str) -> str:
    """The address lockouts and per-client limits are keyed by. X-Forwarded-For is read only when the TCP peer
    is a configured proxy, and then from the right: the first hop that is not itself a trusted proxy is the client."""
    if not config.trust_proxy or not _is_trusted_proxy(config, peer):
        return peer
    for hop in reversed(forwarded_for.split(",")):
        address = _address(hop)
        if address is None:
            return peer
        if not _is_trusted_proxy(config, hop):
            return str(address)
    return peer


def client_ip(request: Request) -> str:
    return forwarded_client(config_of(request), peer_address(request.scope.get("client")),
                            request.headers.get("x-forwarded-for", ""))


def session_cookie(request: Request) -> str:
    return request.cookies.get(SECURE_COOKIE) or request.cookies.get(COOKIE) or ""


def device_cookie(request: Request) -> str | None:
    """The opaque ID of a browser this account signed in from before, if the request carries one."""
    return request.cookies.get(SECURE_DEVICE_COOKIE) or request.cookies.get(DEVICE_COOKIE) or None


def session_credential(request: Request) -> Callable[[], Actor]:
    accounts, session_id = service_of(request).accounts, session_cookie(request)

    def credential() -> Actor:
        return accounts.session(session_id, touch=False).actor
    return credential


def error_response(exc: Exception) -> JSONResponse:
    if isinstance(exc, ValidationError):
        message = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors(include_input=False))
        return JSONResponse({"error": message or "Invalid request", "code": "invalid_request"}, status_code=400)
    code = getattr(exc, "code", "invalid_request")
    return JSONResponse({"error": str(exc), "code": code}, status_code=STATUS.get(code, 400))


async def _call(function: Callable, *args):
    if inspect.iscoroutinefunction(function):
        return await function(*args)
    return await asyncio.to_thread(function, *args)


async def parse(request: Request, model: type[Model]) -> Model:
    if request.method in SAFE_METHODS:
        payload: dict = dict(request.query_params)
    else:
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            raise DomainError("Content-Type must be application/json")
        try:
            payload = json.loads(await request.body() or b"{}")
        except json.JSONDecodeError as exc:
            raise DomainError("Malformed JSON body") from exc
        if not isinstance(payload, dict):
            raise DomainError("JSON body must be an object")
    return model.model_validate({**payload, **request.path_params})


async def authenticate(request: Request, capability: Capability = Capability.authenticated) -> Session:
    service = service_of(request)
    session_id = session_cookie(request)
    if not session_id:
        raise Unauthorized("Sign in required")
    session = await asyncio.to_thread(service.accounts.session, session_id)
    if request.method not in SAFE_METHODS and not hmac.compare_digest(
            request.headers.get(CSRF_HEADER, "").encode(), session.csrf.encode()):
        service.metrics.count("csrf_rejected", route=route_class(request.url.path))
        raise Forbidden("Missing or invalid CSRF token")
    granted = await asyncio.to_thread(capabilities, service.registry, session.actor)
    if capability not in granted:
        raise Forbidden("This section is not available for your role")
    request.state.session = session
    return session


def session_endpoint(capability: Capability = Capability.authenticated, *, password_only: bool = False):
    """``password_only`` refuses sessions opened with an access token: an agent token that leaks must not be
    able to mint further credentials, reset passwords or change settings."""
    def decorate(handler: Callable[[Request, Actor], Awaitable | object]):
        @wraps(handler)
        async def endpoint(request: Request) -> Response:
            try:
                session = await authenticate(request, capability)
                if password_only and session.method == TOKEN_METHOD:
                    service_of(request).metrics.count("token_session_denied", route=route_class(request.url.path))
                    raise Forbidden(TOKEN_SESSION_DENIED)
                result = await _call(handler, request, session.actor)
            except (DomainError, ValidationError, ValueError) as exc:
                return secured(error_response(exc))
            return secured(JSONResponse(result.model_dump(mode="json") if isinstance(result, BaseModel) else result))
        return endpoint
    return decorate


def start_session(request: Request, session: Session, payload: dict) -> Response:
    response = secured(JSONResponse(payload))
    secure = is_https(request)
    max_age = service_of(request).accounts.policy.session_max_seconds
    response.set_cookie(SECURE_COOKIE if secure else COOKIE, session.session_id, max_age=max_age, path="/",
                        httponly=True, samesite="strict", secure=secure)
    if session.device_id is not None:
        response.set_cookie(SECURE_DEVICE_COOKIE if secure else DEVICE_COOKIE, session.device_id,
                            max_age=DEVICE_TTL_SECONDS, path="/", httponly=True, samesite="strict", secure=secure)
    return response


def login(model: type[BaseModel], method: str, *, remembers_device: bool = True):
    async def endpoint(request: Request) -> Response:
        service = service_of(request)
        try:
            payload = await parse(request, model)
            arguments = (payload, client_ip(request), *((device_cookie(request),) if remembers_device else ()))
            session = await asyncio.to_thread(getattr(service, method), *arguments)
            overview = await asyncio.to_thread(service.overview, session)
        except (DomainError, ValidationError, ValueError) as exc:
            return secured(error_response(exc))
        return start_session(request, session, overview)
    return endpoint


@session_endpoint()
async def session_info(request: Request, _actor: Actor):
    return await asyncio.to_thread(service_of(request).overview, request.state.session)


async def logout(request: Request) -> Response:
    try:
        session = await authenticate(request)
    except (DomainError, ValueError) as exc:
        return secured(error_response(exc))
    await asyncio.to_thread(service_of(request).accounts.logout, session.session_id)
    response = secured(JSONResponse({"signed_out": True}))
    for name in (COOKIE, SECURE_COOKIE):
        response.delete_cookie(name, path="/", secure=name == SECURE_COOKIE, httponly=True, samesite="strict")
    return response


@session_endpoint()
async def memory(request: Request, _actor: Actor):
    call = await parse(request, MemoryCall)
    return await service_of(request).call_memory(session_credential(request), call)


@session_endpoint(password_only=True)
async def change_password(request: Request, _actor: Actor):
    change = await parse(request, PasswordChange)
    return await asyncio.to_thread(service_of(request).change_password, request.state.session, change,
                                   client_ip(request), device_cookie(request))


async def index(_request: Request) -> Response:
    return secured(FileResponse(STATIC_DIR / "index.html", media_type="text/html; charset=utf-8"))


async def static(request: Request) -> Response:
    name = request.path_params["name"]
    match = STATIC_NAME.fullmatch(name)
    if match is None or not (STATIC_DIR / name).is_file():
        return secured(JSONResponse({"error": "Not found"}, status_code=404))
    response = FileResponse(STATIC_DIR / name, media_type=CONTENT_TYPES[match.group(1)])
    response.headers["Cache-Control"] = (f"public, max-age={ASSET_MAX_AGE_SECONDS}" if match.group(1) in IMMUTABLE_TYPES
                                         else "no-cache")
    return secured(response)


def route(path: str, model: type[BaseModel] | None, method: str, capability: Capability,
          verb: str = "POST", *, password_only: bool = False) -> Route:
    async def handler(request: Request, actor: Actor):
        args = [actor] if model is None else [actor, await parse(request, model)]
        target = service_of(request)
        for part in method.split("."):
            target = getattr(target, part)
        return await _call(target, *args)
    handler.__name__ = method.replace(".", "_")
    return Route(API_PREFIX + path, session_endpoint(capability, password_only=password_only)(handler),
                 methods=[verb])


async def to_index(_request: Request) -> Response:
    return RedirectResponse("/dashboard/", status_code=308)


def routes() -> list[Route]:
    auth, people, company, admin = (Capability.authenticated, Capability.team_people, Capability.company,
                                    Capability.superadmin)
    return [
        Route("/dashboard", to_index),
        Route("/dashboard/", index),
        Route("/dashboard/static/{name:path}", static),
        Route("/dashboard/api/login", login(PasswordLogin, "login_password"), methods=["POST"]),
        Route("/dashboard/api/login/token", login(TokenLogin, "login_token", remembers_device=False), methods=["POST"]),
        Route("/dashboard/api/invite/redeem", login(InviteRedeem, "redeem_invite"), methods=["POST"]),
        Route("/dashboard/api/logout", logout, methods=["POST"]),
        Route("/dashboard/api/session", session_info),
        Route("/dashboard/api/memory", memory, methods=["POST"]),
        Route("/dashboard/api/password", change_password, methods=["POST"]),
        route("/tokens", None, "my_tokens", auth, "GET"),
        route("/tokens", TokenCreate, "create_token", auth, password_only=True),
        route("/tokens/revoke", TokenRef, "revoke_my_token", auth),
        route("/teams/{team_id}/people", TeamRef, "team_people", people, "GET"),
        route("/company", None, "company", company, "GET"),
        route("/overview/me", None, "insights.me", auth, "GET"),
        route("/overview/team/{team_id}", TeamRef, "insights.team", people, "GET"),
        route("/overview/company", None, "insights.company", company, "GET"),
        route("/overview/system", None, "insights.system", admin, "GET"),
        route("/admin/users", None, "users", admin, "GET"),
        route("/admin/users", UserCreate, "create_user", admin, password_only=True),
        route("/admin/users/{user_id}/invite", UserRef, "issue_invite", admin, password_only=True),
        route("/admin/users/{user_id}/active", UserActive, "set_active", admin, password_only=True),
        route("/admin/users/{user_id}/role", OrgRoleChange, "set_org_role", admin, password_only=True),
        route("/admin/teams", None, "teams", admin, "GET"),
        route("/admin/teams", TeamCreate, "create_team", admin, password_only=True),
        route("/admin/teams/{team_id}/rename", TeamRename, "rename_team", admin, password_only=True),
        route("/admin/teams/{team_id}/delete", TeamRef, "delete_team", admin, password_only=True),
        route("/admin/membership", MembershipChange, "change_membership", admin, password_only=True),
        route("/admin/tokens", TokenFilter, "all_tokens", admin, "GET"),
        route("/admin/tokens/revoke", TokenRef, "admin_revoke_token", admin, password_only=True),
        route("/admin/audit", AuditPage, "audit", admin, "GET"),
        route("/admin/settings", None, "settings_view", admin, "GET"),
        route("/admin/settings", SettingsChange, "update_settings", admin, password_only=True),
        route("/admin/settings/test", ProviderTest, "test_provider", admin, password_only=True),
        route("/admin/backup", None, "backup_info", admin, "GET"),
        route("/admin/metrics", None, "metrics_view", admin, "GET"),
        route("/admin/organization", None, "organization", admin, "GET"),
        route("/admin/organization", OrganizationUpdate, "update_organization", admin, password_only=True),
        route("/admin/setup/finish", None, "finish_setup", admin, password_only=True),
        *database_routes(),
    ]


def register_database_section() -> None:
    if not any(section.id == DATABASE_SECTION.id for section in SECTIONS):
        register(DATABASE_SECTION)


def database_routes() -> list[Route]:
    admin = Capability.superadmin

    def at(api: str) -> str:
        return api.removeprefix(API_PREFIX)
    return [
        route(at(DATABASE_API), None, "database_view", admin, "GET"),
        route(at(DATABASE_TEST_API), DatabaseTestRequest, "test_database", admin, password_only=True),
        route(at(DATABASE_PLAN_API), DatabasePlanRequest, "plan_migration", admin, password_only=True),
        route(at(DATABASE_MIGRATE_API), MigrationStartRequest, "start_migration", admin, password_only=True),
        route(at(DATABASE_MIGRATION_API), None, "migration_progress", admin, "GET"),
        route(at(DATABASE_MIGRATION_CANCEL_API), None, "cancel_migration", admin, password_only=True),
        route(at(DATABASE_REPOINT_API), DatabaseRepointRequest, "repoint_database", admin, password_only=True),
        route(at(DATABASE_ROLLBACK_API), DatabaseRollbackRequest, "rollback_database", admin, password_only=True),
    ]
