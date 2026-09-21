import hashlib
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from team_memory.contracts import (
    Actor,
    Conflict,
    Forbidden,
    Scope,
    ScopeKind,
    Unauthorized,
    Workspace,
)

REGISTRY_TIMEOUT_SECONDS = 10
TOKEN_BYTES = 32


class Registry:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "identity.db"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE IF NOT EXISTS teams (id TEXT PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS membership (
                    user_id TEXT REFERENCES users(id), team_id TEXT REFERENCES teams(id),
                    role TEXT NOT NULL CHECK(role IN ('reader','editor')),
                    PRIMARY KEY(user_id,team_id));
                CREATE TABLE IF NOT EXISTS tokens (
                    digest TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    client TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS admin_events (
                    id INTEGER PRIMARY KEY, at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    action TEXT NOT NULL, subject TEXT NOT NULL);
            """)
        self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=REGISTRY_TIMEOUT_SECONDS)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def add_user(self, user_id: str, name: str) -> None:
        self._identifier(user_id)
        self._name(name)
        with self.connect() as db:
            try:
                db.execute("INSERT INTO users(id,name) VALUES (?,?)", (user_id, name))
            except sqlite3.IntegrityError as exc:
                raise Conflict("User already exists") from exc
            self._event(db, "user_created", user_id)

    def add_team(self, team_id: str, name: str) -> None:
        self._identifier(team_id)
        self._name(name)
        with self.connect() as db:
            try:
                db.execute("INSERT INTO teams(id,name) VALUES (?,?)", (team_id, name))
            except sqlite3.IntegrityError as exc:
                raise Conflict("Team already exists") from exc
            self._event(db, "team_created", team_id)

    def membership(self, user_id: str, team_id: str, role: str | None) -> None:
        if role not in (None, "reader", "editor"):
            raise ValueError("role must be reader or editor")
        with self.connect() as db:
            if role is None:
                db.execute("DELETE FROM membership WHERE user_id=? AND team_id=?", (user_id, team_id))
            else:
                db.execute("INSERT INTO membership VALUES (?,?,?) ON CONFLICT(user_id,team_id) "
                           "DO UPDATE SET role=excluded.role", (user_id, team_id, role))
            self._event(db, "membership:" + str(role), user_id + ":" + team_id)

    def issue_token(self, user_id: str, client: str) -> str:
        self._name(client)
        token = secrets.token_urlsafe(TOKEN_BYTES)
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM users WHERE id=? AND active=1", (user_id,)).fetchone():
                raise Forbidden("Unknown or disabled user")
            db.execute("INSERT INTO tokens(digest,user_id,client) VALUES (?,?,?)",
                       (self.digest(token), user_id, client))
            self._event(db, "token_created", user_id + ":" + client)
        return token

    def revoke(self, token: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE tokens SET revoked=1 WHERE digest=?", (self.digest(token),))
            self._event(db, "token_revoked", self.digest(token))

    def authenticate(self, token: str) -> Actor:
        with self.connect() as db:
            row = db.execute("SELECT u.id,u.name,t.client FROM tokens t JOIN users u ON u.id=t.user_id "
                             "WHERE digest=? AND revoked=0 AND active=1", (self.digest(token),)).fetchone()
        if row is None:
            raise Unauthorized("Invalid or revoked token")
        return Actor(user_id=row["id"], display_name=row["name"], client=row["client"])

    def workspaces(self, actor: Actor) -> list[Workspace]:
        with self.connect() as db:
            rows = db.execute("SELECT team_id,role FROM membership WHERE user_id=? ORDER BY team_id",
                              (actor.user_id,)).fetchall()
        result = [Workspace(key="personal_" + self.digest(actor.user_id), scope=Scope(), owner_id=actor.user_id, writable=True)]
        result.extend(Workspace(key="team_" + self.digest(r["team_id"]),
                                scope=Scope(kind=ScopeKind.team, team_id=r["team_id"]),
                                writable=r["role"] == "editor") for r in rows)
        result.append(Workspace(key="shared", scope=Scope(kind=ScopeKind.shared), writable=True))
        return result

    def authorize(self, actor: Actor, scope: Scope, write: bool) -> Workspace:
        for workspace in self.workspaces(actor):
            if workspace.scope == scope:
                if write and not workspace.writable:
                    raise Forbidden("Workspace is read-only")
                return workspace
        raise Forbidden("Workspace unavailable")

    @staticmethod
    def digest(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def _identifier(value: str) -> None:
        import re
        if re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value) is None:
            raise ValueError("ID must contain 1–64 letters, digits, underscores or hyphens")

    @staticmethod
    def _name(value: str) -> None:
        if not value.strip() or len(value) > 128:
            raise ValueError("Name must contain 1–128 characters")

    @staticmethod
    def _event(db: sqlite3.Connection, action: str, subject: str) -> None:
        db.execute("INSERT INTO admin_events(action,subject) VALUES (?,?)", (action, subject))
