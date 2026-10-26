#!/usr/bin/env python3
"""
Dependency Monitor -- scans projects for outdated dependencies.

Runs daily at 7:00 AM. Checks go.mod, composer.json, package.json
for dependency updates via GitHub API.
"""

import json
import os
import re
import sqlite3
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from background_writes import save_background_knowledge  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from paths import memory_dir  # noqa: E402

MEMORY_DIR = memory_dir()
TELEGRAM_ENV_PATH = Path(__file__).parent.parent.parent / "telegram" / ".env"

LOG = lambda msg: sys.stderr.write(f"[dep-monitor] {datetime.now().strftime('%H:%M:%S')} {msg}\n")


def _load_dotenv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip().strip("'\"")
    return result


def _make_ssl_context() -> ssl.SSLContext:
    """Verified TLS context: certifi, else the platform store, never unverified."""
    from tls_context import verified_context

    return verified_context()

def _telegram_send(token: str, chat_id: int, text: str, parse_mode: str = "HTML") -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": text[:4096], "parse_mode": parse_mode}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        ctx = _make_ssl_context()
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            return json.loads(resp.read()).get("ok", False)
    except Exception as e:
        LOG(f"Telegram send error: {e}")
        return False


def _save_knowledge_record(db_path: str, content: str, record_type: str = "fact",
                           tags: list[str] | None = None, source: str = "dep-monitor",
                           confidence: float = 0.6, project: str = "general") -> None:
    db = sqlite3.connect(db_path)
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        save_background_knowledge(
            db, content=content, ktype=record_type, project=project,
            session_id=f"{source}_{now}_{uuid.uuid4().hex[:6]}",
            tags=tags or [], source=source, confidence=confidence, created_at=now,
        )
    except Exception as e:
        LOG(f"Error saving knowledge: {e}")
    finally:
        db.close()


def _check_latest_github_tag(owner: str, repo: str, ctx: ssl.SSLContext) -> str | None:
    """Check latest release tag from GitHub API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    req = urllib.request.Request(url, headers={
        "User-Agent": "ClaudeMemory-DepMonitor/1.0",
        "Accept": "application/vnd.github+json",
    })
    try:
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = json.loads(resp.read())
            return data.get("tag_name", "")
    except Exception:
        return None


def _parse_go_mod(path: str) -> list[dict[str, str]]:
    """Parse go.mod and extract dependencies with versions."""
    go_mod = Path(path) / "go.mod"
    if not go_mod.is_file():
        return []

    deps = []
    in_require = False
    with open(go_mod, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("require ("):
                in_require = True
                continue
            if in_require and line == ")":
                in_require = False
                continue
            if in_require and line and not line.startswith("//"):
                parts = line.split()
                if len(parts) >= 2:
                    module = parts[0]
                    version = parts[1]
                    if not line.endswith("// indirect"):
                        deps.append({"module": module, "version": version, "type": "go"})
    return deps


def _parse_composer_json(path: str) -> list[dict[str, str]]:
    """Parse composer.json and extract dependencies."""
    composer = Path(path) / "composer.json"
    if not composer.is_file():
        return []

    deps = []
    try:
        with open(composer, encoding="utf-8") as f:
            data = json.load(f)
        for section in ("require", "require-dev"):
            for pkg, version in data.get(section, {}).items():
                if pkg == "php" or pkg.startswith("ext-"):
                    continue
                deps.append({"module": pkg, "version": version, "type": "php"})
    except (json.JSONDecodeError, IOError):
        pass
    return deps


def _parse_package_json(path: str) -> list[dict[str, str]]:
    """Parse package.json and extract dependencies."""
    pkg_json = Path(path) / "package.json"
    if not pkg_json.is_file():
        return []

    deps = []
    try:
        with open(pkg_json, encoding="utf-8") as f:
            data = json.load(f)
        for section in ("dependencies", "devDependencies"):
            for pkg, version in data.get(section, {}).items():
                deps.append({"module": pkg, "version": version, "type": "js"})
    except (json.JSONDecodeError, IOError):
        pass
    return deps


def _extract_github_coords(module: str) -> tuple[str, str] | None:
    """Try to extract GitHub owner/repo from a module name."""
    # Go modules: github.com/owner/repo
    m = re.match(r"github\.com/([^/]+)/([^/]+)", module)
    if m:
        return m.group(1), m.group(2)
    # PHP: vendor/package -> try github
    if "/" in module and module.count("/") == 1:
        parts = module.split("/")
        return parts[0], parts[1]
    return None


def run_dependency_monitor(db_path: str) -> None:
    """Scan monitored projects for dependency updates. Runs daily at 7:00 AM."""
    LOG("=== DEPENDENCY MONITOR STARTING ===")

    projects_file = MEMORY_DIR / "monitored_projects.json"
    if not projects_file.is_file():
        LOG("No monitored_projects.json found, skipping")
        return

    try:
        with open(projects_file, encoding="utf-8") as f:
            projects = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        LOG(f"Error reading monitored_projects.json: {e}")
        return

    ctx = _make_ssl_context()
    all_updates: list[dict[str, Any]] = []
    projects_scanned = 0

    for proj in projects:
        proj_path = proj.get("path", "")
        proj_name = proj.get("name", "unknown")

        if not proj_path or not Path(proj_path).is_dir():
            continue

        # Detect and parse dependencies
        deps: list[dict[str, str]] = []
        deps.extend(_parse_go_mod(proj_path))
        deps.extend(_parse_composer_json(proj_path))
        deps.extend(_parse_package_json(proj_path))

        if not deps:
            continue

        projects_scanned += 1
        LOG(f"Scanning {proj_name}: {len(deps)} dependencies")

        # Check top 10 direct dependencies for updates (rate limit friendly)
        checked = 0
        for dep in deps[:10]:
            coords = _extract_github_coords(dep["module"])
            if not coords:
                continue

            owner, repo = coords
            latest = _check_latest_github_tag(owner, repo, ctx)

            if latest and latest != dep["version"] and not dep["version"].startswith("^"):
                all_updates.append({
                    "project": proj_name,
                    "module": dep["module"],
                    "current": dep["version"],
                    "latest": latest,
                    "type": dep["type"],
                })

            checked += 1
            if checked >= 5:  # Max 5 API calls per project
                break

    LOG(f"Scanned {projects_scanned} projects, found {len(all_updates)} updates")

    if all_updates:
        # Save to memory
        update_lines = []
        for u in all_updates:
            update_lines.append(f"  {u['project']}: {u['module']} {u['current']} → {u['latest']}")

        content = f"[Dependency Monitor] {len(all_updates)} updates available across {projects_scanned} projects:\n" + "\n".join(update_lines)

        _save_knowledge_record(
            db_path, content,
            record_type="fact",
            tags=["dependency-monitor", "update-available"],
            source="dep-monitor",
            confidence=0.7,
        )

        # Telegram notification
        env = _load_dotenv(TELEGRAM_ENV_PATH)
        token = env.get("TELEGRAM_BOT_TOKEN", "")
        allowed = env.get("TELEGRAM_ALLOWED_USERS", "")
        if token and allowed:
            msg_lines = [f"<b>📦 Dependency Report</b>", f"{len(all_updates)} updates across {projects_scanned} projects:", ""]
            for u in all_updates[:15]:
                msg_lines.append(f"  <b>{u['project']}</b>: {u['module']}")
                msg_lines.append(f"    {u['current']} → {u['latest']}")

            msg = "\n".join(msg_lines)
            for uid in allowed.split(","):
                uid = uid.strip()
                if uid:
                    try:
                        _telegram_send(token, int(uid), msg)
                    except ValueError:
                        pass

    LOG(f"=== DEPENDENCY MONITOR COMPLETE === ({len(all_updates)} updates)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dependency Monitor")
    parser.add_argument("--db", default=str(MEMORY_DIR / "memory.db"))
    args = parser.parse_args()
    run_dependency_monitor(args.db)
