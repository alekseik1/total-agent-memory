import argparse
import logging
import os
from pathlib import Path

from team_memory.registry import Registry

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 3737


def main():
    parser = argparse.ArgumentParser(prog="tam-team")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("TAM_TEAM_DIR", "~/.tam-server")).expanduser())
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("user-add", "team-add"):
        sub = commands.add_parser(command)
        sub.add_argument("id")
        sub.add_argument("name")
    member = commands.add_parser("member")
    member.add_argument("user")
    member.add_argument("team")
    member.add_argument("role", choices=("reader", "editor", "remove"))
    token = commands.add_parser("token-create")
    token.add_argument("user")
    token.add_argument("--client", required=True)
    token.add_argument("--out", type=Path, required=True)
    revoke = commands.add_parser("token-revoke")
    revoke.add_argument("--file", type=Path, required=True)
    snapshot = commands.add_parser("backup")
    snapshot.add_argument("--out", type=Path, required=True)
    recovery = commands.add_parser("restore")
    recovery.add_argument("--from", dest="snapshot", type=Path, required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--host", default=os.environ.get("MCP_HTTP_HOST", DEFAULT_HOST))
    serve.add_argument("--port", type=int, default=int(os.environ.get("MCP_HTTP_PORT", DEFAULT_PORT)))
    args = parser.parse_args()
    if args.command in ("backup", "restore"):
        from team_memory.lifecycle import backup, restore
        if args.command == "backup":
            backup(args.root, args.out)
        else:
            restore(args.snapshot, args.root)
        return
    registry = Registry(args.root)
    if args.command == "user-add":
        registry.add_user(args.id, args.name)
    elif args.command == "team-add":
        registry.add_team(args.id, args.name)
    elif args.command == "member":
        registry.membership(args.user, args.team, None if args.role == "remove" else args.role)
    elif args.command == "token-create":
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as destination:
            destination.write(registry.issue_token(args.user, args.client) + "\n")
    elif args.command == "token-revoke":
        registry.revoke(args.file.read_text().strip())
    elif args.command == "serve":
        import uvicorn

        from team_memory.app import create_app
        from team_memory.service import MemoryService
        from team_memory.worker import (
            DEFAULT_OPERATION_TIMEOUT,
            DEFAULT_WORKERS,
            WorkerPool,
        )
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        pool = WorkerPool(registry.root, int(os.environ.get("TAM_TEAM_MAX_WORKERS", DEFAULT_WORKERS)),
                          float(os.environ.get("TAM_TEAM_OPERATION_TIMEOUT", DEFAULT_OPERATION_TIMEOUT)))
        uvicorn.run(create_app(MemoryService(registry, pool)), host=args.host, port=args.port, access_log=False)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
