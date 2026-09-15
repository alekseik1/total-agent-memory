import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

DEFAULT_TIMEOUT_SECONDS = 180
MAX_FRAME_BYTES = 1_000_000


def write_frame(frame):
    sys.stdout.buffer.write((json.dumps(frame, ensure_ascii=False) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Remote MCP redirects are disabled; configure the final endpoint")


def endpoint(raw: str) -> str:
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password or parts.fragment or parts.query:
        raise ValueError("TAM_REMOTE_URL must be an HTTP(S) MCP endpoint without credentials or query")
    if parts.scheme != "https" and parts.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Remote connections require HTTPS; HTTP is allowed only on loopback")
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + "/", "", ""))


def main():
    url = endpoint(os.environ["TAM_REMOTE_URL"])
    token_file = Path(os.environ["TAM_REMOTE_TOKEN_FILE"]).expanduser()
    timeout = float(os.environ.get("TAM_REMOTE_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
    if timeout <= 0:
        raise ValueError("TAM_REMOTE_TIMEOUT must be positive")
    opener = urllib.request.build_opener(NoRedirect())
    protocol = "2025-06-18"
    while True:
        line = sys.stdin.buffer.readline(MAX_FRAME_BYTES + 1)
        if not line:
            return
        if len(line) > MAX_FRAME_BYTES:
            raise ValueError("MCP frame exceeds size limit")
        frame = json.loads(line)
        request_id = frame.get("id")
        try:
            token = token_file.read_text().strip()
            if not token:
                raise ValueError("Empty token file")
            request = urllib.request.Request(url, data=line, method="POST", headers={
                "Authorization": "Bearer " + token, "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream", "MCP-Protocol-Version": protocol,
            })
            with opener.open(request, timeout=timeout) as response:
                payload = response.read()
            if payload:
                result = json.loads(payload)
                if frame.get("method") == "initialize":
                    protocol = result.get("result", {}).get("protocolVersion", protocol)
                write_frame(result)
        except (OSError, ValueError, urllib.error.HTTPError) as exc:
            message = "Remote MCP request failed"
            if isinstance(exc, urllib.error.HTTPError):
                message += f" (HTTP {exc.code})"
            sys.stderr.write(json.dumps({"event": "remote_request_failed", "message": message}) + "\n")
            if request_id is not None:
                write_frame({"jsonrpc": "2.0", "id": request_id,
                             "error": {"code": -32000, "message": message}})


if __name__ == "__main__":
    main()
