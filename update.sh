#!/usr/bin/env bash
# Auto-update for total-agent-memory.
#
# Stages (skip-friendly — every stage tolerates absence of prerequisites):
#   1. Pre-flight  — verify dirs, disk space, snapshot DB
#   2. Source      — git pull (if repo) OR curl tarball (if UPDATE_URL set)
#   3. Deps        — pip install only if requirements*.txt hash changed
#   4. Tests       — pytest gate; abort + rollback if red
#   5. Schema      — Store() init applies pending migrations idempotently
#   6. Services    — reload LaunchAgents + restart dashboard
#   7. MCP         — print user instruction (only Claude Code can /mcp reconnect)
#
# Usage:
#   bash update.sh                    # full update
#   bash update.sh --check            # dry-run: report only, no changes
#   bash update.sh --skip-tests       # skip pytest (NOT recommended)
#   UPDATE_URL=https://… bash update.sh   # tarball-based update
#
# Env:
#   ROOT       — install dir (default: dirname of this script)
#   PY         — venv python (default: $ROOT/.venv/bin/python)

set -euo pipefail

# ────────────────────────────────────────────────
# Settings
# ────────────────────────────────────────────────

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd)}"
PY="${PY:-$ROOT/.venv/bin/python}"
PIP="$ROOT/.venv/bin/pip"
# Resolution: TAM_MEMORY_DIR > legacy CLAUDE_MEMORY_DIR > ~/.tam > legacy ~/.claude-memory > fresh ~/.tam
if [ -n "${TAM_MEMORY_DIR:-}" ]; then
  MEMORY_DIR="$TAM_MEMORY_DIR"
elif [ -n "${CLAUDE_MEMORY_DIR:-}" ]; then
  MEMORY_DIR="$CLAUDE_MEMORY_DIR"
  echo "  WARN: CLAUDE_MEMORY_DIR is deprecated, please switch to TAM_MEMORY_DIR" >&2
elif [ -d "$HOME/.tam" ]; then
  MEMORY_DIR="$HOME/.tam"
elif [ -d "$HOME/.claude-memory" ]; then
  MEMORY_DIR="$HOME/.claude-memory"
else
  MEMORY_DIR="$HOME/.tam"
fi
DB="$MEMORY_DIR/memory.db"
BACKUP_DIR="$MEMORY_DIR/backups"
LOG_DIR="$MEMORY_DIR/logs"
LOG="$LOG_DIR/update.log"
DRY_RUN=false
SKIP_TESTS=false

for arg in "$@"; do
  case "$arg" in
    --check|--dry-run)  DRY_RUN=true ;;
    --skip-tests)       SKIP_TESTS=true ;;
    -h|--help)          sed -n '2,30p' "$0"; exit 0 ;;
  esac
done

mkdir -p "$BACKUP_DIR" "$LOG_DIR"
chmod 700 "$LOG_DIR" 2>/dev/null || true

# Single combined logger (stdout + file)
exec > >(tee -a "$LOG") 2>&1

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
say() { echo "[update $(ts)] $*"; }
fail() { echo "[update $(ts)] FAIL: $*" >&2; exit 1; }

# Put the previous source (and the dependencies it pinned) back after a red test gate.
# git installs: `git reset --keep` returns to the recorded commit and keeps uncommitted local edits (it
# refuses rather than overwrite them). Tarball installs: the snapshot taken before the sync is unpacked
# over the tree; files that only exist in the new version stay behind.
rollback_source() {
  if [[ -n "$old_full_sha" ]]; then
    (cd "$ROOT" && git reset --keep "$old_full_sha") || return 1
  elif [[ -n "$source_snapshot" && -f "$source_snapshot" ]]; then
    tar -xzf "$source_snapshot" -C "$ROOT" || return 1
  else
    return 1
  fi
  if [[ -x "$PIP" && -f "$ROOT/requirements.txt" ]]; then
    local reqs=(-r "$ROOT/requirements.txt")
    [[ -f "$ROOT/requirements-dev.txt" ]] && reqs+=(-r "$ROOT/requirements-dev.txt")
    "$PIP" install -q "${reqs[@]}" || say "  WARN: could not reinstall the previous dependencies"
    rm -f "$ROOT/.last-requirements.sha256"
  fi
}

CURRENT_VERSION=$("$PY" -c "import sys; sys.path.insert(0,'$ROOT/src'); import version; print(version.VERSION)" 2>/dev/null || echo "unknown")
say "current version: $CURRENT_VERSION"
say "ROOT=$ROOT  PY=$PY"
$DRY_RUN && say "DRY-RUN — no changes will be applied"

# ────────────────────────────────────────────────
# 1. Pre-flight
# ────────────────────────────────────────────────

say "stage 1/7: pre-flight"

# Disk space (need ≥200MB for backup + pip cache)
free_mb=$(df -m "$HOME" | tail -1 | awk '{print $4}')
if [[ "$free_mb" -lt 200 ]]; then
  fail "low disk space: $free_mb MB free, need at least 200"
fi
say "  disk space ok ($free_mb MB free)"

# DB snapshot (compressed)
if [[ -f "$DB" && "$DRY_RUN" == "false" ]]; then
  snap="$BACKUP_DIR/memory.db.$(date +%Y%m%d_%H%M%S).gz"
  # Online backup: a plain cp of a live WAL database misses the -wal file and can be inconsistent.
  python3 - "$DB" "$snap.tmp" <<'PY' || fail "database snapshot failed"
import sqlite3, sys

source = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True, timeout=30)
target = sqlite3.connect(sys.argv[2])
try:
    source.backup(target)
finally:
    target.close()
    source.close()
PY
  gzip -f "$snap.tmp" && mv "$snap.tmp.gz" "$snap"
  say "  db snapshot: $snap ($(du -h "$snap" | cut -f1))"
  # Keep only last 7 backups
  ls -1t "$BACKUP_DIR"/memory.db.*.gz 2>/dev/null | tail -n +8 | xargs -I{} rm -f {} || true
fi

# ────────────────────────────────────────────────
# 2. Source update
# ────────────────────────────────────────────────

say "stage 2/7: source update"
old_sha=""
new_sha=""
old_full_sha=""
source_snapshot=""

if [[ -d "$ROOT/.git" ]]; then
  cd "$ROOT"
  old_sha=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
  old_full_sha=$(git rev-parse HEAD 2>/dev/null || echo "")
  if $DRY_RUN; then
    git fetch --quiet
    behind=$(git rev-list --count HEAD..@{u} 2>/dev/null || echo "?")
    say "  git: $behind commit(s) behind upstream (current=$old_sha)"
  else
    if git pull --ff-only 2>&1 | tail -3; then
      new_sha=$(git rev-parse --short HEAD)
      if [[ "$old_sha" == "$new_sha" ]]; then
        say "  git: already up to date ($old_sha)"
      else
        say "  git: $old_sha → $new_sha"
      fi
    else
      fail "git pull failed (non-fast-forward? local commits?)"
    fi
  fi
elif [[ -n "${UPDATE_URL:-}" ]]; then
  # Security: https-only, SHA-256 pin required, no setuid preservation,
  # no absolute paths in tarball members.
  if [[ "$UPDATE_URL" != https://* ]]; then
    fail "UPDATE_URL must use https:// (got $UPDATE_URL)"
  fi
  if [[ -z "${UPDATE_URL_SHA256:-}" ]]; then
    fail "UPDATE_URL_SHA256 is required for tarball mode (64-hex SHA-256 of the tarball)"
  fi
  if [[ ! "$UPDATE_URL_SHA256" =~ ^[a-fA-F0-9]{64}$ ]]; then
    fail "UPDATE_URL_SHA256 must be a 64-char hex digest"
  fi

  if $DRY_RUN; then
    say "  tarball mode: would download $UPDATE_URL (pinned sha256=$UPDATE_URL_SHA256)"
  else
    tmpdir=$(mktemp -d)
    say "  downloading $UPDATE_URL"
    curl --proto '=https' --tlsv1.2 -fsSL "$UPDATE_URL" \
      -o "$tmpdir/update.tar.gz" || fail "download failed"

    # Verify SHA-256
    actual=$(/usr/bin/shasum -a 256 "$tmpdir/update.tar.gz" | awk '{print $1}')
    # Lowercase via tr for bash 3.2 compat (macOS default — no ${var,,})
    actual_lc=$(printf '%s' "$actual" | tr '[:upper:]' '[:lower:]')
    expected_lc=$(printf '%s' "$UPDATE_URL_SHA256" | tr '[:upper:]' '[:lower:]')
    if [ "$actual_lc" != "$expected_lc" ]; then
      rm -rf "$tmpdir"
      fail "SHA-256 mismatch: expected $UPDATE_URL_SHA256, got $actual"
    fi
    say "  sha256 verified: $actual"

    # Extract safely: refuse absolute paths and owner/permission preservation
    /usr/bin/tar --no-same-owner --no-same-permissions \
      -xzf "$tmpdir/update.tar.gz" -C "$tmpdir" || fail "extract failed"

    # Guard against ../ escape in tarball member names
    if /usr/bin/tar -tzf "$tmpdir/update.tar.gz" | grep -E '(^/|\.\./)' >/dev/null; then
      rm -rf "$tmpdir"
      fail "tarball contains unsafe paths (absolute or parent refs)"
    fi

    inner=$(ls -d "$tmpdir"/*/ 2>/dev/null | head -1)
    [[ -z "$inner" ]] && { rm -rf "$tmpdir"; fail "no top-level dir in tarball"; }
    source_snapshot="$BACKUP_DIR/source.$(date +%Y%m%d_%H%M%S).tar.gz"
    tar -czf "$source_snapshot" --exclude='./.venv' --exclude='./.git' --exclude='__pycache__' -C "$ROOT" . \
      || fail "could not snapshot the current source for rollback"
    say "  source snapshot for rollback: $source_snapshot"
    say "  rsync into $ROOT (preserving .venv)"
    /usr/bin/rsync -a --exclude '.venv/' --exclude '.claude-memory/' --exclude '.tam/' \
      --exclude '__pycache__/' --exclude '.git/' \
      "$inner" "$ROOT/"
    rm -rf "$tmpdir"
  fi
else
  say "  no .git and no UPDATE_URL — skipping source step"
fi

# ────────────────────────────────────────────────
# 3. Deps (only if requirements*.txt changed)
# ────────────────────────────────────────────────

say "stage 3/7: dependencies"
req_files=("$ROOT/requirements.txt")
if [[ -f "$ROOT/requirements-dev.txt" ]]; then
  req_files+=("$ROOT/requirements-dev.txt")
fi

req_hash_path="$ROOT/.last-requirements.sha256"
if [[ -f "${req_files[0]}" ]]; then
  cur_hash=$(
    cat "${req_files[@]}" | \
    (/sbin/md5 -q 2>/dev/null || md5sum | cut -d' ' -f1)
  )
  prev_hash=$(cat "$req_hash_path" 2>/dev/null || echo "")
  if [[ "$cur_hash" != "$prev_hash" ]]; then
    changed_files=$(printf '%s ' "${req_files[@]##*/}")
    if $DRY_RUN; then
      say "  ${changed_files% }changed — would pip install (hash $prev_hash → $cur_hash)"
    else
      say "  ${changed_files% }changed — installing"
      "$PIP" install -q -r "$ROOT/requirements.txt" -r "$ROOT/requirements-dev.txt" || fail "pip install failed"
      echo "$cur_hash" > "$req_hash_path"
    fi
  else
    say "  no dependency changes"
  fi

  # v9 — always re-run editable install so [project.scripts] entry-points
  # (tam, tam-lookup, lookup-memory, total-agent-memory, plus legacy
  # claude-total-memory/ctm-lookup) stay on PATH after src changes.
  # `pip install -e .` is idempotent and fast (just rewrites the
  # console_scripts shims when pyproject.toml is unchanged).
  if $DRY_RUN; then
    say "  would refresh editable install (tam / tam-lookup / lookup-memory entry-points)"
  else
    "$PIP" install -q -e "$ROOT" 2>&1 | tail -1 || say "  WARN: editable install failed; CLI entry-points may be stale"
  fi
fi

# ────────────────────────────────────────────────
# 4. Tests
# ────────────────────────────────────────────────

say "stage 4/7: tests"
if $SKIP_TESTS; then
  say "  skipped (--skip-tests)"
elif $DRY_RUN; then
  say "  would run pytest"
else
  if "$PY" -m pytest "$ROOT/tests/" --tb=no -q 2>&1 | tail -5; then
    say "  tests green"
  else
    if rollback_source; then
      fail "tests failed — rolled back to the previous source and dependencies (DB snapshot kept in $BACKUP_DIR)"
    fi
    fail "tests failed and the automatic rollback was not possible — restore the source by hand (DB snapshot in $BACKUP_DIR)"
  fi
fi

# ────────────────────────────────────────────────
# 5. Schema migrations (auto via Store())
# ────────────────────────────────────────────────

say "stage 5/7: schema migrations"
if $DRY_RUN; then
  say "  would auto-apply pending migrations on next Store() init"
else
  "$PY" - <<'PY' 2>&1 | sed 's/^/  /'
import sys, os
sys.path.insert(0, "src")
import server
s = server.Store()
print("ok")
PY
fi

# Installs that predate `tam setup` stay single-user: record personal mode, no prompt.
if $DRY_RUN; then
  say "  would record personal mode for this install if no setup record exists"
else
  PYTHONPATH="$ROOT/src" "$PY" -c 'import os; from setup_wizard.upgrade import adopt_existing; r = adopt_existing(os.environ); print("setup record: personal mode recorded" if r else "setup record: unchanged")' 2>&1 | sed 's/^/  /'
fi

# Hook copies in ~/.claude/hooks must match the new src/ (HookCall fields, ambient command names).
if $DRY_RUN; then
  say "  would refresh installed Claude Code hook scripts"
else
  PYTHONPATH="$ROOT/src" "$PY" -m setup_wizard.refresh_hooks 2>&1 | sed 's/^/  /' || say "  WARN: hook refresh failed"
fi

# ────────────────────────────────────────────────
# 6. Services (LaunchAgents + dashboard)
# ────────────────────────────────────────────────

say "stage 6/7: services"
PLISTS_SRC="$ROOT/launchagents"
PLISTS_DST="$HOME/Library/LaunchAgents"

reload_agent() {
  local label="$1"
  local plist="$PLISTS_DST/$label.plist"
  [[ -f "$plist" ]] || { say "    $label: not installed, skip"; return; }
  if $DRY_RUN; then
    say "    $label: would reload"
  else
    /bin/launchctl unload "$plist" 2>/dev/null || true
    /bin/launchctl load "$plist" || say "    $label: load failed"
    say "    $label: reloaded"
  fi
}

reload_agent com.claude.memory.reflection
reload_agent com.claude.memory.orphan-backfill
reload_agent com.claude.memory.check-updates

# Dashboard restart (if running)
if /usr/bin/pgrep -f "dashboard\.py" > /dev/null; then
  if $DRY_RUN; then
    say "  dashboard: would restart"
  else
    /usr/bin/pkill -f "dashboard\.py" || true
    sleep 1
    nohup "$PY" "$ROOT/src/dashboard.py" > "$LOG_DIR/dashboard.log" 2>&1 &
    say "  dashboard: restarted on :37737"
  fi
fi

# ────────────────────────────────────────────────
# 7. MCP — only Claude Code can respawn
# ────────────────────────────────────────────────

say "stage 7/7: MCP server"
NEW_VERSION=$("$PY" -c "import sys; sys.path.insert(0,'$ROOT/src'); import version; print(version.VERSION)" 2>/dev/null || echo "?")
if [[ -n "$old_sha$new_sha" || "$NEW_VERSION" != "$CURRENT_VERSION" ]]; then
  echo
  echo "════════════════════════════════════════════════════════════════"
  echo "  Update done: $CURRENT_VERSION → $NEW_VERSION"
  echo "  In Claude Code:  /mcp  → memory → Reconnect"
  echo "════════════════════════════════════════════════════════════════"
  # macOS notification
  /usr/bin/osascript -e 'display notification "MCP needs /mcp reconnect" with title "total-agent-memory updated" sound name "Glass"' 2>/dev/null || true
else
  say "  no source changes — MCP restart not needed"
fi

say "DONE"
