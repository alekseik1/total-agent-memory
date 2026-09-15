#!/usr/bin/env bash
set -euo pipefail

TAM_CHECK_SOURCE="$(cd "$(dirname "$0")/../.." && pwd)"
export TAM_CHECK_SOURCE
TAM_CHECK_PROJECT="tam-check-$(date +%s)-$$"
export TAM_TEAM_DATA_VOLUME="$TAM_CHECK_PROJECT-data"
export TAM_TEAM_MODEL_VOLUME="$TAM_CHECK_PROJECT-models"
export TAM_TEAM_PORT=0
COMPOSE=(docker compose -p "$TAM_CHECK_PROJECT" -f "$TAM_CHECK_SOURCE/docker-compose.team.yml" -f "$TAM_CHECK_SOURCE/tests/smoke/compose-check.yml")

cleanup() {
  result=$?
  if [ "$result" -ne 0 ]; then
    "${COMPOSE[@]}" logs --no-color
  fi
  "${COMPOSE[@]}" down --remove-orphans
  exit "$result"
}
trap cleanup EXIT

"${COMPOSE[@]}" config --quiet
"${COMPOSE[@]}" up -d --no-build --wait --wait-timeout 180
"${COMPOSE[@]}" exec -T team-memory python /checks/tests/smoke/installed_runtime.py --layout image --root /team-data --server-url http://127.0.0.1:3737
"${COMPOSE[@]}" stop
"${COMPOSE[@]}" up -d --no-build --wait --wait-timeout 180
"${COMPOSE[@]}" exec -T team-memory python /checks/tests/smoke/installed_runtime.py --layout image --root /team-data --server-url http://127.0.0.1:3737 --verify-existing
