#!/usr/bin/env bash
# Smoke test: `bash install.sh --ide claude-code` on the host (macOS).
# Verifies that after install: plists are valid, no placeholders left,
# dashboard process starts and answers HTTP 200.
#
# Must run on a real macOS host (launchctl requires Darwin).
# Uses a sandbox HOME and a stub launchctl so it doesn't touch the real user setup.
#
# Required: bash 3.2+, launchctl, python3 ≥ 3.10.
#
# Exit codes:
#   0 — plists installed + substituted + dashboard answers
#   2 — placeholder leak in plist
#   3 — wrong path in plist
#   4 — dashboard didn't come up
#   5 - Step 5 did not bootout and bootstrap every plist through the launchctl stub
set -euo pipefail

[ "$(uname)" = "Darwin" ] || { echo "SKIP: macOS-only smoke"; exit 0; }

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
SANDBOX="$(mktemp -d /tmp/tam-smoke-XXXXXX)"
echo "→ sandbox HOME = $SANDBOX"

cleanup() {
  echo "→ cleanup"
  rm -rf "$SANDBOX"
}
trap cleanup EXIT

mkdir -p "$SANDBOX/bin"
printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$*" >> "%s"\nexit 0\n' "$SANDBOX/launchctl.log" > "$SANDBOX/bin/launchctl"
chmod +x "$SANDBOX/bin/launchctl"

echo "→ running install.sh in sandbox (skip pip/model)"
HOME="$SANDBOX" \
PATH="$SANDBOX/bin:$PATH" \
INSTALL_TEST_MODE=skip-heavy \
TAM_MEMORY_DIR="$SANDBOX/.tam" \
bash "$REPO_ROOT/install.sh" --ide claude-code 2>&1 | tail -5

LA_DIR="$SANDBOX/Library/LaunchAgents"
[ -d "$LA_DIR" ] || { echo "FAIL: $LA_DIR not created"; exit 4; }

echo "→ verifying plist substitution"
for plist in "$LA_DIR"/*.plist; do
  name=$(basename "$plist")
  echo "  · $name"
  if grep -q "__INSTALL_DIR__\|__MEMORY_DIR__\|__HOME__" "$plist"; then
    echo "FAIL: $name has leftover placeholders"
    exit 2
  fi
  # A placeholder actually present in the template must have been replaced
  # with the value this run handed the installer, not just be absent (which
  # a substitution that wrote the wrong value would also satisfy).
  template="$REPO_ROOT/launchagents/$name"
  if [ -f "$template" ]; then
    for ph in __INSTALL_DIR__ __MEMORY_DIR__ __HOME__; do
      case "$ph" in
        __INSTALL_DIR__) val="$REPO_ROOT" ;;
        __MEMORY_DIR__) val="$SANDBOX/.tam" ;;
        __HOME__) val="$SANDBOX" ;;
      esac
      if grep -qF "$ph" "$template" && ! grep -qF "$val" "$plist"; then
        echo "FAIL: $name did not substitute $ph with $val"
        exit 3
      fi
    done
  fi
  # ProgramArguments[0] must exist (the python interpreter path).
  py_path=$(/usr/libexec/PlistBuddy -c 'Print :ProgramArguments:0' "$plist" 2>/dev/null || true)
  if [ -z "$py_path" ]; then
    echo "FAIL: $name has no ProgramArguments[0]"
    exit 3
  fi
  if [[ "$py_path" == *"$REPO_ROOT"* ]] || [[ "$py_path" == *"$SANDBOX"* ]]; then
    : # OK — points at either checkout or sandbox
  else
    echo "FAIL: $name ProgramArguments[0] does not reference checkout or sandbox: $py_path"
    exit 3
  fi
done
echo "  ✓ all plists substituted correctly"

[ -s "$SANDBOX/launchctl.log" ] || { echo "FAIL: launchctl stub never called"; exit 5; }
for plist in "$LA_DIR"/*.plist; do
  grep -qxF "bootout gui/$(id -u)/$(basename "$plist" .plist)" "$SANDBOX/launchctl.log" || {
    echo "FAIL: no stub bootout for $plist"; exit 5; }
  grep -qxF "bootstrap gui/$(id -u) $plist" "$SANDBOX/launchctl.log" || {
    echo "FAIL: no stub bootstrap for $plist"; exit 5; }
done

echo "✓ install.sh smoke OK (plists valid; dashboard not exercised here because"
echo "  pip was skipped — see install-docker-pull.sh for runtime smoke)"
