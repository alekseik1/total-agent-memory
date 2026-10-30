#!/usr/bin/env bash
# Fails when tracked or new (untracked, not ignored) files carry internal data:
#   rfc1918   private IPv4 addresses (10/8, 172.16/12, 192.168/16)
#   mdns      single-label *.local host names
#   operator  host names from the operator list (one per line, # comments)
#   apikey    credential shapes (Anthropic/OpenAI/GitHub/AWS/Slack/Google keys, JWTs, PEM private keys)
#
# A finding is accepted only by an entry in the allow-list file; every entry needs a reason:
#   <path glob> :: <ERE the matched text must match completely> :: <one-line reason>
#
# Usage: scripts/check-no-internal-data.sh [--fail-on-stale]
#   --fail-on-stale   also fail when an allow-list entry matched nothing (keeps the list honest)
#
# Environment:
#   TAM_INTERNAL_DATA_ALLOWLIST  allow-list file       (default scripts/internal-data-allowlist.txt)
#   TAM_INTERNAL_HOSTS_FILE      operator host list    (default scripts/internal-data-hosts.local.txt,
#                                gitignored: the host names stay out of the repository)
#   TAM_INTERNAL_HOSTS           extra host names, comma separated (CI repository variable TAM_INTERNAL_HOSTS)
#
# Exit status: 0 clean, 1 findings (or stale entries with --fail-on-stale), 2 bad configuration
# or a failing git grep (never read as "clean").
set -euo pipefail

EXIT_FOUND=1
EXIT_CONFIG=2
REASON_MIN_CHARS=12
SEPARATOR=" :: "

cd "$(git rev-parse --show-toplevel)"
ALLOWLIST="${TAM_INTERNAL_DATA_ALLOWLIST:-scripts/internal-data-allowlist.txt}"
HOSTS_FILE="${TAM_INTERNAL_HOSTS_FILE:-scripts/internal-data-hosts.local.txt}"
FAIL_ON_STALE=0
for arg in "$@"; do
    case "$arg" in
        --fail-on-stale) FAIL_ON_STALE=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit $EXIT_CONFIG ;;
    esac
done

config_error() { echo "check-no-internal-data: $*" >&2; exit $EXIT_CONFIG; }

# ── allow-list ───────────────────────────────────────────────────
allow_globs=()
allow_eres=()
allow_hits=()
if [ -f "$ALLOWLIST" ]; then
    number=0
    while IFS= read -r line || [ -n "$line" ]; do
        number=$((number + 1))
        # Windows checkouts with core.autocrlf=true end every line with CR.
        line=${line%$'\r'}
        case "$line" in ''|'#'*) continue ;; esac
        case "$line" in *"$SEPARATOR"*"$SEPARATOR"*) ;; *)
            config_error "$ALLOWLIST:$number: expected '<glob>$SEPARATOR<ERE>$SEPARATOR<reason>'" ;;
        esac
        glob=${line%%"$SEPARATOR"*}
        rest=${line#*"$SEPARATOR"}
        ere=${rest%%"$SEPARATOR"*}
        reason=${rest#*"$SEPARATOR"}
        [ -n "$glob" ] && [ -n "$ere" ] || config_error "$ALLOWLIST:$number: empty glob or pattern"
        [ "${#reason}" -ge "$REASON_MIN_CHARS" ] || config_error "$ALLOWLIST:$number: a one-line reason is required"
        allow_globs+=("$glob")
        allow_eres+=("$ere")
        allow_hits+=(0)
    done < "$ALLOWLIST"
fi

is_allowed() {
    local file="$1" text="$2" i
    for i in "${!allow_globs[@]}"; do
        # shellcheck disable=SC2254 # the glob is data from the allow-list on purpose
        case "$file" in ${allow_globs[$i]}) ;; *) continue ;; esac
        if [[ "$text" =~ ^(${allow_eres[$i]})$ ]]; then
            allow_hits[$i]=$((allow_hits[i] + 1))
            return 0
        fi
    done
    return 1
}

# ── operator host list → one ERE alternation ─────────────────────
operator_hosts=()
if [ -f "$HOSTS_FILE" ]; then
    while IFS= read -r host || [ -n "$host" ]; do
        host=${host%%#*}
        host=$(printf '%s' "$host" | tr -d '[:space:]')
        [ -n "$host" ] && operator_hosts+=("$host")
    done < "$HOSTS_FILE"
fi
if [ -n "${TAM_INTERNAL_HOSTS:-}" ]; then
    IFS=',' read -r -a extra_hosts <<< "$TAM_INTERNAL_HOSTS"
    for host in "${extra_hosts[@]}"; do
        host=$(printf '%s' "$host" | tr -d '[:space:]')
        [ -n "$host" ] && operator_hosts+=("$host")
    done
fi
operator_ere=""
for host in "${operator_hosts[@]+"${operator_hosts[@]}"}"; do
    escaped=$(printf '%s' "$host" | sed -e 's/[][\.*^$+?(){}|/]/\\&/g')
    operator_ere="${operator_ere:+$operator_ere|}$escaped"
done

# ── scanning ─────────────────────────────────────────────────────
findings=0
IP_EDGE='s/^[^0-9]+//; s/[^0-9]+$//'
NAME_EDGE='s/^[^A-Za-z0-9]+//; s/[^A-Za-z0-9]+$//'
NO_EDGE='s/^//'

scan() {
    local class="$1" ere="$2" edge="$3" case_flag="${4:-}" output file rest line text status=0
    # git grep exits 1 for "no match" and >1 for a real error (bad pattern, unreadable index).
    output=$(git grep -I -n -o -E ${case_flag:+"$case_flag"} --untracked --exclude-standard -e "$ere") || status=$?
    if [ "$status" -gt 1 ]; then
        config_error "git grep failed (exit $status) while scanning for class '$class'"
    fi
    [ -n "$output" ] || return 0
    while IFS= read -r row; do
        file=${row%%:*}
        rest=${row#*:}
        line=${rest%%:*}
        text=$(printf '%s' "${rest#*:}" | sed -E "$edge")
        if ! is_allowed "$file" "$text"; then
            printf '%s:%s: %s: %s\n' "$file" "$line" "$class" "$text"
            findings=$((findings + 1))
        fi
    done <<< "$output"
}

PRIVATE_IPV4='(^|[^0-9.])(10(\.[0-9]{1,3}){3}|172\.(1[6-9]|2[0-9]|3[01])(\.[0-9]{1,3}){2}|192\.168(\.[0-9]{1,3}){2})([^0-9]|$)'
MDNS_HOST='(^|[^A-Za-z0-9_.-])[A-Za-z0-9][A-Za-z0-9-]*\.local([^A-Za-z0-9_/.-]|$)'
API_KEYS='sk-ant-[A-Za-z0-9_-]{20,}|sk-(proj-)?[A-Za-z0-9_-]{32,}|gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,}|AKIA[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}|eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----'

scan rfc1918 "$PRIVATE_IPV4" "$IP_EDGE"
scan mdns "$MDNS_HOST" "$NAME_EDGE"
if [ -n "$operator_ere" ]; then
    scan operator "(^|[^A-Za-z0-9_-])($operator_ere)([^A-Za-z0-9_-]|$)" "$NAME_EDGE" -i
fi
scan apikey "$API_KEYS" "$NO_EDGE"

stale=0
for i in "${!allow_globs[@]}"; do
    if [ "${allow_hits[$i]}" -eq 0 ]; then
        echo "stale allow-list entry (matched nothing): ${allow_globs[$i]}$SEPARATOR${allow_eres[$i]}" >&2
        stale=$((stale + 1))
    fi
done

if [ "$findings" -gt 0 ]; then
    echo "check-no-internal-data: $findings finding(s). Remove the data, or add a reasoned entry to $ALLOWLIST." >&2
    exit $EXIT_FOUND
fi
if [ "$stale" -gt 0 ] && [ "$FAIL_ON_STALE" -eq 1 ]; then
    exit $EXIT_FOUND
fi
echo "check-no-internal-data: clean"
