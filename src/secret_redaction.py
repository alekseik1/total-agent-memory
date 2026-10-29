"""Credential redaction shared by every path that writes user text to disk.

Tool arguments, stored records, captured prompts, queued tool output and
transcripts all go through `redact_secrets`, so one list decides what never
reaches `memory.db`, the raw call log or the extract queue.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from privacy_filter import redact_private_sections

REDACTED = "[REDACTED]"


def _luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _redact_card(match: re.Match[str]) -> str:
    # Only numbers that pass the Luhn check are card numbers; order ids,
    # timestamps and other 16-digit values stay readable.
    digits = re.sub(r"\D", "", match.group(0))
    return REDACTED if _luhn_valid(digits) else match.group(0)




def strip_private_blocks(text: str, placeholder: str = REDACTED) -> str:
    """Replace each outermost <private> block with `placeholder`.

    Case-insensitive, tolerant of spaces inside the tags, nesting-safe; an unterminated block hides the rest
    (failing closed beats leaking the tail). This is the single implementation every writer uses.
    """
    return redact_private_sections(text, placeholder)[0]


class _PrivateBlockStripper:
    """`re.Pattern.sub`-shaped adapter over `strip_private_blocks` for callers that hold a pattern name."""

    @staticmethod
    def sub(repl: str, text: str) -> str:
        return strip_private_blocks(text, repl)


PRIVATE_BLOCK_RE = _PrivateBlockStripper()

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# (pattern, replacement). Order matters: specific key formats run before the
# generic `name = value` rules so a key is replaced whole, not half.
SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]], ...] = (
    # PEM private key blocks, including a header whose END line is missing.
    (re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----(?:.*?-----END [A-Z0-9 ]*PRIVATE KEY-----|.*\Z)",
                re.DOTALL), REDACTED),
    # Credentials embedded in a URL: scheme://user:password@host
    (re.compile(r"\b([a-z][a-z0-9+.-]*://)[^\s:/@]+:[^\s@/]+@", re.IGNORECASE), r"\1" + REDACTED + "@"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"), REDACTED),                      # Anthropic
    (re.compile(r"\bsk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}"), REDACTED),   # OpenAI project/service keys
    (re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"), REDACTED),   # Stripe
    (re.compile(r"\b(?:sk|pk|api[_-]?key)[_-]?[A-Za-z0-9]{20,}", re.IGNORECASE), REDACTED),  # generic sk-/pk-/apikey
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}"), REDACTED),                   # GitHub fine-grained PAT
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"), REDACTED),                     # GitHub classic tokens
    (re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"), REDACTED),                       # GitLab PAT
    (re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"), REDACTED),                  # Slack tokens
    (re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]+"), REDACTED),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}"), REDACTED),                          # Google API key
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), REDACTED),                            # Hugging Face
    (re.compile(r"\bnpm_[A-Za-z0-9]{36}"), REDACTED),                            # npm
    (re.compile(r"\b\d{8,10}:AA[0-9A-Za-z_-]{33}\b"), REDACTED),                 # Telegram bot
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), REDACTED),                    # AWS access key id
    (re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}(?:\.[A-Za-z0-9_-]+)?"), REDACTED),  # JWT
    # Bearer values: 8+ token characters containing an upper-case letter, digit or + / =, or 16+ plain
    # lower-case characters. A snake_case identifier ("refresh_token_validation_flow") is prose and survives.
    (re.compile(r"\b(?i:bearer)\s+(?:(?=[A-Za-z0-9._~+/-]*[A-Z0-9+/=])[A-Za-z0-9._~+/-]{8,}"
                r"|(?![a-z]+(?:_[a-z]+)+(?![A-Za-z0-9._~+/=-]))[A-Za-z0-9._~+/-]{16,})=*"), REDACTED),
    (re.compile(r"\bauthorization\s*:\s*(?:(?:basic|bearer|digest|token)\s+\S+|[A-Za-z0-9._~+/=-]{20,})",
                re.IGNORECASE), REDACTED),
    # name = value / name: value for secret-bearing names, incl. aws_secret_access_key, client_secret.
    (re.compile(r"\b(?:[A-Za-z0-9]+[_-])*(?:secret|password|passwd|pwd)(?:[_-][A-Za-z0-9]+)*\s*[:=]\s*\S+",
                re.IGNORECASE), REDACTED),
    # api_key / access_key / private_key: any value. token: any value after "=", and after ":" a
    # credential-shaped one (8+ characters with an upper-case letter, digit or + / =, or 20+ characters),
    # so "token: validation" stays prose while "token=mysecret" does not.
    (re.compile(r"\b(?i:api[_-]?key|access[_-]?key|private[_-]?key)[\"']?\s*[:=]\s*[\"']?[^\s\"']+[\"']?"), REDACTED),
    (re.compile(r"\b(?i:token)[\"']?\s*=\s*[\"']?[^\s\"']+[\"']?"), REDACTED),
    (re.compile(r"\b(?i:token)[\"']?\s*:\s*[\"']?"
                r"(?:(?=[^\s\"']*[A-Z0-9+/=])[^\s\"']{8,}|[^\s\"']{20,})[\"']?"), REDACTED),
    (EMAIL_PATTERN, REDACTED),                                                    # e-mail addresses
    # Payment card numbers: a standalone run of 16 digits, not a piece of a
    # UUID or another hyphenated or alphanumeric identifier.
    (re.compile(r"(?<![\w-])\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}(?![\w-])"), _redact_card),
)

# Argument keys that carry filesystem locations, never user prose. A path may
# legitimately contain an '@' or look like a key=value pair.
PATH_KEYS = frozenset({"path", "paths", "file", "file_path", "root", "dataset_path", "dir", "directory"})


# Argument keys that name things to look up or link (query text, KG subjects and objects,
# entity names). An e-mail address there is an identifier, and replacing it would collapse
# distinct subjects into one "[REDACTED]" node; credentials in them are still redacted.
IDENTIFIER_KEYS = frozenset({"query", "q", "find", "subject", "object", "entity", "entity_name", "name",
                             "source", "target", "from", "to"})


def redact_secrets(text: str, keep_emails: bool = False) -> tuple[str, bool]:
    """Return `text` with every recognised credential replaced, and whether anything changed.

    `keep_emails` leaves e-mail addresses alone (identifier fields); every other pattern still applies.
    """
    if not text or not isinstance(text, str):
        return text, False
    cleaned = text
    for pattern, replacement in SECRET_PATTERNS:
        if keep_emails and pattern is EMAIL_PATTERN:
            continue
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned, cleaned != text


def redact_value(value: Any, key: str | None = None) -> tuple[Any, bool]:
    """Redact every string inside a JSON-like value; path-valued keys are left alone."""
    if key in PATH_KEYS:
        return value, False
    if isinstance(value, str):
        return redact_secrets(value, keep_emails=key in IDENTIFIER_KEYS)
    if isinstance(value, dict):
        changed = False
        out = {}
        for k, v in value.items():
            out[k], hit = redact_value(v, k)
            changed = changed or hit
        return out, changed
    if isinstance(value, (list, tuple)):
        changed = False
        items = []
        for v in value:
            new, hit = redact_value(v, key)
            items.append(new)
            changed = changed or hit
        return (type(value)(items) if isinstance(value, tuple) else items), changed
    return value, False
