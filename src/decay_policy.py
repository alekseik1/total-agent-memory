"""Which record types resist query-time recency decay, and how far.

Recall multiplies each hit by ``exp(-days * ln2 / half_life)``. That is right
for observations and facts that go stale, wrong for a decision or a convention
that stays in force until something replaces it: after a year a 90-day
half-life leaves 6 % of its score and the record falls out of the top results.
Types listed here never decay below ``DEFAULT_EXEMPT_FLOOR``; supersession,
not age, retires them.

Settings (environment / local settings):
  MEMORY_DECAY_EXEMPT_TYPES  comma-separated types (default decision,convention,lesson)
  MEMORY_DECAY_EXEMPT_FLOOR  minimum decay factor for those types, 0..1 (default 0.85)
"""

from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_EXEMPT_TYPES = ("decision", "convention", "lesson")
DEFAULT_EXEMPT_FLOOR = 0.85
EXEMPT_TYPES_ENV = "MEMORY_DECAY_EXEMPT_TYPES"
EXEMPT_FLOOR_ENV = "MEMORY_DECAY_EXEMPT_FLOOR"


def exempt_types(environ: Mapping[str, str] | None = None) -> frozenset[str]:
    env = os.environ if environ is None else environ
    raw = env.get(EXEMPT_TYPES_ENV)
    if raw is None:
        return frozenset(DEFAULT_EXEMPT_TYPES)
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def exempt_floor(environ: Mapping[str, str] | None = None) -> float:
    env = os.environ if environ is None else environ
    raw = env.get(EXEMPT_FLOOR_ENV, "").strip()
    if not raw:
        return DEFAULT_EXEMPT_FLOOR
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{EXEMPT_FLOOR_ENV} must be within [0, 1], got {value}")
    return value


def floor_for(ktype: str | None, environ: Mapping[str, str] | None = None) -> float | None:
    """Decay floor for ``ktype``, or None when the type decays freely."""
    if not ktype or ktype.lower() not in exempt_types(environ):
        return None
    try:
        return exempt_floor(environ)
    except ValueError as error:
        # A bad setting must not break recall: journal it, use the default.
        from error_journal import log_error
        log_error("decay_policy", error, f"invalid {EXEMPT_FLOOR_ENV}, using {DEFAULT_EXEMPT_FLOOR}")
        return DEFAULT_EXEMPT_FLOOR
