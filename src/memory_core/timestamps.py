"""One timestamp format for stored records: UTC, ISO 8601, microseconds, ``Z``.

``2026-09-21T08:21:37.622445Z`` — every value has the same width, so string
order equals time order in SQL, and readers see one unambiguous form.
Migration 034 rewrites older rows into exactly this shape.
"""

from __future__ import annotations

from datetime import UTC, datetime

CANONICAL_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def format_utc(moment: datetime) -> str:
    """Canonical text for an aware datetime; naive values are rejected, not guessed."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("Timestamps must be timezone-aware")
    return moment.astimezone(UTC).strftime(CANONICAL_FORMAT)


def utc_now() -> str:
    return format_utc(datetime.now(UTC))


def normalize_timestamp(value: str) -> str:
    """Canonical form of a stored timestamp.

    Accepts the shapes older versions wrote: ``…Z``, ``…+00:00`` and the
    zone-less ``YYYY-MM-DD HH:MM:SS`` that Python ``datetime.now()`` produced,
    which is the machine's local time and is converted with its DST rules.
    """
    text = value.strip()
    if not text:
        raise ValueError("Empty timestamp")
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return format_utc(moment)
