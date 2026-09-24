"""Resolve relative date phrases in recalled records against the time the record was said.

A record like "[8 May 2023] Evan: my son fell off his bike last Thursday" carries
the answer to "when did it happen", but only after date arithmetic that readers
get wrong surprisingly often (wrong weekday, wrong year across New Year). Context
mode annotates each phrase in place with the date it denotes:
"last Thursday [Thu 4 May 2023]".

The anchor is the timestamp the record opens with ("[1:56 pm on 8 May, 2023]",
"[2023/05/20 (Sat) 02:21]", "[2023-05-20]"), which is how transcripts carry the
time a message was sent; without one, the record's `created_at`.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from memory_core.telemetry import counters
from memory_core.temporal.normalizer import normalize

_STAMP = re.compile(r"^\s*\[([^\]\n]{6,40})\]")
_STAMP_FORMATS = (
    "%I:%M %p on %d %B, %Y",
    "%I:%M %p on %d %b, %Y",
    "%Y/%m/%d (%a) %H:%M",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d",
    "%d %B, %Y",
    "%d %B %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%b %d, %Y",
)

_EN_NUMBER = r"(?:\d{1,2}|a|an|one|two|three|four|five|six|seven|eight|nine|ten)"
_EN_WEEKDAY = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_RU_WEEKDAY = r"(?:понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье)"
_PHRASE = re.compile(
    r"(?<![\w\[])(?:"
    r"(?:the\s+)?day\s+before\s+yesterday|yesterday|today|tonight|tomorrow|last\s+night|"
    r"(?:last|this)\s+weekend|"
    r"(?:last|next)\s+(?:week|month|year)|"
    r"(?:last|this|next)\s+" + _EN_WEEKDAY + r"|"
    + _EN_NUMBER + r"\s+(?:days?|weeks?|months?|years?)\s+ago|"
    r"(?:a\s+)?(?:few|couple\s+of)\s+(?:days|weeks)\s+ago|"
    r"позавчера|вчера|сегодня|завтра|"
    r"на\s+(?:прошлой|следующей)\s+неделе|в\s+(?:прошлом|следующем)\s+(?:месяце|году)|"
    r"в\s+(?:прошлый|прошлую|прошлое|следующий|следующую|следующее)\s+" + _RU_WEEKDAY + r"|"
    r"(?:(?:\d{1,2}|один|одну|два|две|три|четыре|пять|шесть|семь|восемь|девять|десять)\s+)?"
    r"(?:день|дня|дней|неделю|недели|недель|месяц|месяца|месяцев|год|года|лет)\s+назад"
    r")(?![\w\]])(?! \[)",
    re.IGNORECASE,
)


def message_time(content: str, created_at: str | None) -> datetime | None:
    """When the record was said: its leading timestamp, else `created_at`."""
    match = _STAMP.match(content or "")
    if match:
        stamp = " ".join(match.group(1).split())
        for fmt in _STAMP_FORMATS:
            try:
                return datetime.strptime(stamp, fmt)  # noqa: DTZ007 — a transcript stamp carries no zone
            except ValueError:
                continue
    if created_at:
        try:
            return datetime.fromisoformat(str(created_at)).replace(tzinfo=None)
        except ValueError:
            return None
    return None


def _day(value: datetime) -> str:
    return f"{value:%a} {value.day} {value:%B %Y}"


def _resolve(phrase: str, anchor: datetime) -> str | None:
    """The date a phrase denotes. Day-level phrases resolve to a day; week-level ones stay
    relative to the message ("the week before Fri 9 June 2023"), because "last week" means the
    previous calendar week to some speakers and the past seven days to others."""
    text = " ".join(phrase.lower().split())
    today = anchor.replace(hour=0, minute=0, second=0, microsecond=0)
    said = _day(today)
    if text in ("tonight", "last night"):
        return _day(today - timedelta(days=1 if text == "last night" else 0))
    if text.endswith("weekend"):
        if text.startswith("this"):
            return f"the weekend of {said}"
        return f"the weekend before {said}"
    if "few" in text or "couple" in text:
        return f"{text.removesuffix(' ago')} before {said}"
    if text.startswith("the day before"):
        text = text[len("the "):]
    if text.split()[0] in ("день", "неделю", "месяц", "год"):
        text = "1 " + text
    resolved = normalize(text, anchor)
    if resolved is None:
        return None
    value = datetime.fromisoformat(resolved.iso)
    if resolved.kind == "day":
        return _day(value)
    if resolved.kind == "week":
        if text.endswith(("ago", "назад")):
            weeks = round((today - value).days / 7)
            if weeks > 1:
                return f"about {weeks} weeks before {said}, around {value.day} {value:%B %Y}"
        return f"the week {'after' if value > today else 'before'} {said}"
    if resolved.kind == "month":
        return f"around {value:%B %Y}" if text.endswith(("ago", "назад")) else f"{value:%B %Y}"
    if resolved.kind == "year":
        return f"around {value:%Y}" if text.endswith(("ago", "назад")) else f"{value:%Y}"
    return _day(value)


def annotate(content: str, created_at: str | None = None) -> str:
    """`content` with each relative date phrase followed by the date it denotes."""
    if not content:
        return content
    anchor = message_time(content, created_at)
    if anchor is None:
        return content
    stamp = _STAMP.match(content)
    start = stamp.end() if stamp else 0
    pieces, last, added = [content[:start]], start, 0
    for match in _PHRASE.finditer(content, start):
        resolved = _resolve(match.group(0), anchor)
        if resolved is None:
            continue
        pieces.append(content[last:match.end()])
        pieces.append(f" [{resolved}]")
        last = match.end()
        added += 1
    if not added:
        return content
    pieces.append(content[last:])
    counters.bump("relative_dates_resolved", added)
    return "".join(pieces)
