from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta


@dataclass(frozen=True)
class EventTime:
    start: str = ""
    end: str = ""
    precision: str = "unknown"


def normalize_event_time(expression: str, observed_at: str) -> EventTime:
    text = expression.strip().casefold()
    if not text:
        return EventTime()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            day = date.fromisoformat(text)
            return EventTime(day.isoformat(), day.isoformat(), "day")
        if re.fullmatch(r"\d{4}", text):
            year = int(text)
            return EventTime(
                date(year, 1, 1).isoformat(), date(year, 12, 31).isoformat(), "year"
            )
        if re.fullmatch(r"\d{4}-\d{2}", text):
            year, month = map(int, text.split("-"))
            return EventTime(
                date(year, month, 1).isoformat(),
                date(year, month, calendar.monthrange(year, month)[1]).isoformat(),
                "month",
            )
        observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00")).date()
        offsets = {
            "today": 0,
            "сегодня": 0,
            "yesterday": -1,
            "вчера": -1,
            "tomorrow": 1,
            "завтра": 1,
        }
        if text in offsets:
            day = observed + timedelta(days=offsets[text])
            return EventTime(day.isoformat(), day.isoformat(), "day")
        if text in ("last year", "в прошлом году"):
            return EventTime(
                date(observed.year - 1, 1, 1).isoformat(),
                date(observed.year - 1, 12, 31).isoformat(),
                "year",
            )
        if text in ("this week", "на этой неделе", "last week", "на прошлой неделе"):
            start = observed - timedelta(days=observed.weekday())
            if text in ("last week", "на прошлой неделе"):
                start -= timedelta(days=7)
            return EventTime(
                start.isoformat(), (start + timedelta(days=6)).isoformat(), "week"
            )
    except (ValueError, OverflowError):
        return EventTime()
    return EventTime()
