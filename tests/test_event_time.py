import pytest

from memory_core.event_time import normalize_event_time


@pytest.mark.parametrize(
    "expression,observed,start,end,precision",
    [
        ("вчера", "2026-03-01T10:00:00Z", "2026-02-28", "2026-02-28", "day"),
        ("last year", "2026-01-01", "2025-01-01", "2025-12-31", "year"),
        ("2024-02", "", "2024-02-01", "2024-02-29", "month"),
        ("на прошлой неделе", "2026-01-01", "2025-12-22", "2025-12-28", "week"),
        ("2026-05-15", "", "2026-05-15", "2026-05-15", "day"),
        ("recently", "2026-01-01", "", "", "unknown"),
        ("yesterday", "", "", "", "unknown"),
        ("2026-02-30", "", "", "", "unknown"),
    ],
)
def test_time_resolution_is_bounded_and_does_not_guess(
    expression, observed, start, end, precision
):
    value = normalize_event_time(expression, observed)
    assert (value.start, value.end, value.precision) == (start, end, precision)
