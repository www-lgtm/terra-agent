"""Time expression parser for the scheduling module.

Supports:
- Standard 5-field cron: "minute hour dom month dow" (e.g. "0 9 * * *")
- Interval strings: "30m", "2h", "1d", "90s"
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

from croniter import croniter

_INTERVAL_RE = re.compile(r'^(\d+)\s*(s|m|h|d)$', re.IGNORECASE)
_SECONDS_PER_UNIT: dict[str, int] = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_interval(interval_str: str) -> Optional[float]:
    """Parse '30m', '2h', '1d', '90s' into seconds. Returns None on failure."""
    m = _INTERVAL_RE.match(interval_str.strip())
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2).lower()
    return float(value * _SECONDS_PER_UNIT[unit])


def next_cron_time(cron_expr: str, from_time: datetime | None = None) -> datetime:
    """Return the next datetime matching the cron expression."""
    base = from_time or datetime.now()
    return croniter(cron_expr, base).get_next(datetime)  # type: ignore[no-any-return]


def next_interval_time(interval_seconds: float, from_time: datetime | None = None) -> datetime:
    """Return the next datetime after adding the interval."""
    base = from_time or datetime.now()
    return base + timedelta(seconds=interval_seconds)


def calculate_next_run(schedule_type: str, schedule_value: str,
                       from_time: datetime | None = None) -> datetime:
    """Unified entry: calculate the next execution time.

    Args:
        schedule_type: "cron" or "interval"
        schedule_value: Cron expression ("0 9 * * *") or interval string ("30m")
        from_time: Base time for calculation (default: now)

    Returns:
        Next run datetime

    Raises:
        ValueError: If schedule_type is unknown or schedule_value is invalid
    """
    if schedule_type == "cron":
        return next_cron_time(schedule_value, from_time)
    elif schedule_type == "interval":
        seconds = parse_interval(schedule_value)
        if seconds is None:
            raise ValueError(f"Invalid interval value: {schedule_value!r}")
        return next_interval_time(seconds, from_time)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type!r}. Use 'cron' or 'interval'.")
