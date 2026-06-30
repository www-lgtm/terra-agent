"""Unit tests for src.scheduler.time_parser."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.scheduler.time_parser import (
    calculate_next_run,
    next_cron_time,
    next_interval_time,
    parse_interval,
)


class TestParseInterval:
    def test_valid_minutes(self) -> None:
        assert parse_interval("30m") == 1800.0
        assert parse_interval("5m") == 300.0

    def test_valid_hours(self) -> None:
        assert parse_interval("2h") == 7200.0
        assert parse_interval("1h") == 3600.0

    def test_valid_days(self) -> None:
        assert parse_interval("1d") == 86400.0
        assert parse_interval("7d") == 604800.0

    def test_valid_seconds(self) -> None:
        assert parse_interval("90s") == 90.0
        assert parse_interval("10s") == 10.0

    def test_case_insensitive(self) -> None:
        assert parse_interval("30M") == 1800.0
        assert parse_interval("2H") == 7200.0

    def test_with_spaces(self) -> None:
        assert parse_interval(" 30m ") == 1800.0
        assert parse_interval("2 h") == 7200.0

    def test_invalid_empty(self) -> None:
        assert parse_interval("") is None

    def test_invalid_garbage(self) -> None:
        assert parse_interval("foo") is None
        assert parse_interval("30x") is None
        assert parse_interval("abc123") is None


class TestNextCronTime:
    def test_daily_9am(self) -> None:
        now = datetime(2026, 6, 4, 10, 0, 0)
        result = next_cron_time("0 9 * * *", from_time=now)
        assert result.hour == 9
        assert result.minute == 0
        assert result > now  # Should be tomorrow

    def test_daily_9am_from_early_morning(self) -> None:
        now = datetime(2026, 6, 4, 6, 0, 0)
        result = next_cron_time("0 9 * * *", from_time=now)
        assert result.hour == 9
        assert result.minute == 0
        assert result.day == 4  # Same day

    def test_every_30_minutes(self) -> None:
        now = datetime(2026, 6, 4, 10, 12, 0)
        result = next_cron_time("*/30 * * * *", from_time=now)
        assert result.minute == 30

    def test_every_monday(self) -> None:
        # 2026-06-04 is Thursday, next Monday is 06-08
        now = datetime(2026, 6, 4, 10, 0, 0)
        result = next_cron_time("0 8 * * 1", from_time=now)
        assert result.hour == 8
        assert result.minute == 0
        assert result.weekday() == 0  # Monday
        assert result.day == 8


class TestNextIntervalTime:
    def test_30_minutes(self) -> None:
        now = datetime(2026, 6, 4, 10, 0, 0)
        result = next_interval_time(1800.0, from_time=now)
        expected = now + timedelta(seconds=1800)
        assert result == expected

    def test_2_hours(self) -> None:
        now = datetime(2026, 6, 4, 10, 0, 0)
        result = next_interval_time(7200.0, from_time=now)
        expected = now + timedelta(seconds=7200)
        assert result == expected


class TestCalculateNextRun:
    def test_cron_type(self) -> None:
        result = calculate_next_run("cron", "0 9 * * *")
        assert result.hour == 9
        assert result.minute == 0

    def test_interval_type(self) -> None:
        before = datetime.now()
        result = calculate_next_run("interval", "30m")
        after = datetime.now()
        # Result should be ~30 minutes from now
        assert before + timedelta(seconds=1798) < result < after + timedelta(seconds=1802)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown schedule_type"):
            calculate_next_run("unknown", "whatever")

    def test_invalid_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid interval"):
            calculate_next_run("interval", "not-an-interval")
