from datetime import UTC, datetime

import pytest

from c2_overlay import parse_iso8601


def test_parse_iso8601_trims_fractional_seconds() -> None:
    dt = parse_iso8601("2025-12-14T10:41:31.123456789Z")
    assert dt.tzinfo == UTC
    assert dt.microsecond == 123456


def test_parse_iso8601_handles_offset_without_colon() -> None:
    dt = parse_iso8601("2025-12-14T14:41:31.000000+0400")
    assert dt.tzinfo == UTC
    assert (dt.hour, dt.minute, dt.second) == (10, 41, 31)


def test_parse_iso8601_handles_offset_with_colon() -> None:
    dt = parse_iso8601("2025-12-14T14:41:31+04:00")
    assert dt == datetime(2025, 12, 14, 10, 41, 31, tzinfo=UTC)


def test_parse_iso8601_assumes_utc_when_timezone_missing() -> None:
    dt = parse_iso8601("2025-12-14T10:41:31")
    assert dt == datetime(2025, 12, 14, 10, 41, 31, tzinfo=UTC)


def test_parse_iso8601_strips_whitespace() -> None:
    dt = parse_iso8601(" 2025-12-14T10:41:31Z ")
    assert dt == datetime(2025, 12, 14, 10, 41, 31, tzinfo=UTC)


@pytest.mark.parametrize("s", ["", "nope", "2025-99-99T99:99:99Z"])
def test_parse_iso8601_invalid_raises(s: str) -> None:
    with pytest.raises(ValueError):
        parse_iso8601(s)
