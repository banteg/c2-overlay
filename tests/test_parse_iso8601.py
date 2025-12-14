from datetime import UTC

from c2_overlay import parse_iso8601


def test_parse_iso8601_trims_fractional_seconds() -> None:
    dt = parse_iso8601("2025-12-14T10:41:31.123456789Z")
    assert dt.tzinfo == UTC
    assert dt.microsecond == 123456


def test_parse_iso8601_handles_offset_without_colon() -> None:
    dt = parse_iso8601("2025-12-14T14:41:31.000000+0400")
    assert dt.tzinfo == UTC
    assert (dt.hour, dt.minute, dt.second) == (10, 41, 31)
